"""
Training script for Transformer Detectors (DETR & Deformable DETR).
Described in Section 3.2 & 3.3 of the paper.

Usage:
    # Train DETR (N = 100 queries):
    python -m multi_model_detection.scripts.train_detr --model detr

    # Train Deformable DETR (N = 300 queries, multi-scale):
    python -m multi_model_detection.scripts.train_detr --model deformable_detr
"""

import argparse
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from multi_model_detection.config_loader import load_config
from multi_model_detection.configs.constants import NUM_CLASSES
from multi_model_detection.data.dataset import WaymoMultiParadigmDataset
from multi_model_detection.data.split import create_train_val_split
from multi_model_detection.models.detr import DETR, SetCriterion
from multi_model_detection.models.deformable_detr import DeformableDETR
from multi_model_detection.models.matcher import (
    HungarianMatcher,
    box_xyxy_to_cxcywh,
    box_cxcywh_to_xyxy,
)
from multi_model_detection.utils.helpers import (
    get_device,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    collate_detection,
)
from multi_model_detection.utils.metrics import evaluate_detection_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR / Deformable DETR")
    parser.add_argument(
        "--model",
        type=str,
        default="detr",
        choices=["detr", "deformable_detr"],
        help="Model variant: 'detr' or 'deformable_detr'",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return parser.parse_args()


def prepare_transformer_targets(targets, img_w, img_h, device):
    """Normalize box coordinates to [0, 1] cxcywh for DETR set prediction."""
    t_list = []
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    for t in targets:
        boxes = t["boxes"].to(device)
        labels = t["labels"].to(device)
        if len(boxes) > 0:
            norm_xyxy = boxes / scale
            norm_cxcywh = box_xyxy_to_cxcywh(norm_xyxy)
        else:
            norm_cxcywh = torch.zeros((0, 4), dtype=torch.float32, device=device)
        t_list.append({"boxes": norm_cxcywh, "labels": labels})
    return t_list


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.device.preferred)
    logger = setup_logging(cfg.logging.log_dir, name=f"train_{args.model}")

    logger.info("=" * 60)
    logger.info(f"Training Paradigm: {args.model.upper()} (Set-Prediction Transformer)")
    logger.info("=" * 60)
    logger.info(f"Compute Device: {device}")

    img_path = os.path.join(PROJECT_ROOT, cfg.data.img_parquet)
    box_path = os.path.join(PROJECT_ROOT, cfg.data.box_parquet)

    train_idx, val_idx = create_train_val_split(img_path, cfg.data.train_ratio, cfg.data.split_seed)
    shared_boxes = WaymoMultiParadigmDataset._index_boxes(box_path)

    resize = (cfg.image.resize_height, cfg.image.resize_width)
    train_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=train_idx, resize=resize,
        task="detection", augment=True, preloaded_boxes=shared_boxes, generate_instance_masks=False,
    )
    val_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=val_idx, resize=resize,
        task="detection", augment=False, preloaded_boxes=shared_boxes, generate_instance_masks=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, collate_fn=collate_detection,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, collate_fn=collate_detection,
    )

    # Instantiate model
    if args.model == "detr":
        m_cfg = cfg.detr
        model = DETR(
            num_classes=NUM_CLASSES,
            hidden_dim=m_cfg.hidden_dim,
            nheads=m_cfg.nheads,
            num_encoder_layers=m_cfg.num_encoder_layers,
            num_decoder_layers=m_cfg.num_decoder_layers,
            dim_feedforward=m_cfg.dim_feedforward,
            num_queries=m_cfg.num_queries,
        )
        total_epochs = m_cfg.epochs
        lr_trans = m_cfg.lr_transformer
        lr_back = m_cfg.lr_backbone
    else:
        m_cfg = cfg.deformable_detr
        model = DeformableDETR(
            num_classes=NUM_CLASSES,
            hidden_dim=m_cfg.hidden_dim,
            nheads=m_cfg.nheads,
            num_encoder_layers=m_cfg.num_encoder_layers,
            num_decoder_layers=m_cfg.num_decoder_layers,
            dim_feedforward=m_cfg.dim_feedforward,
            num_queries=m_cfg.num_queries,
        )
        total_epochs = m_cfg.epochs
        lr_trans = m_cfg.lr_transformer
        lr_back = m_cfg.lr_backbone

    model.to(device)

    # Loss criterion with Hungarian Matcher
    matcher = HungarianMatcher(
        cost_class=m_cfg.cost_class,
        cost_bbox=m_cfg.cost_bbox,
        cost_giou=m_cfg.cost_giou,
    )
    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
        eos_coef=0.1,
    )
    criterion.to(device)

    # Differential learning rate (Section 3.2: 1e-4 for transformer, 1e-5 for backbone)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": lr_trans},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": lr_back},
    ]
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=m_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    ckpt_dir = os.path.join(PROJECT_ROOT, cfg.training.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"Starting {args.model.upper()} training loop ({total_epochs} epochs)...")
    best_ap = 0.0

    for epoch in range(total_epochs):
        model.train()
        criterion.train()
        total_loss = 0.0
        num_b = 0

        for b_idx, (images, targets) in enumerate(train_loader):
            if len(images) == 0:
                continue

            imgs_tensor = torch.stack(images, dim=0).to(device)
            t_list = prepare_transformer_targets(targets, cfg.image.resize_width, cfg.image.resize_height, device)

            outputs = model(imgs_tensor)
            loss_dict = criterion(outputs, t_list)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()

            total_loss += losses.item()
            num_b += 1

            if (b_idx + 1) % cfg.logging.log_every_n_steps == 0:
                logger.info(
                    f"[{args.model.upper()} Epoch {epoch}][Batch {b_idx+1}/{len(train_loader)}] "
                    f"Loss: {losses.item():.4f} (ce: {loss_dict['loss_ce']:.4f}, "
                    f"bbox: {loss_dict['loss_bbox']:.4f}, giou: {loss_dict['loss_giou']:.4f})"
                )

        scheduler.step()
        avg_l = total_loss / max(num_b, 1)
        logger.info(f"{args.model.upper()} Epoch {epoch} Complete | Avg Loss: {avg_l:.4f}")

        # Quick checkpoint saving
        if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_l, ckpt_dir,
                filename=f"{args.model}_epoch_{epoch:03d}.pth"
            )

    logger.info(f"{args.model.upper()} training process finished.")


if __name__ == "__main__":
    main()
