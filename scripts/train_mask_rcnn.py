"""
Training script for Mask R-CNN (Instance Segmentation) on Waymo Open Dataset.

Usage:
    python -m multi_model_detection.scripts.train_mask_rcnn
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
from multi_model_detection.models.mask_rcnn import build_mask_rcnn
from multi_model_detection.utils.helpers import (
    get_device,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    collate_detection,
)
from multi_model_detection.utils.metrics import evaluate_detection_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on Waymo Open Dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def train_one_epoch(model, optimizer, loader, device, epoch, logger, cfg, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    grad_accum = cfg.training.gradient_accumulation_steps
    log_every = cfg.logging.log_every_n_steps
    use_amp = cfg.training.use_amp and device.type == "cuda"

    optimizer.zero_grad()

    for idx, (images, targets) in enumerate(loader):
        if len(images) == 0:
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(l for l in loss_dict.values())
                scaled_loss = losses / grad_accum
            scaler.scale(scaled_loss).backward()
        else:
            loss_dict = model(images, targets)
            losses = sum(l for l in loss_dict.values())
            scaled_loss = losses / grad_accum
            scaled_loss.backward()

        if (idx + 1) % grad_accum == 0 or (idx + 1) == len(loader):
            if cfg.training.max_grad_norm > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        b_loss = losses.item()
        total_loss += b_loss
        num_batches += 1

        if (idx + 1) % log_every == 0 or (idx + 1) == len(loader):
            details = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
            logger.info(
                f"[Epoch {epoch}][Batch {idx+1}/{len(loader)}] "
                f"Loss: {b_loss:.4f} | Details: {details}"
            )

        del images, targets, loss_dict, losses

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in loader:
        if len(images) == 0:
            continue
        images_dev = [img.to(device) for img in images]
        predictions = model(images_dev)

        for p in predictions:
            all_preds.append({k: v.cpu() for k, v in p.items()})
        for t in targets:
            all_targets.append({k: v.cpu() if hasattr(v, "cpu") else v for k, v in t.items()})

    if not all_preds:
        return {"AP": 0.0, "AP@50": 0.0, "APS": 0.0}

    return evaluate_detection_metrics(all_preds, all_targets)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.device.preferred)
    logger = setup_logging(cfg.logging.log_dir, name="train_mask_rcnn")

    logger.info("=" * 60)
    logger.info("Training Paradigm 1: Mask R-CNN (Instance Detection & Segmentation)")
    logger.info("=" * 60)
    logger.info(f"Compute Device: {device}")
    logger.info(f"Resolution: {cfg.image.resize_width} x {cfg.image.resize_height}")

    img_path = os.path.join(PROJECT_ROOT, cfg.data.img_parquet)
    box_path = os.path.join(PROJECT_ROOT, cfg.data.box_parquet)

    train_idx, val_idx = create_train_val_split(img_path, cfg.data.train_ratio, cfg.data.split_seed)
    shared_boxes = WaymoMultiParadigmDataset._index_boxes(box_path)

    resize = (cfg.image.resize_height, cfg.image.resize_width)
    train_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=train_idx, resize=resize,
        task="instance", augment=True, preloaded_boxes=shared_boxes, generate_instance_masks=True,
    )
    val_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=val_idx, resize=resize,
        task="instance", augment=False, preloaded_boxes=shared_boxes, generate_instance_masks=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, collate_fn=collate_detection,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, collate_fn=collate_detection,
    )

    # Build Mask R-CNN
    model = build_mask_rcnn(
        num_classes=NUM_CLASSES,
        pretrained_coco=cfg.mask_rcnn.pretrained_coco,
        freeze_backbone_up_to=cfg.mask_rcnn.freeze_backbone_up_to,
        box_score_thresh=cfg.mask_rcnn.box_score_thresh,
        box_nms_thresh=cfg.mask_rcnn.box_nms_thresh,
    )
    model.to(device)

    # Optimizer with MultiStepLR as per Section 3.1
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.mask_rcnn.lr,
        momentum=cfg.mask_rcnn.momentum,
        weight_decay=cfg.mask_rcnn.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.mask_rcnn.lr_decay_epochs, gamma=0.1
    )

    scaler = torch.amp.GradScaler("cuda") if (cfg.training.use_amp and device.type == "cuda") else None

    ckpt_dir = os.path.join(PROJECT_ROOT, cfg.training.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    best_ap = 0.0

    if args.resume or cfg.training.resume_from:
        p = args.resume or cfg.training.resume_from
        start_epoch, _ = load_checkpoint(p, model, optimizer, scheduler)
        start_epoch += 1

    logger.info(f"Starting Mask R-CNN training for {cfg.mask_rcnn.epochs} epochs...")

    for epoch in range(start_epoch, cfg.mask_rcnn.epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, logger, cfg, scaler)
        scheduler.step()
        dur = time.time() - t0

        logger.info(f"Epoch {epoch} finished in {dur:.1f}s | Average Loss: {avg_loss:.4f}")

        # Validation
        logger.info("Evaluating Mask R-CNN...")
        metrics = validate(model, val_loader, device)
        cur_ap = metrics["AP"]
        logger.info(
            f"Validation Results: AP (0.50:0.95)={cur_ap:.4f} | "
            f"AP@50={metrics['AP@50']:.4f} | APS={metrics['APS']:.4f} | "
            f"AP_Pedestrian={metrics['AP_pedestrian']:.4f}"
        )

        if cur_ap > best_ap:
            best_ap = cur_ap
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_loss, ckpt_dir, filename="mask_rcnn_best.pth"
            )
            logger.info(f"Saved new best Mask R-CNN (AP={best_ap:.4f})")

    logger.info(f"Mask R-CNN training complete. Best AP: {best_ap:.4f}")


if __name__ == "__main__":
    main()
