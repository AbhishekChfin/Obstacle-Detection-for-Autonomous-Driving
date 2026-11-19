"""
Training script for DeepLabv3+ (Semantic Segmentation) using pseudo-masks.
Described in Section 3.4 of the paper.

Usage:
    python -m multi_model_detection.scripts.train_deeplab
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from multi_model_detection.config_loader import load_config
from multi_model_detection.configs.constants import NUM_CLASSES
from multi_model_detection.data.dataset import WaymoMultiParadigmDataset
from multi_model_detection.data.split import create_train_val_split
from multi_model_detection.models.deeplabv3plus import DeepLabv3Plus
from multi_model_detection.utils.helpers import (
    get_device,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    collate_segmentation,
)
from multi_model_detection.utils.metrics import compute_miou


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabv3+ on Waymo Open Dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return parser.parse_args()


@torch.no_grad()
def validate_segmentation(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)  # [B, H, W]

        all_preds.extend(preds.cpu())
        all_targets.extend(targets.cpu())

    return compute_miou(all_preds, all_targets)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.device.preferred)
    logger = setup_logging(cfg.logging.log_dir, name="train_deeplabv3plus")

    logger.info("=" * 60)
    logger.info("Training Paradigm 3: DeepLabv3+ (Semantic Segmentation)")
    logger.info("=" * 60)
    logger.info(f"Compute Device: {device}")

    img_path = os.path.join(PROJECT_ROOT, cfg.data.img_parquet)
    box_path = os.path.join(PROJECT_ROOT, cfg.data.box_parquet)

    train_idx, val_idx = create_train_val_split(img_path, cfg.data.train_ratio, cfg.data.split_seed)
    shared_boxes = WaymoMultiParadigmDataset._index_boxes(box_path)

    resize = (cfg.image.resize_height, cfg.image.resize_width)
    train_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=train_idx, resize=resize,
        task="segmentation", augment=True, preloaded_boxes=shared_boxes,
    )
    val_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=val_idx, resize=resize,
        task="segmentation", augment=False, preloaded_boxes=shared_boxes,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, collate_fn=collate_segmentation,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, collate_fn=collate_segmentation,
    )

    # Instantiate DeepLabv3+
    model = DeepLabv3Plus(
        num_classes=NUM_CLASSES,
        atrous_rates=cfg.deeplab.aspp_rates,
        aspp_channels=cfg.deeplab.aspp_out_channels,
        low_level_channels=cfg.deeplab.low_level_channels,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.deeplab.lr, weight_decay=cfg.deeplab.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.deeplab.epochs)

    ckpt_dir = os.path.join(PROJECT_ROOT, cfg.training.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    best_miou = 0.0

    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1

    logger.info(f"Starting DeepLabv3+ training for {cfg.deeplab.epochs} epochs...")

    for epoch in range(start_epoch, cfg.deeplab.epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for b_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            if (b_idx + 1) % cfg.logging.log_every_n_steps == 0:
                logger.info(
                    f"[DeepLabv3+ Epoch {epoch}][Batch {b_idx+1}/{len(train_loader)}] "
                    f"Pixel Cross-Entropy Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_loss = total_loss / max(batches, 1)

        # Validation mIoU
        logger.info(f"Evaluating DeepLabv3+ Epoch {epoch}...")
        metrics = validate_segmentation(model, val_loader, device)
        cur_miou = metrics["mIoU"]
        logger.info(
            f"Validation mIoU: {cur_miou:.4f} | "
            f"Vehicle: {metrics['IoU_vehicle']:.4f}, "
            f"Pedestrian: {metrics['IoU_pedestrian']:.4f}, "
            f"Cyclist: {metrics['IoU_cyclist']:.4f}"
        )

        if cur_miou > best_miou:
            best_miou = cur_miou
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_loss, ckpt_dir,
                filename="deeplabv3plus_best.pth",
            )
            logger.info(f"New best DeepLabv3+ model saved (mIoU: {best_miou:.4f})")

    logger.info(f"DeepLabv3+ training complete. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
