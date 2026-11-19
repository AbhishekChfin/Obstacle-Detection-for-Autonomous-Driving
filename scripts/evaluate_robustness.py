"""
Robustness Evaluation under Simulated Adverse Weather & Imaging Conditions (Tables 2 & 3).
Evaluates models across 9 perturbation types and prints relative degradation drops.

Usage:
    python -m multi_model_detection.scripts.evaluate_robustness \
        --model mask_rcnn \
        --checkpoint multi_model_detection/checkpoints/mask_rcnn_best.pth
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from multi_model_detection.config_loader import load_config
from multi_model_detection.configs.constants import NUM_CLASSES, ROBUSTNESS_CONDITIONS
from multi_model_detection.data.dataset import WaymoMultiParadigmDataset
from multi_model_detection.data.split import create_train_val_split
from multi_model_detection.models.mask_rcnn import build_mask_rcnn
from multi_model_detection.models.detr import DETR
from multi_model_detection.models.deformable_detr import DeformableDETR
from multi_model_detection.models.deeplabv3plus import DeepLabv3Plus
from multi_model_detection.utils.helpers import (
    get_device,
    setup_logging,
    load_checkpoint,
    collate_detection,
    collate_segmentation,
)
from multi_model_detection.utils.metrics import compute_relative_drop
from multi_model_detection.scripts.evaluate import evaluate_detection, evaluate_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate robustness under adverse conditions (Tables 2 & 3)")
    parser.add_argument("--model", type=str, required=True, choices=["mask_rcnn", "detr", "deformable_detr", "deeplab"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--max_val_samples", type=int, default=200, help="Subset size for fast evaluation across 9 conditions")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.device.preferred)
    logger = setup_logging(cfg.logging.log_dir, name=f"robustness_{args.model}")

    logger.info("=" * 70)
    logger.info(f"Robustness Benchmark across 9 Adverse Conditions: {args.model.upper()}")
    logger.info("=" * 70)
    logger.info(f"Device: {device} | Checkpoint: {args.checkpoint}")

    img_path = os.path.join(PROJECT_ROOT, cfg.data.img_parquet)
    box_path = os.path.join(PROJECT_ROOT, cfg.data.box_parquet)

    _, val_idx = create_train_val_split(img_path, cfg.data.train_ratio, cfg.data.split_seed)
    if args.max_val_samples and args.max_val_samples < len(val_idx):
        val_idx = val_idx[:args.max_val_samples]
        logger.info(f"Using representative subset of {len(val_idx)} validation frames for robustness suite.")

    shared_boxes = WaymoMultiParadigmDataset._index_boxes(box_path)
    resize = (cfg.image.resize_height, cfg.image.resize_width)

    is_seg = args.model == "deeplab"
    task = "segmentation" if is_seg else "instance"
    collate_fn = collate_segmentation if is_seg else collate_detection

    # Load model
    if args.model == "mask_rcnn":
        model = build_mask_rcnn(num_classes=NUM_CLASSES, pretrained_coco=False, freeze_backbone_up_to="none")
    elif args.model == "detr":
        m_cfg = cfg.detr
        model = DETR(num_classes=NUM_CLASSES, hidden_dim=m_cfg.hidden_dim, nheads=m_cfg.nheads, pretrained_backbone=False)
    elif args.model == "deformable_detr":
        m_cfg = cfg.deformable_detr
        model = DeformableDETR(num_classes=NUM_CLASSES, hidden_dim=m_cfg.hidden_dim, nheads=m_cfg.nheads, pretrained_backbone=False)
    else:
        model = DeepLabv3Plus(num_classes=NUM_CLASSES, pretrained=False)

    load_checkpoint(args.checkpoint, model)
    model.to(device)
    model.eval()

    # Step 1: Clean Baseline
    logger.info("\n--- Evaluating Baseline: Clean Validation Set ---")
    clean_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=val_idx, resize=resize,
        task=task, perturbation="clean", augment=False, preloaded_boxes=shared_boxes,
    )
    clean_loader = DataLoader(clean_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    if is_seg:
        clean_res, _ = evaluate_segmentation(model, clean_loader, device)
        clean_baseline = clean_res["mIoU"]
        logger.info(f"Clean Baseline mIoU: {clean_baseline:.4f}")
    else:
        clean_res, _ = evaluate_detection(model, clean_loader, device, args.model, cfg.image.resize_width, cfg.image.resize_height)
        clean_baseline_ap = clean_res["AP"]
        clean_baseline_ap50 = clean_res["AP@50"]
        logger.info(f"Clean Baseline AP (0.50:0.95): {clean_baseline_ap:.4f} | AP@50: {clean_baseline_ap50:.4f}")

    # Step 2: Loop over all 9 perturbations
    table_rows = []

    for cond in ROBUSTNESS_CONDITIONS:
        if cond == "clean":
            continue

        pert_ds = WaymoMultiParadigmDataset(
            img_path, box_path, indices=val_idx, resize=resize,
            task=task, perturbation=cond, augment=False, preloaded_boxes=shared_boxes,
        )
        pert_loader = DataLoader(pert_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

        if is_seg:
            res, _ = evaluate_segmentation(model, pert_loader, device)
            miou_val = res["mIoU"]
            drop = compute_relative_drop(clean_baseline, miou_val)
            table_rows.append((cond, miou_val, drop))
            logger.info(f"Condition: {cond:<18} | mIoU: {miou_val:.4f} | Drop: {drop:>5.1f}%")
        else:
            res, _ = evaluate_detection(model, pert_loader, device, args.model, cfg.image.resize_width, cfg.image.resize_height)
            ap_val = res["AP"]
            ap50_val = res["AP@50"]
            drop = compute_relative_drop(clean_baseline_ap, ap_val)
            table_rows.append((cond, ap_val, drop, ap50_val))
            logger.info(f"Condition: {cond:<18} | AP: {ap_val:.4f} | Drop: {drop:>5.1f}% | AP@50: {ap50_val:.4f}")

    # Format Summary Table
    logger.info("\n" + "=" * 70)
    if is_seg:
        logger.info(f"Table 3: Robustness Evaluation for DeepLabv3+ (mIoU under Perturbations)")
        logger.info("=" * 70)
        logger.info(f"{'Condition':<20} | {'mIoU':<8} | {'Relative Drop':<14}")
        logger.info("-" * 70)
        logger.info(f"{'Clean':<20} | {clean_baseline:<8.4f} | {'—':<14}")
        for cond, miou_val, drop in table_rows:
            logger.info(f"{cond:<20} | {miou_val:<8.4f} | {drop:>5.1f}%")
    else:
        logger.info(f"Table 2: Robustness Evaluation for {args.model.upper()} (AP under Perturbations)")
        logger.info("=" * 70)
        logger.info(f"{'Condition':<20} | {'AP':<8} | {'Relative Drop':<14} | {'AP@50':<8}")
        logger.info("-" * 70)
        logger.info(f"{'Clean':<20} | {clean_baseline_ap:<8.4f} | {'—':<14} | {clean_baseline_ap50:<8.4f}")
        for cond, ap_val, drop, ap50 in table_rows:
            logger.info(f"{cond:<20} | {ap_val:<8.4f} | {drop:>5.1f}%        | {ap50:<8.4f}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
