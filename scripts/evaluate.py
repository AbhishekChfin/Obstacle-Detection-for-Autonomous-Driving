"""
Clean Validation Benchmark Evaluation (Table 1 in Paper).
Evaluates Mask R-CNN, DETR, Deformable DETR, or DeepLabv3+ on clean Waymo validation images.

Usage:
    python -m multi_model_detection.scripts.evaluate \
        --model mask_rcnn \
        --checkpoint multi_model_detection/checkpoints/mask_rcnn_best.pth \
        --visualize --num_vis 10
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from multi_model_detection.config_loader import load_config
from multi_model_detection.configs.constants import NUM_CLASSES
from multi_model_detection.data.dataset import WaymoMultiParadigmDataset
from multi_model_detection.data.split import create_train_val_split
from multi_model_detection.models.mask_rcnn import build_mask_rcnn
from multi_model_detection.models.detr import DETR
from multi_model_detection.models.deformable_detr import DeformableDETR
from multi_model_detection.models.deeplabv3plus import DeepLabv3Plus
from multi_model_detection.models.matcher import box_cxcywh_to_xyxy
from multi_model_detection.utils.helpers import (
    get_device,
    setup_logging,
    load_checkpoint,
    collate_detection,
    collate_segmentation,
)
from multi_model_detection.utils.metrics import evaluate_detection_metrics, compute_miou
from multi_model_detection.utils.visualization import plot_detection, plot_semantic_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on clean Waymo validation set (Table 1)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mask_rcnn", "detr", "deformable_detr", "deeplab"],
        help="Model to evaluate",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--visualize", action="store_true", help="Save visual prediction plots")
    parser.add_argument("--num_vis", type=int, default=10, help="Number of visualizations to export")
    parser.add_argument("--vis_dir", type=str, default="multi_model_detection/visualizations", help="Output directory")
    return parser.parse_args()


@torch.no_grad()
def evaluate_detection(model, loader, device, model_name, img_w, img_h):
    model.eval()
    all_preds = []
    all_targets = []

    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    for images, targets in loader:
        if len(images) == 0:
            continue

        if model_name in ("detr", "deformable_detr"):
            imgs_tensor = torch.stack(images, dim=0).to(device)
            outputs = model(imgs_tensor)
            pred_logits = outputs["pred_logits"].cpu()  # [B, Q, num_classes]
            pred_boxes = outputs["pred_boxes"].cpu()    # [B, Q, 4] normalized cxcywh

            for b_i in range(len(images)):
                probs = pred_logits[b_i].softmax(-1)
                scores, labels = probs.max(-1)

                # Keep foreground objects only (0 is background/no-object)
                keep = labels > 0
                f_scores = scores[keep]
                f_labels = labels[keep]
                f_boxes_norm = pred_boxes[b_i][keep]
                f_boxes_xyxy = box_cxcywh_to_xyxy(f_boxes_norm) * scale

                all_preds.append({
                    "boxes": f_boxes_xyxy,
                    "labels": f_labels,
                    "scores": f_scores,
                })
        else:
            # Mask R-CNN
            images_dev = [img.to(device) for img in images]
            predictions = model(images_dev)
            for p in predictions:
                all_preds.append({k: v.cpu() for k, v in p.items()})

        for t in targets:
            all_targets.append({k: v.cpu() if hasattr(v, "cpu") else v for k, v in t.items()})

    return evaluate_detection_metrics(all_preds, all_targets), all_preds


@torch.no_grad()
def evaluate_segmentation(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images_dev = images.to(device)
        logits = model(images_dev)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu())
        all_targets.extend(targets.cpu())

    return compute_miou(all_preds, all_targets), all_preds


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.device.preferred)
    logger = setup_logging(cfg.logging.log_dir, name=f"eval_{args.model}")

    logger.info("=" * 60)
    logger.info(f"Waymo Validation Evaluation: {args.model.upper()}")
    logger.info(f"Checkpoint: {args.checkpoint} on {device}")
    logger.info("=" * 60)

    img_path = os.path.join(PROJECT_ROOT, cfg.data.img_parquet)
    box_path = os.path.join(PROJECT_ROOT, cfg.data.box_parquet)

    _, val_idx = create_train_val_split(img_path, cfg.data.train_ratio, cfg.data.split_seed)
    shared_boxes = WaymoMultiParadigmDataset._index_boxes(box_path)
    resize = (cfg.image.resize_height, cfg.image.resize_width)

    is_seg = args.model == "deeplab"
    task = "segmentation" if is_seg else "instance"

    val_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=val_idx, resize=resize,
        task=task, augment=False, preloaded_boxes=shared_boxes, generate_instance_masks=True,
    )

    collate_fn = collate_segmentation if is_seg else collate_detection
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, collate_fn=collate_fn,
    )

    # Load Model
    if args.model == "mask_rcnn":
        model = build_mask_rcnn(num_classes=NUM_CLASSES, pretrained_coco=False, freeze_backbone_up_to="none")
    elif args.model == "detr":
        m_cfg = cfg.detr
        model = DETR(
            num_classes=NUM_CLASSES, hidden_dim=m_cfg.hidden_dim, nheads=m_cfg.nheads,
            num_encoder_layers=m_cfg.num_encoder_layers, num_decoder_layers=m_cfg.num_decoder_layers,
            num_queries=m_cfg.num_queries, pretrained_backbone=False,
        )
    elif args.model == "deformable_detr":
        m_cfg = cfg.deformable_detr
        model = DeformableDETR(
            num_classes=NUM_CLASSES, hidden_dim=m_cfg.hidden_dim, nheads=m_cfg.nheads,
            num_encoder_layers=m_cfg.num_encoder_layers, num_decoder_layers=m_cfg.num_decoder_layers,
            num_queries=m_cfg.num_queries, pretrained_backbone=False,
        )
    else:
        model = DeepLabv3Plus(num_classes=NUM_CLASSES, pretrained=False)

    load_checkpoint(args.checkpoint, model)
    model.to(device)
    logger.info("Model checkpoint successfully loaded.")

    out_dir = os.path.join(PROJECT_ROOT, args.vis_dir)
    os.makedirs(out_dir, exist_ok=True)

    if is_seg:
        metrics, preds_list = evaluate_segmentation(model, val_loader, device)
        logger.info("\n=== DeepLabv3+ Performance (Table 1) ===")
        logger.info(f"mIoU (segm): {metrics['mIoU']:.4f}")
        logger.info(f"  - Vehicle IoU:    {metrics['IoU_vehicle']:.4f}")
        logger.info(f"  - Pedestrian IoU: {metrics['IoU_pedestrian']:.4f}")
        logger.info(f"  - Cyclist IoU:    {metrics['IoU_cyclist']:.4f}")

        if args.visualize:
            logger.info(f"Exporting {args.num_vis} semantic segmentation maps...")
            for v_i in range(min(args.num_vis, len(val_ds))):
                img, tgt_mask = val_ds[v_i]
                pred_mask = preds_list[v_i]
                p = os.path.join(out_dir, f"deeplab_eval_{v_i:03d}.png")
                plot_semantic_segmentation(img, pred_mask, tgt_mask, title=f"Sample {v_i}", save_path=p)
    else:
        metrics, preds_list = evaluate_detection(
            model, val_loader, device, args.model, cfg.image.resize_width, cfg.image.resize_height
        )
        logger.info(f"\n=== {args.model.upper()} Performance (Table 1) ===")
        logger.info(f"AP (0.50:0.95):          {metrics['AP']:.4f}")
        logger.info(f"AP@50:                   {metrics['AP@50']:.4f}")
        logger.info(f"APS (small objects):     {metrics['APS']:.4f}")
        logger.info(f"AP on Pedestrians:       {metrics['AP_pedestrian']:.4f}")
        logger.info(f"AP on Vehicles:          {metrics['AP_vehicle']:.4f}")
        logger.info(f"AP on Cyclists:          {metrics['AP_cyclist']:.4f}")

        if args.visualize:
            logger.info(f"Exporting {args.num_vis} detection visual comparisons...")
            for v_i in range(min(args.num_vis, len(val_ds))):
                img, _ = val_ds[v_i]
                p = preds_list[v_i]
                save_file = os.path.join(out_dir, f"{args.model}_eval_{v_i:03d}.png")
                plot_detection(
                    img, p["boxes"], p["labels"], scores=p.get("scores"),
                    masks=p.get("masks"), title=f"{args.model.upper()} Sample {v_i}",
                    score_threshold=0.4, save_path=save_file,
                )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
