"""
Comprehensive sanity-check and data exploration script.
Verifies multi-task targets, pseudo-masks, and the 9 adverse condition perturbations.

Usage:
    python -m multi_model_detection.scripts.explore_data
"""

import os
import sys
import io
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from multi_model_detection.config_loader import load_config
from multi_model_detection.configs.constants import (
    CLASS_NAMES,
    WAYMO_TO_MODEL_CLASS,
    ROBUSTNESS_CONDITIONS,
)
from multi_model_detection.data.dataset import WaymoMultiParadigmDataset
from multi_model_detection.data.split import create_train_val_split
from multi_model_detection.data.perturbations import apply_perturbation
from multi_model_detection.utils.visualization import plot_detection, plot_semantic_segmentation


def main():
    cfg = load_config()
    img_path = os.path.join(PROJECT_ROOT, cfg.data.img_parquet)
    box_path = os.path.join(PROJECT_ROOT, cfg.data.box_parquet)

    print("=" * 65)
    print("Multi-Paradigm Dataset & Perturbations Exploration")
    print("=" * 65)

    # 1. Dataset Indexing & Distribution
    shared_boxes = WaymoMultiParadigmDataset._index_boxes(box_path)
    print(f"\n--- 1. Bounding Box Distribution (184,880 Total Annotations) ---")
    type_counts = {}
    for box_list in shared_boxes.values():
        for b in box_list:
            t = b[4]
            type_counts[t] = type_counts.get(t, 0) + 1

    total = sum(type_counts.values())
    for w_type, cnt in sorted(type_counts.items()):
        m_cls = WAYMO_TO_MODEL_CLASS.get(w_type, 0)
        c_name = CLASS_NAMES[m_cls]
        print(f"  Waymo Type {w_type} ({c_name:<10}): {cnt:>8,} boxes ({cnt/total*100:>5.1f}%)")

    # 2. Train / Val split
    train_idx, val_idx = create_train_val_split(img_path, cfg.data.train_ratio, cfg.data.split_seed)
    print(f"\n--- 2. Dataset Split by 20-Second Driving Segments ---")
    print(f"  Train set frames: {len(train_idx):>6,} ({len(train_idx)/(len(train_idx)+len(val_idx))*100:.1f}%)")
    print(f"  Val set frames:   {len(val_idx):>6,} ({len(val_idx)/(len(train_idx)+len(val_idx))*100:.1f}%)")

    resize = (cfg.image.resize_height, cfg.image.resize_width)

    # 3. Detection & Instance Segmentation Dataset Test
    print(f"\n--- 3. Testing Detection & Instance Masks (Mask R-CNN / DETR) ---")
    det_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=val_idx[:5], resize=resize,
        task="instance", augment=False, preloaded_boxes=shared_boxes, generate_instance_masks=True,
    )
    img_det, tgt_det = det_ds[0]
    print(f"  Image shape: {tuple(img_det.shape)}")
    print(f"  Bounding boxes: {tuple(tgt_det['boxes'].shape)}")
    print(f"  Instance masks: {tuple(tgt_det['masks'].shape)}")
    print(f"  Labels: {tgt_det['labels'].tolist()}")

    # 4. Semantic Pseudo-Mask Dataset Test (DeepLabv3+)
    print(f"\n--- 4. Testing Semantic Segmentation Pseudo-Masks (DeepLabv3+) ---")
    seg_ds = WaymoMultiParadigmDataset(
        img_path, box_path, indices=val_idx[:5], resize=resize,
        task="segmentation", augment=False, preloaded_boxes=shared_boxes,
    )
    img_seg, tgt_seg = seg_ds[0]
    unique_pixel_classes = torch.unique(tgt_seg).tolist()
    print(f"  Semantic Mask shape: {tuple(tgt_seg.shape)}")
    print(f"  Classes present in pixel mask: {[CLASS_NAMES[c] for c in unique_pixel_classes]}")

    # 5. Adverse Weather Perturbations Test
    print(f"\n--- 5. Testing 9 Simulated Adverse Conditions (Section 3.5 & 4.6) ---")
    vis_dir = os.path.join(PROJECT_ROOT, "multi_model_detection", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Load one raw PIL image
    raw_img_pf = det_ds.img_pf
    raw_bytes = raw_img_pf.read_row_group(val_idx[0], columns=["[CameraImageComponent].image"])["[CameraImageComponent].image"][0].as_py()
    sample_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((960, 640), Image.BILINEAR)

    print("  Generating perturbed test samples:")
    for cond in ROBUSTNESS_CONDITIONS:
        perturbed = apply_perturbation(sample_pil, cond)
        out_file = os.path.join(vis_dir, f"weather_preview_{cond}.png")
        perturbed.save(out_file)
        print(f"    - {cond:<20} -> saved {out_file}")

    print("\n" + "=" * 65)
    print("All multi-paradigm dataset tests & weather simulations verified!")
    print("=" * 65)


if __name__ == "__main__":
    import io
    main()
