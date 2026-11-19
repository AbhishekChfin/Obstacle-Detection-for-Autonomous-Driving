"""
Multi-Paradigm PyTorch Dataset for Waymo Open Dataset.

Supports:
1. Object Detection & Instance Segmentation (Mask R-CNN, DETR, Deformable DETR):
   Returns image tensors, bounding boxes, labels, and instance masks.
2. Semantic Segmentation (DeepLabv3+):
   Generates per-pixel pseudo-segmentation masks by filling bounding boxes with class IDs.
3. Adverse Weather & Imaging Perturbations:
   Applies simulated rain, fog, noise, blur, compression, or contrast shifts on-the-fly.
4. Ultra-Fast & Memory-Efficient Loading:
   Pre-indexes 15 MB box data into RAM; lazily reads images via row-groups in ~0.5 ms.
"""

import io
import os
import sys
import torch
import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.constants import (
    COL_SEGMENT,
    COL_TIMESTAMP,
    COL_CAMERA,
    COL_IMAGE,
    COL_BOX_CX,
    COL_BOX_CY,
    COL_BOX_W,
    COL_BOX_H,
    COL_BOX_TYPE,
    WAYMO_TO_MODEL_CLASS,
    ORIGINAL_WIDTH,
    ORIGINAL_HEIGHT,
)
from data.perturbations import apply_perturbation


class WaymoMultiParadigmDataset(Dataset):
    """
    High-performance PyTorch Dataset supporting detection, instance masks,
    semantic pseudo-segmentation masks, and adverse weather perturbations.
    """

    def __init__(
        self,
        img_parquet_path: str,
        box_parquet_path: str,
        indices: list = None,
        resize: tuple = (768, 1280),  # (height, width) - per paper Section 3.5
        task: str = "detection",       # "detection", "instance", or "segmentation"
        perturbation: str = "clean",   # Any of the 9 simulated adverse conditions
        augment: bool = False,
        aug_config: dict = None,
        preloaded_boxes: dict = None,
        generate_instance_masks: bool = True,
    ):
        super().__init__()
        self.img_parquet_path = img_parquet_path
        self.box_parquet_path = box_parquet_path
        self.resize_h, self.resize_w = resize
        self.task = task
        self.perturbation = perturbation
        self.augment = augment
        self.aug_config = aug_config or {}
        self.generate_instance_masks = generate_instance_masks

        # File reader handle initialized lazily (process/thread-safe)
        self._img_pf = None

        # Build lightweight index from image parquet (keys only, ~100 KB)
        img_ds = ds.dataset(img_parquet_path)
        keys_table = img_ds.to_table(
            columns=[COL_SEGMENT, COL_TIMESTAMP, COL_CAMERA]
        )
        self.all_segments = keys_table[COL_SEGMENT].to_pylist()
        self.all_timestamps = keys_table[COL_TIMESTAMP].to_pylist()
        self.all_cameras = keys_table[COL_CAMERA].to_pylist()

        # Row indices filtering (train/val split)
        if indices is not None:
            self.indices = list(indices)
        else:
            self.indices = list(range(len(self.all_segments)))

        # Pre-indexed box lookup (15 MB RAM total for 185k boxes)
        if preloaded_boxes is not None:
            self.box_index = preloaded_boxes
        else:
            self.box_index = self._index_boxes(box_parquet_path)

        # Coordinate scale factors
        self.scale_x = self.resize_w / ORIGINAL_WIDTH
        self.scale_y = self.resize_h / ORIGINAL_HEIGHT

    @staticmethod
    def _index_boxes(box_parquet_path: str) -> dict:
        """Pre-index boxes into dict (segment, timestamp) -> [(cx, cy, w, h, class_type)]"""
        box_table = pq.read_table(
            box_parquet_path,
            columns=[
                COL_SEGMENT,
                COL_TIMESTAMP,
                COL_BOX_CX,
                COL_BOX_CY,
                COL_BOX_W,
                COL_BOX_H,
                COL_BOX_TYPE,
            ],
        )

        segs = box_table[COL_SEGMENT].to_pylist()
        tss = box_table[COL_TIMESTAMP].to_pylist()
        cxs = box_table[COL_BOX_CX].to_pylist()
        cys = box_table[COL_BOX_CY].to_pylist()
        ws = box_table[COL_BOX_W].to_pylist()
        hs = box_table[COL_BOX_H].to_pylist()
        types = box_table[COL_BOX_TYPE].to_pylist()

        box_dict = defaultdict(list)
        for i in range(len(segs)):
            box_dict[(segs[i], tss[i])].append(
                (cxs[i], cys[i], ws[i], hs[i], types[i])
            )
        return dict(box_dict)

    @property
    def img_pf(self):
        """Lazy ParquetFile accessor."""
        if self._img_pf is None:
            self._img_pf = pq.ParquetFile(self.img_parquet_path)
        return self._img_pf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        actual_idx = self.indices[idx]
        segment = self.all_segments[actual_idx]
        timestamp = self.all_timestamps[actual_idx]

        # 1. Direct row group image read (~0.5 ms)
        row_group = self.img_pf.read_row_group(actual_idx, columns=[COL_IMAGE])
        img_bytes = row_group[COL_IMAGE][0].as_py()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 2. Apply simulated adverse condition (if evaluating robustness)
        if self.perturbation != "clean":
            pil_img = apply_perturbation(pil_img, self.perturbation)

        # 3. Fast box retrieval from memory
        raw_boxes = self.box_index.get((segment, timestamp), [])

        # 4. Resize image
        pil_img = pil_img.resize((self.resize_w, self.resize_h), Image.BILINEAR)

        # 5. Convert boxes from (cx, cy, w, h) to corner (x1, y1, x2, y2)
        boxes = []
        labels = []
        for cx, cy, w, h, class_type in raw_boxes:
            if class_type not in WAYMO_TO_MODEL_CLASS:
                continue

            x1 = (cx - w / 2.0) * self.scale_x
            y1 = (cy - h / 2.0) * self.scale_y
            x2 = (cx + w / 2.0) * self.scale_x
            y2 = (cy + h / 2.0) * self.scale_y

            x1 = max(0.0, min(x1, float(self.resize_w)))
            y1 = max(0.0, min(y1, float(self.resize_h)))
            x2 = max(0.0, min(x2, float(self.resize_w)))
            y2 = max(0.0, min(y2, float(self.resize_h)))

            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(WAYMO_TO_MODEL_CLASS[class_type])

        # 6. Apply data augmentation (if training)
        if self.augment:
            pil_img, boxes, labels = self._augment(pil_img, boxes, labels)

        # Convert image to tensor [3, H, W] in [0, 1]
        img_tensor = T.functional.to_tensor(pil_img)

        # 7. Format target according to task
        if self.task == "segmentation":
            # Semantic pseudo-mask for DeepLabv3+ (Section 3.4)
            # Background is 0; each box region is stamped with its class ID
            mask_arr = np.zeros((self.resize_h, self.resize_w), dtype=np.int64)
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = [int(round(coord)) for coord in box]
                mask_arr[y1:y2, x1:x2] = label
            mask_tensor = torch.from_numpy(mask_arr)
            return img_tensor, mask_tensor

        # Object Detection / Instance Segmentation targets
        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.uint8),
            }
            if self.generate_instance_masks:
                target["masks"] = torch.zeros(
                    (0, self.resize_h, self.resize_w), dtype=torch.uint8
                )
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )

            target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": area,
                "iscrowd": torch.zeros(len(boxes), dtype=torch.uint8),
            }

            # Generate instance masks for Mask R-CNN (Section 3.1)
            if self.generate_instance_masks:
                n_obj = len(boxes)
                masks = torch.zeros(
                    (n_obj, self.resize_h, self.resize_w), dtype=torch.uint8
                )
                for b_i, b in enumerate(boxes):
                    x1, y1, x2, y2 = [int(round(c)) for c in b]
                    masks[b_i, y1:y2, x1:x2] = 1
                target["masks"] = masks

        return img_tensor, target

    def _augment(self, pil_img, boxes, labels):
        """Random horizontal flip & color jitter."""
        if not boxes:
            return pil_img, boxes, labels

        flip_prob = self.aug_config.get("horizontal_flip_prob", 0.5)
        if np.random.random() < flip_prob:
            pil_img = T.functional.hflip(pil_img)
            w = pil_img.width
            new_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                new_boxes.append([w - x2, y1, w - x1, y2])
            boxes = new_boxes

        jitter = T.ColorJitter(
            brightness=self.aug_config.get("brightness", 0.2),
            contrast=self.aug_config.get("contrast", 0.2),
            saturation=self.aug_config.get("saturation", 0.2),
            hue=self.aug_config.get("hue", 0.1),
        )
        pil_img = jitter(pil_img)
        return pil_img, boxes, labels

    def get_image_info(self, idx: int) -> dict:
        """Return image metadata."""
        actual_idx = self.indices[idx]
        return {
            "segment": self.all_segments[actual_idx],
            "timestamp": self.all_timestamps[actual_idx],
            "camera": self.all_cameras[actual_idx],
            "actual_index": actual_idx,
        }
