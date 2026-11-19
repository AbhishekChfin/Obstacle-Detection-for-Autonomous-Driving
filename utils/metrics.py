"""
Evaluation metrics for detection and semantic segmentation.

Implements:
1. AP (0.50:0.95): COCO-style mAP over IoUs [0.50, 0.55, ..., 0.95]
2. AP@50: PASCAL VOC-style Average Precision at IoU = 0.50
3. APS: Average Precision on small obstacles (area < 32^2 = 1024 px^2)
4. Per-class AP: vehicle, pedestrian, cyclist
5. mIoU: Mean Intersection-over-Union for semantic segmentation (DeepLabv3+)
6. Relative degradation drop (%): (clean - perturbed) / clean * 100
"""

import torch
import numpy as np
from collections import defaultdict

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.constants import CLASS_NAMES, NUM_CLASSES


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes [N, 4] and [M, 4] in (x1, y1, x2, y2)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)

    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter

    return inter / (union + 1e-6)


def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute AP via standard 101-point / continuous integration."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]))


def evaluate_detection_metrics(
    all_predictions: list[dict],
    all_targets: list[dict],
    iou_thresholds: list[float] = None,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """
    Computes AP (0.50:0.95), AP@50, APS (< 32^2 px), and per-class AP.

    Args:
        all_predictions: list of {'boxes': [N, 4], 'labels': [N], 'scores': [N]}
        all_targets: list of {'boxes': [M, 4], 'labels': [M]}
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.50, 0.95, 10).tolist()

    # Pre-filter for small objects (< 32^2 = 1024 px^2)
    small_preds = []
    small_targets = []
    for p, t in zip(all_predictions, all_targets):
        # Targets
        t_boxes = t["boxes"]
        t_areas = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
        t_small_mask = t_areas < 1024
        small_targets.append({
            "boxes": t_boxes[t_small_mask],
            "labels": t["labels"][t_small_mask],
        })

        # Predictions
        p_boxes = p["boxes"]
        p_areas = (p_boxes[:, 2] - p_boxes[:, 0]) * (p_boxes[:, 3] - p_boxes[:, 1])
        p_small_mask = p_areas < 1024
        small_preds.append({
            "boxes": p_boxes[p_small_mask],
            "labels": p["labels"][p_small_mask],
            "scores": p["scores"][p_small_mask],
        })

    # Evaluate across all thresholds
    threshold_results = {}
    for iou_thresh in iou_thresholds:
        threshold_results[round(iou_thresh, 2)] = _eval_single_iou(
            all_predictions, all_targets, iou_thresh, num_classes
        )

    # Evaluate small objects at IoU 0.50:0.95
    small_threshold_results = {}
    for iou_thresh in iou_thresholds:
        small_threshold_results[round(iou_thresh, 2)] = _eval_single_iou(
            small_preds, small_targets, iou_thresh, num_classes
        )

    # Summarize metrics (Table 1 format)
    ap_50_95_per_class = defaultdict(list)
    aps_per_class = defaultdict(list)

    for iou_val, res in threshold_results.items():
        for cls_name, val in res["per_class_ap"].items():
            ap_50_95_per_class[cls_name].append(val)

    for iou_val, res in small_threshold_results.items():
        for cls_name, val in res["per_class_ap"].items():
            aps_per_class[cls_name].append(val)

    final_per_class_ap = {k: float(np.mean(v)) for k, v in ap_50_95_per_class.items()}
    final_ap = float(np.mean(list(final_per_class_ap.values()))) if final_per_class_ap else 0.0

    final_ap_50 = threshold_results[0.50]["mAP"]

    small_per_class_ap = {k: float(np.mean(v)) for k, v in aps_per_class.items()}
    final_aps = float(np.mean(list(small_per_class_ap.values()))) if small_per_class_ap else 0.0

    return {
        "AP": final_ap,              # AP (0.50:0.95)
        "AP@50": final_ap_50,        # AP@50
        "APS": final_aps,            # AP on small objects (< 32^2 px)
        "per_class_AP": final_per_class_ap,
        "AP_pedestrian": final_per_class_ap.get("pedestrian", 0.0),
        "AP_vehicle": final_per_class_ap.get("vehicle", 0.0),
        "AP_cyclist": final_per_class_ap.get("cyclist", 0.0),
    }


def _eval_single_iou(
    all_predictions: list[dict],
    all_targets: list[dict],
    iou_threshold: float,
    num_classes: int,
) -> dict:
    class_preds = defaultdict(list)
    class_n_gt = defaultdict(int)

    for preds, targets in zip(all_predictions, all_targets):
        p_boxes = preds["boxes"]
        p_labels = preds["labels"]
        p_scores = preds["scores"]

        t_boxes = targets["boxes"]
        t_labels = targets["labels"]

        for lbl in t_labels.tolist():
            class_n_gt[lbl] += 1

        for cls_id in range(1, num_classes):
            p_mask = p_labels == cls_id
            t_mask = t_labels == cls_id

            cls_p_boxes = p_boxes[p_mask]
            cls_p_scores = p_scores[p_mask]
            cls_t_boxes = t_boxes[t_mask]

            if len(cls_p_boxes) == 0:
                continue
            if len(cls_t_boxes) == 0:
                for sc in cls_p_scores.tolist():
                    class_preds[cls_id].append((sc, False))
                continue

            ious = box_iou_matrix(cls_p_boxes, cls_t_boxes)
            sorted_idx = torch.argsort(cls_p_scores, descending=True)
            matched_gt = set()

            for pred_idx in sorted_idx.tolist():
                score = cls_p_scores[pred_idx].item()
                row = ious[pred_idx]
                best_gt = torch.argmax(row).item()
                best_iou = row[best_gt].item()

                if best_iou >= iou_threshold and best_gt not in matched_gt:
                    class_preds[cls_id].append((score, True))
                    matched_gt.add(best_gt)
                else:
                    class_preds[cls_id].append((score, False))

    per_class_ap = {}
    aps = []
    for cls_id in range(1, num_classes):
        name = CLASS_NAMES[cls_id]
        n_gt = class_n_gt.get(cls_id, 0)
        p_list = sorted(class_preds.get(cls_id, []), key=lambda x: -x[0])

        if n_gt == 0 or len(p_list) == 0:
            per_class_ap[name] = 0.0
            continue

        tp_cum = np.cumsum([int(is_tp) for _, is_tp in p_list])
        fp_cum = np.cumsum([int(not is_tp) for _, is_tp in p_list])
        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / n_gt

        ap = compute_ap_from_pr(precision, recall)
        per_class_ap[name] = ap
        aps.append(ap)

    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "per_class_ap": per_class_ap,
    }


def compute_miou(
    all_pred_masks: list[torch.Tensor],
    all_target_masks: list[torch.Tensor],
    num_classes: int = NUM_CLASSES,
) -> dict:
    """
    Computes mean Intersection-over-Union (mIoU) for semantic segmentation (Section 4.1).
    Classes: background (0), vehicle (1), pedestrian (2), cyclist (3).
    """
    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    for pred_m, target_m in zip(all_pred_masks, all_target_masks):
        p = pred_m.cpu().numpy().flatten()
        t = target_m.cpu().numpy().flatten()

        for c in range(num_classes):
            p_c = p == c
            t_c = t == c
            inter = np.logical_and(p_c, t_c).sum()
            uni = np.logical_or(p_c, t_c).sum()
            intersection[c] += inter
            union[c] += uni

    per_class_iou = {}
    for c in range(num_classes):
        c_name = CLASS_NAMES[c]
        if union[c] == 0:
            per_class_iou[c_name] = 1.0
        else:
            per_class_iou[c_name] = float(intersection[c] / union[c])

    mIoU = float(np.mean(list(per_class_iou.values())))

    return {
        "mIoU": mIoU,
        "per_class_iou": per_class_iou,
        "IoU_vehicle": per_class_iou.get("vehicle", 0.0),
        "IoU_pedestrian": per_class_iou.get("pedestrian", 0.0),
        "IoU_cyclist": per_class_iou.get("cyclist", 0.0),
        "IoU_background": per_class_iou.get("background", 0.0),
    }


def compute_relative_drop(clean_val: float, perturbed_val: float) -> float:
    """Compute relative performance drop in percent: (clean - perturbed) / clean * 100"""
    if clean_val <= 1e-7:
        return 0.0
    return max(0.0, float((clean_val - perturbed_val) / clean_val * 100.0))
