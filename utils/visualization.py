"""
Visualization utilities for multi-model obstacle detection and segmentation.
Generates qualitative visualizations corresponding to Figures 1, 2, 3 in the paper.
"""

import os
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.constants import CLASS_NAMES, CLASS_COLORS_RGB


def plot_detection(
    image,
    boxes,
    labels,
    scores=None,
    masks=None,
    title="",
    figsize=(12, 8),
    score_threshold=0.5,
    save_path=None,
):
    """
    Plot image with bounding boxes, labels, and optional instance masks.
    """
    img_np = _tensor_to_numpy(image)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img_np)

    if masks is not None and len(masks) > 0:
        # Semitransparent instance mask overlay (Figure 1 in paper)
        mask_overlay = np.zeros((*img_np.shape[:2], 4), dtype=np.float32)
        for i, mask in enumerate(masks):
            if scores is not None and scores[i] < score_threshold:
                continue
            lbl = int(labels[i])
            color = np.array(CLASS_COLORS_RGB.get(lbl, (255, 255, 255))) / 255.0
            m_np = mask.cpu().numpy() if hasattr(mask, "cpu") else mask
            m_bool = m_np > 0.5
            mask_overlay[m_bool, :3] = color
            mask_overlay[m_bool, 3] = 0.4  # alpha
        ax.imshow(mask_overlay)

    # Draw boxes
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    if scores is not None and hasattr(scores, "tolist"):
        scores = scores.tolist()

    for i, (b, lbl) in enumerate(zip(boxes, labels)):
        sc = scores[i] if scores is not None else None
        if sc is not None and sc < score_threshold:
            continue

        x1, y1, x2, y2 = b
        color = tuple(c / 255.0 for c in CLASS_COLORS_RGB.get(lbl, (255, 255, 255)))
        name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"cls_{lbl}"

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        txt = name if sc is None else f"{name} {sc:.2f}"
        ax.text(
            x1, y1 - 4, txt,
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
        )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_semantic_segmentation(
    image,
    pred_mask,
    target_mask=None,
    title="",
    figsize=(16, 7),
    save_path=None,
):
    """
    Side-by-side plot of RGB image and predicted semantic segmentation map.
    """
    img_np = _tensor_to_numpy(image)
    pred_np = pred_mask.cpu().numpy() if hasattr(pred_mask, "cpu") else pred_mask

    # Colorize segmentation map
    h, w = pred_np.shape
    colored_pred = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id, rgb in CLASS_COLORS_RGB.items():
        colored_pred[pred_np == c_id] = rgb

    num_plots = 3 if target_mask is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    axes[0].imshow(img_np)
    axes[0].set_title("Original RGB Front Camera")
    axes[0].axis("off")

    axes[1].imshow(colored_pred)
    axes[1].set_title("DeepLabv3+ Predicted Semantic Map")
    axes[1].axis("off")

    if target_mask is not None:
        target_np = target_mask.cpu().numpy() if hasattr(target_mask, "cpu") else target_mask
        colored_tgt = np.zeros((h, w, 3), dtype=np.uint8)
        for c_id, rgb in CLASS_COLORS_RGB.items():
            colored_tgt[target_np == c_id] = rgb
        axes[2].imshow(colored_tgt)
        axes[2].set_title("Ground-Truth Pseudo-Mask")
        axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _tensor_to_numpy(image):
    if hasattr(image, "numpy"):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = np.clip(img_np, 0, 1)
        return img_np
    elif isinstance(image, Image.Image):
        return np.array(image)
    return image
