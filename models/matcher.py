"""
Hungarian Matcher for DETR and Deformable DETR.

Computes bipartite matching between predicted object queries and ground-truth obstacles
by minimizing the matching cost matrix:
    Cost = lambda_cls * cost_class + lambda_bbox * cost_bbox + lambda_giou * cost_giou
"""

import torch
import torch.nn as nn
from torchvision.ops.boxes import box_area

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (GIoU) between two sets of boxes in (x1, y1, x2, y2) format.
    Returns:
        Tensor [N, M] with values in [-1, 1].
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    # Smallest enclosing convex box
    enclosing_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclosing_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = enclosing_wh[:, :, 0] * enclosing_wh[:, :, 1]

    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou


class HungarianMatcher(nn.Module):
    """
    Computes an optimal bipartite assignment between targets and predictions.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]):
        """
        Args:
            outputs: Dict with:
                - "pred_logits": FloatTensor [batch_size, num_queries, num_classes]
                - "pred_boxes": FloatTensor [batch_size, num_queries, 4] in normalized (cx, cy, w, h)
            targets: List of dicts, each with:
                - "labels": Int64Tensor [num_target_boxes]
                - "boxes": FloatTensor [num_target_boxes, 4] normalized (cx, cy, w, h)

        Returns:
            List of size batch_size, containing tuples of (pred_idx, tgt_idx).
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions across batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch * queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)                # [batch * queries, 4]

        # Concatenate ground truth labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(bs)]

        # 1. Classification cost: -prob for true class
        cost_class = -out_prob[:, tgt_ids]

        # 2. L1 box cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 3. GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )

        # Final weighted cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []

        for i, c_matrix in enumerate(C.split(sizes, -1)):
            c_mat = c_matrix[i]
            if c_mat.shape[1] == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            if HAS_SCIPY:
                pred_idx, tgt_idx = linear_sum_assignment(c_mat.numpy())
                indices.append((torch.as_tensor(pred_idx, dtype=torch.int64), torch.as_tensor(tgt_idx, dtype=torch.int64)))
            else:
                # Fast greedy matching fallback if scipy not yet available
                matched_preds = []
                matched_tgts = []
                cost_copy = c_mat.clone()
                for _ in range(min(num_queries, c_mat.shape[1])):
                    min_val, flat_idx = torch.min(cost_copy.view(-1), dim=0)
                    p_idx = flat_idx // cost_copy.shape[1]
                    t_idx = flat_idx % cost_copy.shape[1]
                    matched_preds.append(p_idx.item())
                    matched_tgts.append(t_idx.item())
                    cost_copy[p_idx, :] = 1e6
                    cost_copy[:, t_idx] = 1e6
                indices.append((torch.tensor(matched_preds, dtype=torch.int64), torch.tensor(matched_tgts, dtype=torch.int64)))

        return indices
