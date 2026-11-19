"""
DETR (Detection Transformer) Architecture and Loss.
Described in Section 3.2 of the paper:
- ResNet-50 backbone with 1x1 projection to hidden dimension (256)
- 2D sine/cosine positional encodings
- 6-layer Transformer Encoder & 6-layer Transformer Decoder
- N = 100 learned object queries
- Linear classification head + 3-layer MLP bounding box regression head
- Hungarian matching with GIoU + L1 + Cross-Entropy loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.constants import NUM_CLASSES
from models.matcher import (
    HungarianMatcher,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
)


class PositionEmbeddingSine(nn.Module):
    """2D Sinusoidal Positional Encoding."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: pos [B, 2*num_pos_feats, H, W]
        """
        if mask is None:
            mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """
    Detection Transformer (DETR) model.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        num_queries: int = 100,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 1. ResNet-50 Backbone
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
            resnet = resnet50(weights=weights)
        except Exception:
            resnet = resnet50(weights=None)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # 2. 1x1 Conv projection from 2048 -> hidden_dim (256)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # 3. Positional encoding
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)

        # 4. Standard Transformer Encoder-Decoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )

        # 5. Learned Object Queries (N x hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 6. Prediction heads
        # Class logits: 4 classes (0: no-object/background, 1: vehicle, 2: pedestrian, 3: cyclist)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        # Box regression: normalized (cx, cy, w, h) in [0, 1]
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, samples: torch.Tensor) -> dict:
        """
        samples: FloatTensor [B, 3, H, W]
        returns dict with:
            - 'pred_logits': [B, num_queries, num_classes]
            - 'pred_boxes': [B, num_queries, 4] normalized (cx, cy, w, h)
        """
        # Backbone feature extraction
        features = self.backbone(samples)  # [B, 2048, H/32, W/32]
        proj_features = self.input_proj(features)  # [B, 256, H/32, W/32]

        pos = self.pos_encoder(proj_features)  # [B, 256, H/32, W/32]

        # Flatten spatial dimensions: [B, C, H, W] -> [H*W, B, C]
        bs, c, h, w = proj_features.shape
        src = proj_features.flatten(2).permute(2, 0, 1)
        pos_embed = pos.flatten(2).permute(2, 0, 1)

        # Query embeddings: [num_queries, B, C]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        # Transformer forward pass
        hs = self.transformer(
            src + pos_embed,
            tgt,
            query_pos=query_embed,
        )  # [num_queries, B, C]

        # Permute back to [B, num_queries, C]
        hs = hs.permute(1, 0, 2)

        # Prediction heads
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
        }


class SetCriterion(nn.Module):
    """
    Computes bipartite Hungarian loss for DETR (Section 3.2):
    Classification (Cross-Entropy) + L1 bounding box + GIoU loss.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        matcher: HungarianMatcher = None,
        weight_dict: dict = None,
        eos_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.weight_dict = weight_dict or {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }
        self.eos_coef = eos_coef

        # Class weights: down-weight no-object (background=0)
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: dict, targets: list[dict]) -> dict:
        """
        outputs: dict from DETR
        targets: list of dicts with 'boxes' (normalized cx,cy,w,h) and 'labels'
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        # Compute losses
        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)

        losses = {
            "loss_ce": loss_ce * self.weight_dict["loss_ce"],
            "loss_bbox": loss_bbox * self.weight_dict["loss_bbox"],
            "loss_giou": loss_giou * self.weight_dict["loss_giou"],
        }
        return losses

    def loss_labels(self, outputs: dict, targets: list[dict], indices: list) -> torch.Tensor:
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            0,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )
        return loss_ce

    def loss_boxes(self, outputs: dict, targets: list[dict], indices: list) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        num_boxes = max(len(target_boxes), 1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes

        loss_giou = (
            1.0
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            )
        ).sum() / num_boxes

        return loss_bbox, loss_giou

    def _get_src_permutation_idx(self, indices: list) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
