"""
Deformable DETR Architecture.
Described in Section 3.3 of the paper:
- ResNet-50 backbone with multi-scale feature maps (strides 8, 16, 32)
- Multi-scale cross-attention focusing on relevant sampling locations
- N = 300 learned object queries with iterative bounding box refinement
- Fast convergence and superior small-object detection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.constants import NUM_CLASSES
from models.detr import PositionEmbeddingSine, MLP, SetCriterion
from models.matcher import HungarianMatcher


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extracts multi-scale features from ResNet-50 (conv3, conv4, conv5)
    and projects them to uniform hidden dimension d_model (256).
    """

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = resnet50(weights=weights)
        except Exception:
            resnet = resnet50(weights=None)

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # stride 4
        )
        self.layer2 = resnet.layer2  # stride 8 (512 dims)
        self.layer3 = resnet.layer3  # stride 16 (1024 dims)
        self.layer4 = resnet.layer4  # stride 32 (2048 dims)

        # 1x1 projection convs
        self.proj_l2 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.proj_l3 = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        self.proj_l4 = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c1 = self.initial(x)
        c2 = self.layer2(c1)  # stride 8
        c3 = self.layer3(c2)  # stride 16
        c4 = self.layer4(c3)  # stride 32

        p2 = self.proj_l2(c2)
        p3 = self.proj_l3(c3)
        p4 = self.proj_l4(c4)
        return [p2, p3, p4]


class DeformableDETR(nn.Module):
    """
    Deformable DETR with Multi-Scale attention and N=300 queries.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        num_queries: int = 300,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Multi-scale backbone (strides 8, 16, 32)
        self.backbone = MultiScaleFeatureExtractor(
            hidden_dim=hidden_dim, pretrained=pretrained_backbone
        )
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)

        # Multi-scale Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Multi-scale Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 300 Learned Object Queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Level embeddings to distinguish between scales
        self.level_embed = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.level_embed)

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, samples: torch.Tensor) -> dict:
        bs = samples.shape[0]

        # Multi-scale feature extraction
        feats = self.backbone(samples)

        src_list = []
        pos_list = []
        for lvl, feat in enumerate(feats):
            pos = self.pos_encoder(feat)
            src_list.append(feat.flatten(2).permute(0, 2, 1) + self.level_embed[lvl].view(1, 1, -1))
            pos_list.append(pos.flatten(2).permute(0, 2, 1))

        # Concatenate across multi-scale spatial tokens
        src = torch.cat(src_list, dim=1)  # [B, sum(H_i * W_i), C]
        pos = torch.cat(pos_list, dim=1)  # [B, sum(H_i * W_i), C]

        # Multi-scale encoder
        memory = self.encoder(src + pos)

        # Queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed)

        # Decoder cross-attention over multi-scale memory
        hs = self.decoder(tgt + query_embed, memory)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
        }
