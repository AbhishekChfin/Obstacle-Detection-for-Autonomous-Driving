"""
DeepLabv3+ (Semantic Segmentation) Architecture.
Described in Section 3.4 of the paper:
- ResNet-50 backbone with atrous convolutions
- ASPP (Atrous Spatial Pyramid Pooling) with dilation rates {6, 12, 18} + image-level pooling
- Low-level feature concatenation from early conv layers (conv2)
- Decoder upsampling 4x and predicting per-pixel logits over 4 classes
- Trained using bounding box pseudo-segmentation masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.constants import NUM_CLASSES


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling Module."""

    def __init__(self, in_channels: int, atrous_rates: list[int], out_channels: int = 256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabv3Plus(nn.Module):
    """
    DeepLabv3+ Encoder-Decoder semantic segmentation network.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        atrous_rates: list[int] = [6, 12, 18],
        aspp_channels: int = 256,
        low_level_channels: int = 48,
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = resnet50(weights=weights)
        except Exception:
            resnet = resnet50(weights=None)

        # ResNet-50 Encoder
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # Low-level features: stride 4, 256 channels
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16 (with atrous)
        self.layer4 = resnet.layer4  # stride 16 (dilated)

        # Adjust layer3 and layer4 dilations for output stride 16
        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding = (2, 2)

        # ASPP module on top of layer4 (2048 channels)
        self.aspp = ASPP(2048, atrous_rates, aspp_channels)

        # 1x1 projection for low-level features (256 -> low_level_channels)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder convs
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels + low_level_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Low-level branch
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)
        low_level = self.layer1(c1)  # stride 4, 256 channels

        # High-level branch
        c2 = self.layer2(low_level)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)  # [B, 2048, H/16, W/16]

        aspp_out = self.aspp(c4)  # [B, 256, H/16, W/16]

        # Upsample ASPP output by 4x to match low_level resolution
        aspp_up = F.interpolate(
            aspp_out,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low_proj = self.low_level_proj(low_level)
        concat = torch.cat([aspp_up, low_proj], dim=1)

        # Decoder predictions
        decoder_out = self.decoder(concat)

        # Final upsampling to original input resolution
        out = F.interpolate(decoder_out, size=input_size, mode="bilinear", align_corners=False)
        return out
