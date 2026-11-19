"""
Mask R-CNN (Instance Segmentation) Architecture.
Described in Section 3.1 of the paper:
- ResNet-50 backbone with 5-level FPN
- RPN with candidate anchors {32, 64, 128, 256, 512} and aspect ratios {0.5, 1.0, 2.0}
- RoIAlign (7x7 for box, 14x14 for mask)
- FastRCNNPredictor (4 classes: vehicle, pedestrian, cyclist, background)
- MaskRCNNPredictor (28x28 mask per instance)
"""

import warnings
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.constants import NUM_CLASSES


def build_mask_rcnn(
    num_classes: int = NUM_CLASSES,
    pretrained_coco: bool = True,
    freeze_backbone_up_to: str = "layer2",
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
    **kwargs,
) -> MaskRCNN:
    """
    Build Mask R-CNN model adapted for Waymo 4-class instance detection/segmentation.
    """
    if pretrained_coco:
        try:
            weights = (
                torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            model = maskrcnn_resnet50_fpn(
                weights=weights,
                box_score_thresh=box_score_thresh,
                box_nms_thresh=box_nms_thresh,
                box_detections_per_img=box_detections_per_img,
                **kwargs,
            )
        except Exception as e:
            warnings.warn(
                f"Could not download pre-trained weights ({e}). Initializing offline."
            )
            model = maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                box_score_thresh=box_score_thresh,
                box_nms_thresh=box_nms_thresh,
                box_detections_per_img=box_detections_per_img,
                **kwargs,
            )
    else:
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            **kwargs,
        )

    # 1. Replace Box Predictor with 4-class head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Replace Mask Predictor with 4-class head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced, num_classes
    )

    # Freeze backbone layers to conserve GPU memory
    if freeze_backbone_up_to != "none":
        _freeze_resnet_layers(model.backbone.body, freeze_backbone_up_to)

    return model


def _freeze_resnet_layers(backbone_body, freeze_up_to: str):
    """Freeze ResNet layers up to the designated stage."""
    layers_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    if freeze_up_to == "all":
        limit = len(layers_order)
    else:
        try:
            limit = layers_order.index(freeze_up_to) + 1
        except ValueError:
            return

    for name, param in backbone_body.named_parameters():
        layer_name = name.split(".")[0]
        if layer_name in layers_order[:limit]:
            param.requires_grad = False
