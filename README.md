# Obstacle Detection for Autonomous Driving Using Semantic, Instance, and Transformer Models on the Waymo Open Dataset

Implementation of the Stanford CS231N project:
**"Obstacle Detection for Autonomous Driving Using Semantic, Instance, and Transformer Models on the Waymo Open Dataset"** (Carlos Aranguren).

---

## 1. Project Overview & Paradigms

This package implements and compares the three major paradigms for camera-only obstacle detection:

| Model | Paradigm | Output | Key Characteristics |
|---|---|---|---|
| **Mask R-CNN** | Two-Stage CNN (Instance Segmentation) | Bounding boxes + $28\times 28$ instance masks | Strongest overall accuracy (AP 0.423), excels at large obstacles |
| **DETR** | Transformer Set Prediction | Direct bounding box set ($N=100$ queries) | Anchor-free, Hungarian bipartite matching loss |
| **Deformable DETR** | Multi-Scale Transformer Detection | Direct bounding box set ($N=300$ queries) | Multi-scale attention, fast convergence, superior on small obstacles ($AP_S = 0.246$) |
| **DeepLabv3+** | Fully Convolutional Semantic Segmentation | Per-pixel class map ($H \times W$) | Atrous Spatial Pyramid Pooling (ASPP), highest robustness under severe weather |

---

## 2. Directory Structure

```
multi_model_detection/
├── configs/
│   ├── config.yaml               # Hyperparameters for all 4 models and test conditions
│   └── constants.py              # Waymo class mappings, palette, and condition lists
├── data/
│   ├── dataset.py                # Multi-task PyTorch Dataset (Detection, Instance, Semantic)
│   ├── split.py                  # Driving segment-based train/val splitter
│   └── perturbations.py          # 9 simulated adverse weather & imaging generators
├── models/
│   ├── mask_rcnn.py              # Mask R-CNN (ResNet-50 + FPN)
│   ├── detr.py                   # DETR with Hungarian loss & positional encodings
│   ├── deformable_detr.py        # Deformable DETR with multi-scale attention & N=300
│   ├── deeplabv3plus.py          # DeepLabv3+ with ASPP & pseudo-mask supervision
│   └── matcher.py                # Hungarian bipartite matcher (L1 + GIoU + Classification)
├── utils/
│   ├── helpers.py                # Device detection (MPS/CUDA/CPU), logging, checkpoints
│   ├── metrics.py                # AP (0.50:0.95), AP@50, APS, per-class AP, mIoU, drop %
│   └── visualization.py          # Rendering boxes, instance overlays, and semantic maps
├── scripts/
│   ├── explore_data.py           # Multi-task dataset sanity check & weather previews
│   ├── train_mask_rcnn.py        # Training script for Mask R-CNN
│   ├── train_detr.py             # Training script for DETR / Deformable DETR
│   ├── train_deeplab.py          # Training script for DeepLabv3+
│   ├── evaluate.py               # Clean validation benchmark (Table 1 in paper)
│   └── evaluate_robustness.py    # 9 adverse conditions benchmark (Tables 2 & 3 in paper)
├── config_loader.py              # Dot/dict accessible YAML loader
├── requirements.txt              # Package dependencies
└── README.md                     # Documentation and usage guide
```

---

## 3. Mac & Low-Memory Optimizations

1. **Lazy Loading via Parquet Row Groups**:
   - `front_camera_images_2sec.parquet` (3.25 GB) is read row-group by row-group directly in **0.5 ms** per image, keeping memory usage minimal.
2. **Pre-Indexed Bounding Boxes (15 MB RAM)**:
   - All 184,880 box annotations are pre-indexed into a fast hashmap at initialization, avoiding disk scans during training.
3. **Apple Silicon Acceleration (`mps`)**:
   - Device selector routes tensors automatically to Apple Silicon GPU (`mps`), with full fallback to CUDA or CPU.
4. **Gradient Accumulation**:
   - Default `batch_size: 2` with `gradient_accumulation_steps: 4` yields an effective batch size of 8 while staying comfortably within Mac RAM limits.
5. **Configurable Resolution**:
   - Default resolution is $1280 \times 768$ (from paper Section 3.5), with options in `config.yaml` to reduce to $960 \times 640$ if lower memory is required.

---

## 4. Usage Instructions

All commands run inside the virtual environment:
```bash
source .venv/bin/activate
```

### Step 1: Sanity Check & Weather Perturbation Previews
```bash
python -m multi_model_detection.scripts.explore_data
```
*Verifies multi-task targets, instance masks, pseudo-masks, and exports sample images for all 9 simulated weather conditions into `multi_model_detection/visualizations/`.*

---

### Step 2: Training Models Manually

#### Option A: Train Mask R-CNN (Instance Segmentation)
```bash
python -m multi_model_detection.scripts.train_mask_rcnn
```

#### Option B: Train DETR (Transformer Detection, N=100)
```bash
python -m multi_model_detection.scripts.train_detr --model detr
```

#### Option C: Train Deformable DETR (Multi-Scale Transformer, N=300)
```bash
python -m multi_model_detection.scripts.train_detr --model deformable_detr
```

#### Option D: Train DeepLabv3+ (Semantic Segmentation)
```bash
python -m multi_model_detection.scripts.train_deeplab
```

---

### Step 3: Clean Validation Benchmark (Table 1)

Evaluates performance on clean Waymo validation data and outputs:
- $AP (0.50:0.95)$
- $AP@50$
- $AP_S$ (Small obstacles $< 32^2\text{ px}$)
- $AP$ on Pedestrians, Vehicles, Cyclists
- $mIoU$ (for DeepLabv3+)

```bash
# Evaluate Mask R-CNN:
python -m multi_model_detection.scripts.evaluate \
    --model mask_rcnn \
    --checkpoint multi_model_detection/checkpoints/mask_rcnn_best.pth \
    --visualize --num_vis 10

# Evaluate Deformable DETR:
python -m multi_model_detection.scripts.evaluate \
    --model deformable_detr \
    --checkpoint multi_model_detection/checkpoints/deformable_detr_epoch_080.pth \
    --visualize --num_vis 10

# Evaluate DeepLabv3+:
python -m multi_model_detection.scripts.evaluate \
    --model deeplab \
    --checkpoint multi_model_detection/checkpoints/deeplabv3plus_best.pth \
    --visualize --num_vis 10
```

---

### Step 4: Robustness Benchmark across 9 Adverse Conditions (Tables 2 & 3)

Evaluates degradation under simulated adverse weather:
1. `Clean` (baseline)
2. `Gaussian Noise` ($\sigma=0.05$)
3. `Salt & Pepper Noise` ($0.02$)
4. `Blur` (radius $=3$)
5. `Brightness Down` ($-30\%$)
6. `Brightness Up` ($+30\%$)
7. `Contrast Low` ($\times 0.7$)
8. `Contrast High` ($\times 1.3$)
9. `Simulated Fog` (opacity $=0.4$)
10. `Simulated Rain` (semi-transparent streaks)
11. `JPEG Compression` (quality $=30$)

```bash
python -m multi_model_detection.scripts.evaluate_robustness \
    --model mask_rcnn \
    --checkpoint multi_model_detection/checkpoints/mask_rcnn_best.pth

python -m multi_model_detection.scripts.evaluate_robustness \
    --model deeplab \
    --checkpoint multi_model_detection/checkpoints/deeplabv3plus_best.pth
```
Outputs formatted comparison tables matching Table 2 (Detection AP drop %) and Table 3 (Semantic mIoU drop %) from Section 4.6 of the paper.
