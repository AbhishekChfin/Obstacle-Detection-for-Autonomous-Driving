"""
Constants for Multi-Model Obstacle Detection on Waymo Open Dataset.
Centralizes dataset schema, class mappings, colors, and robustness test conditions.
"""

# Camera definitions
CAMERA_FRONT = 1
CAMERA_FRONT_LEFT = 2
CAMERA_FRONT_RIGHT = 3
CAMERA_SIDE_LEFT = 4
CAMERA_SIDE_RIGHT = 5

CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

# Waymo raw class identifiers in Parquet annotations
WAYMO_TYPE_VEHICLE = 1
WAYMO_TYPE_PEDESTRIAN = 2
WAYMO_TYPE_SIGN = 3
WAYMO_TYPE_CYCLIST = 4

# Model class mapping (0 is background / no-object)
# 1: Vehicle, 2: Pedestrian, 3: Cyclist
WAYMO_TO_MODEL_CLASS = {
    1: 1,  # Vehicle
    2: 2,  # Pedestrian
    4: 3,  # Cyclist
}

MODEL_CLASS_TO_WAYMO = {v: k for k, v in WAYMO_TO_MODEL_CLASS.items()}

# Human-readable labels
CLASS_NAMES = ["background", "vehicle", "pedestrian", "cyclist"]
NUM_CLASSES = 4  # 3 foreground obstacles + 1 background

# RGB visualization palette
CLASS_COLORS_RGB = {
    0: (120, 120, 120),  # Background - neutral gray
    1: (230, 50, 50),    # Vehicle - red
    2: (40, 200, 40),    # Pedestrian - bright green
    3: (40, 100, 240),   # Cyclist - vibrant blue
}

# Parquet dataset column keys
COL_SEGMENT = "key.segment_context_name"
COL_TIMESTAMP = "key.frame_timestamp_micros"
COL_CAMERA = "key.camera_name"
COL_IMAGE = "[CameraImageComponent].image"
COL_BOX_CX = "[CameraBoxComponent].box.center.x"
COL_BOX_CY = "[CameraBoxComponent].box.center.y"
COL_BOX_W = "[CameraBoxComponent].box.size.x"
COL_BOX_H = "[CameraBoxComponent].box.size.y"
COL_BOX_TYPE = "[CameraBoxComponent].type"
COL_OBJECT_ID = "key.camera_object_id"
COL_DIFFICULTY = "[CameraBoxComponent].difficulty_level.detection"

# Image geometry
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1280

# Perturbation conditions for robustness benchmarks (Section 3.5 & Table 2/3)
ROBUSTNESS_CONDITIONS = [
    "clean",
    "gaussian_noise",
    "salt_pepper",
    "blur",
    "brightness_down",
    "brightness_up",
    "contrast_low",
    "contrast_high",
    "fog",
    "rain",
    "jpeg_compression",
]
