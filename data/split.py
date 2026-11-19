"""
Segment-based train/val dataset splitter.
Ensures all frames of a given 20-second driving scene are placed together in either train or val.
"""

import numpy as np
import pyarrow.dataset as ds
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.constants import COL_SEGMENT


def create_train_val_split(
    img_parquet_path: str,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """
    Split the dataset into train and validation sets by driving segment.
    """
    img_dataset = ds.dataset(img_parquet_path)
    segments_col = img_dataset.to_table(columns=[COL_SEGMENT])
    all_segments = segments_col[COL_SEGMENT].to_pylist()

    unique_segments = sorted(set(all_segments))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_segments)

    n_train = int(len(unique_segments) * train_ratio)
    train_segments = set(unique_segments[:n_train])

    train_indices = []
    val_indices = []
    for i, seg in enumerate(all_segments):
        if seg in train_segments:
            train_indices.append(i)
        else:
            val_indices.append(i)

    return train_indices, val_indices
