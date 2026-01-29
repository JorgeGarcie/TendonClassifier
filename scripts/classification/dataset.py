import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# Pattern names (indices must match model output order)
PATTERN_NAMES = ["none", "single", "crossed", "double"]
NUM_PATTERNS = len(PATTERN_NAMES)

# Column names for explicit multi-label data in manifest
MULTI_LABEL_COLS = ["pattern_none", "pattern_single", "pattern_crossed", "pattern_double"]


class TendonDataset(Dataset):
    """Dataset for tendon classification + depth regression from gt_manifest.csv.

    Supports both single-label and multi-label modes:

    Single-label mode (multi_label=False):
        Returns (image, force_vector, label, depth_mm) where:
            label: int64 scalar — tendon_type (0=none, 1=single, 2=crossed, 3=double)

    Multi-label mode (multi_label=True):
        Returns (image, force_vector, label, depth_mm) where:
            label: (num_patterns,) float tensor — multi-hot encoding of patterns
                   e.g., [0, 1, 1, 0] means both "single" and "crossed" present

        If the manifest has columns [pattern_none, pattern_single, pattern_crossed, pattern_double],
        those are used directly. Otherwise, the single tendon_type is converted to one-hot.

    Common return values:
        image:        (3, H, W) float tensor, normalized [0, 1]
        force_vector: (6,) float tensor [fx, fy, fz, tx, ty, tz]
        depth_mm:     float32 scalar — depth below surface in mm (0.0 when no tendon)
    """

    FORCE_COLS = ["fx", "fy", "fz", "tx", "ty", "tz"]

    def __init__(self, manifest_csv, img_size=(224, 224), dataset_root=None,
                 exclude_phantom_types=None, multi_label=False):
        self.df = pd.read_csv(manifest_csv)
        self.img_size = img_size
        self.multi_label = multi_label

        if exclude_phantom_types:
            self.df = self.df[~self.df["phantom_type"].isin(exclude_phantom_types)]
            self.df = self.df.reset_index(drop=True)

        if dataset_root is None:
            self.dataset_root = Path(manifest_csv).parent
        else:
            self.dataset_root = Path(dataset_root)

        # Check if manifest has explicit multi-label columns
        self.has_multi_label_cols = all(col in self.df.columns for col in MULTI_LABEL_COLS)
        if multi_label and self.has_multi_label_cols:
            print(f"Multi-label mode: using explicit columns {MULTI_LABEL_COLS}")
        elif multi_label:
            print("Multi-label mode: converting single labels to one-hot vectors")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = self.dataset_root / row["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Force / torque
        force = torch.tensor(
            [row[c] for c in self.FORCE_COLS], dtype=torch.float32
        )

        # Label
        if self.multi_label:
            if self.has_multi_label_cols:
                # Use explicit multi-label columns
                label = torch.tensor(
                    [float(row[col]) for col in MULTI_LABEL_COLS],
                    dtype=torch.float32
                )
            else:
                # Convert single label to one-hot
                single_label = int(row["tendon_type"])
                label = torch.zeros(NUM_PATTERNS, dtype=torch.float32)
                label[single_label] = 1.0
        else:
            # Single-label mode
            label = torch.tensor(int(row["tendon_type"]), dtype=torch.long)

        # Depth (0.0 for non-present tendons)
        raw_depth = row["depth_mm"]
        depth_mm = torch.tensor(
            float(raw_depth) if raw_depth != "" and not pd.isna(raw_depth) else 0.0,
            dtype=torch.float32,
        )

        return img, force, label, depth_mm
