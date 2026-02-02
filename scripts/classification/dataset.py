import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TendonDataset(Dataset):
    """Dataset for tendon classification + depth regression from gt_manifest.csv.

    Returns (image, force_vector, label, depth_mm) where:
        image:        (3, H, W) float tensor, normalized [0, 1]
        force_vector: (6,) float tensor [fx, fy, fz, tx, ty, tz]
        label:        int64 scalar — tendon_type (0=none, 1=single, 2=crossed)
        depth_mm:     float32 scalar — depth below surface in mm (0.0 when no tendon)
    """

    FORCE_COLS = ["fx", "fy", "fz", "tx", "ty", "tz"]
    # Maybe remove the torque.
    # Have to Classify both tendons

    def __init__(self, manifest_csv, img_size=(224, 224), dataset_root=None,
                 exclude_phantom_types=None):
        self.df = pd.read_csv(manifest_csv)
        self.img_size = img_size

        if exclude_phantom_types:
            self.df = self.df[~self.df["phantom_type"].isin(exclude_phantom_types)]
            self.df = self.df.reset_index(drop=True)

        if dataset_root is None:
            self.dataset_root = Path(manifest_csv).parent
        else:
            self.dataset_root = Path(dataset_root)

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
        label = torch.tensor(int(row["tendon_type"]), dtype=torch.long)

        # Depth (0.0 for non-present tendons)
        raw_depth = row["depth_mm"]
        depth_mm = torch.tensor(
            float(raw_depth) if raw_depth != "" and not pd.isna(raw_depth) else 0.0,
            dtype=torch.float32,
        )

        return img, force, label, depth_mm


class TendonDatasetV2(Dataset):
    """Enhanced dataset with ImageNet normalization, temporal support, and augmentation.

    Returns vary based on mode:
    - Spatial mode: (image, force, label, depth_mm)
    - Temporal mode: (images, force, label, depth_mm, mask) where images is (T, 3, H, W)

    Key improvements over TendonDataset:
    - ImageNet normalization for pretrained encoders
    - Temporal sequence support (previous N frames)
    - Optional subtraction of reference frame to highlight deformations
    - Data augmentation pipeline
    """

    FORCE_COLS = ["fx", "fy", "fz", "tx", "ty", "tz"]

    def __init__(
        self,
        manifest_csv: str,
        img_size: Tuple[int, int] = (224, 224),
        dataset_root: Optional[str] = None,
        exclude_phantom_types: Optional[List[str]] = None,
        normalization: str = "imagenet",  # "imagenet" or "simple"
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        temporal_frames: int = 1,  # 1 = spatial mode, >1 = temporal mode
        subtraction_enabled: bool = False,
        subtraction_reference: str = "first_frame",  # "first_frame", "pre_contact", or path
        augmentation: Optional[dict] = None,
        return_force_sequence: bool = False,  # For temporal_force model
    ):
        """Initialize the dataset.

        Args:
            manifest_csv: Path to gt_manifest.csv
            img_size: Target image size (H, W)
            dataset_root: Root directory for images (default: same as manifest)
            exclude_phantom_types: List of phantom types to exclude
            normalization: "imagenet" for pretrained models, "simple" for 0-1 scaling
            norm_mean: Custom normalization mean (overrides normalization preset)
            norm_std: Custom normalization std (overrides normalization preset)
            temporal_frames: Number of frames for temporal mode (1 = spatial only)
            subtraction_enabled: Whether to subtract reference frame
            subtraction_reference: Reference frame type for subtraction:
                - "first_frame": Use first frame of each run (default)
                - "pre_contact": Use frame before contact (requires 'contact_frame' column)
                - Path string: Use a global untouched image file
            augmentation: Augmentation config dict or None
            return_force_sequence: If True and temporal_frames > 1, return force sequence
        """
        self.df = pd.read_csv(manifest_csv)
        self.img_size = img_size
        self.temporal_frames = temporal_frames
        self.subtraction_enabled = subtraction_enabled
        self.subtraction_reference = subtraction_reference
        self.augmentation = augmentation
        self.return_force_sequence = return_force_sequence

        # Filter phantom types
        if exclude_phantom_types:
            self.df = self.df[~self.df["phantom_type"].isin(exclude_phantom_types)]
            self.df = self.df.reset_index(drop=True)

        # Dataset root
        if dataset_root is None:
            self.dataset_root = Path(manifest_csv).parent
        else:
            self.dataset_root = Path(dataset_root)

        # Normalization
        self.normalization = normalization
        if norm_mean is not None and norm_std is not None:
            self.norm_mean = torch.tensor(norm_mean).view(3, 1, 1)
            self.norm_std = torch.tensor(norm_std).view(3, 1, 1)
        elif normalization == "imagenet":
            self.norm_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            self.norm_std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        else:
            self.norm_mean = None
            self.norm_std = None

        # Build temporal index if needed
        if temporal_frames > 1:
            self._build_temporal_index()

        # Compute reference frames for subtraction if needed
        if subtraction_enabled:
            self._build_reference_index()

    def _build_temporal_index(self):
        """Build index mapping each sample to its previous frames."""
        self.temporal_index = {}

        # Group by run_id
        for run_id, group in self.df.groupby("run_id"):
            # Sort by frame_idx within each run
            sorted_group = group.sort_values("frame_idx")
            indices = sorted_group.index.tolist()

            for i, idx in enumerate(indices):
                # Get previous frames (pad with first frame if not enough)
                start = max(0, i - self.temporal_frames + 1)
                prev_indices = indices[start:i + 1]

                # Pad with first frame if needed
                while len(prev_indices) < self.temporal_frames:
                    prev_indices.insert(0, prev_indices[0])

                self.temporal_index[idx] = prev_indices

    def _build_reference_index(self):
        """Build index mapping each sample to its reference frame for subtraction."""
        self.reference_index = {}
        self.global_reference_path = None

        if self.subtraction_reference == "first_frame":
            # Use first frame of each run as reference
            for run_id, group in self.df.groupby("run_id"):
                sorted_group = group.sort_values("frame_idx")
                first_idx = sorted_group.index[0]
                for idx in sorted_group.index:
                    self.reference_index[idx] = first_idx

        elif self.subtraction_reference == "pre_contact":
            # Use frame just before contact starts
            # Requires 'contact_frame' column in manifest indicating first contact frame
            if "contact_frame" in self.df.columns:
                for run_id, group in self.df.groupby("run_id"):
                    sorted_group = group.sort_values("frame_idx")
                    contact_frame = sorted_group["contact_frame"].iloc[0]
                    # Find frame just before contact
                    pre_contact = sorted_group[sorted_group["frame_idx"] < contact_frame]
                    if len(pre_contact) > 0:
                        ref_idx = pre_contact.index[-1]  # Last frame before contact
                    else:
                        ref_idx = sorted_group.index[0]  # Fallback to first
                    for idx in sorted_group.index:
                        self.reference_index[idx] = ref_idx
            else:
                print("Warning: 'contact_frame' column not found, using first_frame")
                self._build_reference_index_first_frame()

        elif Path(self.subtraction_reference).exists() or "/" in self.subtraction_reference:
            # Global reference image path provided
            self.global_reference_path = Path(self.subtraction_reference)
            if not self.global_reference_path.is_absolute():
                self.global_reference_path = self.dataset_root / self.global_reference_path
            # No per-sample index needed, all use the same global image

        else:
            # Unknown reference type, fall back to first frame
            print(f"Warning: Unknown subtraction_reference '{self.subtraction_reference}', using first_frame")
            self._build_reference_index_first_frame()

    def _build_reference_index_first_frame(self):
        """Build reference index using first frame of each run."""
        for run_id, group in self.df.groupby("run_id"):
            sorted_group = group.sort_values("frame_idx")
            first_idx = sorted_group.index[0]
            for idx in sorted_group.index:
                self.reference_index[idx] = first_idx

    def _load_reference_image(self, idx: int) -> torch.Tensor:
        """Load the reference image for subtraction.

        Args:
            idx: DataFrame index (used if not using global reference)

        Returns:
            Reference image tensor of shape (3, H, W)
        """
        if self.global_reference_path is not None:
            # Load global reference image
            img = cv2.imread(str(self.global_reference_path))
            if img is None:
                raise FileNotFoundError(f"Could not load reference image: {self.global_reference_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            if self.norm_mean is not None and self.norm_std is not None:
                img = (img - self.norm_mean) / self.norm_std
            return img
        else:
            # Load per-sample reference
            ref_idx = self.reference_index[idx]
            return self._load_image(ref_idx)

    def _load_image(self, idx: int) -> torch.Tensor:
        """Load and preprocess a single image.

        Args:
            idx: DataFrame index

        Returns:
            Image tensor of shape (3, H, W)
        """
        row = self.df.iloc[idx]
        img_path = self.dataset_root / row["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Apply normalization
        if self.norm_mean is not None and self.norm_std is not None:
            img = (img - self.norm_mean) / self.norm_std

        return img

    def _apply_augmentation(self, img: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to an image.

        Args:
            img: Image tensor of shape (3, H, W)

        Returns:
            Augmented image tensor
        """
        if self.augmentation is None or not self.augmentation.get("enabled", False):
            return img

        # Horizontal flip
        if self.augmentation.get("horizontal_flip", False):
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])

        # Rotation (simple 90-degree rotations for now)
        rotation_degrees = self.augmentation.get("rotation_degrees", 0)
        if rotation_degrees > 0:
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])

        # Color jitter (simplified)
        color_jitter = self.augmentation.get("color_jitter", {})
        if color_jitter:
            brightness = color_jitter.get("brightness", 0)
            if brightness > 0:
                factor = 1 + (torch.rand(1).item() * 2 - 1) * brightness
                img = img * factor

        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image(s)
        if self.temporal_frames > 1:
            # Temporal mode: load sequence of frames
            frame_indices = self.temporal_index[idx]
            images = torch.stack([self._load_image(i) for i in frame_indices])

            # Create mask (all valid for now, padding handled in _build_temporal_index)
            mask = torch.ones(self.temporal_frames, dtype=torch.bool)

            # Load force sequence if requested (for temporal_force model)
            if self.return_force_sequence:
                forces = []
                for frame_idx in frame_indices:
                    frame_row = self.df.iloc[frame_idx]
                    force_vec = torch.tensor(
                        [frame_row[c] for c in self.FORCE_COLS], dtype=torch.float32
                    )
                    forces.append(force_vec)
                force = torch.stack(forces)  # (T, 6)
            else:
                # Just current frame's force
                force = torch.tensor(
                    [row[c] for c in self.FORCE_COLS], dtype=torch.float32
                )
        else:
            # Spatial mode: single image
            images = self._load_image(idx)

            # Apply subtraction if enabled
            if self.subtraction_enabled:
                ref_img = self._load_reference_image(idx)
                images = images - ref_img

            # Apply augmentation
            images = self._apply_augmentation(images)

            # Force / torque (single frame)
            force = torch.tensor(
                [row[c] for c in self.FORCE_COLS], dtype=torch.float32
            )

        # Label
        label = torch.tensor(int(row["tendon_type"]), dtype=torch.long)

        # Depth
        raw_depth = row["depth_mm"]
        depth_mm = torch.tensor(
            float(raw_depth) if raw_depth != "" and not pd.isna(raw_depth) else 0.0,
            dtype=torch.float32,
        )

        if self.temporal_frames > 1:
            return images, force, label, depth_mm, mask
        return images, force, label, depth_mm

    def get_class_weights(self, num_classes: int = 4) -> torch.Tensor:
        """Compute class weights for imbalanced dataset.

        Args:
            num_classes: Number of classes

        Returns:
            Tensor of class weights (inverse frequency)
        """
        counts = self.df["tendon_type"].value_counts().sort_index()
        # Fill missing classes with 1 to avoid division by zero
        counts_full = np.zeros(num_classes)
        for cls, count in counts.items():
            if cls < num_classes:
                counts_full[cls] = count

        # Replace zeros with 1
        counts_full = np.maximum(counts_full, 1)

        # Inverse frequency weighting
        weights = 1.0 / counts_full
        weights = weights / weights.sum() * num_classes  # Normalize

        return torch.tensor(weights, dtype=torch.float32)

    def get_class_distribution(self) -> dict:
        """Get class distribution statistics.

        Returns:
            Dict with class counts and percentages
        """
        counts = self.df["tendon_type"].value_counts().sort_index()
        total = len(self.df)
        return {
            "counts": counts.to_dict(),
            "percentages": (counts / total * 100).to_dict(),
            "total": total,
        }


def create_dataset(config) -> TendonDatasetV2:
    """Create a dataset from config.

    Args:
        config: DataConfig dataclass or dict

    Returns:
        TendonDatasetV2 instance
    """
    # Handle both dataclass and dict
    if hasattr(config, "manifest"):
        manifest = config.manifest
        img_size = config.img_size
        exclude_phantoms = config.exclude_phantoms
        norm_type = config.normalization.type
        norm_mean = config.normalization.mean
        norm_std = config.normalization.std
        subtraction_enabled = config.subtraction.enabled
        subtraction_ref = config.subtraction.reference
        aug = config.augmentation
        augmentation = {
            "enabled": aug.enabled,
            "horizontal_flip": aug.horizontal_flip,
            "rotation_degrees": aug.rotation_degrees,
            "color_jitter": {
                "brightness": aug.color_jitter.brightness,
                "contrast": aug.color_jitter.contrast,
                "saturation": aug.color_jitter.saturation,
            },
        }
    else:
        manifest = config.get("manifest", "../labeling/output/gt_dataset/gt_manifest.csv")
        img_size = config.get("img_size", 224)
        exclude_phantoms = config.get("exclude_phantoms")
        norm_cfg = config.get("normalization", {})
        norm_type = norm_cfg.get("type", "imagenet")
        norm_mean = norm_cfg.get("mean")
        norm_std = norm_cfg.get("std")
        sub_cfg = config.get("subtraction", {})
        subtraction_enabled = sub_cfg.get("enabled", False)
        subtraction_ref = sub_cfg.get("reference", "first_frame")
        augmentation = config.get("augmentation")

    return TendonDatasetV2(
        manifest_csv=manifest,
        img_size=(img_size, img_size) if isinstance(img_size, int) else img_size,
        exclude_phantom_types=exclude_phantoms,
        normalization=norm_type,
        norm_mean=norm_mean if norm_type != "imagenet" else None,
        norm_std=norm_std if norm_type != "imagenet" else None,
        subtraction_enabled=subtraction_enabled,
        subtraction_reference=subtraction_ref,
        augmentation=augmentation,
    )
