import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils
import cv2
import numpy as np  # Needed for the random rotation

AUGMENTATION_FACTOR = 3


class CVATDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        has_gt=True,
        img_size=(240, 320),
        for_segmentation=False,
        for_test=False,
    ):
        """
        Args:
            dataset_dir: root folder containing 'p1/images' and 'p1/masks'
            has_gt: True if ground truth masks are available
            img_size: (H, W) resize for network input
            for_classification: if True, return binary label 'has_feature' instead of segmentation mask
        """
        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
        self.img_size = img_size  # (H, W)
        self.for_segmentation = for_segmentation
        self.for_test = for_test

        # Store original files and features
        self.original_img_files = []
        self.original_has_feature_list = []
        self.mask_basenames = set()

        # Transform for RGB images
        mean_rgb = [0.442, 0.417, 0.593]
        std_rgb = [0.188, 0.190, 0.155]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_rgb, std=std_rgb),
            ]
        )

        # Collect image and mask names. Use p1-p3 for train/val, p4 for test
        if for_test:
            images_path = os.path.join(self.dataset_dir, "p4", "images")
            masks_path = os.path.join(self.dataset_dir, "p4", "masks")
            self.original_img_files.extend(sorted(os.listdir(images_path)))
            if self.has_gt:
                self.mask_basenames.update(
                    {os.path.splitext(frame)[0] for frame in os.listdir(masks_path)}
                )
        else:
            videos = ["p1", "p2", "p3"]
            for vid in videos:
                images_path = os.path.join(self.dataset_dir, vid, "images")
                masks_path = os.path.join(self.dataset_dir, vid, "masks")
                self.original_img_files.extend(sorted(os.listdir(images_path)))
                if self.has_gt:
                    self.mask_basenames.update(
                        {os.path.splitext(frame)[0] for frame in os.listdir(masks_path)}
                    )

        # first, check if the frame exists in the masks set.
        # If it does, check if the mask has any non-zero pixels.
        for frame in self.original_img_files:
            filename = os.path.splitext(frame)[0]
            if filename in self.mask_basenames:
                subfolder = filename.split("_")[0]
                mask_path = os.path.join(
                    self.dataset_dir, subfolder, "masks", f"{filename}.png"
                )
                mask = utils.read_mask(mask_path)
                if mask is not None and mask.sum() > 0:
                    self.original_has_feature_list.append(1)
                else:
                    self.original_has_feature_list.append(0)
            else:
                self.original_has_feature_list.append(0)

        # Calculate the total augmented length
        self.total_original_files = len(self.original_img_files)
        if self.for_segmentation and not self.for_test:
            print(
                f"Augmenting dataset size by a factor of {1 + AUGMENTATION_FACTOR}..."
            )
        else:
            print(
                f"Dataset initialized with {self.total_original_files} original images."
            )

    def __len__(self):
        # The total length is the original number of files multiplied by (1 + AUGMENTATION_FACTOR)
        if self.for_segmentation and not self.for_test:
            return self.total_original_files * (1 + AUGMENTATION_FACTOR)
        else:
            return self.total_original_files

    def __getitem__(self, idx):
        if self.for_segmentation and not self.for_test:
            # Determine the index of the original file and the augmentation ID
            original_idx = idx // (1 + AUGMENTATION_FACTOR)
            augmentation_id = idx % (
                1 + AUGMENTATION_FACTOR
            )  # 0 is original, 1-3 are augmented
        else:
            original_idx = idx
            augmentation_id = 0

        # --- 1. Load and Resize Original Data ---
        filename = os.path.splitext(self.original_img_files[original_idx])[0]
        subfolder = filename.split("_")[0]
        img_file = self.original_img_files[original_idx]
        img_path = os.path.join(self.dataset_dir, subfolder, "images", img_file)

        # Load and resize image (to H x W)
        img = utils.read_rgb(img_path)
        img = cv2.resize(
            img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR
        )

        # Load and resize mask (to H x W)
        mask = None
        if self.has_gt and self.original_has_feature_list[original_idx] == 1:
            mask_path = os.path.join(
                self.dataset_dir, subfolder, "masks", f"{filename}.png"
            )
            mask = utils.read_mask(mask_path)
            mask[mask == 255] = 1
            mask = cv2.resize(
                mask,
                (self.img_size[1], self.img_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if augmentation_id > 0:
            # Apply random rotation to both image and mask (Output is still H x W)
            img, mask = self.rotate_image_and_mask(
                img, mask, original_idx, augmentation_id
            )

        # Apply the same rotation-aware crop to all images (original and rotated)
        L, start_x, end_x, start_y, end_y = self.get_crop_coordinates()

        # Crop if segmentation
        if self.for_segmentation:
            L, start_x, end_x, start_y, end_y = self.get_crop_coordinates()
            img = img[start_y:end_y, start_x:end_x]
            if mask is not None:
                mask = mask[start_y:end_y, start_x:end_x]

        # Apply final image normalization/conversion
        img = self.transform(img)

        # Basic fields
        sample = {
            "input": img,
            "filename": filename,  # Note: filename refers to the original file
            "aug_id": torch.tensor(augmentation_id, dtype=torch.long),
            "has_feature": torch.tensor(
                self.original_has_feature_list[original_idx], dtype=torch.float32
            ),
        }

        # For classification tasks
        if not self.for_segmentation:
            sample["target"] = sample["has_feature"]

        # For segmentation tasks (if GT available)
        elif self.has_gt and sample["has_feature"] == 1:
            # The mask is already loaded, resized, and potentially rotated
            sample["target"] = torch.LongTensor(mask)

        return sample

    def get_crop_coordinates(self):
        """Calculates and returns the safe crop dimensions L and coordinates (start_x, end_x, start_y, end_y).
        For a 160x160 center crop.
        """
        (h, w) = self.img_size  # h=240, w=320

        start_x = int(w / 2 - 160 / 2)
        end_x = start_x + 160
        start_y = int(h / 2 - 160 / 2)
        end_y = start_y + 160
        L = 160

        return L, start_x, end_x, start_y, end_y

    def rotate_image_and_mask(self, image, mask, original_idx, augmentation_id):
        """
        Applies a random rotation to both image and mask.
        Deterministic based on original_idx and augmentation_id.
        """
        (h, w) = self.img_size

        local_seed = int(original_idx * 1000 + augmentation_id)

        rng = np.random.RandomState(local_seed)

        # e.g., Aug 1: 0-120, Aug 2: 120-240, Aug 3: 240-360
        sectors = 360 / AUGMENTATION_FACTOR
        base_angle = (augmentation_id - 1) * sectors
        angle = base_angle + rng.uniform(0, sectors)

        # --- Perform Rotation ---
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        rotated_image = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        rotated_mask = None
        if mask is not None:
            rotated_mask = cv2.warpAffine(
                mask,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )

        return rotated_image, rotated_mask
