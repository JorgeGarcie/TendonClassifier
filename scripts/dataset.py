import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import utils
import cv2


class CVATDataset(Dataset):
    def __init__(self, dataset_dir, has_gt=True, img_size=(240, 320)):
        """
        Args:
            dataset_dir: root folder containing 'images' and 'masks'
            has_gt: True if ground truth masks are available
            img_size: (H, W) resize for network input
        """
        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
        self.img_size = img_size  # (H, W)

        # Transform for RGB images
        mean_rgb = [0.499, 0.493, 0.598]
        std_rgb = [0.217, 0.212, 0.177]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_rgb, std=std_rgb),
            ]
        )

        self.img_files = sorted(os.listdir(os.path.join(dataset_dir, "images")))
        if has_gt:
            self.mask_files = sorted(os.listdir(os.path.join(dataset_dir, "masks")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = os.path.splitext(self.img_files[idx])[0]

        # Load RGB image
        img_path = os.path.join(self.dataset_dir, "images", self.img_files[idx])
        img = utils.read_rgb(img_path)
        # Resize RGB using OpenCV
        img = cv2.resize(
            img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR
        )
        img = self.transform(img)

        sample = {"input": img, "filename": filename}

        if self.has_gt:
            mask_path = os.path.join(self.dataset_dir, "masks", self.mask_files[idx])
            mask = utils.read_mask(mask_path)
            # Resize mask with nearest neighbor
            mask = cv2.resize(
                mask,
                (self.img_size[1], self.img_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = torch.LongTensor(mask)
            sample["target"] = mask

        return sample
