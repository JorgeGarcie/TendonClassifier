import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from classification_model import ResNetBinaryClassifier, MobileNetBinaryClassifier
from segmentation_model import MiniUNet
from direction_predict_pca import img_pca_dir
from dataset import CVATDataset
import utils


# DATASET WRAPPER (For Testing/Benchmarking)
class PipelineValidationDataset(CVATDataset):
    """
    Inherits from CVATDataset but overrides __getitem__
    to return the FULL 320x240 image (skipping the training crop).
    This simulates the raw camera input for the pipeline benchmark.
    """

    def __len__(self):
        # Ignore augmentation factor
        return len(self.original_img_files)

    def __getitem__(self, idx):
        img_file = self.original_img_files[idx]

        filename = os.path.splitext(img_file)[0]
        subfolder = filename.split("_")[0]
        img_path = os.path.join(self.dataset_dir, subfolder, "images", img_file)

        # Load and resize
        img = utils.read_rgb(img_path)
        img = cv2.resize(
            img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR
        )

        # Transform
        img_tensor = self.transform(img)

        # Return both the tensor for the model and the numpy image for the video
        return {"input": img_tensor, "filename": filename, "vis_img": img}


class ModelPipeline:
    def __init__(self, detector_path, segmenter_path, device="cpu"):
        self.device = device

        print("Initializing models...")
        self.detector = MobileNetBinaryClassifier(
            pretrained=False, freeze_backbone=True
        ).to(device)
        # self.detector = ResNetBinaryClassifier(
        #     pretrained=False, freeze_backbone=True
        # ).to(device)
        self.segmenter = MiniUNet().to(device)

        load_weights(
            self.detector, self.segmenter, detector_path, segmenter_path, device
        )

    def detect(self, full_tensor):
        # input: (1, 3, 240, 320)
        with torch.no_grad():
            output = self.detector(full_tensor)
        return output.item() > 0.5

    def segment_batch(self, batch_tensor):
        # input: (5, 3, 160, 160)
        with torch.no_grad():
            output = self.segmenter(batch_tensor)  # (Batch, 2, 160, 160)
            masks = torch.argmax(output, dim=1)  # (Batch, 160, 160)

        return masks.cpu().numpy().astype(np.uint8)

    def keep_largest_blob(self, mask):
        """
        Keeps only the largest continuous object in the binary mask.
        Removes noise/small disconnected artifacts.
        """
        # connectivity=8 checks all surrounding pixels
        # stats return: [x, y, width, height, area]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # If only background, return mask.
        if num_labels < 2:
            return mask

        # Find the largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned_mask = np.zeros_like(mask)
        cleaned_mask[labels == largest_label] = 1

        return cleaned_mask

    def check_centroid_safe(self, mask, boundary_percent=0.10):
        """
        Returns False if the mask centroid is within the left/right
        boundary percentage (e.g., 10%).
        """
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return False  # Empty mask is unsafe

        cx = int(M["m10"] / M["m00"])
        width = mask.shape[1]

        limit_px = width * boundary_percent

        # Check if too close to Left or Right edge
        if cx < limit_px or cx > (width - limit_px):
            return False

        return True

    def get_pca_dir_centroid(self, mask):
        # mask: (160,160) binary
        # get dir and centroid from PCA. wrapper for helper function
        dir, centroid = img_pca_dir(mask)
        return dir, centroid


def load_weights(
    clas_model, seg_model, classification_ckpt_path, segmentation_ckpt_path, device
):
    """
    Loads classification and segmentation weights from .pth files
    """

    print(f"Loading {classification_ckpt_path} and {segmentation_ckpt_path}...")

    # Load the files, mapping to the current device
    clas_ckpt = torch.load(classification_ckpt_path, map_location=device)
    seg_ckpt = torch.load(segmentation_ckpt_path, map_location=device)

    clas_model.load_state_dict(clas_ckpt["model_state_dict"])
    seg_model.load_state_dict(seg_ckpt["model_state_dict"])

    clas_model.to(device)
    seg_model.to(device)
    clas_model.eval()
    seg_model.eval()
    print("-> Loaded successfully.")
    return clas_model, seg_model


def get_five_crops_tensor(full_tensor):
    """
    Extract 5 crops directly using Tensor slicing.
    """
    t = full_tensor.squeeze(0)  # Remove batch dim

    base_x, base_y = 80, 40
    offset = 40

    coords = [
        (base_x, base_y),  # Center
        (base_x - offset, base_y - offset),  # TL
        (base_x + offset, base_y - offset),  # TR
        (base_x - offset, base_y + offset),  # BL
        (base_x + offset, base_y + offset),  # BR
    ]

    crops_list = []

    for x, y in coords:
        crop = t[:, y : y + 160, x : x + 160]
        crops_list.append(crop)

    return torch.stack(crops_list)


def benchmark_pipeline(
    dataset_dir,
    det_ckpt,
    seg_ckpt,
    multi_crop=False,
    num_samples=50,
    output_video="inference_output.mp4",
):
    device = "cpu"

    file_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(file_dir, dataset_dir)

    test_set = PipelineValidationDataset(dataset_dir, has_gt=False, for_test=True)
    loader = DataLoader(
        test_set, batch_size=1, shuffle=False
    )  # Shuffle False for video continuity

    # Setup Model
    pipeline = ModelPipeline(det_ckpt, seg_ckpt, device)

    # Video Writer Setup
    # Assuming 320x240 based on dataset constants
    frame_width = 320
    frame_height = 240
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    print(f"\n--- Starting Benchmark & Video Gen on {num_samples} frames ---")

    det_times = []
    segment_times = []
    pca_times = []

    # Warmup
    print("Warming up...")
    iter_loader = iter(loader)
    for _ in range(2):
        try:
            batch = next(iter_loader)
            frame = batch["input"].to(device)
            pipeline.detect(frame)
        except StopIteration:
            break

    # Benchmark Loop
    print(f"Running measurements...")
    count = 0

    with torch.no_grad():
        for batch in loader:
            if count >= num_samples:
                break

            frame_tensor = batch["input"].to(device)
            # Retrieve original image for visualization (convert RGB to BGR for OpenCV)
            vis_img = batch["vis_img"][0].numpy()
            final_frame = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

            filename = batch["filename"][0]

            start_det_t = time.time()
            has_obj = pipeline.detect(frame_tensor)
            end_det_t = time.time()

            elapsed_ms_det = (end_det_t - start_det_t) * 1000
            det_times.append(elapsed_ms_det)

            direction = None
            best_mask = None

            if has_obj:
                start_t = time.time()

                if multi_crop:
                    batch_crops = get_five_crops_tensor(frame_tensor)
                    masks = pipeline.segment_batch(batch_crops)
                else:
                    mask = pipeline.segment_batch(frame_tensor)  # (1, 160, 160)
                    masks = [mask[0]]  # Wrap in list to unify logic below

                # Select best mask
                best_score = -1

                if multi_crop:
                    for m in masks:
                        score = np.count_nonzero(m)
                        if score > best_score:
                            best_score = score
                            best_mask = m
                else:
                    best_mask = masks[0]

                # keep only largest contiguous mask
                best_mask = pipeline.keep_largest_blob(best_mask)

                # reject spurious detections near edge
                is_safe = pipeline.check_centroid_safe(best_mask, boundary_percent=0.10)

                if not is_safe:
                    # If object is at the edge, treat as NO OBJECT
                    has_obj = False
                    best_mask = None

                end_t = time.time()
                elapsed_ms_segment = (end_t - start_t) * 1000
                segment_times.append(elapsed_ms_segment)

            if has_obj and best_mask is not None:
                start_pca = time.time()
                direction, centroid = pipeline.get_pca_dir_centroid(best_mask)
                pca_times.append((time.time() - start_pca) * 1000)

                # VIZ
                mask_vis = cv2.resize(
                    best_mask,
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                overlay = final_frame.copy()
                overlay[mask_vis == 1] = (0, 255, 0)  # Green
                cv2.addWeighted(overlay, 0.5, final_frame, 0.5, 0, final_frame)

                if centroid is not None:
                    scale_x = frame_width / best_mask.shape[1]
                    scale_y = frame_height / best_mask.shape[0]
                    cx, cy = int(centroid[0] * scale_x), int(centroid[1] * scale_y)

                    cv2.circle(final_frame, (cx, cy), 5, (0, 0, 255), -1)
                    if direction is not None:
                        end_point = (
                            int(cx + direction[0] * 30),
                            int(cy + direction[1] * 30),
                        )
                        cv2.line(final_frame, (cx, cy), end_point, (0, 0, 255), 2)

                cv2.putText(
                    final_frame,
                    "DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                # if Detection failed initially OR we filtered it out because too close to edge
                cv2.putText(
                    final_frame,
                    "NO OBJECT",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            out_vid.write(final_frame)
            count += 1
            if count % 20 == 0:
                print(f"Processed {count}...")

    out_vid.release()
    print(f"\nSaved inference video to: {output_video}")

    # Report Stats
    print("\n" + "=" * 40)
    print("BENCHMARK RESULTS (CPU)")
    print("=" * 40)

    for idx, times in enumerate([det_times, segment_times, pca_times]):
        if len(times) == 0:
            continue
        task_name = ["Detection", "Segmentation", "PCA Direction"][idx]
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps_est = 1000 / mean_time

        print(f"{task_name}:")
        print(f"  Mean: {mean_time:.2f} ms")
        print(f"  Std:  {std_time:.2f} ms")
        print(f"  FPS:  {fps_est:.2f}")
        print("-" * 20)
    print("=" * 40)


if __name__ == "__main__":
    DATA_DIR = "../data"
    DET_CKPT = "checkpoints/classifier_mobilenetv3_small-2025-12-04.pth.tar"
    SEG_CKPT = "checkpoints/checkpoint-segmentation-2025-12-04.pth.tar"

    benchmark_pipeline(
        DATA_DIR,
        DET_CKPT,
        SEG_CKPT,
        multi_crop=False,
        num_samples=784,
        output_video="output/inference_output.mp4",
    )
