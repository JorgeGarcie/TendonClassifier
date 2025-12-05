"""
Segmentation training script. Inspired by CS227 course material.

Currently supports training on a CPU with 80/20 train/val split.
"""

import os
from argparse import ArgumentParser
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import CVATDataset, AUGMENTATION_FACTOR
from segmentation_model import MiniUNet
import utils
import train_utils


def iou(prediction, target):
    """
    Computes IoU, safely handling division by zero.
    """
    _, pred = torch.max(prediction, dim=1)
    batch_num = prediction.shape[0]
    batch_ious = list()
    epsilon = 1e-6

    for batch_id in range(batch_num):
        # Assuming binary segmentation: class 1 is the object, class 0 is background
        class_id = 1

        mask_pred = (pred[batch_id] == class_id).int()
        mask_target = (target[batch_id] == class_id).int()

        # Check if the ground truth object is present in this image (mask_target.sum() > 0)
        if mask_target.sum() > 0:
            intersection = (mask_pred * mask_target).sum().float()
            union = (mask_pred + mask_target).sum().float() - intersection

            # Prevent Division by Zero
            batch_ious.append(float(intersection) / float(union + epsilon))

    return batch_ious


def save_prediction(model, device, dataloader, dataset, output_dir):
    """
    Save predicted masks for a dataset, handles augmented files by using
    a unique identifier from the batch.
    """
    pred_dir = os.path.join(output_dir, "pred/")
    os.makedirs(pred_dir, exist_ok=True)
    print(f"Saving predicted masks to {pred_dir}")

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            data = batch["input"].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            filenames = batch["filename"]

            if "aug_id" in batch:
                aug_ids = batch["aug_id"]
            else:
                # Fallback for original files
                aug_ids = [0] * pred.shape[0]

            for i in range(pred.shape[0]):
                # Base filename will be the same for all augmented versions
                base_filename = filenames[i]

                # Get the augmentation identifier
                aug_id = (
                    aug_ids[i].item()
                    if isinstance(aug_ids[i], torch.Tensor)
                    else aug_ids[i]
                )

                # Create a unique filename
                if aug_id > 0:
                    unique_filename = f"{base_filename}_aug{aug_id}"
                else:
                    unique_filename = base_filename

                mask = pred[i].cpu().numpy()
                mask_path = os.path.join(pred_dir, f"{unique_filename}_pred.png")

                utils.write_mask(mask, mask_path)
                utils.write_rgb(
                    utils.mask2rgb(mask),
                    mask_path.replace("_pred.png", "_pred_rgb.png"),
                )


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    train_loss_sum, train_iou_sum = 0.0, 0.0
    data_size = 0

    batch_loss_list = []

    train_bar = tqdm(train_loader, desc="Training")

    for batch in train_bar:
        batch_size = batch["input"].size(0)
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Store loss for plotting
        batch_loss_list.append(loss.item())

        data_size += batch_size
        train_loss_sum += loss.item() * batch_size
        train_iou_sum += np.sum(iou(outputs, targets))

        # tqdm progress bar description with the current loss
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
        #    print(f"  Batch {batch_idx+1}/{total_batches} - loss: {loss.item():.4f}")

    train_loss = train_loss_sum / data_size
    train_iou = train_iou_sum / data_size

    # Return the list of batch losses
    return train_loss, train_iou, batch_loss_list


def val(model, device, val_loader, criterion):
    """
    Similar to train(), but no need to backward and optimize.
    """
    model.eval()
    val_loss_sum, val_iou_sum = 0, 0
    data_size = 0

    val_bar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for batch in val_bar:
            batch_size = batch["input"].size(0)
            output = model(batch["input"].to(device))
            target = batch["target"].to(device)
            loss = criterion(output, target)

            data_size += batch_size
            val_loss_sum += loss.item() * batch_size
            val_iou_sum += np.sum(iou(output, target))

    val_loss = val_loss_sum / data_size
    val_iou = val_iou_sum / data_size
    return val_loss, val_iou


def test(model, test_loader, device):
    """
    Test the model on the test dataset and return mIoU.
    """
    model.eval()
    test_iou_sum = 0
    data_size = 0

    test_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for batch in test_bar:
            batch_size = batch["input"].size(0)
            output = model(batch["input"].to(device))
            target = batch["target"].to(device)

            data_size += batch_size
            test_iou_sum += np.sum(iou(output, target))

    test_miou = test_iou_sum / data_size
    return test_miou


def main():
    train_utils.setup_single_threaded_torch()

    # Load arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint file to initialize the model with. Training will resume from the epoch saved in the checkpoint.",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="If set, only run testing on the test set after loading the checkpoint.",
    )
    args = parser.parse_args()

    # Check if GPU is being detected
    device = train_utils.get_device()

    # Define directories (running from scripts/)
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/")

    # Create Datasets.
    # Load all dataset (assumes has_gt=True)
    full_dataset = CVATDataset(dataset_dir=root_dir, has_gt=True, for_segmentation=True)
    test_dataset = CVATDataset(
        dataset_dir=root_dir, has_gt=True, for_segmentation=True, for_test=True
    )

    # Filter to only indices with objects
    original_indices_with_objects = [
        i
        for i, has_feat in enumerate(full_dataset.original_has_feature_list)
        if has_feat == 1
    ]
    print(
        f"Using {len(original_indices_with_objects)} of {len(full_dataset.original_has_feature_list)} original frames (with objects)."
    )

    # Shuffle and split indices
    random.seed(42)
    random.shuffle(original_indices_with_objects)
    split = int(0.8 * len(original_indices_with_objects))

    original_train_indices = original_indices_with_objects[:split]
    original_val_indices = original_indices_with_objects[split:]

    for idx_list in [original_train_indices, original_val_indices]:
        final_indices = []
        copies = 1 + AUGMENTATION_FACTOR
        for idx in idx_list:
            start_idx = idx * copies
            final_indices.extend(range(start_idx, start_idx + copies))
        if idx_list is original_train_indices:
            train_indices = final_indices
        else:
            val_indices = final_indices

    print(f"Total Training Samples (Original + Augmented): {len(train_indices)}")
    print(f"Total Validation Samples (Original + Augmented): {len(val_indices)}")

    # Create Subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Prepare Dataloaders. You can use check_dataloader() to check your implementation.
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Prepare model
    model = MiniUNet().to(device)

    # Define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load checkpoint if provided
    if args.checkpoint:
        model, epoch, metric = train_utils.load_checkpoint(
            model, args.checkpoint, device
        )
        best_miou = metric["model_miou"]
        train_loss_list = metric["train_loss_list"]
        train_batch_loss_list = metric["train_batch_loss_list"]
        train_miou_list = metric["train_miou_list"]
        val_loss_list = metric["val_loss_list"]
        val_miou_list = metric["val_miou_list"]
        print(
            f"Loaded a checkpoint from {args.checkpoint} at epoch {epoch} with best mIoU {best_miou}."
        )
        epoch += 1  # start training from the next epoch
    else:
        epoch = 1
        best_miou = float("-inf")
        train_loss_list, train_miou_list, val_loss_list, val_miou_list = (
            list(),
            list(),
            list(),
            list(),
        )
        train_batch_loss_list = []
        print(
            f"No checkpoint provided. Starting training from scratch at epoch {epoch}."
        )

    if not args.test_only:
        # Train and validate the model
        print(f"Training from epoch {epoch}...")
        max_epochs = 29
        newly_saved = False
        while epoch <= max_epochs:
            print("Epoch (", epoch, "/", max_epochs, ")")
            train_loss, train_miou, train_batch_loss = train(
                model, device, train_loader, criterion, optimizer
            )
            val_loss, val_miou = val(model, device, val_loader, criterion)
            train_loss_list.append(train_loss)
            train_miou_list.append(train_miou)
            train_batch_loss_list.extend(train_batch_loss)
            val_loss_list.append(val_loss)
            val_miou_list.append(val_miou)
            print("Train loss & mIoU: %0.2f %0.2f" % (train_loss, train_miou))
            print("Validation loss & mIoU: %0.2f %0.2f" % (val_loss, val_miou))
            if val_miou > best_miou:
                best_miou = val_miou
                train_utils.save_checkpoint(
                    model,
                    epoch,
                    {
                        "model_miou": best_miou,
                        "train_loss_list": train_loss_list,
                        "train_batch_loss_list": train_batch_loss_list,
                        "train_miou_list": train_miou_list,
                        "val_loss_list": val_loss_list,
                        "val_miou_list": val_miou_list,
                    },
                    filename="checkpoint.pth.tar",
                )
                newly_saved = True
            train_utils.save_learning_curve(
                metrics={
                    "train_loss": train_loss_list,
                    "val_loss": val_loss_list,
                    "train_miou": train_miou_list,
                    "val_miou": val_miou_list,
                },
                filename="learning_curve.png",
                title="Segmentation Training Curve",
            )
            train_utils.save_batch_loss_curve(
                train_batch_loss_list,
                train_loss_list,
                total_batches_per_epoch=len(train_loader),
                filename="batch_loss_curve.png",
                plot_interval=10,
            )
            print("---------------------------------")
            epoch += 1
    else:
        newly_saved = False

    # Load the best checkpoint, use save_prediction() on the validation set and test set
    load_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../checkpoints/"
    )
    if newly_saved:
        model, _, _ = train_utils.load_checkpoint(
            model, os.path.join(load_dir, "checkpoint.pth.tar"), device
        )
    else:
        model, _, _ = train_utils.load_checkpoint(model, args.checkpoint, device)
    save_prediction(model, device, val_loader, val_dataset, root_dir)
    test_dir = os.path.join(root_dir, "test_results/")
    os.makedirs(test_dir, exist_ok=True)
    save_prediction(model, device, test_loader, test_dataset, test_dir)
    train_utils.save_learning_curve(
        metrics={
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "train_miou": train_miou_list,
            "val_miou": val_miou_list,
        },
        filename="learning_curve.png",
        title="Segmentation Training Curve",
    )
    test_miou = test(model, test_loader, device)
    print(f"Test mIoU: {test_miou:.4f}")


if __name__ == "__main__":
    main()
