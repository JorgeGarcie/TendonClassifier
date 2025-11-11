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

from dataset import CVATDataset
from segmentation_model import MiniUNet
import utils
import train_utils


def iou(prediction, target):
    """
    In:
        prediction: Tensor [batchsize, class, height, width], predicted mask.
        target: Tensor [batchsize, height, width], ground truth mask.
    Out:
        batch_ious: a list of floats, storing IoU on each batch.
    Purpose:
        Compute IoU on each data and return as a list.
    """
    _, pred = torch.max(prediction, dim=1)
    batch_num = prediction.shape[0]
    class_num = prediction.shape[1]
    batch_ious = list()
    for batch_id in range(batch_num):
        class_ious = list()
        for class_id in range(1, class_num):  # class 0 is background
            mask_pred = (pred[batch_id] == class_id).int()
            mask_target = (target[batch_id] == class_id).int()
            if mask_target.sum() == 0:  # skip the occluded object
                continue
            intersection = (mask_pred * mask_target).sum()
            union = (mask_pred + mask_target).sum() - intersection
            class_ious.append(float(intersection) / float(union))
        batch_ious.append(np.mean(class_ious))
    return batch_ious


def save_prediction(model, device, dataloader, dataset, output_dir):
    """
    Save predicted masks for a dataset, works with Subset or full dataset.

    Args:
        model: trained MiniUNet
        device: torch.device
        dataloader: DataLoader
        dataset: Dataset or Subset (used to get original indices / filenames)
        output_dir: output directory for predictions
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

            for i in range(pred.shape[0]):
                if isinstance(dataset, Subset):
                    original_idx = dataset.indices[batch_id * dataloader.batch_size + i]
                    filename = dataset.dataset[original_idx]["filename"]
                else:
                    filename = batch["filename"][i]

                mask = pred[i].cpu().numpy()
                mask_path = os.path.join(pred_dir, f"{filename}_pred.png")

                utils.write_mask(mask, mask_path)
                utils.write_rgb(
                    utils.mask2rgb(mask),
                    mask_path.replace("_pred.png", "_pred_rgb.png"),
                )


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    train_loss_sum, train_iou_sum = 0.0, 0.0
    data_size = 0
    total_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        batch_size = batch["input"].size(0)
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        data_size += batch_size
        train_loss_sum += loss.item() * batch_size
        train_iou_sum += np.sum(iou(outputs, targets))

        if (batch_idx + 1) % 3 == 0 or (batch_idx + 1) == total_batches:
            print(f"  Batch {batch_idx+1}/{total_batches} - loss: {loss.item():.4f}")

    train_loss = train_loss_sum / data_size
    train_iou = train_iou_sum / data_size
    return train_loss, train_iou


def val(model, device, val_loader, criterion):
    """
    Similar to train(), but no need to backward and optimize.
    """
    model.eval()
    val_loss_sum, val_iou_sum = 0, 0
    data_size = 0

    with torch.no_grad():
        for batch in val_loader:
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


def main():
    train_utils.setup_single_threaded_torch()
    # Load arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint file to initialize the model with. Training will resume from the epoch saved in the checkpoint.",
    )
    args = parser.parse_args()

    # Check if GPU is being detected
    device = train_utils.get_device()

    # Define directories (running from scripts/)
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/")

    # Create Datasets.
    # Load all dataset (assumes has_gt=True)
    full_dataset = CVATDataset(
        dataset_dir=root_dir, has_gt=True, for_classification=False
    )

    # Filter to only indices with objects
    indices_with_objects = [
        i for i, has_feat in enumerate(full_dataset.has_feature_list) if has_feat == 1
    ]
    print(
        f"Using {len(indices_with_objects)} of {len(full_dataset)} frames with objects."
    )

    # Shuffle and split indices
    random.seed(42)
    random.shuffle(indices_with_objects)
    split = int(0.8 * len(indices_with_objects))
    train_indices = indices_with_objects[:split]
    val_indices = indices_with_objects[split:]

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
    # test_loader = DataLoader(
    #     test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    # )

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
        train_miou_list = metric["train_miou_list"]
        val_loss_list = metric["val_loss_list"]
        val_miou_list = metric["val_miou_list"]
        print(
            f"Loaded a checkpoint from {args.checkpoint} at epoch {epoch} with best mIoU {best_miou}. Resuming training from epoch {epoch + 1}."
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
        print(
            f"No checkpoint provided. Starting training from scratch at epoch {epoch}."
        )

    # Train and validate the model
    max_epochs = 30
    newly_saved = False
    while epoch <= max_epochs:
        print("Epoch (", epoch, "/", max_epochs, ")")
        train_loss, train_miou = train(
            model, device, train_loader, criterion, optimizer
        )
        val_loss, val_miou = val(model, device, val_loader, criterion)
        train_loss_list.append(train_loss)
        train_miou_list.append(train_miou)
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
        print("---------------------------------")
        epoch += 1

    # Load the best checkpoint, use save_prediction() on the validation set and test set
    load_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../checkpoints/"
    )
    if newly_saved:
        model, _, _ = train_utils.load_checkpoint(
            model, os.path.join(load_dir, "checkpoint.pth.tar"), device
        )
    else:
        model, _, _ = train_utils.load_checkpoint(
            model, os.path.join(load_dir, args.checkpoint), device
        )
    save_prediction(model, device, val_loader, val_dataset, root_dir)
    # save_prediction(model, device, test_loader, test_dir)
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


if __name__ == "__main__":
    main()
