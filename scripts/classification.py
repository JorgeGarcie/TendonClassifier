import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from classification_model import ResNetBinaryClassifier, MobileNetBinaryClassifier
from dataset import CVATDataset
import train_utils


# Setup
train_utils.setup_single_threaded_torch()
DEVICE = train_utils.get_device()

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATASET_DIR = os.path.join(ROOT_DIR, "data")
IMG_SIZE = (240, 320)
SAVE_NAME = "classifier_mobilenetv3_small.pth.tar"

EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4
VAL_SPLIT = 0.2


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    train_bar = tqdm(loader, desc="Training")

    for batch in train_bar:
        imgs = batch["input"].to(device)
        labels = batch["has_feature"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

        running_loss += loss.item()

        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return running_loss / len(loader), correct / total


def visualize_batch(batch, num_samples=4):
    """
    Visualize a batch of images with their labels.

    Args:
        batch: dict with 'input' tensor and 'has_feature' labels
        num_samples: number of images to display
    """
    imgs = batch["input"]  # (B, 3, H, W)
    labels = batch["has_feature"]  # (B,)

    # Denormalize images
    mean = np.array([0.442, 0.417, 0.593]).reshape(3, 1, 1)
    std = np.array([0.188, 0.190, 0.155]).reshape(3, 1, 1)

    num_samples = min(num_samples, imgs.shape[0])
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        img = imgs[i].cpu().numpy()  # (3, H, W)

        # Denormalize
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Convert to (H, W, 3) for display
        img = np.transpose(img, (1, 2, 0))

        axes[i].imshow(img)
        axes[i].set_title(f"Label: {labels[i].item()}\n(1=has_obj, 0=no_obj)")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("classification_batch_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved visualization to classification_batch_visualization.png")


def test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    test_bar = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for batch in test_bar:
            imgs = batch["input"].to(device)
            labels = batch["has_feature"].float().unsqueeze(1).to(device)
            outputs = model(imgs)

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    acc = correct / total
    print(f"Test Accuracy: {acc:.3f}")
    return acc


def check_dataset_balance(dataset):
    """
    Check the distribution of positive/negative samples.
    """
    has_feature_count = sum(dataset.original_has_feature_list)
    total = len(dataset.original_has_feature_list)

    print("\n" + "=" * 50)
    print("DATASET BALANCE CHECK")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Has feature (1): {has_feature_count} ({100*has_feature_count/total:.1f}%)")
    print(
        f"No feature (0): {total - has_feature_count} ({100*(total-has_feature_count)/total:.1f}%)"
    )
    print("=" * 50 + "\n")


def validate(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    val_bar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for batch in val_bar:
            imgs = batch["input"].to(device)
            labels = batch["has_feature"].float().unsqueeze(1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return val_loss / len(loader), correct / total


def main():
    # Load args
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

    # Dataset
    full_dataset = CVATDataset(
        DATASET_DIR, has_gt=True, img_size=IMG_SIZE, for_segmentation=False
    )

    test_dataset = CVATDataset(
        DATASET_DIR, has_gt=True, img_size=IMG_SIZE, for_test=True
    )

    # Train/val split
    random_generator = torch.Generator().manual_seed(42)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=random_generator
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Check dataset balance
    check_dataset_balance(full_dataset)
    check_dataset_balance(test_dataset)

    # Visualize a batch from training set
    sample_batch = next(iter(train_loader))
    print(f"Sample batch image tensor size: {sample_batch['input'].shape}")
    print(f"Sample batch labels: {sample_batch['has_feature']}")
    visualize_batch(sample_batch, num_samples=min(4, BATCH_SIZE))

    # Model, Loss, Optimizer
    model = MobileNetBinaryClassifier(pretrained=True, freeze_backbone=True).to(DEVICE)
    # model = ResNetBinaryClassifier(pretrained=True, freeze_backbone=True).to(DEVICE)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Load checkpoint if provided
    if args.checkpoint:
        model, epoch, metric = train_utils.load_checkpoint(
            model, args.checkpoint, DEVICE
        )
        best_acc = metric["best_acc"]
        train_loss_list = metric["train_loss_list"]
        val_loss_list = metric["val_loss_list"]
        train_acc_list = metric["train_acc_list"]
        val_acc_list = metric["val_acc_list"]
        print(
            f"Loaded a checkpoint from {args.checkpoint} at epoch {epoch} with best acc {best_acc}. Resuming training from epoch {epoch + 1}."
        )
        epoch += 1
    else:
        epoch = 1
        best_acc = float("-inf")
        train_loss_list, val_loss_list, train_acc_list, val_acc_list = (
            list(),
            list(),
            list(),
            list(),
        )
        print(
            f"No checkpoint provided. Starting training from scratch at epoch {epoch}."
        )

    if not args.test_only:
        while epoch <= EPOCHS:
            print(f"\nEpoch ({epoch}/{EPOCHS})")
            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
            )

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                train_utils.save_checkpoint(
                    model,
                    epoch,
                    {
                        "train_loss_list": train_loss_list,
                        "train_acc_list": train_acc_list,
                        "val_loss_list": val_loss_list,
                        "val_acc_list": val_acc_list,
                        "best_acc": best_acc,
                    },
                    filename=SAVE_NAME,
                )

            # Update learning curve plot
            train_utils.save_learning_curve(
                {
                    "train_loss": train_loss_list,
                    "val_loss": val_loss_list,
                    "train_acc": train_acc_list,
                    "val_acc": val_acc_list,
                },
                filename="classification_curve.png",
                title="Classification Training Progress",
            )

            epoch += 1

    print("\n*** TESTING MODEL ***")
    test_acc = test(model, test_loader, DEVICE)
    print(f"Testing complete. Acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()
