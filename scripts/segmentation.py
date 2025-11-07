import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import random

import utils
from dataset import CVATDataset
from model import MiniUNet
from tqdm import tqdm


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


def save_chkpt(
    model,
    epoch,
    val_miou,
    train_loss_list,
    train_miou_list,
    val_loss_list,
    val_miou_list,
):
    """
    In:
        model: MiniUNet instance in this homework, trained model.
        epoch: int, current epoch number.
        val_miou: float, mIoU of the validation set.
        train_loss_list: list of training loss over past epochs
        train_miou_list: list of training mIoU over past epochs
        val_loss_list: list of validation loss over past epochs
        val_miou_list: list of validation mIoU over past epochs
    Out:
        None.
    Purpose:
        Save parameters of the trained model.
    """
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "model_miou": val_miou,
        "train_loss_list": train_loss_list,
        "train_miou_list": train_miou_list,
        "val_loss_list": val_loss_list,
        "val_miou_list": val_miou_list,
    }
    torch.save(state, "checkpoint.pth.tar")
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    """
    In:
        model: MiniUNet instance in this homework to accept the saved parameters.
        chkpt_path: string, path of the checkpoint to be loaded.
    Out:
        model: MiniUNet instance in this homework, with its parameters loaded from the checkpoint.
        epoch: int, epoch at which the checkpoint is saved.
        model_miou: float, mIoU on the validation set at the checkpoint.
        train_loss_list: list of training loss over past epochs
        train_miou_list: list of training mIoU over past epochs
        val_loss_list: list of validation loss over past epochs
        val_miou_list: list of validation mIoU over past epochs
    Purpose:
        Load model parameters from saved checkpoint.
    """
    checkpoint = torch.load(chkpt_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    model_miou = checkpoint["model_miou"]
    train_loss_list = checkpoint["train_loss_list"]
    train_miou_list = checkpoint["train_miou_list"]
    val_loss_list = checkpoint["val_loss_list"]
    val_miou_list = checkpoint["val_miou_list"]

    return (
        model,
        epoch,
        model_miou,
        train_loss_list,
        train_miou_list,
        val_loss_list,
        val_miou_list,
    )


def save_learning_curve(train_loss_list, train_miou_list, val_loss_list, val_miou_list):
    """
    In:
        train_loss, train_miou, val_loss, val_miou: list of floats, where the length is how many epochs you trained.
    Out:
        None.
    Purpose:
        Plot and save the learning curve.
    """
    epochs = np.arange(1, len(train_loss_list) + 1)
    plt.figure()
    lr_curve_plot = plt.plot(epochs, train_loss_list, color="navy", label="train_loss")
    plt.plot(epochs, train_miou_list, color="teal", label="train_mIoU")
    plt.plot(epochs, val_loss_list, color="orange", label="val_loss")
    plt.plot(epochs, val_miou_list, color="gold", label="val_mIoU")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.xticks(epochs, epochs)
    plt.yticks(np.arange(10) * 0.1, [f"0.{i}" for i in range(10)])
    plt.xlabel("epoch")
    plt.ylabel("mIoU")
    plt.grid(True)
    plt.savefig("learning_curve.png", bbox_inches="tight")
    plt.close()


def save_prediction(model, device, dataloader, dataset, dataset_dir):
    """
    Save predicted masks for a dataset, works with Subset or full dataset.

    Args:
        model: trained MiniUNet
        device: torch.device
        dataloader: DataLoader
        dataset: Dataset or Subset (used to get original indices / filenames)
        dataset_dir: output directory for predictions
    """
    pred_dir = os.path.join(dataset_dir, "pred/")
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

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
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
    # Load arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint file to initialize the model with. Training will resume from the epoch saved in the checkpoint.",
    )
    args = parser.parse_args()

    # Check if GPU is being detected
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except:
            device = torch.device("cpu")
    print("device:", device)

    # Define directories (running from scripts/)
    root_dir = "../data/"

    # Create Datasets.
    # Load all dataset (assumes has_gt=True)
    full_dataset = CVATDataset(dataset_dir=root_dir, has_gt=True)

    # Shuffle and split indices
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    random.seed(42)  # for reproducibility
    random.shuffle(indices)

    split = int(0.8 * num_samples)  # 80% train, 20% val
    train_indices, val_indices = indices[:split], indices[split:]

    # Create Subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Prepare Dataloaders. You can use check_dataloader() to check your implementation.
    BATCH_SIZE = 4
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
    # print('\ncheck_dataloader: train_dataloader:')
    # check_dataloader(train_loader)
    # print('\ncheck_dataloader: val_dataloader:')
    # check_dataloader(val_loader)
    # print('\ncheck_dataloader: test_dataloader:')
    # check_dataloader(test_loader)

    # Prepare model
    model = MiniUNet().to(device)

    # Define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load checkpoint if provided
    if args.checkpoint:
        (
            model,
            epoch,
            best_miou,
            train_loss_list,
            train_miou_list,
            val_loss_list,
            val_miou_list,
        ) = load_chkpt(model, args.checkpoint, device)
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
            save_chkpt(
                model,
                epoch,
                val_miou,
                train_loss_list,
                train_miou_list,
                val_loss_list,
                val_miou_list,
            )
        save_learning_curve(
            train_loss_list, train_miou_list, val_loss_list, val_miou_list
        )
        print("---------------------------------")
        epoch += 1

    # Load the best checkpoint, use save_prediction() on the validation set and test set
    model, _, _, _, _, _, _ = load_chkpt(model, "checkpoint.pth.tar", device)
    save_prediction(model, device, val_loader, val_dataset, root_dir)
    # save_prediction(model, device, test_loader, test_dir)
    save_learning_curve(train_loss_list, train_miou_list, val_loss_list, val_miou_list)


if __name__ == "__main__":
    main()
