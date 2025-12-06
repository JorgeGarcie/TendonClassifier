import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse


def setup_single_threaded_torch():
    """Force single-threaded execution for deterministic behavior."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def get_device():
    """Return an available torch.device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def save_checkpoint(model, epoch, metrics_dict, filename="checkpoint.pth.tar"):
    """
    Save model checkpoint including training metrics.

    Args:
        model: torch.nn.Module
        epoch: int
        metrics_dict: dict of any lists or values you want to save
                      (e.g., {'train_loss_list': [...], 'val_acc_list': [...]})
        filename: output checkpoint name
    """
    state = {"model_state_dict": model.state_dict(), "epoch": epoch, **metrics_dict}

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../checkpoints/"
    )
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch}: {path}")


def load_checkpoint(model, chkpt_path, device):
    """
    Load a checkpoint and return (model, epoch, metrics_dict).

    Args:
        model: torch.nn.Module
        chkpt_path: str, path to saved .pth.tar
        device: torch.device

    Returns:
        model, epoch, metrics_dict
    """
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    metrics = {
        k: v for k, v in checkpoint.items() if k not in ["model_state_dict", "epoch"]
    }

    print(f"Loaded checkpoint '{chkpt_path}' (epoch {epoch})")
    return model, epoch, metrics


def save_learning_curve(metrics, filename="learning_curve.png", title=None):
    """
    Plot and save training curves from a dict of metric lists.

    Args:
        metrics: dict
            Example:
                {
                    "train_loss": [...],
                    "val_loss": [...],
                    "train_acc": [...],
                    "val_acc": [...]
                }
        filename: name of the output image
        title: optional plot title
    """
    epochs = np.arange(1, len(next(iter(metrics.values()))) + 1)

    plt.figure(figsize=(8, 5))
    for key, values in metrics.items():
        plt.plot(epochs, values, label=key)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    if title:
        plt.title(title)
    plt.grid(True)

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../learning_curves/"
    )
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Learning curve saved: {path}")


def save_batch_loss_curve(
    batch_loss_list,
    epoch_loss_list,
    train_miou_list,
    val_miou_list,
    total_batches_per_epoch,
    filename="learning_curves/batch_loss_curve.png",
    plot_interval=10,
):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    num_batches = len(batch_loss_list)
    batch_iterations = np.arange(1, num_batches + 1)

    # --- Plotting Loss on the Left Axis (ax1) ---
    ax1.plot(
        batch_iterations,
        batch_loss_list,
        label="Batch Train Loss",
        color="gray",
        alpha=0.5,
        linewidth=0.5,
    )

    if epoch_loss_list:
        epoch_iterations = (
            np.arange(1, len(epoch_loss_list) + 1) * total_batches_per_epoch
            - 0.5 * total_batches_per_epoch
        )
        ax1.plot(
            epoch_iterations,
            epoch_loss_list,
            label="Epoch Mean Train Loss",
            color="red",
            linewidth=2.0,
        )

    ax1.set_xlabel("Batch Number", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.grid(True, linestyle="--")

    # --- Plotting mIoU on the Right Axis (ax2) ---
    if train_miou_list and val_miou_list:
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

        # Recalculate iterations for epoch-based metrics if needed
        # (Assuming same x-axis alignment as epoch loss)
        epoch_iterations = (
            np.arange(1, len(train_miou_list) + 1) * total_batches_per_epoch
            - 0.5 * total_batches_per_epoch
        )

        ax2.plot(
            epoch_iterations,
            train_miou_list,
            label="Train IoU",
            color="blue",
            linestyle="--",
            linewidth=2.0,
        )
        ax2.plot(
            epoch_iterations,
            val_miou_list,
            label="Val IoU",
            color="green",
            linestyle="--",
            linewidth=2.0,
        )
        ax2.set_ylabel("IoU", fontsize=16)
        ax2.set_ylim(0.5, 1)  # mIoU ranges from 0 to 1

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = [], []
    if train_miou_list and val_miou_list:
        lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=12)

    plt.savefig(filename)
    plt.close()
    print(f"Batch loss curve saved: {filename}")


def get_ckpt_info(chkpt_path):
    """
    Extract epoch and metrics info from a checkpoint file.

    Args:
        chkpt_path: str, path to saved .pth.tar
    """
    checkpoint = torch.load(chkpt_path, map_location="cpu")
    epoch = checkpoint["epoch"]
    metrics = {
        k: v for k, v in checkpoint.items() if k not in ["model_state_dict", "epoch"]
    }
    # print metrics
    print(f"Checkpoint info from '{chkpt_path}':")
    print(f"  Epoch: {epoch}")
    for key, value in metrics.items():
        if key != "train_batch_loss_list":
            print(f"  {key}: {value}")
        else:
            save_batch_loss_curve(
                value,
                metrics.get("train_loss_list", []),
                metrics.get("train_miou_list", []),
                metrics.get("val_miou_list", []),
                total_batches_per_epoch=len(metrics.get("train_batch_loss_list", []))
                // epoch,
                filename="learning_curves/batch_loss_curve_from_chkpt.png",
            )


def main():
    parser = argparse.ArgumentParser(
        description="Extract and display checkpoint information."
    )
    parser.add_argument("--chkpt_path", type=str, help="Path to the checkpoint file.")
    args = parser.parse_args()

    get_ckpt_info(args.chkpt_path)


if __name__ == "__main__":
    main()
