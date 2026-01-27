import os
import torch
import matplotlib.pyplot as plt
import numpy as np


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
