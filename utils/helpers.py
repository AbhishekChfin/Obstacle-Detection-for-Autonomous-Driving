"""
General helper functions: device selection, checkpointing, logging, and collate.
"""

import os
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import logging
from datetime import datetime
import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Select best available compute device (MPS on Apple Silicon, CUDA, or CPU)."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preferred)


def setup_logging(log_dir: str, name: str = "train") -> logging.Logger:
    """Set up dual console and file logging."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = None,
) -> str:
    """Save model checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:03d}.pth"
    path = os.path.join(checkpoint_dir, filename)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
) -> tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))


def collate_detection(batch):
    """Collate function for detection and instance segmentation models."""
    images = []
    targets = []
    for img, tgt in batch:
        if img is not None and tgt is not None:
            images.append(img)
            targets.append(tgt)
    return images, targets


def collate_segmentation(batch):
    """Collate function for semantic segmentation (DeepLabv3+)."""
    images = []
    masks = []
    for img, mask in batch:
        images.append(img)
        masks.append(mask)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0)
