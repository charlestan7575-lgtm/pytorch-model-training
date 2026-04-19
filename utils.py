import json
import time
from datetime import datetime
from pathlib import Path

import torch


def setup_output_dir(output_dir: str, experiment_name: str) -> Path:
    """Create and return the experiment output directory."""
    out = Path(output_dir) / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def format_time(seconds: float) -> str:
    """Convert elapsed seconds to a human-readable string."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def save_checkpoint(model, optimizer, scheduler, epoch: int,
                    val_loss: float, val_accuracy: float, filepath: Path):
    """Save model, optimizer, and scheduler states to a .pth file."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }
    torch.save(state, filepath)


def load_checkpoint(filepath: Path, model, optimizer=None, scheduler=None):
    """Load checkpoint and return the epoch number."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0)


class JsonLogger:
    """
    Writes per-epoch metrics to a JSON file after every epoch.
    The file is fully rewritten each time so it is always valid JSON.
    """

    def __init__(self, filepath: Path, config: dict, metadata: dict):
        self.filepath = filepath
        self._data = {
            "config": config,
            "metadata": {**metadata, "start_time": datetime.now().isoformat()},
            "epochs": [],
            "result": None,
        }
        self._start = time.time()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: float,
        is_best: bool,
    ):
        elapsed = time.time() - self._start
        self._data["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
            "learning_rate": learning_rate,
            "is_best": is_best,
            "elapsed_seconds": round(elapsed, 2),
        })
        self._save()

    def log_result(
        self,
        best_epoch: int,
        best_val_loss: float,
        best_val_accuracy: float,
        total_epochs_run: int,
        stopped_early: bool,
    ):
        self._data["result"] = {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "best_val_accuracy": round(best_val_accuracy, 6),
            "total_epochs_run": total_epochs_run,
            "stopped_early": stopped_early,
            "total_time_seconds": round(time.time() - self._start, 2),
        }
        self._save()

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)
