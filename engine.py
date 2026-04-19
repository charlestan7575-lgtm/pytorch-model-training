import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import JsonLogger, format_time, save_checkpoint


class EarlyStopping:
    """
    Stops training if the monitored metric does not improve by at least
    min_delta for `patience` consecutive epochs.
    """

    def __init__(self, patience: int, min_delta: float, monitor: str):
        """
        Args:
            patience:  Number of epochs with no improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
            monitor:   'val_loss' (lower is better) or 'val_accuracy' (higher is better).
        """
        if monitor not in ("val_loss", "val_accuracy"):
            raise ValueError(f"monitor must be 'val_loss' or 'val_accuracy', got '{monitor}'")

        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def check(self, metric_value: float) -> bool:
        """Return True if training should stop."""
        if self.best_score is None:
            self.best_score = metric_value
            return False

        if self.monitor == "val_loss":
            improved = metric_value < (self.best_score - self.min_delta)
        else:  # val_accuracy
            improved = metric_value > (self.best_score + self.min_delta)

        if improved:
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Run one validation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,           # any torch LR scheduler or None
    scheduler_type: str,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
    early_stopper: Optional[EarlyStopping],
    epochs: int,
    device: torch.device,
    logger: JsonLogger,
    checkpoint_path: Path,
) -> dict:
    """
    Full training loop.

    Handles:
    - Linear LR warmup for the first `warmup_epochs` epochs
    - LR scheduler stepping (ReduceLROnPlateau gets val_loss, others step per epoch)
    - Best-model checkpointing
    - Early stopping

    Returns a summary dict with best epoch and metrics.
    """
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 1
    stopped_early = False

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Linear warmup: manually set LR for the first warmup_epochs ---
        if epoch <= warmup_epochs and warmup_epochs > 0:
            warmup_lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # --- Train and validate ---
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # --- Step LR scheduler (after warmup) ---
        if scheduler is not None and epoch > warmup_epochs:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # --- Checkpoint best model ---
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, checkpoint_path)

        # --- Log epoch ---
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, is_best)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:>4d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}  "
            f"{'* best ' if is_best else '       '}"
            f"[{format_time(elapsed)}]"
        )

        # --- Early stopping ---
        if early_stopper is not None and early_stopper.check(val_loss):
            print(f"Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {early_stopper.patience} epochs).")
            stopped_early = True
            break

    result = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "total_epochs_run": epoch,
        "stopped_early": stopped_early,
    }

    logger.log_result(**result)
    return result
