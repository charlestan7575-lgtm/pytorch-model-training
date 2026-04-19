import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders
from engine import EarlyStopping, train
from model import build_model, count_parameters
from utils import JsonLogger, setup_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Image Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default="resnet50",
                        help="timm model name (e.g. resnet50, vit_base_patch16_224)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Load pretrained ImageNet weights")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Train from scratch (no pretrained weights)")
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="Freeze all layers except the classifier head")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Path to training ImageFolder root")
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Path to validation ImageFolder root "
                             "(if omitted, auto-split from train-dir)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of train-dir to use as validation (when val-dir is omitted)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size (square)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (train and val)")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adam", "sgd"],
                        help="Optimizer")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (ignored for Adam/AdamW)")

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "plateau", "none"],
                        help="LR scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup epochs (0 to disable)")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum LR for cosine scheduler and warmup floor")
    parser.add_argument("--scheduler-step-size", type=int, default=30,
                        help="StepLR: decay LR every N epochs")
    parser.add_argument("--scheduler-gamma", type=float, default=0.1,
                        help="StepLR / ExponentialLR: multiplicative decay factor")
    parser.add_argument("--scheduler-patience", type=int, default=10,
                        help="ReduceLROnPlateau: epochs with no improvement before reducing LR")

    # ── Early Stopping ────────────────────────────────────────────────────────
    parser.add_argument("--early-stopping", action="store_true", default=True,
                        help="Enable early stopping")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false",
                        help="Disable early stopping")
    parser.add_argument("--early-stopping-patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001,
                        help="Minimum improvement to reset early stopping counter")
    parser.add_argument("--early-stopping-monitor", type=str, default="val_loss",
                        choices=["val_loss", "val_accuracy"],
                        help="Metric to monitor for early stopping")

    # ── Augmentation ──────────────────────────────────────────────────────────
    parser.add_argument("--augmentation", type=str, default="auto",
                        choices=["auto", "none"],
                        help="'auto' = AutoAugment ImageNet policy; 'none' = no augmentation")
    parser.add_argument("--no-augmentation", dest="augmentation",
                        action="store_const", const="none",
                        help="Disable data augmentation")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Base directory for experiment outputs")
    parser.add_argument("--experiment-name", type=str, default="experiment",
                        help="Subdirectory name under output-dir")

    # ── System ────────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(params, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def build_scheduler(optimizer, args, steps_after_warmup: int):
    """Build scheduler that runs AFTER warmup."""
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        # T_max = epochs remaining after warmup
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(steps_after_warmup, 1),
            eta_min=args.min_lr,
        )
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=args.scheduler_patience,
            factor=args.scheduler_gamma,
            min_lr=args.min_lr,
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"\n{'='*60}")
    print(f"  PyTorch Image Classification Training")
    print(f"{'='*60}")
    print(f"  Model      : {args.model} (pretrained={args.pretrained})")
    print(f"  Device     : {device}")
    print(f"  Optimizer  : {args.optimizer}  lr={args.lr}  wd={args.weight_decay}")
    print(f"  Scheduler  : {args.scheduler}  warmup={args.warmup_epochs} epochs")
    print(f"  Augment    : {args.augmentation}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    train_loader, val_loader, num_classes, class_names = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_split=args.val_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        augmentation=args.augmentation,
        seed=args.seed,
    )
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train samples : {n_train}")
    print(f"  Val samples   : {n_val}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Building model: {args.model} ...")
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}\n")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model.parameters(), args)
    steps_after_warmup = args.epochs - args.warmup_epochs
    scheduler = build_scheduler(optimizer, args, steps_after_warmup)

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            monitor=args.early_stopping_monitor,
        )

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = setup_output_dir(args.output_dir, args.experiment_name)
    checkpoint_path = output_dir / "best_model.pth"
    log_path = output_dir / "metrics.json"
    print(f"Output directory: {output_dir}\n")

    # ── Logger ────────────────────────────────────────────────────────────────
    config_dict = vars(args)
    metadata = {
        "num_classes": num_classes,
        "class_names": class_names,
        "train_samples": n_train,
        "val_samples": n_val,
        "device": str(device),
    }
    logger = JsonLogger(filepath=log_path, config=config_dict, metadata=metadata)

    # ── Training ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    print(f"Starting training for up to {args.epochs} epochs...\n")

    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_type=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
        early_stopper=early_stopper,
        epochs=args.epochs,
        device=device,
        logger=logger,
        checkpoint_path=checkpoint_path,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best epoch      : {result['best_epoch']}")
    print(f"  Best val loss   : {result['best_val_loss']:.4f}")
    print(f"  Best val acc    : {result['best_val_accuracy']:.4f}")
    print(f"  Epochs run      : {result['total_epochs_run']}")
    print(f"  Early stopped   : {result['stopped_early']}")
    print(f"  Checkpoint      : {checkpoint_path}")
    print(f"  Metrics log     : {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
