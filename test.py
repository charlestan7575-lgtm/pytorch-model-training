import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from dataset import get_val_transforms
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth checkpoint file")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to test ImageFolder root")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Input image size (default: read from metrics.json, else 224)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', 'cuda', or 'cuda:N'")
    # Override model config when metrics.json is not available
    parser.add_argument("--model", type=str, default=None,
                        help="timm model name (auto-detected from metrics.json if available)")
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config_from_metrics(checkpoint_path: Path):
    """Try to load model config from metrics.json next to the checkpoint."""
    metrics_path = checkpoint_path.parent / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    return data.get("config"), data.get("metadata")


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = resolve_device(args.device)

    # ── Resolve model config ─────────────────────────────────────────────────
    config, metadata = None, None
    metrics_cfg = load_config_from_metrics(checkpoint_path)
    if metrics_cfg is not None:
        config, metadata = metrics_cfg

    model_name = args.model or (config and config.get("model"))
    image_size = args.image_size or (config and config.get("image_size")) or 224

    if model_name is None:
        print("Error: could not determine model name. "
              "Provide --model or ensure metrics.json is next to the checkpoint.")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    # Infer num_classes from the classifier layer in the state dict
    # timm models use 'fc.weight', 'classifier.weight', 'head.weight', or 'head.fc.weight'
    num_classes = None
    for key in ("fc.weight", "classifier.weight", "head.weight", "head.fc.weight"):
        if key in state_dict:
            num_classes = state_dict[key].shape[0]
            break
    if num_classes is None:
        print("Error: could not infer num_classes from checkpoint.")
        sys.exit(1)

    print(f"\nConfig: {vars(args)}\n")

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ── Load test data ───────────────────────────────────────────────────────
    transform = get_val_transforms(image_size)
    test_dataset = ImageFolder(root=args.test_dir, transform=transform)
    class_names = test_dataset.classes
    n_classes = len(class_names)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes ({n_classes}): {class_names}\n")

    # ── Inference ────────────────────────────────────────────────────────────
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # ── Metrics ──────────────────────────────────────────────────────────────
    accuracy = accuracy_score(all_targets, all_preds)

    avg = "binary" if n_classes == 2 else "macro"
    f1 = f1_score(all_targets, all_preds, average=avg, zero_division=0)
    precision = precision_score(all_targets, all_preds, average=avg, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=avg, zero_division=0)

    # AUROC & AUPRC
    if n_classes == 2:
        try:
            auroc = roc_auc_score(all_targets, all_probs[:, 1])
        except ValueError:
            auroc = float("nan")
        try:
            auprc = average_precision_score(all_targets, all_probs[:, 1])
        except ValueError:
            auprc = float("nan")
    else:
        try:
            auroc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
        except ValueError:
            auroc = float("nan")
        try:
            # one-hot encode targets for multiclass AUPRC
            targets_onehot = np.eye(n_classes)[all_targets]
            auprc = average_precision_score(targets_onehot, all_probs, average="macro")
        except ValueError:
            auprc = float("nan")

    # ── Per-class statistics ─────────────────────────────────────────────────
    actual_counts = np.bincount(all_targets, minlength=n_classes)
    pred_counts = np.bincount(all_preds, minlength=n_classes)

    correct_per_class = np.zeros(n_classes, dtype=int)
    for t, p in zip(all_targets, all_preds):
        if t == p:
            correct_per_class[t] += 1

    # ── Print results ────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  Overall Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1        : {f1:.4f}  ({avg})")
    print(f"  Precision : {precision:.4f}  ({avg})")
    print(f"  Recall    : {recall:.4f}  ({avg})")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  AUPRC     : {auprc:.4f}")
    print()

    # Prediction distribution
    name_width = max(len(name) for name in class_names)
    header = f"  {'Class':<{name_width}}   {'Predicted':>10}   {'Actual':>10}"
    print(f"{'='*60}")
    print(f"  Prediction Distribution")
    print(f"{'='*60}")
    print(header)
    print(f"  {'-'*(name_width + 26)}")
    for i, name in enumerate(class_names):
        print(f"  {name:<{name_width}}   {pred_counts[i]:>10}   {actual_counts[i]:>10}")
    print()

    # Correct predictions per class
    print(f"{'='*60}")
    print(f"  Correct Predictions Per Class")
    print(f"{'='*60}")
    header = f"  {'Class':<{name_width}}   {'Correct':>8} / {'Total Pred':>10}   {'Acc':>7}"
    print(header)
    print(f"  {'-'*(name_width + 34)}")
    for i, name in enumerate(class_names):
        total_pred = int(pred_counts[i])
        correct = int(correct_per_class[i])
        cls_acc = correct / total_pred if total_pred > 0 else 0.0
        print(f"  {name:<{name_width}}   {correct:>8} / {total_pred:>10}   {cls_acc:>7.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
