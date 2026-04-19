# PyTorch Image Classification Training

A clean, standard-level training script for image classification. All hyperparameters are passed via CLI arguments, making it easy to run experiments from shell scripts.

---

## Features

- **700+ model architectures** via [timm](https://github.com/huggingface/pytorch-image-models) — ResNet, EfficientNet, VGG, ConvNeXt, ViT, DeiT, Swin, and more
- **Transfer learning** — pretrained ImageNet weights out of the box
- **Backbone freezing** — optionally freeze all layers except the classifier head
- **AutoAugment** — ImageNet-optimized augmentation policy by default
- **LR scheduling** — cosine (with linear warmup), StepLR, ReduceLROnPlateau, or none
- **Early stopping** — configurable patience on val_loss or val_accuracy
- **JSON metric log** — every epoch written atomically (always valid JSON)
- **Best-model checkpoint** — saves model + optimizer state for the best validation epoch

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.0`, `torchvision>=0.15`, `timm>=0.9`

---

## Dataset Setup

Organize your images in the **ImageFolder** format:

```
data/
├── train/
│   ├── class_a/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_b/
│       ├── image3.jpg
│       └── image4.jpg
└── val/           ← optional; if omitted, train/ is auto-split
    ├── class_a/
    └── class_b/
```

The number of classes is detected automatically from the subdirectory names.

---

## Quick Start

```bash
# Fine-tune ResNet-50 on your dataset
python train.py \
  --model resnet50 \
  --train-dir data/train \
  --val-dir data/val \
  --epochs 50 \
  --batch-size 32 \
  --experiment-name my_first_run

# Or use the example shell script
bash run.sh
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| **Model** | | |
| `--model` | `resnet50` | timm model name |
| `--pretrained` / `--no-pretrained` | `True` | Load ImageNet pretrained weights |
| `--freeze-backbone` | `False` | Freeze all layers except classifier head |
| **Data** | | |
| `--train-dir` | *(required)* | Path to training ImageFolder root |
| `--val-dir` | `None` | Validation ImageFolder root (auto-split if omitted) |
| `--val-split` | `0.2` | Fraction of train data to use as val (when val-dir is omitted) |
| `--image-size` | `224` | Input image size (square) |
| `--num-workers` | `4` | DataLoader worker processes |
| **Training** | | |
| `--epochs` | `100` | Maximum training epochs |
| `--batch-size` | `32` | Batch size |
| `--optimizer` | `adamw` | Optimizer: `adamw`, `adam`, `sgd` |
| `--lr` | `1e-3` | Peak learning rate |
| `--weight-decay` | `0.05` | L2 weight decay |
| `--momentum` | `0.9` | SGD momentum (ignored for Adam/AdamW) |
| **LR Scheduler** | | |
| `--scheduler` | `cosine` | Scheduler: `cosine`, `step`, `plateau`, `none` |
| `--warmup-epochs` | `5` | Linear warmup epochs (0 to disable) |
| `--min-lr` | `1e-6` | Minimum LR for cosine and warmup floor |
| `--scheduler-step-size` | `30` | StepLR: decay every N epochs |
| `--scheduler-gamma` | `0.1` | StepLR/Plateau: multiplicative decay factor |
| `--scheduler-patience` | `10` | Plateau: epochs with no improvement before reducing LR |
| **Early Stopping** | | |
| `--early-stopping` / `--no-early-stopping` | `True` | Enable/disable early stopping |
| `--early-stopping-patience` | `15` | Epochs with no improvement before stopping |
| `--early-stopping-min-delta` | `0.001` | Minimum change to qualify as improvement |
| `--early-stopping-monitor` | `val_loss` | Metric to monitor: `val_loss` or `val_accuracy` |
| **Augmentation** | | |
| `--augmentation` | `auto` | `auto` = AutoAugment ImageNet; `none` = no augmentation |
| `--no-augmentation` | — | Alias for `--augmentation none` |
| **Output** | | |
| `--output-dir` | `runs` | Base directory for experiment outputs |
| `--experiment-name` | `experiment` | Subdirectory name under output-dir |
| **System** | | |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `--seed` | `42` | Random seed |

---

## Output Structure

Each experiment creates:

```
runs/
└── my_experiment/
    ├── best_model.pth    ← model + optimizer state for the best val epoch
    └── metrics.json      ← full metrics log (see below)
```

### metrics.json Format

```json
{
  "config": { "model": "resnet50", "lr": 0.001, "optimizer": "adamw", "...": "..." },
  "metadata": {
    "start_time": "2026-04-19T15:30:00",
    "num_classes": 5,
    "class_names": ["cat", "dog", "bird", "fish", "horse"],
    "train_samples": 4000,
    "val_samples": 1000,
    "device": "cuda"
  },
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 2.1345,
      "train_accuracy": 0.3210,
      "val_loss": 1.8901,
      "val_accuracy": 0.4120,
      "learning_rate": 0.001,
      "is_best": true,
      "elapsed_seconds": 45.2
    }
  ],
  "result": {
    "best_epoch": 42,
    "best_val_loss": 0.3210,
    "best_val_accuracy": 0.9120,
    "total_epochs_run": 45,
    "stopped_early": true,
    "total_time_seconds": 2034.5
  }
}
```

---

## Code Architecture

```
train.py        Entry point — parse args, wire components, print summary
  ├── dataset.py  ImageFolder loading, transforms, auto-split
  ├── model.py    timm model factory, backbone freezing
  ├── engine.py   Training loop, validation loop, EarlyStopping
  └── utils.py    JsonLogger, save_checkpoint, helpers
```

| File | Key components |
|------|----------------|
| `train.py` | `main()`, `build_optimizer()`, `build_scheduler()` |
| `model.py` | `build_model()`, `count_parameters()` |
| `dataset.py` | `get_dataloaders()`, `get_train_transforms()`, `get_val_transforms()` |
| `engine.py` | `train()`, `train_one_epoch()`, `validate()`, `EarlyStopping` |
| `utils.py` | `JsonLogger`, `save_checkpoint()`, `load_checkpoint()`, `setup_output_dir()` |

---

## Example Recipes

### Fine-tune any model with 2 lines

```bash
python train.py --model resnet50 --train-dir data/train --val-dir data/val
```

### Vision Transformer (ViT)

```bash
python train.py \
  --model vit_base_patch16_224 \
  --pretrained \
  --train-dir data/train \
  --val-dir data/val \
  --batch-size 16 \
  --lr 1e-4 \
  --warmup-epochs 10 \
  --experiment-name vit_base
```

### Freeze backbone (fast linear probe)

```bash
python train.py \
  --model resnet50 \
  --pretrained \
  --freeze-backbone \
  --train-dir data/train \
  --val-split 0.2 \
  --epochs 20 \
  --lr 1e-2 \
  --experiment-name resnet50_linear_probe
```

### Train from scratch (SGD)

```bash
python train.py \
  --model resnet18 \
  --no-pretrained \
  --train-dir data/train \
  --val-dir data/val \
  --optimizer sgd \
  --lr 0.01 \
  --scheduler step \
  --scheduler-step-size 60 \
  --epochs 200 \
  --experiment-name resnet18_scratch
```

### Loading a checkpoint

```python
import torch
from model import build_model

model = build_model("resnet50", num_classes=5, pretrained=False)
checkpoint = torch.load("runs/my_experiment/best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## Tips

- **ViTs need warmup** — use `--warmup-epochs 10` or more; ViTs are sensitive to early LR
- **Lower LR for pretrained models** — `1e-4` or `5e-5` is safer than `1e-3` for large ViTs
- **Frozen backbone for small datasets** — `--freeze-backbone` trains only the head, much faster
- **SGD for training from scratch** — AdamW is better for fine-tuning; SGD often wins from scratch
- **Finding model names** — run `python -c "import timm; print(timm.list_models())"` or check [timm docs](https://huggingface.co/docs/timm)
