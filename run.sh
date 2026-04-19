#!/bin/bash
# Example training script. Copy and modify for your experiments.
# Usage: bash run.sh

# ── Fine-tune ResNet-50 on a custom dataset ─────────────────────────────────
python train.py \
  --model resnet50 \
  --pretrained \
  --train-dir data/train \
  --val-dir data/val \
  --image-size 224 \
  --epochs 100 \
  --batch-size 32 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --scheduler cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --early-stopping \
  --early-stopping-patience 15 \
  --augmentation auto \
  --output-dir runs \
  --experiment-name resnet50_finetune \
  --device auto \
  --seed 42


# ── Train a Vision Transformer (ViT-Base) ───────────────────────────────────
# python train.py \
#   --model vit_base_patch16_224 \
#   --pretrained \
#   --train-dir data/train \
#   --val-dir data/val \
#   --image-size 224 \
#   --epochs 100 \
#   --batch-size 16 \
#   --optimizer adamw \
#   --lr 1e-4 \
#   --weight-decay 0.05 \
#   --scheduler cosine \
#   --warmup-epochs 10 \
#   --early-stopping \
#   --early-stopping-patience 20 \
#   --experiment-name vit_base_finetune \
#   --device cuda


# ── Train Swin Transformer with frozen backbone ──────────────────────────────
# python train.py \
#   --model swin_tiny_patch4_window7_224 \
#   --pretrained \
#   --freeze-backbone \
#   --train-dir data/train \
#   --val-split 0.2 \
#   --epochs 50 \
#   --batch-size 32 \
#   --optimizer adamw \
#   --lr 5e-4 \
#   --scheduler cosine \
#   --warmup-epochs 3 \
#   --experiment-name swin_tiny_frozen


# ── Train ResNet from scratch (no pretrained weights) ───────────────────────
# python train.py \
#   --model resnet18 \
#   --no-pretrained \
#   --train-dir data/train \
#   --val-dir data/val \
#   --epochs 200 \
#   --batch-size 64 \
#   --optimizer sgd \
#   --lr 0.01 \
#   --weight-decay 1e-4 \
#   --momentum 0.9 \
#   --scheduler step \
#   --scheduler-step-size 60 \
#   --scheduler-gamma 0.1 \
#   --experiment-name resnet18_scratch
