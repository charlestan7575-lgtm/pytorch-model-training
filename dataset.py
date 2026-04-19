import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int, augmentation: str) -> transforms.Compose:
    """
    Build training transforms.

    augmentation:
        'auto'  — AutoAugment with ImageNet policy (default)
        'none'  — no augmentation, just resize + center crop
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if augmentation == "none":
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    # Default: AutoAugment ImageNet policy
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        normalize,
    ])


def get_val_transforms(image_size: int) -> transforms.Compose:
    """Build deterministic validation transforms."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(
    train_dir: str,
    val_dir: Optional[str],
    val_split: float,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    augmentation: str,
    seed: int,
):
    """
    Build and return (train_loader, val_loader, num_classes, class_names).

    If val_dir is provided, load two separate ImageFolder datasets.
    If val_dir is None, split train_dir using val_split — using two separate
    ImageFolder instances with shared index lists so each gets its own transforms.
    """
    train_tfm = get_train_transforms(image_size, augmentation)
    val_tfm = get_val_transforms(image_size)

    train_dir = Path(train_dir)

    if val_dir is not None:
        # Explicit separate validation directory
        val_dir = Path(val_dir)
        train_dataset = ImageFolder(root=str(train_dir), transform=train_tfm)
        val_dataset = ImageFolder(root=str(val_dir), transform=val_tfm)
    else:
        # Auto-split: build two ImageFolder instances with shared index lists
        # so train and val each get their own transform.
        full_for_split = ImageFolder(root=str(train_dir))  # no transform — just for indexing
        n_total = len(full_for_split)
        indices = list(range(n_total))

        rng = random.Random(seed)
        rng.shuffle(indices)

        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        if n_val == 0:
            raise ValueError(
                f"val_split={val_split} produces 0 validation samples from {n_total} total."
            )

        train_indices = indices[n_val:]   # indices[n_val:] -> train
        val_indices = indices[:n_val]     # indices[:n_val]  -> val

        train_full = ImageFolder(root=str(train_dir), transform=train_tfm)
        val_full = ImageFolder(root=str(train_dir), transform=val_tfm)

        train_dataset = Subset(train_full, train_indices)
        val_dataset = Subset(val_full, val_indices)

    # Detect class info from the underlying ImageFolder
    if isinstance(train_dataset, Subset):
        base = train_dataset.dataset
    else:
        base = train_dataset
    num_classes = len(base.classes)
    class_names = base.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, num_classes, class_names
