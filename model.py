import timm
import torch.nn as nn


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build and return a model using timm.

    timm.create_model() handles the classifier head replacement internally
    via the num_classes parameter, covering CNNs, ViTs, Swin, and 700+ others.

    Args:
        model_name:      Any timm model name, e.g. 'resnet50', 'vit_base_patch16_224',
                         'swin_tiny_patch4_window7_224', 'efficientnet_b0'.
        num_classes:     Number of output classes.
        pretrained:      Whether to load pretrained ImageNet weights.
        freeze_backbone: If True, freeze all layers except the classifier head.

    Returns:
        nn.Module ready to be moved to a device.
    """
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception as e:
        raise ValueError(
            f"Could not create model '{model_name}'. "
            f"Check that it is a valid timm model name.\n"
            f"Original error: {e}"
        )

    if freeze_backbone:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the classifier head (timm provides a unified API)
        classifier = model.get_classifier()
        if classifier is None:
            raise RuntimeError(
                f"Model '{model_name}' has no classifier head accessible via "
                f"get_classifier(). Cannot freeze backbone."
            )
        for param in classifier.parameters():
            param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> tuple:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
