"""U-Net for the model stage (Stage 3): RGB patch -> EI map in [0, 1].

A U-Net (Ronneberger et al. 2015) with a ResNet-34 encoder pretrained on ImageNet,
via segmentation_models_pytorch. Single-channel output with a sigmoid, matching the
[0, 1]-normalised EI target. The encoder's ImageNet preprocessing is applied to the
RGB input in the Dataset (src.normalization.preprocess_rgb_imagenet).
"""

import torch

import config


def build_unet(encoder_name=config.ENCODER_NAME,
               encoder_weights=config.ENCODER_WEIGHTS):
    """Create the U-Net (1-channel sigmoid output).

    Parameters
    ----------
    encoder_name : str
        Backbone, e.g. "resnet34".
    encoder_weights : str or None
        Pretrained weights, e.g. "imagenet" (None for random init).
    """
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )


def get_device():
    """Return the best available torch device: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
