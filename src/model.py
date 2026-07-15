"""U-Net (ResNet-34 encoder) mapping RGB to a single-channel EI map in [0, 1].

Public API:
    build_unet(encoder_name, encoder_weights) -> torch.nn.Module
    get_device() -> torch.device
"""

import torch

import config


def build_unet(encoder_name=config.ENCODER_NAME,
               encoder_weights=config.ENCODER_WEIGHTS):
    """Build the U-Net with a single-channel sigmoid output.

    Parameters
    ----------
    encoder_name : str
        Encoder backbone, e.g. "resnet34".
    encoder_weights : str or None
        Pretrained encoder weights, e.g. "imagenet" (None for random init).

    Returns
    -------
    torch.nn.Module
        A segmentation_models_pytorch U-Net taking (N, 3, H, W) RGB and
        returning (N, 1, H, W) predictions in [0, 1].
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
    """Select the best available torch device.

    Returns
    -------
    torch.device
        "cuda" if a CUDA GPU is available, otherwise "mps" on Apple Silicon,
        otherwise "cpu".
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
