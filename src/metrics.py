"""Masked evaluation metrics for the model stage (Stage 3).

All metrics are computed over skin pixels only (mask == 1), matching the training
loss. MAE is the model-selection metric during training;
MSE and SSIM are reported at evaluation. MAE/MSE are computed on the normalised
[0, 1] EI space (denormalise by multiplying by the p99-p1 range for EI units);
SSIM uses data_range=1.0 on the normalised maps.
"""

import numpy as np
import torch


def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean absolute error over skin pixels."""
    err = (pred - target).abs() * mask
    return float(err.sum() / (mask.sum() + eps))


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean squared error over skin pixels."""
    err = (pred - target).pow(2) * mask
    return float(err.sum() / (mask.sum() + eps))


def masked_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Structural similarity averaged over skin pixels.

    Computes the per-pixel SSIM map over the whole image (needed for SSIM's local
    windows), then averages it over the skin region only — measuring whether the
    predicted map has the right spatial *structure*, not just the right level.

    Parameters
    ----------
    pred, target : np.ndarray
        2-D maps in [0, 1] (normalised EI).
    mask : np.ndarray
        2-D boolean skin mask.
    """
    from skimage.metrics import structural_similarity

    _, ssim_map = structural_similarity(
        target, pred, data_range=1.0, full=True,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    return float(ssim_map[mask].mean())