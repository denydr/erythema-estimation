"""Masked evaluation metrics over skin pixels: MAE, MSE, and SSIM.

Public API:
    masked_mae(pred, target, mask, eps)  -> float
    masked_mse(pred, target, mask, eps)  -> float
    masked_ssim(pred, target, mask)      -> float
"""

import numpy as np
import torch


def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean absolute error over skin pixels only.

    Parameters
    ----------
    pred, target, mask : torch.Tensor
        Same shape (..., H, W). mask is binary (0/1); pred and target in [0, 1].
    eps : float
        Guards against division by zero when there are no skin pixels.

    Returns
    -------
    float
        Mean absolute error averaged over the skin pixels (mask == 1).
    """
    err = (pred - target).abs() * mask
    return float(err.sum() / (mask.sum() + eps))


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean squared error over skin pixels only.

    Parameters
    ----------
    pred, target, mask : torch.Tensor
        Same shape (..., H, W). mask is binary (0/1); pred and target in [0, 1].
    eps : float
        Guards against division by zero when there are no skin pixels.

    Returns
    -------
    float
        Mean squared error averaged over the skin pixels (mask == 1).
    """
    err = (pred - target).pow(2) * mask
    return float(err.sum() / (mask.sum() + eps))


def masked_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Structural similarity averaged over skin pixels.

    Parameters
    ----------
    pred, target : np.ndarray
        2-D maps in [0, 1] (normalised EI).
    mask : np.ndarray
        2-D boolean skin mask.

    Returns
    -------
    float
        Mean SSIM over the skin pixels (mask == 1); 1.0 is a perfect match.
    """
    from skimage.metrics import structural_similarity

    _, ssim_map = structural_similarity(
        target, pred, data_range=1.0, full=True,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    return float(ssim_map[mask].mean())
