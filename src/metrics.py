"""Masked evaluation metrics for the model stage (Stage 3).

All metrics are computed over skin pixels only (mask == 1), matching the training
loss and the README contract. MAE is the model-selection metric during training;
MSE is reported alongside it. SSIM is added with the
evaluation script. Inputs are in the normalised [0, 1] EI space.
"""

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