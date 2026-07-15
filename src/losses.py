"""Masked L1 loss, averaged over skin pixels only.

Public API:
    masked_l1_loss(pred, target, mask, eps) -> torch.Tensor
"""

import torch


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean absolute error over skin pixels only.

        loss = sum(|pred - target| * mask) / sum(mask)

    Parameters
    ----------
    pred, target, mask : torch.Tensor
        Same shape (N, 1, H, W). mask is binary (0/1); pred and target in [0, 1].
    eps : float
        Guards against division by zero when a batch has no skin pixels.

    Returns
    -------
    torch.Tensor
        Scalar loss: the mean absolute error over the skin pixels (mask == 1).
    """
    abs_err = (pred - target).abs() * mask
    return abs_err.sum() / (mask.sum() + eps)
