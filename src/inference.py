"""Tiled inference for the model stage (Stage 3).

The model is trained on CROP_SIZE patches, but predictions are needed for whole
1024x1024 images (validation and test). tiled_predict covers the image with a grid
of CROP_SIZE tiles, predicts each, and stitches them back into a full-size map —
every pixel predicted once. Tile starts are aligned so the last row/column reaches
the image edge even when the size is not an exact multiple of the tile.
"""

import torch

import config


def _tile_starts(length: int, tile: int):
    """Tile start positions covering [0, length) with the last tile flush to the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def tiled_predict(model, rgb, device, tile=config.CROP_SIZE):
    """Predict a full EI map for one image by tiling.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net (expects (N, 3, tile, tile), returns (N, 1, tile, tile)).
    rgb : torch.Tensor
        One preprocessed image, shape (3, H, W).
    device : torch.device
    tile : int
        Tile side length (defaults to the training crop size).

    Returns
    -------
    torch.Tensor
        Predicted map, shape (1, H, W), on CPU.
    """
    model.eval()
    _, h, w = rgb.shape
    pred = torch.zeros((1, h, w))
    for y in _tile_starts(h, tile):
        for x in _tile_starts(w, tile):
            patch = rgb[:, y:y + tile, x:x + tile].unsqueeze(0).to(device)
            out = model(patch).squeeze(0).cpu()
            pred[:, y:y + tile, x:x + tile] = out
    return pred