
"""Mask-guided random cropping for the model stage (Stage 3).

Full 1024x1024 maps do not fit in memory, so training samples random square crops.
Because facial skin is only ~26% of the frame (and off-centre on side profiles),
crops are guided by the mask: a crop is accepted only if enough of it is skin,
resampling a few times, then falling back to a crop centred on the mask centroid.

These helpers are pure NumPy (no torch) so the crop logic is testable on its own.
The same crop coordinates must be applied to the RGB, EI, and mask together — the
Dataset does that via apply_crop; here we only choose the coordinates.
"""

import numpy as np

import config


def centroid_crop_coords(mask: np.ndarray, size: int) -> tuple:
    """Top-left (y, x) of a `size` crop centred on the mask's skin centroid.

    Clamped so the crop stays inside the image. If the mask is empty, centres
    the crop on the image instead.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = int(ys.mean()), int(xs.mean())
    y = int(np.clip(cy - size // 2, 0, h - size))
    x = int(np.clip(cx - size // 2, 0, w - size))
    return y, x


def random_crop_coords(mask: np.ndarray, size: int = config.CROP_SIZE,
                       min_skin_frac: float = config.CROP_MIN_SKIN_FRAC,
                       max_tries: int = config.CROP_MAX_TRIES,
                       rng: np.random.Generator = None) -> tuple:
    """Top-left (y, x) of a random `size` crop with at least `min_skin_frac` skin.

    Draws up to `max_tries` random positions and returns the first whose skin
    fraction (mask mean over the crop) reaches the threshold. If none qualifies,
    falls back to the centroid crop so the crop still lands on the face.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) binary mask, 1 = skin.
    size : int
        Crop side length.
    min_skin_frac : float
        Minimum fraction of skin pixels required to accept a crop.
    max_tries : int
        Number of random positions to try before the centroid fallback.
    rng : np.random.Generator
        Random source (a fresh default_rng() is used if None).
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = mask.shape
    if h < size or w < size:
        raise ValueError(f"Crop size {size} exceeds mask shape {mask.shape}")

    for _ in range(max_tries):
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
        if mask[y:y + size, x:x + size].mean() >= min_skin_frac:
            return y, x
    return centroid_crop_coords(mask, size)


def apply_crop(arr: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    """Crop `arr` (2-D or 3-D H,W-first) to the `size` square at (y, x)."""
    return arr[y:y + size, x:x + size]


def hflip(arr: np.ndarray) -> np.ndarray:
    """Horizontal flip of a 2-D or 3-D (H, W, ...) array."""
    return np.flip(arr, axis=1)