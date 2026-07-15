"""Mask-guided random cropping helpers (pure NumPy).

Public API:
    centroid_crop_coords(mask, size) -> (y, x)
    random_crop_coords(mask, size, min_skin_frac, max_tries, rng) -> (y, x)
    apply_crop(arr, y, x, size) -> np.ndarray
    hflip(arr) -> np.ndarray
"""

import numpy as np

import config


def centroid_crop_coords(mask: np.ndarray, size: int) -> tuple:
    """Top-left (y, x) of a crop centred on the mask's skin centroid.

    Clamped so the crop stays inside the image. If the mask is empty, the crop is
    centred on the image instead.

    Parameters
    ----------
    mask : np.ndarray
        Shape (H, W) binary mask, 1 = skin.
    size : int
        Crop side length.

    Returns
    -------
    tuple of int
        (y, x) top-left corner of the crop.
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
    """Top-left (y, x) of a random crop with at least `min_skin_frac` skin.

    Draws up to `max_tries` random positions and returns the first whose skin
    fraction (mask mean over the crop) reaches the threshold. If none qualifies,
    falls back to the centroid crop so the crop still lands on the face.

    Parameters
    ----------
    mask : np.ndarray
        Shape (H, W) binary mask, 1 = skin.
    size : int
        Crop side length.
    min_skin_frac : float
        Minimum fraction of skin pixels required to accept a crop.
    max_tries : int
        Number of random positions to try before the centroid fallback.
    rng : np.random.Generator, optional
        Random source (a fresh default_rng() is used if None).

    Returns
    -------
    tuple of int
        (y, x) top-left corner of the accepted (or fallback) crop.

    Raises
    ------
    ValueError
        If the crop size exceeds the mask dimensions.
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
    """Crop an array to the `size` square at (y, x).

    Parameters
    ----------
    arr : np.ndarray
        2-D or 3-D array with the spatial dimensions first (H, W, ...).
    y, x : int
        Top-left corner of the crop.
    size : int
        Crop side length.

    Returns
    -------
    np.ndarray
        The cropped view, shape (size, size, ...).
    """
    return arr[y:y + size, x:x + size]


def hflip(arr: np.ndarray) -> np.ndarray:
    """Horizontally flip an array.

    Parameters
    ----------
    arr : np.ndarray
        2-D or 3-D array with the spatial dimensions first (H, W, ...).

    Returns
    -------
    np.ndarray
        The array mirrored along the width axis.
    """
    return np.flip(arr, axis=1)