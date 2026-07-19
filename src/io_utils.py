"""Handles all file I/O for raw data: RGB images and hyperspectral cubes.

Public API:
    load_cube(mat_path) -> np.ndarray          shape (H, W, B), float32
    load_rgb(jpg_path)  -> np.ndarray          shape (H, W, 3),  uint8
    get_wavelengths()   -> np.ndarray          1-D array of wavelengths in nm
    band_index_for_wavelength(wavelength_nm) -> int
"""

import numpy as np
from PIL import Image

import config


def load_cube(mat_path: str) -> np.ndarray:
    """Load a .mat hyperspectral cube and return shape (H, W, B) float32.

    The Hyper-Skin cubes are MATLAB v7.3 files, read with h5py.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file.

    Returns
    -------
    np.ndarray
        Shape (1024, 1024, 31), dtype float32.

    Raises
    ------
    RuntimeError
        If no data array is found in the file.
    ValueError
        If the loaded array shape doesn't match config.CUBE_SHAPE.
    """
    import h5py

    path = str(mat_path)
    with h5py.File(path, "r") as f:
        data_keys = [k for k in f.keys() if not k.startswith("#")]
        if not data_keys:
            raise RuntimeError(f"No data keys found in {path}")
        cube = np.array(f[data_keys[0]])
        # h5py reads MATLAB arrays transposed: (B, W, H) → need (H, W, B)
        cube = _orient_to_hwb(cube)
        _check_shape(cube, path)
        return cube.astype(np.float32)


def _orient_to_hwb(cube: np.ndarray) -> np.ndarray:
    """Transpose cube to (H, W, B) if bands appear to be on axis 0."""
    if cube.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {cube.shape}")
    # Heuristic: if the smallest dimension is first, it's likely the band axis
    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
        cube = np.transpose(cube, (2, 1, 0))
    return cube


def _check_shape(cube: np.ndarray, path: str) -> None:
    """Raise ValueError if cube shape != config.CUBE_SHAPE."""
    if cube.shape != config.CUBE_SHAPE:
        raise ValueError(
            f"Unexpected cube shape {cube.shape} in {path}; "
            f"expected {config.CUBE_SHAPE}"
        )


def load_rgb(jpg_path: str) -> np.ndarray:
    """Load a .jpg RGB image and return as (H, W, 3) uint8 array.

    Parameters
    ----------
    jpg_path : str
        Path to the .jpg file.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype uint8.
    """
    img = Image.open(str(jpg_path)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def get_wavelengths() -> np.ndarray:
    """Return the wavelength array corresponding to each band index.

    Uses config.WAVELENGTH_START_NM, WAVELENGTH_STEP_NM, and CUBE_SHAPE[2].

    Returns
    -------
    np.ndarray
        1-D array of length CUBE_SHAPE[2], e.g. [400, 410, ..., 700].
    """
    n_bands = config.CUBE_SHAPE[2]
    stop = config.WAVELENGTH_START_NM + config.WAVELENGTH_STEP_NM * n_bands
    return np.arange(config.WAVELENGTH_START_NM, stop, config.WAVELENGTH_STEP_NM)


def band_index_for_wavelength(wavelength_nm: int) -> int:
    """Convert a wavelength in nm to its zero-based band index.

    Parameters
    ----------
    wavelength_nm : int
        Target wavelength in nm.

    Returns
    -------
    int
        Zero-based index into the cube's band axis.

    Raises
    ------
    ValueError
        If the wavelength falls outside the cube's spectral range.
    """
    wavelengths = get_wavelengths()
    if wavelength_nm not in wavelengths:
        raise ValueError(
            f"Wavelength {wavelength_nm} nm is not in the cube range "
            f"[{int(wavelengths[0])}–{int(wavelengths[-1])} nm, "
            f"step {config.WAVELENGTH_STEP_NM} nm]"
        )
    return int((wavelength_nm - config.WAVELENGTH_START_NM) // config.WAVELENGTH_STEP_NM)
