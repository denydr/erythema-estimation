"""Implements the Dawson erythema index formula, destriping, and batch processing.

Public API:
    compute_ei_map(cube, wavelengths) -> np.ndarray   shape (H, W), float32
    batch_compute_ei_maps(manifest, output_dir)
    destripe_ei(ei_map, window) -> np.ndarray          shape (H, W), float32
    batch_destripe_ei_maps(manifest, input_dir, output_dir)
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

import config
from src.io_utils import load_cube, band_index_for_wavelength


def compute_ei_map(cube: np.ndarray, wavelengths: list) -> np.ndarray:
    """Compute a 2D Dawson erythema index map from a hyperspectral cube.

    Parameters
    ----------
    cube : np.ndarray
        Shape (H, W, B), float32 reflectance values in [0, 1].
    wavelengths : list[int]
        Five wavelengths in nm ordered as [p, q, r, s, t]:
        [510, 540, 560, 580, 610].

    Returns
    -------
    np.ndarray
        Shape (H, W), dtype float32, Dawson EI values.

    Raises
    ------
    ValueError
        If wavelengths list does not have exactly 5 entries.
    """
    if len(wavelengths) != 5:
        raise ValueError(
            f"Expected exactly 5 wavelengths [p,q,r,s,t], got {len(wavelengths)}"
        )

    indices = [band_index_for_wavelength(w) for w in wavelengths]

    # Clip to REFLECTANCE_FLOOR before log to avoid log10(0)
    bands = [np.clip(cube[:, :, i], config.REFLECTANCE_FLOOR, None) for i in indices]

    # Log reciprocal reflectance: log10(1/R) = -log10(R)
    p, q, r, s, t = [-np.log10(b) for b in bands]

    ei = 100.0 * (r + 1.5 * (q + s) - 2.0 * (p + t))
    return ei.astype(np.float32)


def batch_compute_ei_maps(manifest, output_dir: str) -> None:
    """Compute and save EI maps for every row in the manifest.

    Output files are named <subject_id>_<pose>_<view>.npy and saved
    in output_dir. Already-existing files are skipped (resumable).

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest with at least columns:
        subject_id, pose, view, vis_path.
    output_dir : str
        Directory to write .npy EI map files into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = len(manifest)
    computed = 0
    skipped = 0

    for _, row in manifest.iterrows():
        stem = f"{row['subject_id']}_{row['pose']}_{row['view']}"
        out_path = out / f"{stem}.npy"

        if out_path.exists():
            skipped += 1
            continue

        cube = load_cube(row["vis_path"])
        ei_map = compute_ei_map(cube, config.DAWSON_WAVELENGTHS)
        np.save(out_path, ei_map)
        computed += 1
        print(f"  [{computed + skipped}/{total}] {stem}")

    print(
        f"\nDone. Computed: {computed}, "
        f"skipped (already exist): {skipped}, "
        f"total: {total}"
    )


def destripe_ei(ei_map: np.ndarray, window: int = config.DESTRIPE_MEDIAN_WINDOW) -> np.ndarray:
    """Remove the pushbroom vertical stripe from an EI map.

    Parameters
    ----------
    ei_map : np.ndarray
        Shape (H, W), a Dawson EI map from compute_ei_map.
    window : int
        Median-filter window in columns used to isolate the stripe offset.

    Returns
    -------
    np.ndarray
        Shape (H, W), dtype float32, destriped EI map.
    """
    col = np.median(ei_map, axis=0)
    offset = col - median_filter(col, size=window)
    return (ei_map - offset[np.newaxis, :]).astype(np.float32)


def batch_destripe_ei_maps(manifest, input_dir: str, output_dir: str) -> None:
    """Destripe every EI map referenced by the manifest.

    Reads raw EI maps <subject_id>_<pose>_<view>.npy from input_dir, applies
    destripe_ei, and writes the destriped maps under the same names in
    output_dir. Runs offline (no hyperspectral cubes needed). Already-existing
    output files are skipped (resumable).

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest with columns subject_id, pose, view.
    input_dir : str
        Directory of raw EI map .npy files (data/processed/ei_maps).
    output_dir : str
        Directory to write destriped .npy EI maps into.
    """
    src_dir = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = len(manifest)
    computed = 0
    skipped = 0

    for _, row in manifest.iterrows():
        stem = f"{row['subject_id']}_{row['pose']}_{row['view']}"
        out_path = out / f"{stem}.npy"

        if out_path.exists():
            skipped += 1
            continue

        ei_map = np.load(src_dir / f"{stem}.npy")
        np.save(out_path, destripe_ei(ei_map))
        computed += 1
        print(f"  [{computed + skipped}/{total}] {stem}")

    print(
        f"\nDone. Destriped: {computed}, "
        f"skipped (already exist): {skipped}, "
        f"total: {total}"
    )
