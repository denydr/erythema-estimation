"""Implements the Dawson erythema index formula and batch processing.

Public API:
    compute_ei_map(cube, wavelengths) -> np.ndarray   shape (H, W), float32
    batch_compute_ei_maps(manifest, output_dir)
"""

from pathlib import Path

import numpy as np

import config
from src.io_utils import load_cube, band_index_for_wavelength


def compute_ei_map(cube: np.ndarray, wavelengths: list) -> np.ndarray:
    """Compute a 2D Dawson erythema index map from a hyperspectral cube.

    Formula (Abdlaty et al. 2021, Eq. 3 — citing Dawson et al. 1980):
        DEI = 100 × [r + (3/2)(q + s) − 2(p + t)]
    where p, q, r, s, t = log10(1 / R) at 510, 540, 560, 580, 610 nm.

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
