"""Compute the EI normalisation statistics from train-split skin pixels.

Usage:
    python scripts/compute_norm_stats.py

Reads the destriped EI maps and masks, takes the 1st/99th EI percentiles over
train-split skin pixels, and writes data/processed/norm_stats.json. Run after
compute_masks.py and destripe_ei_maps.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.manifest import load_manifest
from src.normalization import compute_ei_norm_stats, save_stats


def main() -> None:
    """Compute and save EI normalisation statistics from train-split skin pixels."""
    processed = Path(config.LOCAL_PROCESSED_DIR)
    manifest_path = processed / "manifest.csv"
    ei_dir = processed / "ei_maps_destriped"
    mask_dir = processed / "masks"

    for path, what in [
        (manifest_path, "manifest.csv (run scripts/build_manifest.py)"),
        (ei_dir, "destriped EI maps (run scripts/destripe_ei_maps.py)"),
        (mask_dir, "masks (run scripts/compute_masks.py)"),
    ]:
        if not path.exists():
            print(f"Not found: {path}  -  {what}")
            sys.exit(1)

    manifest = load_manifest(str(manifest_path))
    stats = compute_ei_norm_stats(manifest, str(ei_dir), str(mask_dir))

    out_path = processed / "norm_stats.json"
    save_stats(stats, str(out_path))
    print(
        f"EI norm stats (train skin, p{stats['percentiles']}): "
        f"low={stats['low']:.2f}  high={stats['high']:.2f}  "
        f"from {stats['n_pixels']:,} px / {stats['n_images']} imgs -> {out_path}"
    )


if __name__ == "__main__":
    main()
