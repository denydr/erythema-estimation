"""CLI entry point: batch-compute EI maps for all samples in the manifest.

Usage:
    python scripts/compute_ei_maps.py

Loads manifest.csv from data/processed/, computes a Dawson EI map for
each sample, and saves .npy files to data/processed/ei_maps/.
Skips files that already exist (resumable).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.manifest import load_manifest
from src.ei_computation import batch_compute_ei_maps


def main() -> None:
    """Load the manifest and run batch EI map computation."""
    manifest_path = Path(config.LOCAL_PROCESSED_DIR) / "manifest.csv"

    if not manifest_path.exists():
        print(
            f"Manifest not found at {manifest_path}.\n"
            "Run 'python scripts/build_manifest.py' first."
        )
        sys.exit(1)

    manifest = load_manifest(str(manifest_path))
    output_dir = Path(config.LOCAL_PROCESSED_DIR) / "ei_maps"

    print(f"Computing EI maps for {len(manifest)} samples → {output_dir}")
    batch_compute_ei_maps(manifest, str(output_dir))


if __name__ == "__main__":
    main()
