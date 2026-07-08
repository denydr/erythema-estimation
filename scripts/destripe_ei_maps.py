"""CLI entry point: batch-destripe the EI maps for all samples in the manifest.

Usage:
    python scripts/destripe_ei_maps.py

Loads manifest.csv from data/processed/, reads each raw EI map from
data/processed/ei_maps/, removes the push-broom stripe (notebook 01c), and
saves the destriped .npy files to data/processed/ei_maps_destriped/.
Runs offline (no hyperspectral cubes needed). Skips files that already exist
(resumable). The destriped maps are the model target and the input to masking.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.manifest import load_manifest
from src.ei_computation import batch_destripe_ei_maps


def main() -> None:
    """Load the manifest and run batch EI map destriping."""
    processed = Path(config.LOCAL_PROCESSED_DIR)
    manifest_path = processed / "manifest.csv"
    input_dir = processed / "ei_maps"

    if not manifest_path.exists():
        print(
            f"Manifest not found at {manifest_path}.\n"
            "Run 'python scripts/build_manifest.py' first."
        )
        sys.exit(1)

    if not input_dir.exists():
        print(
            f"Raw EI maps not found at {input_dir}.\n"
            "Run 'python scripts/compute_ei_maps.py' first."
        )
        sys.exit(1)

    manifest = load_manifest(str(manifest_path))
    output_dir = processed / "ei_maps_destriped"

    print(f"Destriping EI maps for {len(manifest)} samples → {output_dir}")
    batch_destripe_ei_maps(manifest, str(input_dir), str(output_dir))


if __name__ == "__main__":
    main()