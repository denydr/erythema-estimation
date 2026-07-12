"""CLI entry point: compute binary face-skin masks for all manifest samples.

Usage:
    python scripts/compute_masks.py

Loads manifest.csv from data/processed/, reads each RGB image, runs the
MediaPipe face-skin segmenter (notebook 02_skin_masking), and saves a binary
(0/1) mask .npy per image to data/processed/masks/. Skips files that already
exist (resumable). Needs the RGB images (the dataset SSD). The masks are the
region selector multiplied into the EI target / RGB downstream.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.manifest import load_manifest
from src.masking import batch_compute_masks


def main() -> None:
    """Load the manifest and run batch face-skin masking."""
    processed = Path(config.LOCAL_PROCESSED_DIR)
    manifest_path = processed / "manifest.csv"

    if not manifest_path.exists():
        print(
            f"Manifest not found at {manifest_path}.\n"
            "Run 'python scripts/build_manifest.py' first."
        )
        sys.exit(1)

    manifest = load_manifest(str(manifest_path))
    output_dir = processed / "masks"

    print(f"Computing face-skin masks for {len(manifest)} samples -> {output_dir}")
    batch_compute_masks(manifest, str(output_dir))


if __name__ == "__main__":
    main()