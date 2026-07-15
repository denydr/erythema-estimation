"""Generate the dataset manifest CSV from the dataset folder structure.

Usage:
    python scripts/build_manifest.py

Reads paths and split overrides from config.py; writes data/processed/manifest.csv
and prints a summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.manifest import build_manifest, save_manifest


def main() -> None:
    """Build and save the dataset manifest, then print a summary."""
    print(f"Building manifest from: {config.DATA_ROOT}")
    print(f"Split overrides: {config.SPLIT_OVERRIDE}")

    df = build_manifest(config.DATA_ROOT, config.SPLIT_OVERRIDE)

    output_path = Path(config.LOCAL_PROCESSED_DIR) / "manifest.csv"
    save_manifest(df, str(output_path))

    print(f"\nSaved → {output_path}")
    print(f"Total samples : {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"\nSamples per split:\n{df['split'].value_counts().to_string()}")

    print("\nSplit override verification:")
    for subj, expected in config.SPLIT_OVERRIDE.items():
        actual = df.loc[df["subject_id"] == subj, "split"].unique().tolist()
        status = "OK" if actual == [expected] else "MISMATCH"
        print(f"  {subj} → {expected}: {status}  (found: {actual})")


if __name__ == "__main__":
    main()
