"""Builds and manages the dataset manifest CSV.

Public API:
    parse_filename(filename) -> dict
    build_manifest(ssd_root, split_override) -> pd.DataFrame
    save_manifest(df, output_path)
    load_manifest(path) -> pd.DataFrame
"""

import re
from pathlib import Path

import pandas as pd


def parse_filename(filename: str) -> dict:
    """Extract subject_id, pose, and view from a filename string.

    Parameters
    ----------
    filename : str
        File stem such as "p001_neutral_front" (extension optional).

    Returns
    -------
    dict
        Keys: subject_id, pose, view.

    Raises
    ------
    ValueError
        If the filename doesn't match the expected pattern.
    """
    stem = Path(filename).stem
    m = re.match(r"^(p\d+)_([^_]+)_(.+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse filename: {filename!r}")
    return {"subject_id": m.group(1), "pose": m.group(2), "view": m.group(3)}


def build_manifest(ssd_root: str, split_override: dict) -> pd.DataFrame:
    """Walk the SSD folder structure and build the dataset manifest.

    Scans {data_root}/{train,test,valid}/RGB/*.jpg, pairs each with the
    corresponding VIS/*.mat file, applies split_override, and returns
    one row per sample.

    Parameters
    ----------
    data_root : str
        Path to the Hyper-Skin(RGB, VIS) directory (contains train/, test/, valid/).
    split_override : dict
        subject_id → split mapping that overrides the folder-based split
        (e.g. {"p027": "test", "p019": "test", "p012": "test"}).

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, pose, view, split, rgb_path, vis_path.
    """
    root = Path(ssd_root)
    rows = []

    for folder_split in ("train", "test", "valid"):
        rgb_dir = root / folder_split / "RGB"
        vis_dir = root / folder_split / "VIS"
        if not rgb_dir.exists():
            continue

        for rgb_file in sorted(rgb_dir.glob("*.jpg")):
            try:
                parsed = parse_filename(rgb_file.stem)
            except ValueError:
                continue

            vis_file = vis_dir / (rgb_file.stem + ".mat")
            assigned_split = split_override.get(parsed["subject_id"], folder_split)

            rows.append({
                "subject_id": parsed["subject_id"],
                "pose": parsed["pose"],
                "view": parsed["view"],
                "split": assigned_split,
                "rgb_path": str(rgb_file),
                "vis_path": str(vis_file),
            })

    columns = ["subject_id", "pose", "view", "split", "rgb_path", "vis_path"]
    return pd.DataFrame(rows, columns=columns)


def save_manifest(df: pd.DataFrame, output_path: str) -> None:
    """Save manifest DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest to save.
    output_path : str
        Destination file path (parent directories are created if needed).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_manifest(path: str) -> pd.DataFrame:
    """Load a previously saved manifest CSV.

    Parameters
    ----------
    path : str
        Path to the manifest CSV.

    Returns
    -------
    pd.DataFrame
        Loaded manifest.
    """
    return pd.read_csv(path)
