"""Face-skin masking (notebook 02_skin_masking).

The mask is produced from the RGB image alone, by MediaPipe's multiclass selfie
segmenter, keeping the face-skin class only (per-pixel; hair and background are
excluded by class — no threshold or morphology). The output is a binary (0/1)
uint8 array. Masks are saved one .npy per image and multiplied into the EI
target / RGB downstream (never pre-baked into them).
"""

import urllib.request
from pathlib import Path

import numpy as np

import config
from src.io_utils import load_rgb

_SEGMENTER = None


def _get_segmenter():
    """Lazily create the MediaPipe segmenter, downloading the model if absent."""
    global _SEGMENTER
    if _SEGMENTER is None:
        from mediapipe.tasks.python import BaseOptions, vision

        model = Path(config.SEG_MODEL_PATH)
        if not model.exists():
            model.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading segmenter model -> {model}")
            urllib.request.urlretrieve(config.SEG_MODEL_URL, model)
        _SEGMENTER = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(model)),
                output_category_mask=True,
            )
        )
    return _SEGMENTER


def compute_mask(rgb: np.ndarray) -> np.ndarray:
    """Binary face-skin mask for one RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        Shape (H, W, 3), uint8 RGB image.

    Returns
    -------
    np.ndarray
        Shape (H, W), uint8 in {0, 1}: 1 = face skin, 0 = everything else.
    """
    import mediapipe as mp

    segmenter = _get_segmenter()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    category_mask = segmenter.segment(image).category_mask.numpy_view()
    if category_mask.ndim == 3:
        category_mask = category_mask[..., 0]
    return (category_mask == config.FACE_SKIN_CLASS).astype(np.uint8)


def batch_compute_masks(manifest, output_dir: str) -> None:
    """Compute and save a binary face-skin mask for every image in the manifest.

    Reads each RGB image (rgb_path), runs compute_mask, and writes
    <subject_id>_<pose>_<view>.npy to output_dir. Already-existing outputs are
    skipped (resumable). Needs the RGB images (the dataset SSD).

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest with columns subject_id, pose, view, rgb_path.
    output_dir : str
        Directory to write binary mask .npy files into.
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

        rgb = load_rgb(str(row["rgb_path"]))
        np.save(out_path, compute_mask(rgb))
        computed += 1
        print(f"  [{computed + skipped}/{total}] {stem}")

    print(
        f"\nDone. Masks: {computed}, "
        f"skipped (already exist): {skipped}, total: {total}"
    )