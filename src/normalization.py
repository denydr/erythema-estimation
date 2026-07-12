"""Normalisation for the model inputs and target (notebook 03).

Technique: min-max normalisation to [0,1] for both streams — fixed-range scaling
(/255, known uint8 bounds) for RGB; robust percentile-based min-max (p1/p99 with
clipping, bounds measured from the data) for the EI target. Chosen over z-score
standardisation because a bounded [0,1] target pairs with a sigmoid output and a
fixed scale for SSIM/error maps; the percentiles keep log(1/R) outlier tails from
defining the scale.

RGB input is scaled by 1/255 at load time (no saved statistics needed).

The EI target (destriped EI) is scaled to [0, 1] with robust percentiles. The
percentiles are computed from the TRAIN split, over SKIN pixels only (mask==1),
so the scale reflects the erythema signal rather than the background — background
EI (a different distribution plus corrupted extremes) would otherwise skew it.
The same saved statistics are applied to every split at load time.
"""

import json
from pathlib import Path

import numpy as np

import config


def compute_ei_norm_stats(manifest, ei_dir, mask_dir,
                          percentiles=config.NORM_PERCENTILES) -> dict:
    """Robust low/high EI percentiles over train-split skin pixels.

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest (needs subject_id, pose, view, split).
    ei_dir : str
        Directory of destriped EI maps (data/processed/ei_maps_destriped).
    mask_dir : str
        Directory of binary masks (data/processed/masks).
    percentiles : tuple
        (low, high) percentiles, default config.NORM_PERCENTILES = (1, 99).

    Returns
    -------
    dict
        {percentiles, low, high, n_pixels, n_images, split}.
    """
    ei_dir, mask_dir = Path(ei_dir), Path(mask_dir)
    train = manifest[manifest["split"] == "train"]

    skin_values = []
    for _, row in train.iterrows():
        stem = f"{row['subject_id']}_{row['pose']}_{row['view']}"
        ei = np.load(ei_dir / f"{stem}.npy")
        mask = np.load(mask_dir / f"{stem}.npy").astype(bool)
        skin_values.append(ei[mask])

    values = np.concatenate(skin_values)
    low, high = np.percentile(values, percentiles)
    return {
        "percentiles": list(percentiles),
        "low": float(low),
        "high": float(high),
        "n_pixels": int(values.size),
        "n_images": int(len(train)),
        "split": "train",
    }


def save_stats(stats: dict, path: str) -> None:
    """Write normalisation statistics to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: str) -> dict:
    """Load normalisation statistics from JSON."""
    with open(path) as f:
        return json.load(f)


def normalize_ei(ei: np.ndarray, stats: dict) -> np.ndarray:
    """Scale EI to [0, 1] using saved (low, high) percentiles, clipped.

    Applied to the whole map; only skin pixels are used downstream (masked).
    """
    low, high = stats["low"], stats["high"]
    return np.clip((ei - low) / (high - low), 0.0, 1.0).astype(np.float32)


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Scale a uint8 [0, 255] RGB image to float32 [0, 1]."""
    return rgb.astype(np.float32) / 255.0


# ImageNet channel statistics — the preprocessing a resnet encoder pretrained on
# ImageNet expects (matches segmentation_models_pytorch's resnet preprocessing_fn).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_rgb_imagenet(rgb: np.ndarray) -> np.ndarray:
    """Scale a uint8 [0, 255] RGB image to float32 and ImageNet-normalise it.

    (rgb/255 - mean) / std, per channel. This is the input transform for the
    pretrained encoder — use instead of normalize_rgb when the encoder carries
    ImageNet weights. Shape (H, W, 3) in, same shape out.
    """
    return ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
