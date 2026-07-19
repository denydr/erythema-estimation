"""Normalisation of the model input (RGB) and target (EI).

Public API:
    compute_ei_norm_stats(manifest, ei_dir, mask_dir, percentiles) -> dict
    save_stats(stats, path)
    load_stats(path) -> dict
    normalize_ei(ei, stats) -> np.ndarray
    preprocess_rgb_imagenet(rgb) -> np.ndarray
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
    """Write normalisation statistics to JSON.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_ei_norm_stats.
    path : str
        Destination JSON path (parent directories are created if needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: str) -> dict:
    """Load normalisation statistics from JSON.

    Parameters
    ----------
    path : str
        Path to the statistics JSON written by save_stats.

    Returns
    -------
    dict
        The saved statistics dictionary.
    """
    with open(path) as f:
        return json.load(f)


def normalize_ei(ei: np.ndarray, stats: dict) -> np.ndarray:
    """Scale an EI map to [0, 1] using saved (low, high) percentiles, clipped.

    Applied to the whole map; only skin pixels are used downstream (masked).

    Parameters
    ----------
    ei : np.ndarray
        A destriped EI map.
    stats : dict
        Statistics with "low" and "high" keys (from load_stats).

    Returns
    -------
    np.ndarray
        float32 map scaled to [0, 1] and clipped, same shape as `ei`.
    """
    low, high = stats["low"], stats["high"]
    return np.clip((ei - low) / (high - low), 0.0, 1.0).astype(np.float32)


# ImageNet channel statistics — the preprocessing a resnet encoder pretrained on
# ImageNet expects (matches segmentation_models_pytorch's resnet preprocessing_fn).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_rgb_imagenet(rgb: np.ndarray) -> np.ndarray:
    """ImageNet-standardise a uint8 RGB image for the pretrained encoder.

    Computes (rgb/255 - mean) / std per channel — the model input transform
    expected by the ImageNet-pretrained encoder.

    Parameters
    ----------
    rgb : np.ndarray
        Shape (H, W, 3), uint8.

    Returns
    -------
    np.ndarray
        float32 ImageNet-standardised image, same shape (values not in [0, 1]).
    """
    return ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
