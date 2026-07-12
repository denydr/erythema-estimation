"""PyTorch Dataset for the U-Net stage (Stage 3): RGB -> normalised EI map.

Each sample yields (rgb, ei, mask) as float32 CHW tensors, pixel-aligned:
    rgb  : (3, H, W)  ImageNet-normalised for the pretrained encoder
    ei   : (1, H, W)  destriped EI normalised to [0, 1] with train-skin stats
    mask : (1, H, W)  binary 0/1 skin selector (for the masked loss)

Two modes:
    "train" -> a mask-guided random CROP_SIZE crop + optional horizontal flip
               (memory management + geometric augmentation).
    "full"  -> the whole 1024x1024 map, no crop, no flip (for tiled inference /
               validation; see the tiling utility in the training stage).

The mask is returned as a separate channel, never baked into the EI — the loss
and metrics select skin pixels with it. RGB uses ImageNet preprocessing (the
encoder is pretrained); the EI target keeps the pipeline's [0, 1] normalisation.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from src.cropping import apply_crop, hflip, random_crop_coords
from src.io_utils import load_rgb
from src.normalization import normalize_ei, preprocess_rgb_imagenet


class ErythemaDataset(Dataset):
    """Paired (RGB, EI, mask) samples for one split."""

    def __init__(self, manifest, ei_dir, mask_dir, stats, split=None,
                 mode="train", crop_size=config.CROP_SIZE,
                 min_skin_frac=config.CROP_MIN_SKIN_FRAC,
                 max_tries=config.CROP_MAX_TRIES, flip=True,
                 crops_per_image=1, seed=None):
        """
        Parameters
        ----------
        manifest : pd.DataFrame
            Dataset manifest (subject_id, pose, view, split, rgb_path).
        ei_dir, mask_dir : str
            Directories of destriped EI maps and binary masks.
        stats : dict
            EI normalisation stats from src.normalization.load_stats.
        split : str, optional
            If given, keep only rows with this split value.
        mode : {"train", "full"}
            "train" applies random crop + flip; "full" returns whole maps.
        crop_size, min_skin_frac, max_tries : crop policy (train mode).
        flip : bool
            Enable random horizontal flip (train mode only).
        crops_per_image : int
            Random crops drawn per image per epoch (train mode). Multiplies the
            dataset length; each draw re-crops the same image independently.
        seed : int, optional
            Seed for this dataset's RNG. With multiple DataLoader workers, use
            worker_init_fn (below) so each worker gets a distinct crop stream.
        """
        if mode not in ("train", "full"):
            raise ValueError(f"mode must be 'train' or 'full', got {mode!r}")

        df = manifest if split is None else manifest[manifest["split"] == split]
        self.rows = df.reset_index(drop=True)
        self.ei_dir = Path(ei_dir)
        self.mask_dir = Path(mask_dir)
        self.stats = stats
        self.mode = mode
        self.crop_size = crop_size
        self.min_skin_frac = min_skin_frac
        self.max_tries = max_tries
        self.flip = flip
        self.crops_per_image = crops_per_image if mode == "train" else 1
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows) * self.crops_per_image

    def __getitem__(self, i):
        row = self.rows.iloc[i % len(self.rows)]
        stem = f"{row['subject_id']}_{row['pose']}_{row['view']}"
        rgb = load_rgb(str(row["rgb_path"]))                       # (H, W, 3) uint8
        ei = np.load(self.ei_dir / f"{stem}.npy")                 # (H, W) float32
        mask = np.load(self.mask_dir / f"{stem}.npy")            # (H, W) uint8 {0,1}

        if self.mode == "train":
            y, x = random_crop_coords(mask, self.crop_size, self.min_skin_frac,
                                      self.max_tries, self._rng)
            rgb = apply_crop(rgb, y, x, self.crop_size)
            ei = apply_crop(ei, y, x, self.crop_size)
            mask = apply_crop(mask, y, x, self.crop_size)
            if self.flip and self._rng.random() < 0.5:
                rgb, ei, mask = hflip(rgb), hflip(ei), hflip(mask)

        rgb_n = preprocess_rgb_imagenet(np.ascontiguousarray(rgb))    # (H, W, 3)
        ei_n = normalize_ei(np.ascontiguousarray(ei), self.stats)     # (H, W)
        mask_f = np.ascontiguousarray(mask).astype(np.float32)        # (H, W)

        rgb_t = torch.from_numpy(rgb_n.transpose(2, 0, 1)).contiguous()
        ei_t = torch.from_numpy(ei_n[None]).contiguous()
        mask_t = torch.from_numpy(mask_f[None]).contiguous()
        return rgb_t, ei_t, mask_t


def worker_init_fn(worker_id):
    """Give each DataLoader worker a distinct, reproducible crop RNG.

    PyTorch does not reseed NumPy per worker, so without this every worker would
    draw identical crops. Pass as DataLoader(worker_init_fn=worker_init_fn).
    """
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(info.seed)