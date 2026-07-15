"""Evaluate the trained U-Net on the test split.

Usage:
    python scripts/evaluate.py                       # uses outputs/best_model.pt
    python scripts/evaluate.py --checkpoint path.pt

Predicts every test image by tiling and reports masked MAE/MSE/SSIM over skin,
stratified by view and pose, in normalised and EI units. Writes per-subject,
per-view/pose, and aggregate metric tables plus a qualitative figure to OUTPUT_DIR.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

import config
from src.dataset import ErythemaDataset
from src.inference import tiled_predict
from src.io_utils import load_rgb
from src.metrics import masked_mae, masked_mse, masked_ssim
from src.model import build_unet, get_device
from src.normalization import load_stats, normalize_ei

DISCLOSURE = ["p012", "p019", "p027"]   # only subjects permitted for display


def parse_args() -> argparse.Namespace:
    """Parse the evaluation command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments (checkpoint path).
    """
    p = argparse.ArgumentParser(description="Evaluate the erythema U-Net on the test split.")
    p.add_argument("--checkpoint", type=str,
                   default=str(Path(config.OUTPUT_DIR) / "best_model.pt"))
    return p.parse_args()


def load_model(checkpoint: str, device) -> torch.nn.Module:
    """Build the U-Net architecture and load trained weights.

    The encoder is built with random init (no ImageNet download) since the
    checkpoint supplies all weights.

    Parameters
    ----------
    checkpoint : str
        Path to the saved state_dict (outputs/best_model.pt).
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    torch.nn.Module
        The trained U-Net in eval mode on `device`.
    """
    model = build_unet(encoder_weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, manifest, ei_dir, mask_dir, stats, device):
    """Compute per-image masked MAE/MSE/SSIM on the test split.

    MAE/MSE are recorded in normalised [0,1] units (mae_norm/mse_norm — the scale
    to compare against normalised literature values) and in denormalised EI units
    (mae_ei/mse_ei — interpretable magnitude). SSIM is on the normalised maps.

    Parameters
    ----------
    model : torch.nn.Module
        The trained U-Net.
    manifest : pd.DataFrame
        Dataset manifest.
    ei_dir, mask_dir : Path or str
        Directories of destriped EI maps and binary masks.
    stats : dict
        EI normalisation statistics (from load_stats).
    device : torch.device
        Device to run on.

    Returns
    -------
    pd.DataFrame
        One row per test image with subject_id, pose, view and the metrics
        (mae_norm, mse_norm, ssim, mae_ei, mse_ei).
    """
    test = manifest[manifest["split"] == "test"]
    ds = ErythemaDataset(test, ei_dir, mask_dir, stats, mode="full")
    rng = stats["high"] - stats["low"]   # denormalisation factor for EI units

    rows = []
    for i, r in enumerate(test.itertuples(index=False)):
        rgb, ei, mask = ds[i]
        pred = tiled_predict(model, rgb, device)
        mae_n = masked_mae(pred, ei, mask)
        mse_n = masked_mse(pred, ei, mask)
        ssim = masked_ssim(pred[0].numpy(), ei[0].numpy(), mask[0].numpy().astype(bool))
        rows.append({
            "subject_id": r.subject_id, "pose": r.pose, "view": r.view,
            "mae_norm": mae_n, "mse_norm": mse_n, "ssim": ssim,
            "mae_ei": mae_n * rng, "mse_ei": mse_n * rng * rng,
        })
    return pd.DataFrame(rows)


def qualitative_figure(model, manifest, ei_dir, mask_dir, stats, device, path):
    """Save a four-panel figure (RGB | GT | prediction | error) per disclosure subject.

    Each row is one display-permitted image (p012/p019/p027), annotated with its
    SSIM and MAE. GT/prediction share a [0, 1] colour scale; the error map uses a
    fixed scale across rows.

    Parameters
    ----------
    model : torch.nn.Module
        The trained U-Net.
    manifest : pd.DataFrame
        Dataset manifest.
    ei_dir, mask_dir : Path or str
        Directories of destriped EI maps and binary masks.
    stats : dict
        EI normalisation statistics.
    device : torch.device
        Device to run on.
    path : Path or str
        Output path for the PNG figure.
    """
    import matplotlib.pyplot as plt

    rows = manifest[(manifest["subject_id"].isin(DISCLOSURE)) &
                    (manifest["split"] == "test")].reset_index(drop=True)
    if len(rows) == 0:
        print("No disclosure subjects in the test split; skipping figure.")
        return

    rng = stats["high"] - stats["low"]
    err_vmax = 0.3   # fixed error scale (normalised units) across all rows
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.4 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for i, r in enumerate(rows.itertuples(index=False)):
        stem = f"{r.subject_id}_{r.pose}_{r.view}"
        rgb = load_rgb(str(r.rgb_path))
        ei = normalize_ei(np.load(Path(ei_dir) / f"{stem}.npy"), stats)
        mask = np.load(Path(mask_dir) / f"{stem}.npy").astype(bool)
        pred = tiled_predict(model, _preprocess_for_model(rgb), device)[0].numpy()

        gt_disp = np.where(mask, ei, np.nan)
        pred_disp = np.where(mask, pred, np.nan)
        err_disp = np.where(mask, np.abs(ei - pred), np.nan)
        mae_ei = float(np.abs(ei - pred)[mask].mean() * rng)
        ssim = masked_ssim(pred, ei, mask)

        panels = [
            (rgb, f"{stem}\nRGB", None, {}),
            (gt_disp, "GT EI", "magma", dict(vmin=0, vmax=1)),
            (pred_disp, f"pred EI\nSSIM={ssim:.3f}  MAE={mae_ei:.2f}", "magma", dict(vmin=0, vmax=1)),
            (err_disp, "|GT - pred|", "inferno", dict(vmin=0, vmax=err_vmax)),
        ]
        for ax, (im, title, cmap, kw) in zip(axes[i], panels):
            h = ax.imshow(im, cmap=cmap, **kw)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if cmap in ("magma", "inferno"):
                plt.colorbar(h, ax=ax, fraction=0.046)

    plt.suptitle("Test predictions on display-permitted subjects "
                 "(RGB | ground-truth EI | prediction | error)", y=1.002)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Qualitative figure -> {path}")


def _preprocess_for_model(rgb):
    """ImageNet-standardise a raw RGB image into a model-ready (3,H,W) tensor."""
    from src.normalization import preprocess_rgb_imagenet
    arr = preprocess_rgb_imagenet(np.ascontiguousarray(rgb))
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def target_ei_stats(manifest, ei_dir, mask_dir, stats) -> pd.DataFrame:
    """Descriptive stats of the ground-truth EI over test-split SKIN pixels only.

    Context for the error metrics: MAE/MSE are only meaningful against the spread of
    the values the target actually takes on skin. Reported in both EI units and
    normalised [0,1] — the same two scales as the metric tables — so an error can be
    read against the signal on whichever scale (a `scale` column labels each row).

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest.
    ei_dir, mask_dir : Path or str
        Directories of destriped EI maps and binary masks.
    stats : dict
        EI normalisation statistics (used for the normalised-scale row).

    Returns
    -------
    pd.DataFrame
        Two rows (scale = "EI" and "norm"), columns mean/std/p1/p99/min/max.
    """
    test = manifest[manifest["split"] == "test"]
    vals = []
    for r in test.itertuples(index=False):
        stem = f"{r.subject_id}_{r.pose}_{r.view}"
        ei = np.load(Path(ei_dir) / f"{stem}.npy")
        m = np.load(Path(mask_dir) / f"{stem}.npy").astype(bool)
        vals.append(ei[m])
    v = np.concatenate(vals)

    def describe(arr, scale):
        return {
            "scale": scale,
            "mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4),
            "p1": round(float(np.percentile(arr, 1)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
            "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
        }

    return pd.DataFrame([describe(v, "EI"),
                         describe(normalize_ei(v, stats), "norm")])


def summarise_by_view_pose(df) -> pd.DataFrame:
    """Mean and std of each metric per (view, pose) group, plus per-view 'all'.

    Views and poses are taken from the data, not hard-coded. Metrics are aggregated
    ACROSS the images in each group (std = subject-to-subject variation). Both EI
    units and normalised [0,1] are included. Values rounded to 4 decimals.

    Parameters
    ----------
    df : pd.DataFrame
        Per-image metrics from evaluate().

    Returns
    -------
    pd.DataFrame
        One row per (view, pose) group and per-view "all", with mean/std of each
        metric in EI units and normalised.
    """
    def stats(sub, view, pose):
        return {
            "view": view, "pose": pose, "n": len(sub),
            "mae_ei_mean": round(sub.mae_ei.mean(), 4), "mae_ei_std": round(sub.mae_ei.std(), 4),
            "mse_ei_mean": round(sub.mse_ei.mean(), 4), "mse_ei_std": round(sub.mse_ei.std(), 4),
            "mae_norm_mean": round(sub.mae_norm.mean(), 4), "mae_norm_std": round(sub.mae_norm.std(), 4),
            "mse_norm_mean": round(sub.mse_norm.mean(), 4), "mse_norm_std": round(sub.mse_norm.std(), 4),
            "ssim_mean": round(sub.ssim.mean(), 4), "ssim_std": round(sub.ssim.std(), 4),
        }

    out = []
    for view in sorted(df.view.unique()):
        for pose in sorted(df.pose.unique()):
            sub = df[(df.view == view) & (df.pose == pose)]
            if len(sub):
                out.append(stats(sub, view, pose))
        out.append(stats(df[df.view == view], view, "all"))
    return pd.DataFrame(out)


def aggregate_metrics(df) -> pd.DataFrame:
    """Overall mean/std across all test subjects, in EI units and normalised.

    SSIM is unitless (no EI-unit form), so its EI columns are left blank.

    Parameters
    ----------
    df : pd.DataFrame
        Per-image metrics from evaluate().

    Returns
    -------
    pd.DataFrame
        One row per metric (MAE, MSE, SSIM) with mean/std in EI and normalised units.
    """
    rows = [
        {"metric": "MAE", "mean_ei": round(df.mae_ei.mean(), 4), "std_ei": round(df.mae_ei.std(), 4),
         "mean_norm": round(df.mae_norm.mean(), 4), "std_norm": round(df.mae_norm.std(), 4)},
        {"metric": "MSE", "mean_ei": round(df.mse_ei.mean(), 4), "std_ei": round(df.mse_ei.std(), 4),
         "mean_norm": round(df.mse_norm.mean(), 4), "std_norm": round(df.mse_norm.std(), 4)},
        {"metric": "SSIM", "mean_ei": np.nan, "std_ei": np.nan,
         "mean_norm": round(df.ssim.mean(), 4), "std_norm": round(df.ssim.std(), 4)},
    ]
    return pd.DataFrame(rows)


def print_tables(df, kind="ei") -> None:
    """Print one metrics table per view; rows = each pose present, plus 'all'.

    Parameters
    ----------
    df : pd.DataFrame
        Per-image metrics from evaluate().
    kind : {"ei", "norm"}
        "ei" prints denormalised EI units; "norm" prints normalised [0, 1].
        Values are shown to 4 decimals.
    """
    mae, mse = ("mae_ei", "mse_ei") if kind == "ei" else ("mae_norm", "mse_norm")
    unit = "EI units" if kind == "ei" else "normalised"
    poses = sorted(df.pose.unique())
    for view in sorted(df.view.unique()):
        print(f"\n{view.capitalize() + f' ({unit})':<18s}"
              f"{'MAE':>18s}{'MSE':>18s}{'SSIM':>18s}")
        for pose in poses + ["all"]:
            sub = df[df.view == view] if pose == "all" \
                else df[(df.view == view) & (df.pose == pose)]
            if not len(sub):
                continue
            print(f"{pose:<10s}"
                  f"{sub[mae].mean():.4f}±{sub[mae].std():<9.4f}"
                  f"{sub[mse].mean():.4f}±{sub[mse].std():<9.4f}"
                  f"{sub.ssim.mean():.4f}±{sub.ssim.std():.4f}")


def main() -> None:
    args = parse_args()
    processed = Path(config.LOCAL_PROCESSED_DIR)
    ei_dir = processed / "ei_maps_destriped"
    mask_dir = processed / "masks"
    stats = load_stats(str(processed / "norm_stats.json"))
    manifest = pd.read_csv(processed / "manifest.csv")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}  (run scripts/train.py first)")
        sys.exit(1)

    device = get_device()
    model = load_model(str(ckpt), device)

    # Context: the range of EI values the target spans on test skin (both scales).
    ctx = target_ei_stats(manifest, ei_dir, mask_dir, stats)
    print("\n--- Ground-truth EI on test skin (context for MAE/MSE; skin pixels only) ---")
    print(ctx.to_string(index=False))

    df = evaluate(model, manifest, ei_dir, mask_dir, stats, device)

    out_dir = Path(config.OUTPUT_DIR)
    ctx.to_csv(out_dir / "test_target_ei_stats.csv", index=False, float_format="%.4f")
    df.to_csv(out_dir / "test_metrics_per_subject.csv", index=False, float_format="%.4f")
    summarise_by_view_pose(df).to_csv(out_dir / "test_metrics_per_view&pose.csv",
                                      index=False, float_format="%.4f")
    aggregate_metrics(df).to_csv(out_dir / "test_metrics_aggregate.csv",
                                 index=False, float_format="%.4f")

    print("\n--- Per view & pose, EI units ---")
    print_tables(df, "ei")
    print("\n--- Per view & pose, normalised [0,1] ---")
    print_tables(df, "norm")

    print("\n--- Aggregate over all test subjects ---")
    print(f"{'metric':<8s}{'mean (EI)':>12s}{'std (EI)':>12s}"
          f"{'mean (norm)':>14s}{'std (norm)':>12s}")
    for _, r in aggregate_metrics(df).iterrows():
        ei_m = "-" if pd.isna(r.mean_ei) else f"{r.mean_ei:.4f}"
        ei_s = "-" if pd.isna(r.std_ei) else f"{r.std_ei:.4f}"
        print(f"{r.metric:<8s}{ei_m:>12s}{ei_s:>12s}{r.mean_norm:>14.4f}{r.std_norm:>12.4f}")

    print(f"\nTarget EI stats -> {out_dir / 'test_target_ei_stats.csv'}")
    print(f"Per-subject     -> {out_dir / 'test_metrics_per_subject.csv'}")
    print(f"Per view & pose -> {out_dir / 'test_metrics_per_view&pose.csv'}")
    print(f"Aggregate       -> {out_dir / 'test_metrics_aggregate.csv'}")

    qualitative_figure(model, manifest, ei_dir, mask_dir, stats, device,
                       out_dir / "qualitative_test.png")


if __name__ == "__main__":
    main()