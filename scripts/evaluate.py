"""Evaluate the trained U-Net on the test split.

Usage:
    python scripts/evaluate.py       # uses outputs/best_model.pt

Predicts every test image by tiling and reports masked MAE/MSE/SSIM over skin on
the normalised [0, 1] scale, stratified by view and pose. Writes per-subject,
per-view/pose, and aggregate metric tables plus a qualitative figure to OUTPUT_DIR.
"""

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
    """Compute per-image masked MAE/MSE/SSIM on the test split (normalised [0, 1]).

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
        (mae, mse, ssim).
    """
    test = manifest[manifest["split"] == "test"]
    ds = ErythemaDataset(test, ei_dir, mask_dir, stats, mode="full")

    rows = []
    for i, r in enumerate(test.itertuples(index=False)):
        rgb, ei, mask = ds[i]
        pred = tiled_predict(model, rgb, device)
        rows.append({
            "subject_id": r.subject_id, "pose": r.pose, "view": r.view,
            "mae": masked_mae(pred, ei, mask),
            "mse": masked_mse(pred, ei, mask),
            "ssim": masked_ssim(pred[0].numpy(), ei[0].numpy(), mask[0].numpy().astype(bool)),
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
        mae = float(np.abs(ei - pred)[mask].mean())
        ssim = masked_ssim(pred, ei, mask)

        panels = [
            (rgb, f"{stem}\nRGB", None, {}),
            (gt_disp, "GT EI", "magma", dict(vmin=0, vmax=1)),
            (pred_disp, f"pred EI\nSSIM={ssim:.3f}  MAE={mae:.3f}", "magma", dict(vmin=0, vmax=1)),
            (err_disp, "|GT - pred|", "inferno", dict(vmin=0, vmax=err_vmax)),
        ]
        for ax, (im, title, cmap, kw) in zip(axes[i], panels):
            h = ax.imshow(im, cmap=cmap, **kw)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if cmap in ("magma", "inferno"):
                plt.colorbar(h, ax=ax, fraction=0.046)

    plt.suptitle("Test predictions on p012, p019, and p027"
                 "(RGB image | Ground-truth EI | Predicted EI | Error Map)", y=1.002)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Qualitative figure -> {path}")


def _preprocess_for_model(rgb):
    """ImageNet-standardise a raw RGB image into a model-ready (3,H,W) tensor."""
    from src.normalization import preprocess_rgb_imagenet
    arr = preprocess_rgb_imagenet(np.ascontiguousarray(rgb))
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def summarise_by_view_pose(df) -> pd.DataFrame:
    """Mean and std of each metric per (view, pose) group, plus per-view 'all'.

    Views and poses are taken from the data, not hard-coded. Metrics are aggregated
    ACROSS the images in each group (std = subject-to-subject variation). Values
    rounded to 4 decimals.

    Parameters
    ----------
    df : pd.DataFrame
        Per-image metrics from evaluate().

    Returns
    -------
    pd.DataFrame
        One row per (view, pose) group and per-view "all", with mean/std of each metric.
    """
    def stats(sub, view, pose):
        return {
            "view": view, "pose": pose, "n": len(sub),
            "mae_mean": round(sub.mae.mean(), 4), "mae_std": round(sub.mae.std(), 4),
            "mse_mean": round(sub.mse.mean(), 4), "mse_std": round(sub.mse.std(), 4),
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


def summarise_by_subject(df) -> pd.DataFrame:
    """Mean and std of each metric per subject, aggregated over the subject's images.

    Each subject contributes all its images (every view and pose); the std captures
    within-subject variation across them. Values rounded to 4 decimals, rows sorted
    by MAE (lowest error first).

    Parameters
    ----------
    df : pd.DataFrame
        Per-image metrics from evaluate().

    Returns
    -------
    pd.DataFrame
        One row per subject with n and the mean/std of mae, mse, ssim.
    """
    def stats(sub, sid):
        return {
            "subject_id": sid, "n": len(sub),
            "mae_mean": round(sub.mae.mean(), 4), "mae_std": round(sub.mae.std(), 4),
            "mse_mean": round(sub.mse.mean(), 4), "mse_std": round(sub.mse.std(), 4),
            "ssim_mean": round(sub.ssim.mean(), 4), "ssim_std": round(sub.ssim.std(), 4),
        }

    out = [stats(df[df.subject_id == sid], sid) for sid in sorted(df.subject_id.unique())]
    return pd.DataFrame(out).sort_values("mae_mean").reset_index(drop=True)


def aggregate_metrics(df) -> pd.DataFrame:
    """Overall mean/std across all test subjects.

    Parameters
    ----------
    df : pd.DataFrame
        Per-image metrics from evaluate().

    Returns
    -------
    pd.DataFrame
        One row per metric (MAE, MSE, SSIM) with mean/std.
    """
    rows = [
        {"metric": "MAE", "mean": round(df.mae.mean(), 4), "std": round(df.mae.std(), 4)},
        {"metric": "MSE", "mean": round(df.mse.mean(), 4), "std": round(df.mse.std(), 4)},
        {"metric": "SSIM", "mean": round(df.ssim.mean(), 4), "std": round(df.ssim.std(), 4)},
    ]
    return pd.DataFrame(rows)


def print_tables(df) -> None:
    """Print one metrics table per view; rows = each pose present, plus 'all'.

    Parameters
    ----------
    df : pd.DataFrame
        Per-image metrics from evaluate(). Values shown to 4 decimals.
    """
    def cell(mean, std):
        return f"{mean:.4f}±{std:.4f}"

    poses = sorted(df.pose.unique())
    for view in sorted(df.view.unique()):
        print(f"\n{view.capitalize():<10s}{'MAE':>16s}{'MSE':>16s}{'SSIM':>16s}")
        for pose in poses + ["all"]:
            sub = df[df.view == view] if pose == "all" \
                else df[(df.view == view) & (df.pose == pose)]
            if not len(sub):
                continue
            print(f"{pose:<10s}"
                  f"{cell(sub.mae.mean(), sub.mae.std()):>16s}"
                  f"{cell(sub.mse.mean(), sub.mse.std()):>16s}"
                  f"{cell(sub.ssim.mean(), sub.ssim.std()):>16s}")


def main() -> None:
    processed = Path(config.LOCAL_PROCESSED_DIR)
    ei_dir = processed / "ei_maps_destriped"
    mask_dir = processed / "masks"
    stats = load_stats(str(processed / "norm_stats.json"))
    manifest = pd.read_csv(processed / "manifest.csv")

    ckpt = Path(config.OUTPUT_DIR) / "best_model.pt"
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}  (run scripts/train.py first)")
        sys.exit(1)

    device = get_device()
    model = load_model(str(ckpt), device)

    df = evaluate(model, manifest, ei_dir, mask_dir, stats, device)

    out_dir = Path(config.OUTPUT_DIR)
    summarise_by_subject(df).to_csv(out_dir / "test_metrics_per_subject.csv",
                                    index=False, float_format="%.4f")
    summarise_by_view_pose(df).to_csv(out_dir / "test_metrics_per_view&pose.csv",
                                      index=False, float_format="%.4f")
    aggregate_metrics(df).to_csv(out_dir / "test_metrics_aggregate.csv",
                                 index=False, float_format="%.4f")

    print("\n--- Per view & pose (normalised [0, 1]) ---")
    print_tables(df)

    print("\n--- Aggregate over all test subjects ---")
    print(f"{'metric':<8s}{'mean':>12s}{'std':>12s}")
    for _, r in aggregate_metrics(df).iterrows():
        print(f"{r.metric:<8s}{r['mean']:>12.4f}{r['std']:>12.4f}")

    print(f"\nPer-subject     -> {out_dir / 'test_metrics_per_subject.csv'}")
    print(f"Per view & pose -> {out_dir / 'test_metrics_per_view&pose.csv'}")
    print(f"Aggregate       -> {out_dir / 'test_metrics_aggregate.csv'}")

    qualitative_figure(model, manifest, ei_dir, mask_dir, stats, device,
                       out_dir / "qualitative_test.png")


if __name__ == "__main__":
    main()