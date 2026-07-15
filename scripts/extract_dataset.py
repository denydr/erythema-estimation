"""Download and extract the Hyper-Skin dataset from Google Drive.

Usage:
    python scripts/extract_dataset.py [--output-dir /path/to/destination]

Fill in HYPERSKIN_PASS and RCLONE_REMOTE in your .env file (see .env.example).
See README.md Setup section for the full rclone configuration steps.

Requires:
    pip install -r requirements.txt
    brew install p7zip   (macOS)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ARCHIVE_NAME = "Hyper-Skin.7z"
EXPECTED_SUBDIRS = [
    "Hyper-Skin(RGB, VIS)/train",
    "Hyper-Skin(RGB, VIS)/test",
    "Hyper-Skin(RGB, VIS)/valid",
]


def check_dependency(cmd: str, install_hint: str) -> None:
    """Exit with an error if a required CLI command is not on PATH.

    Parameters
    ----------
    cmd : str
        Command to look for (e.g. "rclone", "7z").
    install_hint : str
        Message shown to the user if the command is missing.
    """
    if shutil.which(cmd) is None:
        print(f"ERROR: '{cmd}' not found. {install_hint}")
        sys.exit(1)


def check_rclone_configured(remote: str) -> bool:
    """Check whether a named rclone remote exists.

    Parameters
    ----------
    remote : str
        The rclone remote name to look for.

    Returns
    -------
    bool
        True if the remote is configured, otherwise False.
    """
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    return f"{remote}:" in result.stdout


def download_archive(dest_path: Path, rclone_remote: str) -> None:
    """Download the dataset archive from Google Drive with rclone.

    Skips the download if the archive already exists; exits with an error if rclone
    is missing, the remote is not configured, or the download fails.

    Parameters
    ----------
    dest_path : Path
        Target path for the downloaded Hyper-Skin.7z.
    rclone_remote : str
        Name of the configured rclone Google Drive remote.
    """
    if dest_path.exists():
        print(f"Archive already present at {dest_path} — skipping download.")
        return

    check_dependency("rclone", "Install on macOS with:  brew install rclone")

    if not check_rclone_configured(rclone_remote):
        print(f"ERROR: rclone has no remote named '{rclone_remote}'.")
        print("Run the one-time setup before re-running this script:")
        print("  rclone config")
        print("  → n  (new remote)")
        print(f"  → name: {rclone_remote}")
        print("  → storage: drive")
        print("  → leave client_id and client_secret blank")
        print("  → follow the browser login prompt")
        sys.exit(1)

    print(f"Downloading Hyper-Skin.7z from Google Drive → {dest_path} ...")
    result = subprocess.run([
        "rclone", "copy",
        "--drive-shared-with-me",
        "--progress",
        f"{rclone_remote}:Hyper-Skin.7z",
        str(dest_path.parent),
    ])

    if result.returncode != 0 or not dest_path.exists():
        print("ERROR: Download failed.")
        print("Ensure 'Hyper-Skin.7z' appears in your Google Drive 'Shared with me' section.")
        sys.exit(1)

    print("Download complete.")


def extract_archive(archive_path: Path, password: str, output_dir: Path) -> None:
    """Extract the password-protected 7z archive (RGB/VIS split only).

    Parameters
    ----------
    archive_path : Path
        Path to the downloaded Hyper-Skin.7z.
    password : str
        Archive password (from HYPERSKIN_PASS).
    output_dir : Path
        Directory to extract into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path.name} → {output_dir} ...")
    print("Extracting Hyper-Skin(RGB, VIS) only. Files marked with '.' are skipped (MSI/NIR), '-' are extracted.")

    result = subprocess.run(
        ["7z", "x", f"-p{password}", str(archive_path), f"-o{output_dir}", "-y",
         "-xr!Hyper-Skin(MSI, NIR)"],
    )

    if result.returncode != 0:
        print("ERROR: Extraction failed — check the password and archive integrity.")
        sys.exit(1)

    print("Extraction complete.")


def verify_structure(output_dir: Path) -> None:
    """Warn if the expected RGB/VIS split directories are missing after extraction.

    Parameters
    ----------
    output_dir : Path
        Directory the archive was extracted into.
    """
    print("Verifying dataset structure ...")
    missing = [d for d in EXPECTED_SUBDIRS if not (output_dir / d).exists()]

    if missing:
        print("WARNING: The following expected directories were not found:")
        for d in missing:
            print(f"  {output_dir / d}")
        print("Check that the archive extracted correctly and the folder names match.")
    else:
        print("Structure OK — train / test / valid splits found for RGB and VIS.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to download and extract the dataset into (default: current directory).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()

    password = os.environ.get("HYPERSKIN_PASS")
    rclone_remote = os.environ.get("RCLONE_REMOTE")

    if not password:
        print("ERROR: HYPERSKIN_PASS is not set.")
        print('Set it in .env:  HYPERSKIN_PASS=your_password_here')
        sys.exit(1)

    if not rclone_remote:
        print("ERROR: RCLONE_REMOTE is not set.")
        print('Set it in .env to the name you gave your Google Drive remote during rclone config.')
        print('Example:  RCLONE_REMOTE=gdrive')
        sys.exit(1)

    check_dependency("7z", "Install on macOS with:  brew install p7zip")

    archive_path = output_dir / ARCHIVE_NAME
    download_archive(archive_path, rclone_remote)
    extract_archive(archive_path, password, output_dir)
    verify_structure(output_dir)

    vis_root = output_dir / "Hyper-Skin(RGB, VIS)"
    print(f"\nDataset ready. Add the following line to your .env file:")
    print(f'  DATA_ROOT="{vis_root}"')


if __name__ == "__main__":
    main()