"""Download and extract the Hyper-Skin dataset from Google Drive.

Usage:
    python scripts/extract_dataset.py [--output-dir /path/to/destination]

Fill in HYPERSKIN_PASS and HYPERSKIN_GDRIVE_URL in your .env file (see .env.example).
Both values are provided in the dataset access email from the Hyper-Skin authors.

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
    if shutil.which(cmd) is None:
        print(f"ERROR: '{cmd}' not found. {install_hint}")
        sys.exit(1)


def download_archive(gdrive_url: str, dest_path: Path) -> None:
    """Download the archive from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown is not installed. Run:  pip install gdown")
        sys.exit(1)

    if dest_path.exists():
        print(f"Archive already present at {dest_path} — skipping download.")
        return

    print(f"Downloading from Google Drive → {dest_path} ...")
    try:
        gdown.download(gdrive_url, str(dest_path), quiet=False)
    except Exception as e:
        print(f"ERROR: Download failed — {e}")
        print("If Drive quota is exceeded, download Hyper-Skin.7z manually from the browser,")
        print(f"save it to {dest_path}, then re-run this script.")
        sys.exit(1)

    if not dest_path.exists():
        print("ERROR: Download finished but archive not found at the expected path.")
        sys.exit(1)

    size_mb = dest_path.stat().st_size / (1024 ** 2)
    if size_mb < 1:
        dest_path.unlink()
        print(f"ERROR: Downloaded file is only {size_mb:.2f} MB — not a valid archive.")
        print("The URL may be an Outlook SafeLinks wrapper. Copy the actual Google Drive")
        print("URL (starts with https://drive.google.com/) into HYPERSKIN_GDRIVE_URL in .env.")
        sys.exit(1)

    print("Download complete.")


def extract_archive(archive_path: Path, password: str, output_dir: Path) -> None:
    """Extract the password-protected 7z archive using the 7z CLI."""
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
    """Confirm expected RGB/VIS split directories exist after extraction."""
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
    gdrive_url = os.environ.get("HYPERSKIN_GDRIVE_URL")

    if not password:
        print("ERROR: HYPERSKIN_PASS is not set.")
        print('Set it before running:  export HYPERSKIN_PASS="your_password_here"')
        sys.exit(1)

    if not gdrive_url:
        print("ERROR: HYPERSKIN_GDRIVE_URL is not set.")
        print('Set it before running:  export HYPERSKIN_GDRIVE_URL="https://drive.google.com/..."')
        sys.exit(1)

    check_dependency("7z", "Install on macOS with:  brew install p7zip")

    archive_path = output_dir / ARCHIVE_NAME
    download_archive(gdrive_url, archive_path)
    extract_archive(archive_path, password, output_dir)
    verify_structure(output_dir)

    vis_root = output_dir / "Hyper-Skin(RGB, VIS)"
    print(f"\nDataset ready. Add the following line to your .env file:")
    print(f'  DATA_ROOT="{vis_root}"')


if __name__ == "__main__":
    main()