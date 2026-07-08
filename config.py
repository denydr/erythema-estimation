"""Single source of truth for all paths, constants, and hyperparameters.

Every other module imports from here. No logic — only constants.

Dataset path configuration
--------------------------
Copy .env.example to .env and set DATA_ROOT to the Hyper-Skin(RGB, VIS)
directory on your machine (the folder that directly contains train/, test/,
and valid/ subfolders). The .env file is gitignored and never committed.

Dataset access: https://hyperskinsiteapp--hyperskinwebapp.asia-east1.hosted.app/dataAccess
"""

import os
from dotenv import load_dotenv

load_dotenv()

_data_root = os.environ.get("DATA_ROOT")
if not _data_root:
    raise EnvironmentError(
        "DATA_ROOT is not set.\n"
        "Copy .env.example to .env and set DATA_ROOT to your local "
        "Hyper-Skin(RGB, VIS) directory."
    )

DATA_ROOT = _data_root
LOCAL_PROCESSED_DIR = "./data/processed"

CUBE_SHAPE = (1024, 1024, 31)
WAVELENGTH_START_NM = 400
WAVELENGTH_STEP_NM = 10  # confirmed in notebook

# Subjects overridden to test split regardless of folder location
SPLIT_OVERRIDE = {"p027": "test", "p019": "test", "p012": "test"}

# Dawson (1980) EI formula wavelengths (nm), ordered as [p, q, r, s, t]
# p=510 (baseline), q=540, r=560, s=580 (Hb peaks), t=610 (baseline)
# Band indices (10 nm step from 400 nm):
#   510 → 11, 540 → 14, 560 → 16, 580 → 18, 610 → 21
DAWSON_WAVELENGTHS = [510, 540, 560, 580, 610]

# Reflectance values clipped to this floor before log10 to avoid log(0)
REFLECTANCE_FLOOR = 1e-6

# Destriping: window (in columns) of the median filter used to isolate the
# high-frequency per-column stripe offset in an EI map (notebook 01c).
DESTRIPE_MEDIAN_WINDOW = 100
