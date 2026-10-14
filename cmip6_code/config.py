"""
Configuration for the CMIP6 -> East Antarctic Atlantis projection forcing workflow.

All user-editable paths and settings are in this file. Paths can be overridden
with environment variables, for example:

    export CMIP6_ROOT=/path/to/CMCC-CM2-SR5
    export ATLANTIS_DIR=/path/to/atlantis_model
    export OUTPUT_DIR=/path/to/outputs
"""

import logging
import os
import sys
from pathlib import Path


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Paths (edit here or set environment variables) ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
_REPO_ROOT = Path(__file__).resolve().parent.parent

CMIP6_ROOT = Path(os.environ.get(
    "CMIP6_ROOT", _REPO_ROOT / "cmip6_data" / "CMCC-CM2-SR5"))

ATLANTIS_DIR = Path(os.environ.get(
    "ATLANTIS_DIR", _REPO_ROOT / "atlantis_model"))

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR", _REPO_ROOT / "outputs"))

# Sub-folders of OUTPUT_DIR
BOX_SERIES_DIR = OUTPUT_DIR / "box_series"        # monthly CMIP6 box/layer series
SIGNAL_DIR = OUTPUT_DIR / "climate_signal"        # daily CMIP6 deltas per scenario
FORCING_DIR = OUTPUT_DIR / "atlantis_forcing"     # final Atlantis NetCDF + force.prm
VALIDATION_DIR = OUTPUT_DIR / "validation"        # plots and check reports


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Atlantis model files ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
BGM_FILE = ATLANTIS_DIR / "Antarctica_28.bgm"
INIT_FILE = ATLANTIS_DIR / "init_squid_230809.nc"   # source of nominal_dz / numlayers
FORCE_PRM_FILE = ATLANTIS_DIR / "EA_force.prm"      # template for the scenario force.prm


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ CMIP6 model, member and scenarios ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
CMIP6_MODEL = "CMCC-CM2-SR5"
CMIP6_MEMBER = "r1i1p1f1"
CMIP6_GRID_LABEL = "gn"
HISTORICAL_EXPERIMENT = "historical"

# ssp585 was downloaded but is deliberately NOT part of the final workflow
SCENARIOS = ["ssp126", "ssp245", "ssp370"]


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Variable mapping: Atlantis forcing <- CMIP6 ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
# Both variables use an ADDITIVE climate delta (see README, ASSUMPTION 5).
VARIABLES = {
    "temperature": {
        "cmip6_name": "thetao",
        "cmip6_table": "Omon",
        "cmip6_standard_name": "sea_water_potential_temperature",
        "cmip6_units": "degC",            # "K" would be converted to degC
        "atlantis_file": "EAtemp.nc",
        "atlantis_variable": "temperature",
        "units_label": "degC",
        "plausible_range": (-2.2, 6.0),   # East Antarctic water masses
    },
    "salinity": {
        "cmip6_name": "so",
        "cmip6_table": "Omon",
        "cmip6_standard_name": "sea_water_salinity",
        "cmip6_units": "0.001",           # practical salinity, same scale as Atlantis PSU
        "atlantis_file": "EAsalt.nc",
        "atlantis_variable": "salinity",
        "units_label": "PSU",
        "plausible_range": (33.0, 35.2),
    },
}


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Atlantis simulation timeline ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
# Atlantis forcing years are 365 days (the baseline year has 365 daily records),
# which matches the CMCC-CM2-SR5 "365_day" calendar.
DAYS_PER_YEAR = 365
SPINUP_YEARS = 10                   # labelled 1985-1994, repeated baseline only
HINDCAST_FIRST_YEAR = 1995          # repeated baseline only
HINDCAST_LAST_YEAR = 2014
PROJECTION_FIRST_YEAR = 2015        # baseline + CMIP6 climate signal
PROJECTION_LAST_YEAR = 2100

SIMULATION_FIRST_YEAR = HINDCAST_FIRST_YEAR - SPINUP_YEARS
N_SIMULATION_YEARS = PROJECTION_LAST_YEAR - SIMULATION_FIRST_YEAR + 1     # 116


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Climate signal settings ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
# CMIP6 reference climatology: the baseline year is assumed to represent this period
REFERENCE_FIRST_YEAR = 1995
REFERENCE_LAST_YEAR = 2014

# Years of CMIP6 data used to estimate the smooth signal (historical + scenario)
SIGNAL_FIRST_YEAR = REFERENCE_FIRST_YEAR
SIGNAL_LAST_YEAR = PROJECTION_LAST_YEAR

# Width of the LOWESS smoothing window (years). ~30 yr removes interannual and
# decadal noise but keeps the scenario-dependent curvature (e.g. ssp126 levelling off).
LOWESS_WINDOW_YEARS = 30

# The delta is forced to zero at this year so the hindcast -> projection
# transition is continuous (see README, ASSUMPTION 1).
ANCHOR_YEAR = PROJECTION_FIRST_YEAR

# Remove the within-year linear drift of the baseline in spin-up and hindcast too
# (see README, ASSUMPTION 2). False = keep the original baseline year there.
DETREND_HINDCAST = True


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Spatial processing settings ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
# Degrees added around the Atlantis domain when cropping the global CMIP6 grid
REGION_BUFFER_DEGREES = 3.0

# If a box contains no CMIP6 ocean cell centre, use this many nearest ocean cells
N_NEAREST_FALLBACK_CELLS = 3

# Representative boxes for the validation plots (shelf, slope, deep ocean)
VALIDATION_BOXES = [3, 9, 15]


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Logging ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def setup_logging(log_name, verbose=True):
    """Log to the console and to OUTPUT_DIR/logs/<log_name>.log."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cmip6_atlantis")
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"{log_name}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
