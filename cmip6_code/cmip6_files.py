"""
Locate CMIP6 files and decode their time axis.

Expected layout (as produced by download_all.sh):
    CMIP6_ROOT/<experiment>/<variable>/<variable>_<table>_<model>_<experiment>_<member>_<grid>_YYYYMM-YYYYMM.nc
"""

import re

import cftime
import numpy as np

import config


def find_cmip6_files(cmip6_name, experiment):
    """Sorted list of CMIP6 files for one variable and experiment (fails if none)."""
    variable_info = [info for info in config.VARIABLES.values()
                     if info["cmip6_name"] == cmip6_name][0]
    pattern = (f"{cmip6_name}_{variable_info['cmip6_table']}_{config.CMIP6_MODEL}_"
               f"{experiment}_{config.CMIP6_MEMBER}_{config.CMIP6_GRID_LABEL}_*.nc")
    folder = config.CMIP6_ROOT / experiment / cmip6_name
    files = sorted(folder.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return files


def years_in_filename(file_path):
    """(first_year, last_year) from the YYYYMM-YYYYMM part of a CMIP6 file name."""
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", file_path.name)
    if match is None:
        raise ValueError(f"Cannot read the period from file name {file_path.name}")
    return int(match.group(1)), int(match.group(2))


def decode_time(time_variable):
    """Decode a netCDF4 time variable into cftime dates (keeps the model calendar)."""
    calendar = getattr(time_variable, "calendar", "standard")
    return cftime.num2date(np.asarray(time_variable[:]), time_variable.units, calendar=calendar)
