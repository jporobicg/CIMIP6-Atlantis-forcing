"""
Inspect CMIP6 NetCDF files before they are used for the Atlantis forcing.

Reports, per file: variable name, standard name, units, dimensions, lon/lat
convention and orientation, depth levels, calendar, time coverage, fill values,
member / parent metadata, and data range inside the Atlantis domain.
Then checks consistency between files: same grid and levels, continuous time,
and that each SSP branches from the historical run of the same member.

Usage:
    python inspect_cmip6.py                          # all files under CMIP6_ROOT
    python inspect_cmip6.py --files file1.nc file2.nc
"""

import argparse
from pathlib import Path

import netCDF4
import numpy as np

import atlantis_geometry
import cmip6_files
import config


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Inspect a single file ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def inspect_file(file_path, domain, logger):
    """Log the metadata of one CMIP6 file and return a summary dict."""
    with netCDF4.Dataset(file_path) as cmip_nc:
        variable_name = cmip_nc.getncattr("variable_id")
        data_variable = cmip_nc[variable_name]
        latitude = np.asarray(cmip_nc["latitude"][:])
        longitude = np.asarray(cmip_nc["longitude"][:])
        depth = np.asarray(cmip_nc["lev"][:])
        depth_bounds = np.asarray(cmip_nc["lev_bnds"][:])
        dates = cmip6_files.decode_time(cmip_nc["time"])
        time_values = np.asarray(cmip_nc["time"][:])

        summary = {
            "file": file_path,
            "variable": variable_name,
            "experiment": cmip_nc.getncattr("experiment_id"),
            "member": cmip_nc.getncattr("variant_label"),
            "source_id": cmip_nc.getncattr("source_id"),
            "grid_label": cmip_nc.getncattr("grid_label"),
            "parent_experiment": cmip_nc.getncattr("parent_experiment_id"),
            "parent_member": cmip_nc.getncattr("parent_variant_label"),
            "branch_time_in_parent": cmip_nc.getncattr("branch_time_in_parent"),
            "parent_time_units": cmip_nc.getncattr("parent_time_units"),
            "time_units": cmip_nc["time"].units,
            "calendar": cmip_nc["time"].calendar,
            "first_date": dates[0],
            "last_date": dates[-1],
            "first_time": time_values[0],
            "last_time_bound": float(cmip_nc["time_bnds"][-1, 1]),
            "n_months": len(dates),
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
        }

        logger.info("-" * 78)
        logger.info(f"File: {file_path.name}")
        logger.info(f"  variable {variable_name}: standard_name = {data_variable.standard_name}, "
                    f"units = {data_variable.units}, dims = {data_variable.dimensions}, "
                    f"shape = {data_variable.shape}")
        logger.info(f"  _FillValue = {getattr(data_variable, '_FillValue', None)}, "
                    f"missing_value = {getattr(data_variable, 'missing_value', None)}")
        logger.info(f"  experiment = {summary['experiment']}, member = {summary['member']}, "
                    f"source = {summary['source_id']}, grid = {summary['grid_label']}")
        logger.info(f"  parent = {summary['parent_experiment']} {summary['parent_member']}, "
                    f"branch_time_in_parent = {summary['branch_time_in_parent']} "
                    f"({summary['parent_time_units']})")
        logger.info(f"  time: {summary['time_units']}, calendar = {summary['calendar']}, "
                    f"{summary['n_months']} records, {dates[0]} -> {dates[-1]}")
        logger.info(f"  longitude range {np.nanmin(longitude):.2f} to {np.nanmax(longitude):.2f} "
                    f"(convention {'0-360' if np.nanmin(longitude) >= 0 else '-180-180'})")
        logger.info(f"  latitude range {np.nanmin(latitude):.2f} to {np.nanmax(latitude):.2f}; "
                    f"latitude increases along first horizontal dim: "
                    f"{bool(np.nanmean(np.diff(latitude, axis=0)) > 0)}")
        logger.info(f"  depth: {len(depth)} levels, centres {depth[0]:.2f} - {depth[-1]:.1f} m, "
                    f"bounds {depth_bounds[0, 0]:.1f} - {depth_bounds[-1, 1]:.1f} m, "
                    f"positive = {cmip_nc['lev'].positive}")

        # Data inside the Atlantis domain at the first time step
        longitude_360 = np.mod(longitude, 360.0)
        in_domain = ((longitude_360 >= domain["lon_min"]) & (longitude_360 <= domain["lon_max"])
                     & (latitude >= domain["lat_min"]) & (latitude <= domain["lat_max"]))
        first_field = np.ma.masked_invalid(data_variable[0])     # [lev, i, j], fill -> masked
        surface_in_domain = first_field[0][in_domain]
        bottom_layers = first_field[:, in_domain]
        logger.info(f"  Atlantis domain: {in_domain.sum()} grid cells, "
                    f"{surface_in_domain.count()} ocean at surface; "
                    f"first-month values {bottom_layers.min():.3f} to {bottom_layers.max():.3f}")
    return summary


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Consistency checks between files ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def check_consistency(summaries, logger):
    """Check grids, members and time continuity across all inspected files."""
    problems = []
    reference = summaries[0]

    for summary in summaries:
        name = summary["file"].name
        if summary["member"] != config.CMIP6_MEMBER:
            problems.append(f"{name}: member {summary['member']} != {config.CMIP6_MEMBER}")
        if summary["source_id"] != config.CMIP6_MODEL:
            problems.append(f"{name}: model {summary['source_id']} != {config.CMIP6_MODEL}")
        if summary["calendar"] != reference["calendar"]:
            problems.append(f"{name}: calendar {summary['calendar']} differs")
        if summary["time_units"] != reference["time_units"]:
            problems.append(f"{name}: time units {summary['time_units']} differ")
        same_grid = (summary["latitude"].shape == reference["latitude"].shape
                     and np.allclose(summary["latitude"], reference["latitude"], equal_nan=True)
                     and np.allclose(summary["longitude"], reference["longitude"], equal_nan=True)
                     and np.allclose(summary["depth"], reference["depth"]))
        if not same_grid:
            problems.append(f"{name}: horizontal grid or depth levels differ from {reference['file'].name}")
        if summary["experiment"] in config.SCENARIOS + ["ssp585"]:
            if (summary["parent_experiment"] != config.HISTORICAL_EXPERIMENT
                    or summary["parent_member"] != config.CMIP6_MEMBER):
                problems.append(f"{name}: parent is {summary['parent_experiment']} "
                                f"{summary['parent_member']}, expected historical {config.CMIP6_MEMBER}")

    # Time continuity within each variable/experiment, and historical -> SSP branch
    groups = {}
    for summary in summaries:
        groups.setdefault((summary["variable"], summary["experiment"]), []).append(summary)
    for (variable, experiment), group in sorted(groups.items()):
        group = sorted(group, key=lambda s: s["first_time"])
        logger.info(f"{variable} {experiment}: {len(group)} file(s), "
                    f"{group[0]['first_date']} -> {group[-1]['last_date']}")
        for previous, following in zip(group[:-1], group[1:]):
            gap_days = following["first_time"] - previous["last_time_bound"]
            if not 0 <= gap_days < 31:
                problems.append(f"{variable} {experiment}: gap of {gap_days} days between "
                                f"{previous['file'].name} and {following['file'].name}")
        historical = groups.get((variable, config.HISTORICAL_EXPERIMENT))
        if experiment != config.HISTORICAL_EXPERIMENT and historical is not None:
            historical_end = max(s["last_time_bound"] for s in historical)
            if not np.isclose(group[0]["branch_time_in_parent"], historical_end):
                problems.append(f"{variable} {experiment}: branch_time_in_parent "
                                f"{group[0]['branch_time_in_parent']} != historical end {historical_end}")

    logger.info("=" * 78)
    if problems:
        for problem in problems:
            logger.warning(f"PROBLEM: {problem}")
    else:
        logger.info("All consistency checks passed")
    return problems


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Main ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", type=Path,
                        help="Inspect only these files (default: all files under CMIP6_ROOT)")
    arguments = parser.parse_args()

    logger = config.setup_logging("inspect_cmip6")

    bgm = atlantis_geometry.add_lonlat_to_boxes(atlantis_geometry.read_bgm_boxes(config.BGM_FILE))
    domain = atlantis_geometry.domain_lonlat_bounds(bgm, config.REGION_BUFFER_DEGREES)
    logger.info(f"Atlantis domain (with buffer): lon {domain['lon_min']:.1f}-{domain['lon_max']:.1f} E, "
                f"lat {domain['lat_min']:.1f} to {domain['lat_max']:.1f}")

    if arguments.files:
        file_list = arguments.files
    else:
        logger.info(f"CMIP6_ROOT = {config.CMIP6_ROOT}")
        file_list = []
        for variable_info in config.VARIABLES.values():
            for experiment in [config.HISTORICAL_EXPERIMENT] + config.SCENARIOS:
                file_list += cmip6_files.find_cmip6_files(variable_info["cmip6_name"], experiment)

    summaries = [inspect_file(Path(file_path), domain, logger) for file_path in file_list]
    check_consistency(summaries, logger)


if __name__ == "__main__":
    main()
