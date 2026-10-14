"""
Extract monthly CMIP6 series for every Atlantis box and layer.

Spatial method (preserved from the CMIP5 workflow, box_avg_VAR_CC.m):
    CMIP6 ocean cell centres that fall inside an Atlantis polygon are averaged
    with equal weights (unweighted mean, as the CMIP5 code did).
Vertical method (corrected relative to CMIP5):
    each Atlantis layer of each box is mapped by its REAL depth range
    (from nominal_dz / numlayers), using a thickness-weighted mean of the CMIP6
    levels overlapping it (from lev_bnds). The CMIP5 code mapped depth bands to
    fixed z indices, which is wrong for the shallow shelf boxes.
Not carried over from CMIP5:
    the 0.2 degree grid refinement and the monthly -> daily interpolation of
    the raw fields. They add no information; the daily interpolation is done
    later on the smooth climate signal.

Output (small file, can be copied off the HPC):
    BOX_SERIES_DIR/<cmip6_name>_<experiment>_box_layer_monthly.nc
    with dims (time, b, z) in Atlantis z order (z = 0 is the deepest layer).

Usage:
    python extract_cmip6_boxes.py --variable thetao --experiment ssp126
    python extract_cmip6_boxes.py --all
"""

import argparse

import netCDF4
import numpy as np
import xarray as xr
from matplotlib.path import Path as PolygonPath

import atlantis_geometry
import cmip6_files
import config


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Map CMIP6 grid cells onto Atlantis boxes ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def crop_indices(latitude, longitude, domain):
    """
    Index ranges (i, j) of the smallest CMIP6 grid rectangle covering the Atlantis domain.
    The ORCA1 grid seam is near 73 E, inside the EA domain, so the j range spans
    the whole grid; only the latitude rows are effectively cropped.
    """
    longitude_360 = np.mod(longitude, 360.0)
    in_domain = ((longitude_360 >= domain["lon_min"]) & (longitude_360 <= domain["lon_max"])
                 & (latitude >= domain["lat_min"]) & (latitude <= domain["lat_max"]))
    i_index, j_index = np.where(in_domain)
    return slice(i_index.min(), i_index.max() + 1), slice(j_index.min(), j_index.max() + 1)


def assign_cells_to_boxes(cell_lon, cell_lat, ocean_at_surface, bgm, logger):
    """
    For each box, the flat indices of the cropped CMIP6 cells used for its mean.

    Cells are tested in the BGM's own projection (LAEA), so polygon edges are the
    same straight lines Atlantis uses and no longitude convention issue arises.
    Boxes with no ocean cell centre inside use the nearest ocean cells instead.
    """
    to_bgm = atlantis_geometry.make_lonlat_to_bgm_transformer(bgm["projection"])
    cell_x, cell_y = to_bgm.transform(cell_lon.ravel(), cell_lat.ravel())
    cell_points = np.column_stack([cell_x, cell_y])
    ocean_flat = ocean_at_surface.ravel()

    cells_per_box = []
    used_fallback = []
    for box in bgm["boxes"]:
        polygon = PolygonPath(np.column_stack([box["x"], box["y"]]))
        inside = polygon.contains_points(cell_points) & ocean_flat
        box_cells = np.where(inside)[0]
        fallback = len(box_cells) == 0
        if fallback:
            # Nearest ocean cells to the polygon centre (vertex mean)
            distance = np.hypot(cell_x - box["x"].mean(), cell_y - box["y"].mean())
            distance[~ocean_flat] = np.inf
            box_cells = np.argsort(distance)[:config.N_NEAREST_FALLBACK_CELLS]
            logger.warning(f"  box {box['number']:2d}: no CMIP6 ocean cell inside polygon, "
                           f"using {len(box_cells)} nearest ocean cells "
                           f"(max distance {distance[box_cells].max() / 1000:.0f} km)")
        cells_per_box.append(box_cells)
        used_fallback.append(fallback)
    return cells_per_box, np.array(used_fallback)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Map CMIP6 levels onto Atlantis layers ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def level_overlap_weights(layer_top, layer_bottom, level_bounds):
    """
    Thickness (m) of every CMIP6 level that overlaps each Atlantis layer.
    Returns [n_boxes, n_atlantis_layers, n_cmip_levels]; unused layers get 0.
    """
    level_top = level_bounds[:, 0][None, None, :]
    level_bottom = level_bounds[:, 1][None, None, :]
    overlap = (np.minimum(layer_bottom[:, :, None], level_bottom)
               - np.maximum(layer_top[:, :, None], level_top))
    overlap = np.where(np.isfinite(overlap), np.clip(overlap, 0.0, None), 0.0)
    return overlap


def box_level_wet_mask(ocean_by_level, cells_per_box):
    """[n_boxes, n_levels]: True where at least one of the box's cells is ocean at that level."""
    return np.array([ocean_by_level[:, box_cells].any(axis=1) for box_cells in cells_per_box])


def plan_layer_fill(weights, box_wet_levels, numlayers, logger):
    """
    Decide where each Atlantis layer gets its value from.

    A layer with no wet CMIP6 level in its depth range (e.g. the CMIP6
    bathymetry is shallower than the Atlantis box) copies the nearest layer
    above in the SAME box. This replaces the CMIP5 behaviour of reusing the
    previous box's trend.
    Returns source_layer [n_boxes, n_layers]: own index, a copied index, or -1 (unused).
    """
    n_boxes, n_layers, _ = weights.shape
    source_layer = np.full((n_boxes, n_layers), -1, dtype=int)
    for box_number in range(n_boxes):
        surface_index = numlayers[box_number] - 1
        has_data = (weights[box_number] * box_wet_levels[box_number][None, :]).sum(axis=1) > 0
        if not has_data[surface_index]:
            raise ValueError(f"Box {box_number}: no CMIP6 data even in the surface layer")
        # Walk from the surface down, so a layer without data copies the one above
        for k in range(surface_index, -1, -1):
            if has_data[k]:
                source_layer[box_number, k] = k
            else:
                source_layer[box_number, k] = source_layer[box_number, k + 1]
                logger.warning(f"  box {box_number:2d} layer z={k}: no wet CMIP6 level in its "
                               f"depth range, copying layer z={source_layer[box_number, k]}")
    return source_layer


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Average one block of CMIP6 months ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def average_block(field_block, cells_per_box, weights, box_wet_levels, source_layer):
    """
    field_block : [n_months, n_levels, n_cells] cropped CMIP6 data, NaN on land
    returns     : [n_months, n_boxes, n_layers] Atlantis box/layer values
    """
    n_months = field_block.shape[0]
    n_boxes, n_layers = source_layer.shape
    box_values = np.full((n_months, n_boxes, n_layers), np.nan)

    for box_number, box_cells in enumerate(cells_per_box):
        # 1. Unweighted mean over the box's wet cells, level by level
        box_field = field_block[:, :, box_cells]
        is_wet = np.isfinite(box_field)
        n_wet_cells = is_wet.sum(axis=2)
        level_sum = np.where(is_wet, box_field, 0.0).sum(axis=2)
        level_profile = level_sum / np.maximum(n_wet_cells, 1)                # [months, levels]
        wet = box_wet_levels[box_number]
        level_profile = np.where(wet[None, :], level_profile, 0.0)

        # 2. Thickness-weighted mean of the CMIP6 levels overlapping each layer
        layer_weights = weights[box_number] * wet[None, :]                   # [layers, levels]
        weight_sum = layer_weights.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            layer_values = (level_profile @ layer_weights.T) / weight_sum[None, :]

        # 3. Layers without data copy the nearest layer above (same box)
        for k in range(n_layers):
            if source_layer[box_number, k] >= 0:
                box_values[:, box_number, k] = layer_values[:, source_layer[box_number, k]]
    return box_values


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Extract one variable for one experiment ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def extract_variable(variable_key, experiment, logger):
    variable_info = config.VARIABLES[variable_key]
    cmip6_name = variable_info["cmip6_name"]

    if experiment == config.HISTORICAL_EXPERIMENT:
        first_year, last_year = config.SIGNAL_FIRST_YEAR, config.HINDCAST_LAST_YEAR
    else:
        first_year, last_year = config.PROJECTION_FIRST_YEAR, config.SIGNAL_LAST_YEAR
    logger.info(f"=== {cmip6_name} ({variable_key}) {experiment}: years {first_year}-{last_year}")

    # Only the files that overlap the years needed
    all_files = cmip6_files.find_cmip6_files(cmip6_name, experiment)
    files = [f for f in all_files
             if cmip6_files.years_in_filename(f)[1] >= first_year
             and cmip6_files.years_in_filename(f)[0] <= last_year]
    for file_path in files:
        logger.info(f"  source file: {file_path}")

    # Atlantis geometry
    bgm = atlantis_geometry.add_lonlat_to_boxes(atlantis_geometry.read_bgm_boxes(config.BGM_FILE))
    layers = atlantis_geometry.read_atlantis_layers(config.INIT_FILE, bgm)
    domain = atlantis_geometry.domain_lonlat_bounds(bgm, config.REGION_BUFFER_DEGREES)

    # Grid, crop, ocean mask and mapping (static: taken from the first file)
    with netCDF4.Dataset(files[0]) as first_nc:
        latitude = np.asarray(first_nc["latitude"][:])
        longitude = np.asarray(first_nc["longitude"][:])
        level_bounds = np.asarray(first_nc["lev_bnds"][:])
        if first_nc[cmip6_name].dimensions != ("time", "lev", "i", "j"):
            raise ValueError(f"Unexpected dimensions {first_nc[cmip6_name].dimensions}")
        i_crop, j_crop = crop_indices(latitude, longitude, domain)
        first_field = np.ma.filled(first_nc[cmip6_name][0, :, i_crop, j_crop].astype(float), np.nan)

    cell_lat = latitude[i_crop, j_crop]
    cell_lon = longitude[i_crop, j_crop]
    ocean_by_level = np.isfinite(first_field).reshape(first_field.shape[0], -1)   # [levels, cells]
    logger.info(f"  cropped grid: {cell_lat.shape} cells, {ocean_by_level[0].sum()} ocean at surface")

    cells_per_box, used_fallback = assign_cells_to_boxes(
        cell_lon, cell_lat, ocean_by_level[0].reshape(cell_lat.shape), bgm, logger)
    n_cells_per_box = np.array([len(c) for c in cells_per_box])
    logger.info(f"  CMIP6 cells per box: {n_cells_per_box.tolist()}")

    weights = level_overlap_weights(layers["layer_top"], layers["layer_bottom"], level_bounds)
    box_wet_levels = box_level_wet_mask(ocean_by_level, cells_per_box)
    source_layer = plan_layer_fill(weights, box_wet_levels, layers["numlayers"], logger)
    logger.info(f"  Atlantis boxes processed: {len(bgm['boxes'])}, water layers per box: "
                f"{layers['numlayers'].tolist()}")

    # Read the data month by month block and average
    all_values = []
    all_time = []
    for file_path in files:
        with netCDF4.Dataset(file_path) as cmip_nc:
            data_variable = cmip_nc[cmip6_name]
            kelvin_temperature = variable_key == "temperature" and data_variable.units == "K"
            if data_variable.units != variable_info["cmip6_units"] and not kelvin_temperature:
                raise ValueError(f"{file_path.name}: units {data_variable.units}, "
                                 f"expected {variable_info['cmip6_units']}")
            if data_variable.standard_name != variable_info["cmip6_standard_name"]:
                raise ValueError(f"{file_path.name}: standard_name {data_variable.standard_name}, "
                                 f"expected {variable_info['cmip6_standard_name']}")
            if (not np.allclose(np.asarray(cmip_nc["latitude"][i_crop, j_crop]), cell_lat)
                    or not np.allclose(np.asarray(cmip_nc["lev_bnds"][:]), level_bounds)):
                raise ValueError(f"{file_path.name}: grid differs from {files[0].name}")
            dates = cmip6_files.decode_time(cmip_nc["time"])
            keep = np.where([first_year <= d.year <= last_year for d in dates])[0]
            for block_start in range(0, len(keep), 12):
                months = keep[block_start:block_start + 12]
                block = data_variable[months[0]:months[-1] + 1, :, i_crop, j_crop]
                block = np.ma.filled(block.astype(float), np.nan)
                if kelvin_temperature:
                    block = block - 273.15       # Kelvin -> degC (as in the CMIP5 code)
                block = block.reshape(block.shape[0], block.shape[1], -1)
                all_values.append(average_block(block, cells_per_box, weights,
                                                box_wet_levels, source_layer))
            all_time += [dates[m] for m in keep]
            logger.info(f"  {file_path.name}: {len(keep)} months used "
                        f"({dates[keep[0]]} -> {dates[keep[-1]]})")

    box_values = np.concatenate(all_values, axis=0)
    years = np.array([d.year for d in all_time])
    months = np.array([d.month for d in all_time])
    expected_months = (last_year - first_year + 1) * 12
    if len(years) != expected_months:
        logger.warning(f"  {len(years)} months extracted, {expected_months} expected for "
                       f"{first_year}-{last_year} (incomplete CMIP6 coverage)")
    if np.any(np.diff(years * 12 + months) != 1):
        raise ValueError("Extracted months are not consecutive")

    # Unused layers must stay NaN, used layers must be finite
    used_layer = source_layer >= 0
    if np.isnan(box_values[:, used_layer]).any():
        raise ValueError("NaN values in existing Atlantis layers after extraction")

    ## ~ Write the box/layer series ~ ##
    output = xr.Dataset(
        {
            cmip6_name: (("time", "b", "z"), box_values, {
                "units": variable_info["units_label"],
                "long_name": f"CMIP6 {cmip6_name} averaged per Atlantis box and layer",
                "comment": "z = 0 is the deepest water layer of each box; NaN = layer not present"}),
            "n_cells": ("b", n_cells_per_box, {
                "long_name": "number of CMIP6 cells averaged per box"}),
            "nearest_cell_fallback": ("b", used_fallback.astype(int), {
                "long_name": "1 = no CMIP6 cell inside polygon, nearest ocean cells used"}),
            "source_layer": (("b", "z"), source_layer, {
                "long_name": "z index the value was taken from (-1 = layer not present)"}),
            "layer_top": (("b", "z"), layers["layer_top"], {"units": "m"}),
            "layer_bottom": (("b", "z"), layers["layer_bottom"], {"units": "m"}),
        },
        coords={"time": np.arange(len(years)), "year": ("time", years), "month": ("time", months)},
        attrs={
            "source_model": config.CMIP6_MODEL,
            "member": config.CMIP6_MEMBER,
            "experiment": experiment,
            "source_files": ", ".join(f.name for f in files),
            "atlantis_geometry": config.BGM_FILE.name,
            "method": "unweighted mean of CMIP6 ocean cells inside each polygon; "
                      "thickness-weighted mean of CMIP6 levels within each Atlantis layer",
        },
    )
    config.BOX_SERIES_DIR.mkdir(parents=True, exist_ok=True)
    output_file = config.BOX_SERIES_DIR / f"{cmip6_name}_{experiment}_box_layer_monthly.nc"
    output.to_netcdf(output_file)
    logger.info(f"  written: {output_file}")


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Main ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def main():
    cmip6_to_key = {info["cmip6_name"]: key for key, info in config.VARIABLES.items()}
    experiments = [config.HISTORICAL_EXPERIMENT] + config.SCENARIOS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", choices=sorted(cmip6_to_key))
    parser.add_argument("--experiment", choices=experiments)
    parser.add_argument("--all", action="store_true", help="all variables and experiments")
    arguments = parser.parse_args()

    if arguments.all:
        jobs = [(key, experiment) for key in config.VARIABLES for experiment in experiments]
        log_name = "extract_all"
    elif arguments.variable and arguments.experiment:
        jobs = [(cmip6_to_key[arguments.variable], arguments.experiment)]
        log_name = f"extract_{arguments.variable}_{arguments.experiment}"
    else:
        parser.error("give --variable and --experiment, or --all")

    logger = config.setup_logging(log_name)
    for variable_key, experiment in jobs:
        extract_variable(variable_key, experiment, logger)


if __name__ == "__main__":
    main()
