"""
Build the East Antarctic Atlantis temperature and salinity forcing for one scenario.

Timeline (365-day years, daily records):
    spin-up    10 years (1985-1994)  baseline year repeated
    hindcast   20 years (1995-2014)  baseline year repeated
    projection 86 years (2015-2100)  detrended baseline year + CMIP6 delta

    F_projection(day) = F_detrended_baseline(day of year) + delta_CMIP6(day)

The delta is zero throughout 2015 (anchored), so there is no jump between the
hindcast and the projection.

Detrending: the baseline year (EAtemp.nc / EAsalt.nc) does not end where it
starts, so repeating it gives a small step every New Year (a drift). That drift
is removed with a zero-mean ramp (see detrend_baseline_year), keeping the annual
mean and seasonal cycle, so the only long-term change comes from CMIP6.

Outputs (FORCING_DIR):
    EAtemp_<scenario>.nc, EAsalt_<scenario>.nc   same structure as the baseline
    EA_force_<scenario>.prm                      force file using them, rewind off

Usage:
    python build_atlantis_forcing.py --scenario ssp126
"""

import argparse
import datetime
import re

import netCDF4
import numpy as np
import xarray as xr

import atlantis_geometry
import config


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Read and check the Atlantis baseline year ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def read_baseline(variable_key, layers):
    """Baseline forcing year and its NetCDF metadata, with structural checks."""
    variable_info = config.VARIABLES[variable_key]
    baseline_file = config.ATLANTIS_DIR / variable_info["atlantis_file"]
    variable_name = variable_info["atlantis_variable"]

    with netCDF4.Dataset(baseline_file) as baseline_nc:
        baseline_nc.set_auto_mask(False)
        baseline = {
            "file": baseline_file,
            "file_format": baseline_nc.file_format,
            "values": np.array(baseline_nc[variable_name][:], dtype=float),     # [t, b, z]
            "time": np.array(baseline_nc["t"][:], dtype=float),
            "time_units": baseline_nc["t"].units,
            "time_dt": float(baseline_nc["t"].dt),
            "fill_value": baseline_nc[variable_name]._FillValue,
            "dimensions": baseline_nc[variable_name].dimensions,
            "global_attributes": {name: baseline_nc.getncattr(name) for name in baseline_nc.ncattrs()},
        }

    n_days, n_boxes, n_z = baseline["values"].shape
    if baseline["dimensions"] != ("t", "b", "z"):
        raise ValueError(f"{baseline_file.name}: dimensions {baseline['dimensions']}, expected (t, b, z)")
    if n_days != config.DAYS_PER_YEAR or baseline["time_dt"] != 86400:
        raise ValueError(f"{baseline_file.name}: expected {config.DAYS_PER_YEAR} daily records")
    if n_z != layers["n_water_layers"] + 1 or n_boxes != len(layers["numlayers"]):
        raise ValueError(f"{baseline_file.name}: shape {baseline['values'].shape} does not match the "
                         f"init file ({len(layers['numlayers'])} boxes, {layers['n_water_layers']} + 1 layers)")

    # Layers that exist must have data; layers that do not exist hold 0 (checked, not assumed).
    # An exact 0 in a temperature layer can be a real sample, because the file uses
    # 0 as _FillValue. Those samples are nudged later so Atlantis does not drop them.
    exists = layer_exists_mask(layers)
    unused_values = baseline["values"][:, ~exists]
    if np.any(unused_values != 0):
        raise ValueError(f"{baseline_file.name}: non-zero values in layers that do not exist")
    existing_values = baseline["values"][:, exists]
    if np.isnan(existing_values).any():
        raise ValueError(f"{baseline_file.name}: NaN in existing layers")
    if baseline["fill_value"] != 0 and np.any(existing_values == 0):
        raise ValueError(f"{baseline_file.name}: zeros in existing layers")
    sediment = layers["sediment_index"]
    if not np.array_equal(baseline["values"][:, :, sediment], baseline["values"][:, :, 0]):
        raise ValueError(f"{baseline_file.name}: sediment layer is not a copy of layer 0")
    return baseline


def layer_exists_mask(layers):
    """[b, z] True for the water layers that exist in each box and for the sediment layer."""
    n_boxes = len(layers["numlayers"])
    z_index = np.arange(layers["n_water_layers"] + 1)[None, :]
    exists = np.broadcast_to(z_index < layers["numlayers"][:, None],
                             (n_boxes, layers["n_water_layers"] + 1)).copy()
    exists[:, layers["sediment_index"]] = True
    return exists


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Detrend the baseline year ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def detrend_baseline_year(values, exists, edge_days=7):
    """
    Remove the drift of the baseline year, keeping its annual mean and seasonal cycle.

    A linear regression over a single year is NOT used: it would mistake part
    of the seasonal cycle (e.g. warm January, cold spring) for a trend and
    create a jump at every New Year. Instead, the drift is the mismatch at the
    point where the repeated year wraps from 31 Dec back to 1 Jan:

        expected 1 Jan value = end-of-year mean + local daily slope * edge_days
        drift over the year  = expected 1 Jan value - actual start-of-year mean

    A zero-mean linear ramp of that size is removed, so the repeated year joins
    smoothly and its annual mean is unchanged.

    values [days, b, z]; only existing layers are changed.
    Returns the detrended year and the drift over the year [b, z].
    """
    n_days = values.shape[0]
    start_mean = values[:edge_days].mean(axis=0)                       # centred on day 3
    end_mean = values[-edge_days:].mean(axis=0)                        # centred on day 361
    start_slope = (values[edge_days:2 * edge_days].mean(axis=0) - start_mean) / edge_days
    end_slope = (end_mean - values[-2 * edge_days:-edge_days].mean(axis=0)) / edge_days
    local_slope = (start_slope + end_slope) / 2.0

    expected_start_mean = end_mean + local_slope * edge_days           # continuation past 31 Dec
    drift_over_year = expected_start_mean - start_mean

    centred_day = (np.arange(n_days) - (n_days - 1) / 2.0) / n_days     # zero mean, spans ~1 year
    ramp = centred_day[:, None, None] * drift_over_year[None, :, :]
    return np.where(exists[None, :, :], values - ramp, values), drift_over_year


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Build the full 116-year series ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def build_series(baseline, layers, delta_daily, logger):
    """Returns the full daily series [n_days, b, z] and the detrended baseline year."""
    values = baseline["values"]
    sediment = layers["sediment_index"]
    exists = layer_exists_mask(layers)
    water_exists = exists[:, :sediment]

    detrended, drift_over_year = detrend_baseline_year(values, exists)
    logger.info(f"  baseline drift over the year removed: mean |drift| = "
                f"{np.abs(drift_over_year[exists]).mean():.4f}, max |drift| = "
                f"{np.abs(drift_over_year[exists]).max():.4f}")

    n_pre_projection_years = config.PROJECTION_FIRST_YEAR - config.SIMULATION_FIRST_YEAR
    n_projection_years = config.PROJECTION_LAST_YEAR - config.PROJECTION_FIRST_YEAR + 1
    if delta_daily.shape[0] != n_projection_years * config.DAYS_PER_YEAR:
        raise ValueError(f"delta has {delta_daily.shape[0]} days, expected "
                         f"{n_projection_years * config.DAYS_PER_YEAR}")

    # Spin-up and hindcast: the baseline year repeated
    hindcast_year = detrended if config.DETREND_HINDCAST else values
    pre_projection = np.tile(hindcast_year, (n_pre_projection_years, 1, 1))

    # Projection: detrended baseline + CMIP6 delta (only in layers that exist)
    projection = np.tile(detrended, (n_projection_years, 1, 1))
    water_delta = np.where(water_exists[None, :, :], delta_daily, 0.0)
    projection[:, :, :sediment] += water_delta

    full_series = np.concatenate([pre_projection, projection], axis=0)

    # Sediment layer stays a copy of the bottom water layer, as in the baseline
    full_series[:, :, sediment] = full_series[:, :, 0]

    logger.info(f"  series built: {n_pre_projection_years} years repeated baseline "
                f"({config.SIMULATION_FIRST_YEAR}-{config.HINDCAST_LAST_YEAR}) + "
                f"{n_projection_years} projection years "
                f"({config.PROJECTION_FIRST_YEAR}-{config.PROJECTION_LAST_YEAR}) = "
                f"{full_series.shape[0]} days")
    return full_series, detrended


def protect_fill_value(full_series, exists, fill_value, logger):
    """
    Atlantis treats the _FillValue as missing data. A real value exactly equal
    to it (e.g. a temperature of exactly 0.0 degC) is nudged by 1e-6.
    """
    collision = exists[None, :, :] & (full_series == fill_value)
    if collision.any():
        logger.warning(f"  {collision.sum()} values exactly equal to _FillValue {fill_value}; "
                       f"nudged by +1e-6 so Atlantis does not read them as missing")
        full_series[collision] += 1e-6
    return full_series


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Write the Atlantis NetCDF file ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def write_forcing_file(output_file, variable_key, baseline, full_series, scenario, signal_attributes):
    """Same dimensions, names, _FillValue, time units and file format as the baseline."""
    variable_info = config.VARIABLES[variable_key]
    variable_name = variable_info["atlantis_variable"]
    n_days, n_boxes, n_z = full_series.shape
    time_seconds = baseline["time"][0] + np.arange(n_days) * baseline["time_dt"]

    with netCDF4.Dataset(output_file, "w", format=baseline["file_format"]) as output_nc:
        output_nc.createDimension("t", None)
        output_nc.createDimension("b", n_boxes)
        output_nc.createDimension("z", n_z)

        time_variable = output_nc.createVariable("t", "f8", ("t",))
        time_variable.units = baseline["time_units"]
        time_variable.dt = baseline["time_dt"]

        data_variable = output_nc.createVariable(variable_name, "f8", ("t", "b", "z"),
                                                 fill_value=baseline["fill_value"])
        data_variable.units = variable_info["units_label"]
        data_variable.long_name = (f"{variable_name}: EA baseline year + CMIP6 "
                                   f"{config.CMIP6_MODEL} {scenario} climate delta")

        # Keep the baseline global attributes (title, geometry, parameters) and add provenance
        for name, value in baseline["global_attributes"].items():
            output_nc.setncattr(name, value)
        output_nc.scenario = scenario
        output_nc.cmip6_source = (f"{config.CMIP6_MODEL} {config.CMIP6_MEMBER} historical + {scenario}; "
                                  f"files: {signal_attributes['source_files']}")
        output_nc.baseline_file = baseline["file"].name
        output_nc.timeline = (f"day 0 = 1 Jan {config.SIMULATION_FIRST_YEAR} (nominal); "
                              f"spin-up {config.SIMULATION_FIRST_YEAR}-{config.HINDCAST_FIRST_YEAR - 1}, "
                              f"hindcast {config.HINDCAST_FIRST_YEAR}-{config.HINDCAST_LAST_YEAR}, "
                              f"projection {config.PROJECTION_FIRST_YEAR}-{config.PROJECTION_LAST_YEAR}; "
                              f"365-day years")
        output_nc.method = (f"projection = detrended baseline year + LOWESS-smoothed CMIP6 monthly anomaly "
                            f"(reference {signal_attributes['reference_period']}, "
                            f"{signal_attributes['lowess_window_years']}-yr window, per box/layer/month) "
                            f"anchored to zero in {signal_attributes['anchor_year']}; "
                            f"detrended baseline in spin-up/hindcast: {config.DETREND_HINDCAST}")
        output_nc.history = (f"Created {datetime.date.today().isoformat()} by "
                             f"cmip6_code/build_atlantis_forcing.py")

        time_variable[:] = time_seconds
        data_variable[:] = full_series


def write_force_prm(scenario, output_names, logger):
    """Copy EA_force.prm, pointing temperature/salinity to the new files with rewind off."""
    prm_text = config.FORCE_PRM_FILE.read_text()
    replacements = {
        r"^Temperature0\.name\s+\S+": f"Temperature0.name {output_names['temperature']}",
        r"^temp_rewind\s+\S+": "temp_rewind 0",
        r"^Salinity0\.name\s+\S+": f"Salinity0.name {output_names['salinity']}",
        r"^salt_rewind\s+\S+": "salt_rewind 0",
    }
    for pattern, new_line in replacements.items():
        prm_text, n_replaced = re.subn(pattern, new_line, prm_text, flags=re.MULTILINE)
        if n_replaced != 1:
            raise ValueError(f"Expected exactly one line matching '{pattern}' in "
                             f"{config.FORCE_PRM_FILE.name}, found {n_replaced}")

    header = (f"# Generated from {config.FORCE_PRM_FILE.name} by cmip6_code/build_atlantis_forcing.py\n"
              f"# CMIP6 {config.CMIP6_MODEL} {scenario}: temperature and salinity run continuously for "
              f"{config.N_SIMULATION_YEARS} years ({config.N_SIMULATION_YEARS * config.DAYS_PER_YEAR} days); "
              f"tstop in run.prm must not exceed this.\n")
    output_file = config.FORCING_DIR / f"EA_force_{scenario}.prm"
    output_file.write_text(header + prm_text)
    logger.info(f"  written: {output_file}")


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Main ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, choices=config.SCENARIOS)
    parser.add_argument("--variable", choices=sorted(config.VARIABLES),
                        help="default: all variables (the force.prm is only written for all)")
    arguments = parser.parse_args()
    scenario = arguments.scenario

    logger = config.setup_logging(f"build_forcing_{scenario}")
    logger.info(f"=== Building Atlantis forcing for {scenario}")

    bgm = atlantis_geometry.read_bgm_boxes(config.BGM_FILE)
    layers = atlantis_geometry.read_atlantis_layers(config.INIT_FILE, bgm)
    config.FORCING_DIR.mkdir(parents=True, exist_ok=True)

    variable_keys = [arguments.variable] if arguments.variable else list(config.VARIABLES)
    output_names = {}
    for variable_key in variable_keys:
        variable_info = config.VARIABLES[variable_key]
        logger.info(f"--- {variable_key}")

        baseline = read_baseline(variable_key, layers)
        logger.info(f"  baseline: {baseline['file']} {baseline['values'].shape}")

        signal_file = config.SIGNAL_DIR / f"{variable_info['cmip6_name']}_{scenario}_climate_signal.nc"
        with xr.open_dataset(signal_file) as signal:
            delta_daily = signal["delta_daily"].values
            signal_attributes = dict(signal.attrs)
        logger.info(f"  climate signal: {signal_file}")
        if signal_attributes.get("incomplete_test_run", 0):
            logger.warning("  climate signal comes from an INCOMPLETE test run -- not for use in Atlantis")

        full_series, _ = build_series(baseline, layers, delta_daily, logger)
        full_series = protect_fill_value(full_series, layer_exists_mask(layers),
                                         baseline["fill_value"], logger)
        if np.isnan(full_series).any():
            raise ValueError("NaN in the final forcing series")

        output_name = f"{baseline['file'].stem}_{scenario}.nc"
        output_file = config.FORCING_DIR / output_name
        write_forcing_file(output_file, variable_key, baseline, full_series, scenario, signal_attributes)
        output_names[variable_key] = output_name
        logger.info(f"  written: {output_file}")

    if set(output_names) == set(config.VARIABLES):
        write_force_prm(scenario, output_names, logger)
    else:
        logger.info("  force.prm not written (not all variables were built)")


if __name__ == "__main__":
    main()
