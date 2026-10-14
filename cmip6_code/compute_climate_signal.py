"""
Compute the daily CMIP6 climate-change signal (delta) for one scenario.

For every Atlantis box, layer and calendar month:
    1. Join CMIP6 historical (1995-2014) and the scenario (2015-2100).
    2. Reference climatology = mean of that calendar month over 1995-2014.
    3. Anomaly(year) = CMIP6(year) - reference climatology.
    4. Smooth the anomaly across years with LOWESS (~30 year window), so the
       signal is the gradual climate change, not interannual model noise.
    5. Anchor: delta(year) = smooth(year) - smooth(2015), so the delta is exactly
       zero in 2015 and the hindcast -> projection transition is continuous.
    6. Interpolate the monthly deltas (mid-month) to daily values over 365-day years.

Why not the CMIP5 method (Main_get_physics.m): it multiplied the baseline by a
linear ratio of degC values and added the HP-filtered CMIP residual, which
re-added the CMIP seasonal cycle and produced a jump at the projection start.

Output: SIGNAL_DIR/<cmip6_name>_<scenario>_climate_signal.nc

Usage:
    python compute_climate_signal.py --scenario ssp126
    python compute_climate_signal.py --scenario ssp126 --variable temperature
"""

import argparse

import numpy as np
import xarray as xr

import config


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Smoothing ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def lowess_smooth(years, values, window_years):
    """
    LOWESS (local linear regression with tricube weights, no robustness
    iterations) along the first axis.

    years        : [n_years]
    values       : [n_years, ...] any number of trailing dimensions (boxes, layers)
    window_years : number of nearest years used for each local fit
    """
    years = np.asarray(years, dtype=float)
    n_years = len(years)
    n_neighbours = min(int(window_years), n_years)
    flat_values = values.reshape(n_years, -1)
    smoothed = np.full_like(flat_values, np.nan, dtype=float)

    for target_index, target_year in enumerate(years):
        distance = np.abs(years - target_year)
        bandwidth = np.sort(distance)[n_neighbours - 1] * 1.0001 + 1e-9
        scaled = distance / bandwidth
        weights = np.where(scaled < 1.0, (1.0 - scaled ** 3) ** 3, 0.0)

        # Weighted least squares for intercept and slope, centred on the target year
        offset = years - target_year
        sum_w = weights.sum()
        sum_wx = (weights * offset).sum()
        sum_wxx = (weights * offset ** 2).sum()
        sum_wy = weights @ flat_values
        sum_wxy = (weights * offset) @ flat_values
        denominator = sum_w * sum_wxx - sum_wx ** 2
        smoothed[target_index] = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denominator

    return smoothed.reshape(values.shape)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Read and join historical + scenario box series ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def read_joined_series(cmip6_name, scenario, allow_incomplete, logger):
    """Monthly CMIP6 box/layer series from 1995 to 2100: values [months, b, z], years, months."""
    historical_file = config.BOX_SERIES_DIR / f"{cmip6_name}_{config.HISTORICAL_EXPERIMENT}_box_layer_monthly.nc"
    scenario_file = config.BOX_SERIES_DIR / f"{cmip6_name}_{scenario}_box_layer_monthly.nc"
    for file_path in [historical_file, scenario_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found; run extract_cmip6_boxes.py first")
        logger.info(f"  box series: {file_path}")

    with xr.open_dataset(historical_file) as historical, xr.open_dataset(scenario_file) as projection:
        # Historical and scenario must be the same model, member and Atlantis mapping
        for attribute in ["source_model", "member", "atlantis_geometry"]:
            if historical.attrs[attribute] != projection.attrs[attribute]:
                raise ValueError(f"{attribute} differs: {historical.attrs[attribute]} vs "
                                 f"{projection.attrs[attribute]}")
        if not np.array_equal(historical["source_layer"].values, projection["source_layer"].values):
            raise ValueError("Historical and scenario use a different layer mapping")

        values = np.concatenate([historical[cmip6_name].values, projection[cmip6_name].values])
        years = np.concatenate([historical["year"].values, projection["year"].values])
        months = np.concatenate([historical["month"].values, projection["month"].values])
        source_layer = projection["source_layer"].values
        source_files = (historical.attrs["source_files"] + ", " + projection.attrs["source_files"])

    # The historical run must end in Dec 2014 and the scenario start in Jan 2015
    month_counter = years * 12 + months
    if np.any(np.diff(month_counter) != 1):
        raise ValueError("Historical + scenario months are not consecutive (gap or overlap at 2014/2015?)")

    required_years = set(range(config.SIGNAL_FIRST_YEAR, config.SIGNAL_LAST_YEAR + 1))
    missing_years = sorted(required_years - set(years.tolist()))
    if missing_years:
        message = (f"{len(missing_years)} CMIP6 years missing between {config.SIGNAL_FIRST_YEAR} "
                   f"and {config.SIGNAL_LAST_YEAR} (first missing {missing_years[0]}, "
                   f"last missing {missing_years[-1]})")
        if not allow_incomplete:
            raise ValueError(message)
        logger.warning(f"  {message} -- continuing because --allow-incomplete (TEST ONLY)")

    logger.info(f"  CMIP6 years covered: {years.min()}-{years.max()} ({len(years)} months)")
    return values, years, months, source_layer, source_files


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Monthly anomalies, smoothing and anchoring ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def monthly_delta(values, years, months, logger):
    """
    Returns, all as [n_years, 12, b, z] on the annual axis signal_years:
        anomaly, smoothed anomaly, anchored delta, plus the reference climatology [12, b, z].
    """
    signal_years = np.arange(years.min(), years.max() + 1)
    n_years = len(signal_years)
    by_year_month = np.full((n_years, 12) + values.shape[1:], np.nan)
    by_year_month[years - signal_years[0], months - 1] = values

    # Reference climatology per calendar month (the baseline year is assumed to represent it)
    in_reference = (signal_years >= config.REFERENCE_FIRST_YEAR) & (signal_years <= config.REFERENCE_LAST_YEAR)
    reference_climatology = by_year_month[in_reference].mean(axis=0)              # [12, b, z]
    logger.info(f"  reference climatology: {signal_years[in_reference].min()}-"
                f"{signal_years[in_reference].max()} ({in_reference.sum()} years)")

    anomaly = by_year_month - reference_climatology[None]

    # Smooth each calendar month separately so the seasonal structure of the change is kept
    smoothed = np.full_like(anomaly, np.nan)
    for month_index in range(12):
        smoothed[:, month_index] = lowess_smooth(signal_years, anomaly[:, month_index],
                                                 config.LOWESS_WINDOW_YEARS)
    logger.info(f"  LOWESS smoothing per box, layer and calendar month "
                f"({config.LOWESS_WINDOW_YEARS}-year window)")

    # Anchor at the first projection year: delta = 0 throughout 2015
    anchor_index = int(np.where(signal_years == config.ANCHOR_YEAR)[0][0])
    delta = smoothed - smoothed[anchor_index][None]
    logger.info(f"  anchored: delta = smooth(year) - smooth({config.ANCHOR_YEAR}); "
                f"CMIP6 change between reference period and {config.ANCHOR_YEAR} "
                f"(domain mean, all layers) = {np.nanmean(smoothed[anchor_index]):+.3f} (not applied)")
    return signal_years, anomaly, smoothed, delta, reference_climatology


def monthly_to_daily(delta, signal_years):
    """
    Interpolate monthly deltas (placed at mid-month) to daily values for the
    projection years, using 365-day years. Returns [n_days, b, z].
    """
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    mid_month_day = np.cumsum(days_in_month) - days_in_month / 2.0               # day of year

    projection_years = np.arange(config.PROJECTION_FIRST_YEAR, config.PROJECTION_LAST_YEAR + 1)
    year_index = projection_years - signal_years[0]
    available = (year_index >= 0) & (year_index < len(signal_years))
    monthly_values = delta[year_index[available]]                                  # [years, 12, b, z]
    monthly_time = ((projection_years[available] - config.PROJECTION_FIRST_YEAR)[:, None]
                    * config.DAYS_PER_YEAR + mid_month_day[None, :]).ravel()
    monthly_values = monthly_values.reshape((-1,) + delta.shape[2:])

    n_days = len(projection_years) * config.DAYS_PER_YEAR
    daily_time = np.arange(n_days) + 0.5
    flat_monthly = monthly_values.reshape(len(monthly_time), -1)
    daily = np.full((n_days, flat_monthly.shape[1]), np.nan)
    for series in range(flat_monthly.shape[1]):
        if np.all(np.isfinite(flat_monthly[:, series])):
            # Before mid-Jan 2015 the delta stays 0; after mid-Dec 2100 it stays constant
            daily[:, series] = np.interp(daily_time, monthly_time, flat_monthly[:, series])
    return daily.reshape((n_days,) + delta.shape[2:])


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ One variable for one scenario ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def compute_signal(variable_key, scenario, allow_incomplete, logger):
    variable_info = config.VARIABLES[variable_key]
    cmip6_name = variable_info["cmip6_name"]
    logger.info(f"=== {scenario}: {variable_key} ({cmip6_name})")

    values, years, months, source_layer, source_files = read_joined_series(
        cmip6_name, scenario, allow_incomplete, logger)
    signal_years, anomaly, smoothed, delta, reference_climatology = monthly_delta(
        values, years, months, logger)
    daily_delta = monthly_to_daily(delta, signal_years)

    # Existing Atlantis layers must have a complete delta
    used_layer = source_layer >= 0
    if np.isnan(daily_delta[:, used_layer]).any():
        raise ValueError("NaN in the daily delta of an existing Atlantis layer")

    # Short summary: annual-mean delta of the surface layer at start / middle / end
    surface_index = np.array([np.where(used_layer[b])[0].max() for b in range(used_layer.shape[0])])
    for year in [config.PROJECTION_FIRST_YEAR,
                 (config.PROJECTION_FIRST_YEAR + config.PROJECTION_LAST_YEAR) // 2,
                 config.PROJECTION_LAST_YEAR]:
        if year in signal_years:
            year_delta = delta[year - signal_years[0]].mean(axis=0)                 # [b, z]
            surface_delta = year_delta[np.arange(len(surface_index)), surface_index]
            bottom_delta = year_delta[:, 0]
            logger.info(f"  {year}: surface delta mean {np.mean(surface_delta):+.3f} "
                        f"(range {surface_delta.min():+.3f} to {surface_delta.max():+.3f}), "
                        f"bottom-layer delta mean {np.mean(bottom_delta):+.3f} {variable_info['units_label']}")

    ## ~ Write the signal file ~ ##
    units = variable_info["units_label"]
    output = xr.Dataset(
        {
            "reference_climatology": (("month_of_year", "b", "z"), reference_climatology, {"units": units}),
            "anomaly": (("year", "month_of_year", "b", "z"), anomaly, {
                "units": units, "long_name": "CMIP6 monthly anomaly vs reference climatology"}),
            "smoothed_anomaly": (("year", "month_of_year", "b", "z"), smoothed, {
                "units": units, "long_name": "LOWESS-smoothed anomaly"}),
            "delta_monthly": (("year", "month_of_year", "b", "z"), delta, {
                "units": units, "long_name": f"smoothed anomaly minus its {config.ANCHOR_YEAR} value"}),
            "delta_daily": (("day", "b", "z"), daily_delta, {
                "units": units,
                "long_name": "daily climate delta added to the detrended Atlantis baseline",
                "comment": f"day 0 = 1 Jan {config.PROJECTION_FIRST_YEAR}, 365-day years"}),
            "source_layer": (("b", "z"), source_layer),
        },
        coords={"year": signal_years, "month_of_year": np.arange(1, 13),
                "day": np.arange(daily_delta.shape[0])},
        attrs={
            "scenario": scenario,
            "variable": cmip6_name,
            "source_model": config.CMIP6_MODEL,
            "member": config.CMIP6_MEMBER,
            "source_files": source_files,
            "reference_period": f"{config.REFERENCE_FIRST_YEAR}-{config.REFERENCE_LAST_YEAR}",
            "anchor_year": config.ANCHOR_YEAR,
            "lowess_window_years": config.LOWESS_WINDOW_YEARS,
            "incomplete_test_run": int(allow_incomplete),
        },
    )
    config.SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    output_file = config.SIGNAL_DIR / f"{cmip6_name}_{scenario}_climate_signal.nc"
    output.to_netcdf(output_file)
    logger.info(f"  written: {output_file}")


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Main ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, choices=config.SCENARIOS)
    parser.add_argument("--variable", choices=sorted(config.VARIABLES),
                        help="default: all variables")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="TEST ONLY: accept CMIP6 series that do not cover 1995-2100")
    arguments = parser.parse_args()

    logger = config.setup_logging(f"climate_signal_{arguments.scenario}")
    variable_keys = [arguments.variable] if arguments.variable else list(config.VARIABLES)
    for variable_key in variable_keys:
        compute_signal(variable_key, arguments.scenario, arguments.allow_incomplete, logger)


if __name__ == "__main__":
    main()
