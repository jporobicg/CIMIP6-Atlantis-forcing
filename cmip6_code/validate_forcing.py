"""
Validate the CMIP6 Atlantis forcing files for all scenarios.

Checks (written to VALIDATION_DIR/validation_report.txt):
    - dimensions, variable names, _FillValue, time units and format as in the baseline
    - time strictly increasing, daily, expected length
    - no NaN, layers that do not exist unchanged (0), sediment = bottom layer
    - no real value equal to the _FillValue
    - values inside a plausible range
    - no jump at the hindcast -> projection boundary (compared with a normal New Year)
    - baseline variability retained (projection - delta = detrended baseline year)
    - no meaningful seasonal cycle collapses
    - climate delta at the start / middle / end of the projection
    - scenarios ordered and diverging by 2100

Plots (VALIDATION_DIR):
    <variable>_box<N>.png   original / detrended baseline, CMIP6 signal,
                            final forcing and transition zoom, for surface / mid / bottom
    <variable>_scenario_divergence.png   domain-mean delta of the three scenarios

Usage:
    python validate_forcing.py
"""

import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import atlantis_geometry
import build_atlantis_forcing
import config


SCENARIO_COLOURS = {"ssp126": "tab:blue", "ssp245": "tab:orange", "ssp370": "tab:red"}


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Read the files of one variable ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def read_forcing(file_path, variable_name):
    """Forcing values and metadata, read without masking so fill values are visible."""
    with netCDF4.Dataset(file_path) as forcing_nc:
        forcing_nc.set_auto_mask(False)
        return {
            "values": np.array(forcing_nc[variable_name][:], dtype=float),
            "time": np.array(forcing_nc["t"][:], dtype=float),
            "time_units": forcing_nc["t"].units,
            "dimensions": forcing_nc[variable_name].dimensions,
            "fill_value": forcing_nc[variable_name]._FillValue,
            "file_format": forcing_nc.file_format,
            "geometry": forcing_nc.getncattr("geometry"),
        }


def surface_mid_bottom(numlayers_box):
    """z indices of the surface, a middle and the bottom layer of a box."""
    return {"surface": numlayers_box - 1, "mid": numlayers_box // 2, "bottom": 0}


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Numerical checks for one scenario ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def check_forcing(variable_key, scenario, forcing, baseline, detrended, signal, layers, report):
    """Append PASS / FAIL lines to the report list; returns the number of failures."""
    variable_info = config.VARIABLES[variable_key]
    values = forcing["values"]
    exists = build_atlantis_forcing.layer_exists_mask(layers)
    sediment = layers["sediment_index"]
    n_expected_days = config.N_SIMULATION_YEARS * config.DAYS_PER_YEAR
    projection_start = (config.PROJECTION_FIRST_YEAR - config.SIMULATION_FIRST_YEAR) * config.DAYS_PER_YEAR
    low, high = variable_info["plausible_range"]

    checks = []
    checks.append(("dimensions / names / fill / time units / format as baseline",
                   forcing["dimensions"] == baseline["dimensions"]
                   and values.shape[1:] == baseline["values"].shape[1:]
                   and forcing["fill_value"] == baseline["fill_value"]
                   and forcing["time_units"] == baseline["time_units"]
                   and forcing["file_format"] == baseline["file_format"]
                   and forcing["geometry"] == baseline["global_attributes"]["geometry"], ""))
    time_steps = np.diff(forcing["time"])
    checks.append(("time strictly increasing, daily, expected length",
                   len(forcing["time"]) == n_expected_days and np.all(time_steps == 86400),
                   f"{len(forcing['time'])} records"))
    checks.append(("no NaN", not np.isnan(values).any(), ""))
    checks.append(("layers that do not exist are unchanged (0)",
                   np.all(values[:, ~exists] == 0), ""))
    checks.append(("sediment = bottom water layer",
                   np.array_equal(values[:, :, sediment], values[:, :, 0]), ""))
    collisions = int((values[:, exists] == forcing["fill_value"]).sum())
    checks.append(("no real value equal to _FillValue", collisions == 0, f"{collisions} values"))
    existing_values = values[:, exists]
    checks.append((f"values within plausible range {low} to {high}",
                   existing_values.min() >= low and existing_values.max() <= high,
                   f"min {existing_values.min():.3f}, max {existing_values.max():.3f}"))

    # Transition: the step on 1 Jan 2015 should equal a normal hindcast New Year step
    step_2015 = values[projection_start] - values[projection_start - 1]
    step_hindcast = values[projection_start - config.DAYS_PER_YEAR] - values[projection_start - config.DAYS_PER_YEAR - 1]
    extra_step = np.abs(step_2015 - step_hindcast)[exists].max()
    checks.append(("no extra jump at hindcast -> projection (vs a hindcast New Year)",
                   extra_step < 1e-6, f"max extra step {extra_step:.2e}"))

    # Baseline variability retained exactly: projection - delta = detrended baseline year
    water = slice(0, sediment)
    water_exists = exists[:, water]
    projection = values[projection_start:, :, water]
    delta_daily = np.where(water_exists[None], signal["delta_daily"].values, 0.0)
    n_projection_years = projection.shape[0] // config.DAYS_PER_YEAR
    reconstructed = (projection - delta_daily).reshape(n_projection_years, config.DAYS_PER_YEAR,
                                                       *projection.shape[1:])
    reconstruction_error = np.abs(reconstructed - detrended[None, :, :, water])[:, :, water_exists].max()
    checks.append(("projection - CMIP6 delta = detrended baseline (variability retained)",
                   reconstruction_error < 1e-5, f"max difference {reconstruction_error:.1e}"))

    # Seasonal amplitude change caused by the month-specific CMIP6 delta (information)
    detrended_amplitude = np.ptp(detrended[:, :, water], axis=0)
    projection_amplitude = np.ptp(projection.reshape(n_projection_years, config.DAYS_PER_YEAR,
                                                     *projection.shape[1:]), axis=1)
    amplitude_change = (projection_amplitude - detrended_amplitude[None])[:, water_exists]
    lost_cycle = ((detrended_amplitude[None] > 0.05)
                  & (projection_amplitude < 0.5 * detrended_amplitude[None]))[:, water_exists]
    n_failures = 0
    for name, passed, detail in checks:
        report.append(f"  [{'PASS' if passed else 'FAIL'}] {scenario} {variable_key}: {name} {detail}")
        n_failures += 0 if passed else 1

    # A month-specific CMIP6 delta can legitimately change the seasonal amplitude
    # (e.g. summer warming > winter warming), so this is flagged for inspection, not failed
    lost_cases = np.argwhere(lost_cycle)
    status = "PASS" if len(lost_cases) == 0 else "CHECK"
    report.append(f"  [{status}] {scenario} {variable_key}: seasonal amplitude change range "
                  f"{amplitude_change.min():+.3f} to {amplitude_change.max():+.3f}; "
                  f"{len(lost_cases)} box/layer/years where a cycle >0.05 loses more than half its amplitude")
    if len(lost_cases) > 0:
        exists_index = np.argwhere(water_exists)
        affected = sorted({tuple(int(i) for i in exists_index[case[1]]) for case in lost_cases})
        report.append(f"          affected (box, z): {affected[:10]}{' ...' if len(affected) > 10 else ''}")
    return n_failures


def delta_table(variable_key, signals, layers, report):
    """Annual-mean delta at start / middle / end of the projection (domain mean and validation boxes)."""
    units = config.VARIABLES[variable_key]["units_label"]
    years = [config.PROJECTION_FIRST_YEAR,
             (config.PROJECTION_FIRST_YEAR + config.PROJECTION_LAST_YEAR) // 2,
             config.PROJECTION_LAST_YEAR]
    n_boxes = len(layers["numlayers"])
    surface_index = layers["numlayers"] - 1
    report.append(f"  Annual-mean CMIP6 delta ({units}): domain mean surface / bottom layer; "
                  f"surface of boxes {config.VALIDATION_BOXES}")
    end_surface_delta = {}
    for scenario, signal in signals.items():
        for year in years:
            if year not in signal["year"].values:
                continue
            year_delta = signal["delta_monthly"].sel(year=year).mean("month_of_year").values   # [b, z]
            surface = year_delta[np.arange(n_boxes), surface_index]
            box_values = ", ".join(f"{surface[b]:+.3f}" for b in config.VALIDATION_BOXES)
            report.append(f"    {scenario} {year}: surface {surface.mean():+.3f}, "
                          f"bottom {year_delta[:, 0].mean():+.3f}; boxes [{box_values}]")
            if year == config.PROJECTION_LAST_YEAR:
                end_surface_delta[scenario] = surface.mean()
    return end_surface_delta


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Plots ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def plot_box(variable_key, box_number, baseline, detrended, signals, forcings, layers):
    """4 columns (baseline year, CMIP6 signal, full forcing, transition) x 3 layers."""
    units = config.VARIABLES[variable_key]["units_label"]
    layer_choice = surface_mid_bottom(layers["numlayers"][box_number])
    n_days_total = config.N_SIMULATION_YEARS * config.DAYS_PER_YEAR
    simulation_year = config.SIMULATION_FIRST_YEAR + np.arange(n_days_total) / config.DAYS_PER_YEAR
    projection_start = (config.PROJECTION_FIRST_YEAR - config.SIMULATION_FIRST_YEAR) * config.DAYS_PER_YEAR
    zoom = slice(projection_start - 3 * config.DAYS_PER_YEAR, projection_start + 4 * config.DAYS_PER_YEAR)

    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    for row, (layer_name, z) in enumerate(layer_choice.items()):
        depth_label = (f"{layer_name}: z={z}, {layers['layer_top'][box_number, z]:.0f}-"
                       f"{layers['layer_bottom'][box_number, z]:.0f} m")

        # 1. Original and detrended baseline year
        ax = axes[row, 0]
        ax.plot(baseline["values"][:, box_number, z], color="black", label="original baseline")
        ax.plot(detrended[:, box_number, z], color="tab:green", linestyle="--", label="detrended baseline")
        ax.set_title(f"Box {box_number} {depth_label}", fontsize=9)
        ax.set_xlabel("day of baseline year")
        ax.set_ylabel(units)

        # 2. CMIP6 signal: annual-mean anomaly (thin) and anchored smooth delta (thick)
        ax = axes[row, 1]
        for scenario, signal in signals.items():
            colour = SCENARIO_COLOURS[scenario]
            anomaly = signal["anomaly"].isel(b=box_number, z=z).mean("month_of_year")
            smoothed = signal["smoothed_anomaly"].isel(b=box_number, z=z).mean("month_of_year")
            delta = signal["delta_monthly"].isel(b=box_number, z=z).mean("month_of_year")
            ax.plot(signal["year"], anomaly, color=colour, linewidth=0.6, alpha=0.5)
            ax.plot(signal["year"], smoothed, color=colour, linewidth=1.0, linestyle=":")
            projection_part = signal["year"] >= config.PROJECTION_FIRST_YEAR
            ax.plot(signal["year"][projection_part], delta[projection_part], color=colour,
                    linewidth=2.0, label=f"{scenario} delta")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(config.PROJECTION_FIRST_YEAR, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title("CMIP6 anomaly (thin), LOWESS (dotted), anchored delta (thick)", fontsize=9)
        ax.set_ylabel(f"{units} vs {config.REFERENCE_FIRST_YEAR}-{config.REFERENCE_LAST_YEAR}")

        # 3. Final forcing over the whole simulation
        ax = axes[row, 2]
        for scenario, forcing in forcings.items():
            series = forcing["values"][:, box_number, z]
            annual_mean = series.reshape(-1, config.DAYS_PER_YEAR).mean(axis=1)
            ax.plot(simulation_year, series, color=SCENARIO_COLOURS[scenario], linewidth=0.3, alpha=0.4)
            ax.plot(config.SIMULATION_FIRST_YEAR + np.arange(len(annual_mean)) + 0.5, annual_mean,
                    color=SCENARIO_COLOURS[scenario], linewidth=1.5, label=scenario)
        ax.axvline(config.HINDCAST_FIRST_YEAR, color="grey", linestyle=":", linewidth=0.8)
        ax.axvline(config.PROJECTION_FIRST_YEAR, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title("Final forcing: daily (thin) and annual mean (thick)", fontsize=9)
        ax.set_ylabel(units)

        # 4. Zoom on the hindcast -> projection transition
        ax = axes[row, 3]
        for scenario, forcing in forcings.items():
            ax.plot(simulation_year[zoom], forcing["values"][zoom, box_number, z],
                    color=SCENARIO_COLOURS[scenario], linewidth=0.8, label=scenario)
        ax.axvline(config.PROJECTION_FIRST_YEAR, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title("Transition hindcast -> projection", fontsize=9)
        ax.set_ylabel(units)

    for ax in axes[0]:
        ax.legend(fontsize=7)
    fig.suptitle(f"{variable_key}: Atlantis box {box_number} -- {config.CMIP6_MODEL} scenarios")
    fig.tight_layout()
    output_file = config.VALIDATION_DIR / f"{variable_key}_box{box_number}.png"
    fig.savefig(output_file, dpi=110)
    plt.close(fig)
    return output_file


def plot_divergence(variable_key, signals, layers):
    """Domain-mean annual delta of surface, mid and bottom layers for the three scenarios."""
    units = config.VARIABLES[variable_key]["units_label"]
    n_boxes = len(layers["numlayers"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, layer_name in zip(axes, ["surface", "mid", "bottom"]):
        z_per_box = np.array([surface_mid_bottom(n)[layer_name] for n in layers["numlayers"]])
        for scenario, signal in signals.items():
            annual_delta = signal["delta_monthly"].mean("month_of_year").values         # [year, b, z]
            domain_delta = annual_delta[:, np.arange(n_boxes), z_per_box].mean(axis=1)
            projection_part = signal["year"].values >= config.PROJECTION_FIRST_YEAR
            ax.plot(signal["year"].values[projection_part], domain_delta[projection_part],
                    color=SCENARIO_COLOURS[scenario], linewidth=2, label=scenario)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title(f"Domain-mean delta, {layer_name} layer", fontsize=10)
        ax.set_xlabel("year")
        ax.set_ylabel(units)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output_file = config.VALIDATION_DIR / f"{variable_key}_scenario_divergence.png"
    fig.savefig(output_file, dpi=110)
    plt.close(fig)
    return output_file


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Main ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def main():
    logger = config.setup_logging("validate_forcing")
    config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    bgm = atlantis_geometry.read_bgm_boxes(config.BGM_FILE)
    layers = atlantis_geometry.read_atlantis_layers(config.INIT_FILE, bgm)
    report = [f"Validation of CMIP6 {config.CMIP6_MODEL} Atlantis forcing", ""]
    n_failures = 0

    for variable_key, variable_info in config.VARIABLES.items():
        logger.info(f"=== {variable_key}")
        baseline = build_atlantis_forcing.read_baseline(variable_key, layers)
        detrended, _ = build_atlantis_forcing.detrend_baseline_year(
            baseline["values"], build_atlantis_forcing.layer_exists_mask(layers))

        forcings = {}
        signals = {}
        for scenario in config.SCENARIOS:
            forcing_file = config.FORCING_DIR / f"{baseline['file'].stem}_{scenario}.nc"
            signal_file = config.SIGNAL_DIR / f"{variable_info['cmip6_name']}_{scenario}_climate_signal.nc"
            if not forcing_file.exists() or not signal_file.exists():
                logger.warning(f"  {scenario}: {forcing_file.name} or {signal_file.name} missing, skipped")
                report.append(f"  [MISSING] {scenario} {variable_key}: forcing or signal file not found")
                n_failures += 1
                continue
            logger.info(f"  {scenario}: {forcing_file}")
            forcings[scenario] = read_forcing(forcing_file, variable_info["atlantis_variable"])
            signals[scenario] = xr.load_dataset(signal_file)
            if signals[scenario].attrs.get("incomplete_test_run", 0):
                report.append(f"  [WARNING] {scenario} {variable_key}: signal from an INCOMPLETE test run")
            n_failures += check_forcing(variable_key, scenario, forcings[scenario], baseline,
                                        detrended, signals[scenario], layers, report)

        if not forcings:
            continue

        end_surface_delta = delta_table(variable_key, signals, layers, report)
        if variable_key == "temperature" and len(end_surface_delta) == len(config.SCENARIOS):
            ordered = (end_surface_delta["ssp126"] < end_surface_delta["ssp245"]
                       < end_surface_delta["ssp370"])
            report.append(f"  [{'PASS' if ordered else 'CHECK'}] surface warming in "
                          f"{config.PROJECTION_LAST_YEAR} ordered ssp126 < ssp245 < ssp370: "
                          + ", ".join(f"{s} {v:+.3f}" for s, v in end_surface_delta.items()))

        for box_number in config.VALIDATION_BOXES:
            plot_file = plot_box(variable_key, box_number, baseline, detrended, signals, forcings, layers)
            logger.info(f"  plot: {plot_file}")
        plot_file = plot_divergence(variable_key, signals, layers)
        logger.info(f"  plot: {plot_file}")
        report.append("")

    report.append(f"Total failed or missing checks: {n_failures}")
    report_file = config.VALIDATION_DIR / "validation_report.txt"
    report_file.write_text("\n".join(report) + "\n")
    for line in report:
        logger.info(line)
    logger.info(f"report: {report_file}")


if __name__ == "__main__":
    main()
