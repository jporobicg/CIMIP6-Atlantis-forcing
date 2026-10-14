# CMIP6 forcing for the East Antarctic Atlantis model

Build daily temperature and salinity hydrodynamic forcing for the 28-box East
Antarctic Atlantis model from CMIP6 **CMCC-CM2-SR5** (`r1i1p1f1`, grid `gn`).

Scenarios produced:

- SSP1-2.6: `ssp126`
- SSP2-4.5: `ssp245`
- SSP3-7.0: `ssp370`

`ssp585` is not used.

The Atlantis baseline year is kept. Its small drift across the year is removed
so the repeated year joins smoothly. During the projection, a smoothed CMIP6
change is added. That change is zero in 2015, so the hindcast does not jump
into the projection.

The scientific method, the CMIP5 comparison, and the assumptions are in
[cmip6_code/README.md](cmip6_code/README.md).

## What you need

Python packages: `numpy`, `xarray`, `netCDF4`, `cftime`, `pyproj`, `matplotlib`.

**CMIP6 data** (`CMIP6_ROOT`), member `r1i1p1f1`, grid `gn`:

```text
CMIP6_ROOT/
    historical/
        thetao/   thetao_Omon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_*.nc
        so/       so_Omon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_*.nc
    ssp126/
        thetao/
        so/
    ssp245/
        thetao/
        so/
    ssp370/
        thetao/
        so/
```

Historical files cover 1850–2014. The workflow reads 1990–2014 and uses
1995–2014 as the reference climatology. Each SSP covers 2015–2100.
`thetao` and `so` are monthly ocean fields (`Omon`). Sea ice is not used.

**Atlantis files** (`ATLANTIS_DIR`):

```text
Antarctica_28.bgm
init_squid_230809.nc
EAtemp.nc
EAsalt.nc
EA_force.prm
```

## Paths

Set these before running. Defaults, if you do not set them, are
`cmip6_data/CMCC-CM2-SR5`, `atlantis_model` and `outputs` inside this repository.

```bash
export CMIP6_ROOT=/path/to/CMCC-CM2-SR5
export ATLANTIS_DIR=/path/to/atlantis_model
export OUTPUT_DIR=/path/to/outputs
```

## Run

From `cmip6_code/`:

```bash
python inspect_cmip6.py
python extract_cmip6_boxes.py --all
for s in ssp126 ssp245 ssp370; do
    python compute_climate_signal.py --scenario "$s"
    python build_atlantis_forcing.py --scenario "$s"
done
python validate_forcing.py
```

`extract_cmip6_boxes.py` is the heavy step. On a batch system, edit the three
paths at the top of `cmip6_code/run_cmip6_hpc.slurm` and submit that script.
It runs the same steps.

One scenario or one variable can be run on its own:

```bash
python extract_cmip6_boxes.py --variable thetao --experiment historical
python compute_climate_signal.py --scenario ssp126 --variable temperature
python build_atlantis_forcing.py --scenario ssp126
```

## Timeline

Each forcing file is 116 years of daily data (42,340 days), using 365-day years:

| Period | Years | Content |
|--------|-------|---------|
| Spin-up | 1985–1994 | detrended baseline year, repeated |
| Hindcast | 1995–2014 | detrended baseline year, repeated |
| Projection | 2015–2100 | detrended baseline year plus the CMIP6 delta |

The run length in the Atlantis run parameter file must not exceed 42,340 days.
Temperature and salinity rewind must be off. The exchange and sea-ice files
are not changed by this workflow.

## Outputs

Under `OUTPUT_DIR`:

```text
box_series/          monthly CMIP6 values on Atlantis boxes and layers
climate_signal/      daily CMIP6 delta for each scenario
atlantis_forcing/    EAtemp_<scenario>.nc, EAsalt_<scenario>.nc, EA_force_<scenario>.prm
validation/          plots and validation_report.txt
logs/
```

`--allow-incomplete` on `compute_climate_signal.py` is only for tests with a
partial CMIP6 series. Files made that way are marked and must not be used in Atlantis.
