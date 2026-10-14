# CMIP6 projection forcing for the East Antarctic Atlantis model

Python workflow that builds daily temperature and salinity forcing for the
28-box East Antarctic (EA) Atlantis model from CMIP6 **CMCC-CM2-SR5**
(`r1i1p1f1`, native grid `gn`) for **ssp126, ssp245 and ssp370**.
`ssp585` is downloaded but deliberately not used.

The original CMIP5 code (MATLAB, JFRE model) is kept unchanged in
`../projections_code/` for comparison.

---

## 1. The old CMIP5 workflow (projections_code/)

Written for the Juan Fernandez Ridge (JFRE, 59 boxes) model, temperature only.

| Step | Script | What it did |
|------|--------|-------------|
| 1 | `projection.sh` | `ncks` subsets of GFDL-ESM2M `thetao` (monthly, K), RCP4.5 / RCP8.5, 2011-2100 |
| 2 | `CMIP5_HR.m` | K -> degC, monthly -> daily linear interpolation, refine grid to 0.2 deg |
| 3 | `code_CC/box_avg_VAR_CC.m` | grid points inside each polygon (`inpolygon`), unweighted mean per depth band `dlev = [0 20 50 150 250 400 650 1000 4300]` |
| 4 | `Main_get_physics.m` | new = (3-year baseline climatology) x (linear CMIP trend / its first value) + HP-filtered CMIP residual; appended to the original hindcast |

Problems found (fixed in the new workflow, not carried over):

- ~0.5 degC jump at the start of the projection (checked in `CC_output/scenarios/rcp45.nc`).
- CMIP depth bands were written to fixed z indices, but z = 0 is the *deepest*
  layer of each box, so shallow boxes received trends from the wrong depth.
- Layers that do not exist got non-zero values (0 x ratio + residual).
- Boxes without CMIP data reused the trend of the previous box.
- Multiplicative trend on degC (ratio of Celsius values is not physical).
- The HP-filtered residual still contained the CMIP seasonal cycle, which was
  added on top of the baseline seasonal cycle.

Kept from the CMIP5 method: unweighted mean of grid cells inside each polygon,
overlap of model levels with Atlantis depth bands, and the idea of
"baseline + CMIP-derived change" after an unchanged historical period.

## 2. The EA Atlantis target

- `EAtemp.nc` / `EAsalt.nc`: one 365-day year, `(t, b=28, z=11)`, time units
  `seconds since 2000-01-01 00:00:00 +10`, `dt = 86400`, format NETCDF3_CLASSIC.
  `EA_force.prm` repeats it with `temp_rewind 1` / `salt_rewind 1`.
- z = 0 is the deepest water layer of each box, z = numlayers-1 the surface,
  z = 10 the sediment (an exact copy of z = 0). Boxes 1-6 have 7 water layers,
  the others 10 (`numlayers`, `nominal_dz` in `init_squid_230809.nc`).
- Depth bands (top to bottom): 0-20, 20-50, 50-100, 100-200, 200-300, 300-400,
  400-750, 750-1000, 1000-2000, 2000-botz m.
- Layers that do not exist hold exactly 0 in both files (temperature
  `_FillValue = 0`; salinity declares `_FillValue = 35` but also uses 0).
- The BGM (`Antarctica_28.bgm`) is in LAEA metres (`+lat_0=-63 +lon_0=82`).

## 3. Variable mapping

| Atlantis | CMIP5 (old) | CMIP6 | CMIP6 units | Transformation |
|----------|-------------|-------|-------------|----------------|
| `temperature` (degC) | `thetao` (K) | `thetao`, Omon, `sea_water_potential_temperature` | `degC` | none (K -> degC if ever needed); additive delta |
| `salinity` (PSU) | not used | `so`, Omon, `sea_water_salinity` | `0.001` (practical salinity) | none; additive delta |
| sea ice | not used | `siconc` downloaded | `%` | not used (out of scope) |

Checked in the CMIP6 files: 365_day calendar, time `days since 1850-01-01`,
2D `latitude`/`longitude` (0-360 E) on dims `(time, lev, i=292, j=362)` (ORCA1),
50 levels with `lev_bnds` (0-6110 m, positive down), `_FillValue = 1e20`,
ssp126 parent = historical `r1i1p1f1`, `branch_time_in_parent = 60225`
(= end of historical, 2015-01-01).

## 4. New workflow

```
inspect_cmip6.py         metadata, member/parent, grid and time continuity checks
        |
extract_cmip6_boxes.py   CMIP6 -> monthly (time, box, layer) series   [heavy, HPC]
        |
compute_climate_signal.py --scenario   anomaly vs 1995-2014, LOWESS, anchored, daily
        |
build_atlantis_forcing.py --scenario   detrended baseline + delta -> Atlantis NetCDF
        |
validate_forcing.py      checks + diagnostic plots for all scenarios
```

Supporting modules: `config.py` (all paths and settings), `atlantis_geometry.py`
(BGM and layer structure), `cmip6_files.py` (file lookup, time decoding).

**Spatial mapping.** CMIP6 cell centres are projected into the BGM's own LAEA
projection and tested against the polygons (no longitude convention issues).
Ocean cells inside a box are averaged with equal weights (as in CMIP5). If a
box had no cell inside, the nearest ocean cells would be used (logged; with
CMCC-CM2-SR5 every box has 4-238 cells, so no fallback is needed).

**Vertical mapping.** Each Atlantis layer of each box uses the thickness-weighted
mean of the CMIP6 levels overlapping its real depth range (`lev_bnds`), wet
cells only. A layer with no wet CMIP6 level would copy the layer above in the
same box (logged; not needed with CMCC-CM2-SR5).

**Climate signal.** Per box, layer and calendar month:
`anomaly = CMIP6 - mean(1995-2014)` -> LOWESS (30 nearest years, local linear,
tricube weights) -> `delta(year) = smooth(year) - smooth(2015)` -> mid-month
values interpolated to daily (365-day years).

**Baseline detrending.** The baseline year does not end where it starts. The
drift is measured as the mismatch at the 31 Dec -> 1 Jan wrap (7-day means on
both sides plus the local slope) and removed with a zero-mean linear ramp.
A linear regression over the year is NOT used: it confuses the seasonal cycle
with a trend (it gave ~1 degC "drift" at the surface).
Result: the New Year step drops from 0.023 to 0.002 degC on average (typical
daily change 0.0014 degC); the annual mean is unchanged.

**Final forcing.**

| Period | Years | Days | Forcing |
|--------|-------|------|---------|
| spin-up | 1985-1994 (nominal) | 0-3649 | detrended baseline year |
| hindcast | 1995-2014 | 3650-10949 | detrended baseline year |
| projection | 2015-2100 | 10950-42339 | detrended baseline year + CMIP6 delta |

Layers that do not exist keep 0, the sediment layer is set equal to z = 0, and
dimensions, variable names, `_FillValue`, time units and file format are those
of the baseline. `EA_force_<scenario>.prm` points to the new files with
`temp_rewind 0` / `salt_rewind 0`.

## 5. Assumptions

**ASSUMPTION 1.** The EA baseline year represents the 1995-2014 climate (the
CMIP6 reference period). The delta is anchored to zero in 2015, so the CMIP6
change between the middle of the reference period and 2015 is not added (it is
logged by `compute_climate_signal.py`; +0.03 degC domain mean in the partial
local test, to be confirmed on the full data). Consequence: a continuous
transition, at the cost of slightly underestimating the total change.

**ASSUMPTION 2.** The detrended baseline is also used in spin-up and hindcast
(`DETREND_HINDCAST = True` in `config.py`). It differs from the original by
at most 0.046 degC / 0.009 PSU and removes the small step at every New Year.
Set it to False to keep the original year in spin-up/hindcast.

**ASSUMPTION 3.** The forcing files keep the baseline time units
(`since 2000-01-01`) and start at the same first time value; they run
continuously for 42,340 days with rewind off. Atlantis counts time from the
run start, so day 0 is nominally 1 Jan 1985. `EAflow.nc` (hydrodynamic
exchanges) and the sea-ice file keep repeating their single year. `tstop` in
`EA_run.prm` must be at most 42,340 days (currently 36,500).

**ASSUMPTION 4.** Unweighted means of CMIP6 cells within each polygon (as in
CMIP5). Small boxes contain few cells (box 22: 4; boxes 1 and 4: 9).

**ASSUMPTION 5.** Additive deltas for temperature and salinity. The CMIP5
multiplicative ratio is not used because a ratio of degC values is not physical.

**ASSUMPTION 6.** A single member (`r1i1p1f1`) is used, so the delta includes
some internal variability of that member even after the 30-year LOWESS. The
month-specific deltas are noisier than an annual delta; inspect the
`<variable>_box<N>.png` plots and the `CHECK` lines of the validation report.

## 6. How to run

See the repository `README.md` for the directory layout and the run commands.
Set `CMIP6_ROOT`, `ATLANTIS_DIR` and `OUTPUT_DIR` before running. `ATLANTIS_DIR`
must contain the BGM, the initial-conditions file, `EAtemp.nc`, `EAsalt.nc` and
`EA_force.prm`.

The box series (`OUTPUT_DIR/box_series/*.nc`, a few MB) can be copied to another
machine and the climate-signal, forcing and validation steps run there.

Outputs in `OUTPUT_DIR`: `box_series/`, `climate_signal/`, `atlantis_forcing/`
(`EAtemp_<ssp>.nc`, `EAsalt_<ssp>.nc`, `EA_force_<ssp>.prm`), `validation/`
(plots, `validation_report.txt`) and `logs/`.

`--allow-incomplete` in `compute_climate_signal.py` is for testing with partial
CMIP6 data only; files produced this way are flagged and must not be used in Atlantis.

Python packages: numpy, xarray, netCDF4, cftime, pyproj, matplotlib.
