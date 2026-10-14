#!/bin/bash
#SBATCH --job-name=cmip6_atlantis
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=cmip6_atlantis_%j.out

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ CMIP6 -> East Antarctic Atlantis forcing ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
# Submit from the cmip6_code folder:   sbatch run_cmip6_hpc.slurm
# Edit the paths below. ATLANTIS_DIR must contain Antarctica_28.bgm,
# init_squid_230809.nc, EAtemp.nc, EAsalt.nc and EA_force.prm.
set -euo pipefail

export CMIP6_ROOT=/path/to/CMCC-CM2-SR5
export ATLANTIS_DIR=/path/to/atlantis_model
export OUTPUT_DIR=/path/to/outputs

# Python environment with numpy, xarray, netCDF4, cftime, pyproj, matplotlib
# module load python      # or:  source activate <your_env>

## ~ 1. Check metadata of all CMIP6 files ~ ##
python inspect_cmip6.py

## ~ 2. Box/layer extraction (the only heavy step) ~ ##
python extract_cmip6_boxes.py --all

## ~ 3. Climate signal, Atlantis forcing and validation per scenario ~ ##
for scenario in ssp126 ssp245 ssp370; do
    python compute_climate_signal.py --scenario "$scenario"
    python build_atlantis_forcing.py --scenario "$scenario"
done
python validate_forcing.py
