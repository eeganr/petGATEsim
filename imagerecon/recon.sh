#!/bin/bash
#
#SBATCH --job-name=recon
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,FAIL,END

module load python/3.12
python3 lm_to_recon.py