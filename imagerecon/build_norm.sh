#!/bin/bash
#
#SBATCH --job-name=buildnorm
#
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,FAIL,END

module load python/3.12
python3 build_norm_map_chunk.py -i /scratch/groups/cslevin/eeganr/flangeless_corr/actualcorr.lm -o /scratch/groups/cslevin/eeganr/reconfiles/actual_flange