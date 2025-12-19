#!/bin/bash
#
#SBATCH --job-name=remap
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 remap.py -a $GROUP_SCRATCH/eeganr/cylwater/cylwat_nocorr/cylwat.lm -b $GROUP_SCRATCH/eeganr/grantdata/everything.lm
