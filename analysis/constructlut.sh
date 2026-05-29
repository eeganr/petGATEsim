#!/bin/bash
#
#SBATCH --job-name=constructlut
#
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 constructlut.py -f scatter/scatter_nocorr/$1/