#!/bin/bash
#
#SBATCH --job-name=eval
#
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 eval.py $GROUP_SCRATCH/eeganr/cylinder/cyl_eval
