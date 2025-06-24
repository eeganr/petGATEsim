#!/bin/bash
#
#SBATCH --job-name=analyze
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=BEGIN,FAIL,END

module load python/3.12
python3 findrandoms.py -o estimates1mCi -t 1
