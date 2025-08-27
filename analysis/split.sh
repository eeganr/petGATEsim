#!/bin/bash
#
#SBATCH --job-name=split
#
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 split.py -f eeganr/crc/crc_nocorr -n crc
