#!/bin/bash
#
#SBATCH --job-name=split
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 split3p.py -f eeganr/crc_sp -n coin_corrected -i spcorr -r
