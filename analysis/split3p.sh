#!/bin/bash
#
#SBATCH --job-name=split
#
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 split3p.py -f eeganr/crc2/crc2_nocorr -i crc_delay -n delay
