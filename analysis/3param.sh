#!/bin/bash
#
#SBATCH --job-name=3param
#
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 3param.py -f eeganr/crc2/crc2_nocorr -n crc
