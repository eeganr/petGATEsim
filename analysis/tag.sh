#!/bin/bash
#
#SBATCH --job-name=tag
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 tag.py -f crc2/crc2_nocorr2/ -n crc
# python3 tag.py -f scatter/scatter_nocorr/$1/ -n scatter$1
