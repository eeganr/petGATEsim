#!/bin/bash
#
#SBATCH --job-name=eval
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
mkdir $GROUP_SCRATCH/eeganr/crc2/crc2_eval
mv $GROUP_SCRATCH/eeganr/crc2/crc2_nocorr/split $GROUP_SCRATCH/eeganr/crc2/crc2_eval
python3 eval.py $GROUP_SCRATCH/eeganr/crc2/crc2_eval
