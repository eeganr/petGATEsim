#!/bin/bash
#
#SBATCH --job-name=eval
#
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 kde.py /scratch/groups/cslevin/eeganr/crc_sp
