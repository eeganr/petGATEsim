#!/bin/bash
#
#SBATCH --job-name=combine
#
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,FAIL,END

module load python/3.12
python3 combine.py -i /scratch/groups/cslevin/eeganr/annulus_corr/ -n sp