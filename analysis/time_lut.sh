#!/bin/bash
#
#SBATCH --job-name=timelut
#
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 $HOME/petGATEsim/analysis/time_lut.py -s 1 -e 120 -f scatter/scatter_nocorr/10 -n scatter10 -i scatter/singles/10