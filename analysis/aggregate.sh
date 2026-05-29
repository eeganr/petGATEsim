#!/bin/bash
#
#SBATCH --job-name=aggregate
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

module load python/3.12
python3 $HOME/petGATEsim/analysis/aggregate.py -s 1 -e 120 -i crc2/singles/output -o crc2/crc2_nocorr2/ -n crc
# python3 $HOME/petGATEsim/analysis/aggregate.py -s 1 -e 120 -i scatter/singles/$1/output -o scatter/scatter_nocorr/$1/ -n scatter$1