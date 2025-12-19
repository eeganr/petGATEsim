#!/bin/bash
#
#SBATCH --job-name=aggregate
#
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,FAIL,END

module load python/3.12
python3 $HOME/petGATEsim/analysis/aggregate.py -s 1 -e 120