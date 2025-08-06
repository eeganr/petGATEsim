#!/bin/bash
#
#SBATCH --job-name=analyze
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32GB
#SBATCH --mail-type=BEGIN,FAIL,END

module load python/3.12
python3 /home/users/eeganr/petGATEsim/analysis/gatebin_pdpkl.py -s 1 -e 1