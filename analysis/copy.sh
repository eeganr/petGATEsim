#!/bin/bash
#
#SBATCH --job-name=copy
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=FAIL

cp -r /scratch/groups/cslevin/eeganr/cylinder/cylinder_singles /scratch/groups/cslevin/eeganr/cylinder/cylinder_singles2
