#!/bin/bash
#
#SBATCH --job-name=correct
#
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,FAIL,END

BASE=$1

module load python/3.12

for i in $(seq $((10*BASE)) $((10*BASE + 9)));
do
    python3 correct.py -i /scratch/groups/cslevin/eeganr/flangeless/ -o /scratch/groups/cslevin/eeganr/flangeless_corr/ -l $i
done
