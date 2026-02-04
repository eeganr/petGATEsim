#!/bin/bash
#
#SBATCH --job-name=correct
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=FAIL

BASE=$1

module load python/3.12

for i in $(seq $((10*BASE)) $((10*BASE + 9)));
do
    python3 correct.py -i /scratch/groups/cslevin/eeganr/cylwater/cylwat_nocorr/ -o /scratch/groups/cslevin/eeganr/cylwater/cylwat_corr/split/ -l $i
done
