#!/bin/bash
#
#SBATCH --job-name=PhytoPET
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=BEGIN,FAIL,END

for i in $(seq 1 30);
do
    echo $i
done