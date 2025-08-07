#!/bin/bash
#SBATCH --job-name=GDriveCopy
#
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,FAIL,END

ml rclone

for i in $(seq 61 120); 
do
rclone copy $SCRATCH/aug1flange/output${i}Singles.dat gdrive:Example_Data/Eegan/FlangelessEsserCoins
done