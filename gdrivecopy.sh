#!/bin/bash
#SBATCH --job-name=GDriveCopy
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,FAIL,END

ml rclone

for i in $(seq 1 60); 
do
rclone copy $SCRATCH/aug6annulus/output${i}Coincidences.dat gdrive:James/Simulation_Data/Annulus_Simulation
rclone copy $SCRATCH/aug6annulus/output${i}Singles.dat gdrive:James/Simulation_Data/Annulus_Simulation
rclone copy $SCRATCH/aug6annulus/output${i}Hits.dat gdrive:James/Simulation_Data/Annulus_Simulation
done