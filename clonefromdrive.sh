#!/bin/bash

# This script clones a Git repository from Google Drive using rclone

for i in $(seq 1 15);
do
    rclone copy gdrive:/Example_Data/Sarah/Single_5min_Na22_250uCi/Axial_5_minus/Na22_250uCi_Single_06212024_minus5_PETA${i}_ch0_C_PCIe.dat /scratch/users/eeganr/realpointsource
    echo "copied ${i}!"
done