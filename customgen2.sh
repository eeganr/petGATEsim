#!/bin/bash
#
#SBATCH --job-name=GetData
#
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=BEGIN,FAIL,END

singularity exec -B /home/users/jamesdho/collabGATE:/home/users/jamesdho/collabGATE /home/groups/cslevin/mhchin/gate/gate_latest.sif /home/users/jamesdho/collabGATE/runtrain1.sh $1 /home/users/jamesdho/collabGATE/macros/sourcesNorm.mac 10.0 /home/users/jamesdho/collabGATE/macros/Geometry.mac
