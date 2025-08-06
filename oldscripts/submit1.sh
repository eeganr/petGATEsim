#!/bin/bash
#
#SBATCH --job-name=PhytoPET
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=BEGIN,FAIL,END

index=$1
source=$2
duration=$3
rotation=$4

# -B arg binds path to image / -c executes ...gate.sh file (seems like -c is redundant but kept here because it worked)

# ###################### EDIT ##################################
# change all refereneces to /home/users/fwdesai/PhytoPET to the directory where you keep the simulation files
# do not change the path /home/groups/cslevin/mhchin/gate/gate_latest.sif
# make sure you are calling runtrain1.sh as done below
################################################################

singularity exec -B /home/users/eeganr/petGATEsim:/home/users/eeganr/petGATEsim /home/groups/cslevin/mhchin/gate/gate_latest.sif /home/users/eeganr/petGATEsim/runtrain1.sh $index $source $duration $rotation

