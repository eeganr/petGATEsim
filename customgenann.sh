#!/bin/bash
#
#SBATCH --job-name=GetDataAnn
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=BEGIN,FAIL,END

singularity exec -B /home/users/eeganr/petGATEsim:/home/users/eeganr/petGATEsim /home/groups/cslevin/mhchin/gate/gate_latest.sif /home/users/eeganr/petGATEsim/runtrainann.sh $1 /home/users/eeganr/petGATEsim/macros/annulus.mac 10.0 /home/users/eeganr/petGATEsim/macros/Geometry.mac
