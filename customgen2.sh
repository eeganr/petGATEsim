#!/bin/bash
#
#SBATCH --job-name=GetData
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=BEGIN,FAIL,END

singularity exec -B /home/users/eeganr/petGATEsim:/home/users/eeganr/petGATEsim /home/groups/cslevin/mhchin/gate/gate_latest.sif /home/users/eeganr/petGATEsim/runtrain1.sh $1 /home/users/eeganr/petGATEsim/macros/flangeless.mac 10.0 /home/users/eeganr/petGATEsim/macros/Geometry.mac
