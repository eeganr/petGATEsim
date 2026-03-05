#!/bin/bash
#
#SBATCH --job-name=GetDataCyl
#
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=FAIL

singularity exec -B /home/users/eeganr/petGATEsim:/home/users/eeganr/petGATEsim /home/groups/cslevin/mhchin/gate/gate_latest.sif /home/users/eeganr/petGATEsim/runtrain.sh $1 /home/users/eeganr/petGATEsim/macros/scatterphantom.mac 10.0 /home/users/eeganr/petGATEsim/macros/Geometry.mac
