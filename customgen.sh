#!/bin/bash
#
#SBATCH --job-name=GetData
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=BEGIN,FAIL,END


for i in $(seq 1 30);
do
    singularity exec -B /home/users/eeganr/petGATEsim:/home/users/eeganr/petGATEsim /home/groups/cslevin/mhchin/gate/gate_latest.sif /home/users/eeganr/petGATEsim/runtrain1.sh $i /home/users/eeganr/petGATEsim/testcylinder.mac 60.0 /home/users/eeganr/petGATEsim/Geometry.mac
done