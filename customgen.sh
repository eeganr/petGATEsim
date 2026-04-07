#!/bin/bash
#
#SBATCH --job-name=GetDataScatterArr10
#SBATCH --output=scatter_%A_%a.out
#
#SBATCH --time=12:00:00
#SBATCH --array=1-120
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=FAIL

i=$SLURM_ARRAY_TASK_ID

singularity exec -B /home/users/eeganr/petGATEsim:/home/users/eeganr/petGATEsim /home/groups/cslevin/mhchin/gate/gate_latest.sif /home/users/eeganr/petGATEsim/runtrain.sh $i /home/users/eeganr/petGATEsim/macros/scattermods/scatterphantom10.mac 1.0 /home/users/eeganr/petGATEsim/macros/Geometry.mac scatter/singles/10
