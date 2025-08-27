#!/bin/bash
#
#SBATCH --job-name=pateint_tof_S
#SBATCH --chdir=/scratch/users/eeganr/aug27delay
#SBATCH --time=960:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=gpu
#SBATCH --gpus=1

source /home/users/gchinn/config/cudarecon.sh 

/home/users/eeganr/petGATEsim/imagerecon/recon_train.sh \
  /scratch/groups/cslevin/eeganr/crc/crc_corr/delaycorr.lm \
  /scratch/groups/cslevin/eeganr/annulus/annulus_corr/delaycorr.lm\
