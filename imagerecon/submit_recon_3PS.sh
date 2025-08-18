#!/bin/bash
#
#SBATCH --job-name=pateint_tof_S
#
#SBATCH --time=960:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=gpu
#SBATCH --gpus=1

source /home/users/gchinn/config/cudarecon.sh 

/home/users/eeganr/petGATEsim/imagerecon/recon_train.sh \
  /scratch/groups/cslevin/eeganr/flangeless_corr/actualcorr.lm \
  /scratch/groups/cslevin/eeganr/annulus_corr/actualcorr.lm\
