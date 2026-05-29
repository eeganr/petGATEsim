#!/bin/bash

mkdir $GROUP_SCRATCH/eeganr/scatter/scatter_nocorr/$1
job_id=$(sbatch --parsable aggregate.sh $1)
sbatch --dependency=afterok:$job_id tag.sh $1
sbatch --dependency=afterok:$job_id constructlut.sh $1