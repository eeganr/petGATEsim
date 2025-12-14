#!/bin/bash

for i in $(seq 61 122);
do
    sbatch customgenann.sh $i
done
