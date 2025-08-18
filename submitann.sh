#!/bin/bash

for i in $(seq 61 120);
do
    sbatch customgenann.sh $i
done