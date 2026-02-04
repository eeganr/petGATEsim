#!/bin/bash

for i in $(seq 1 120);
do
    sbatch customgen.sh $i
done
