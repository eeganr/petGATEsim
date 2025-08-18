#!/bin/bash

for i in $(seq 0 11);
do
    sbatch correct.sh $i
done
