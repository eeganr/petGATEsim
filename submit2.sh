#!/bin/bash

for i in $(seq 61 120);
do
    sbatch customgen2.sh $i
done
