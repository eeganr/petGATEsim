#!/bin/bash

for i in $(seq 301 360);
do
    sbatch customgen2.sh $i
done
