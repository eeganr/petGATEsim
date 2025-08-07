#!/bin/bash

for i in $(seq 1 60);
do
    sbatch customgen2.sh $i
done