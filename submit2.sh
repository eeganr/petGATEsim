#!/bin/bash

for i in $(seq 241 300);
do
    sbatch customgen2.sh $i
done
