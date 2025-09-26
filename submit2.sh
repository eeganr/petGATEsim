#!/bin/bash

for i in $(seq 181 240);
do
    sbatch customgen2.sh $i
done
