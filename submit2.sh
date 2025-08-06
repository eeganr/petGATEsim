#!/bin/bash

for i in $(seq 121 180);
do
    sbatch customgen2.sh $i
done