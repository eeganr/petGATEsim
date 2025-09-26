#!/bin/bash

for i in $(seq 181 240);
do
    sbatch customgenann.sh $i
done
