#!/bin/bash

singularity exec -B /home/users/eeganr/petGATEsim:/home/users/eeganr/petGATEsim /home/groups/cslevin/mhchin/gate/gate_latest.sif /home/users/eeganr/petGATEsim/runtrain1.sh $1 /home/users/eeganr/petGATEsim/testcylinder.mac 1.0 /home/users/eeganr/petGATEsim/Geometry.mac

rm /home/users/eeganr/petGATEsim/pastoutput/output$1Run.bin

rm /home/users/eeganr/petGATEsim/pastoutput/output$1delay.dat
