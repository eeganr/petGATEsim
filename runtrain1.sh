#!/bin/bash


i=$1
j=$2
x=$3
r=$4

SOURCE="${j##*/}"
NAME="PhytoPET_${SOURCE}_$r"
PREFIX=ps

BASE_SEED=123456789
INCR_SEED=3527
MODE="WARM"


echo "runtrain index= "$i

let "RANDOM_SEED=$BASE_SEED+$i*$INCR_SEED"
  
echo "Random seed ="$RANDOM_SEED

# If Sherlock cluster, set to 1
SHERLOCK=1
GATE=Gate

TSTART=0.0
TSLICE=$x
TSTOP=$x
#RANDOM_SEED=123456789
#INCR_SEED=3527
CAMERA_FILE=$r

# Hot sources only

SOURCE_FILE="sources.mac"
BASE_FILE="${PREFIX}_{$NAME}_hotr"

if [ $SHERLOCK -gt 0 ]
then
  # sourcing 2 sh files is necessary for running the Gate
  source /geant4/geant4.10.05-install/bin/geant4.sh
  source /cern/root-install/bin/thisroot.sh

  # moving to the path where mac files are located. Seems like if you run the program on the paths other than the folder you keep your mac files, Gate can't find the files...
  ############## EDIT #################
  # change the path /home/users/fwdesai/PhytoPET below to the directory where you will keep the files for this simulation
  ####################################

  cd /home/users/eeganr/petGATEsim
  GATE=/gate/gate_8.2-install/bin/Gate
fi

echo $MODE

if [ "$MODE" = "HOT" ]
then
  OUTPUT_FILE="${BASE_FILE}${i}"
  echo $OUTPUT_FILE
  $GATE -a [randomseed,$RANDOM_SEED][filename,$OUTPUT_FILE][camerafile,$CAMERA_FILE][sourcefile,$SOURCE_FILE][timestart,$TSTART][timeslice,$TSLICE][timestop,$TSTOP] mmRPET.mac > train1_hotr$i.out
  echo $RANDOM_SEED
  
#  RANDOM_SEED=`expr $RANDOM_SEED + $INCR_SEED`
fi

# Normalization / Attenuation phantom

SOURCE_FILE="sourcesNorm.mac"
BASE_FILE="${PREFIX}_${NAME}_normr"

if [ "$MODE" = "NORM" ]
then
  OUTPUT_FILE="/home/users/eeganr/PhytoPET_Simulation/${BASE_FILE}${i}"
  echo $CAMERA_FILE
  echo $SOURCE_FILE
  echo $OUTPUT_FILE
  $GATE -a [randomseed,$RANDOM_SEED][filename,$OUTPUT_FILE][camerafile,$CAMERA_FILE][sourcefile,$SOURCE_FILE][timestart,$TSTART][timeslice,$TSLICE][timestop,$TSTOP] simu_pet.mac > /home/users/eeganr/PhytoPET_Simulation/train1_normr$i.out
  echo $RANDOM_SEED
#  RANDOM_SEED=`expr $RANDOM_SEED + $INCR_SEED`
fi

# Main phantom

SOURCE_FILE=$j
BASE_FILE="${PREFIX}_${NAME}_r"

if [ "$MODE" = "WARM" ]
then
################ EDIT #############################
# change the OUTPUT_FILE to the location where you want data outputted, see step 3 in README to pick data output format
# make sure path ends with ${BASE_FILE}${i} as seen below
# also change the path /scratch/users/fwdesai/PhytoPET/spatialres/train1_r$i.out to the same location you used for OUTPUT_FILE, 
#	make sure this path ends with train1_r$i.out as seen below
####################################################
  OUTPUT_FILE="/home/users/eeganr/petGATEsim/output/${BASE_FILE}${i}"
  echo $SOURCE_FILE  
  echo $OUTPUT_FILE
  echo $CAMERA_FILE
  $GATE -a [randomseed,$RANDOM_SEED][filename,$OUTPUT_FILE][camerafile,$CAMERA_FILE][sourcefile,$SOURCE_FILE][timestart,$TSTART][timeslice,$TSLICE][timestop,$TSTOP] simu_pet.mac > /home/users/eeganr/PhytoPET_Simulation/train1_r$i.out
  echo $RANDOM_SEED
  
#  RANDOM_SEED=`expr $RANDOM_SEED + $INCR_SEED`
fi