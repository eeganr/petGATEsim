#!/bin/bash
singularity run /home/groups/cslevin/mhchin/gate/gate_latest.sif
source /geant4/geant4.10.05-install/bin/geant4.sh
source /cern/root-install/bin/thisroot.sh
GATE=/gate/gate_8.2-install/bin/Gate

$GATE