#!/bin/bash

# ======================= SYSTEM CONFIG ===========================
GPU=0
######################################################
# This section includes system configurations.       #
#         ** ! Please don't modify ! **              #
#source /home/products/cudarecon/include.sh          #

MAX_TEMP=85
GPUOSEM=

echo
echo "============ GPU Status begins ==============="
TEMPERATURE=$(nvidia-smi -q -i $GPU -d TEMPERATURE | awk '/Gpu/ {print $3}')
echo "Current GPU Temperature: $TEMPERATURE C."
echo "Maximum allowed GPU Temperature: $MAX_TEMP C."
if [ "$TEMPERATURE" -gt "$MAX_TEMP" ]; then
    echo "GPU Temperature exceeds maximum allowed temperature ($MAX_TEMP C)."
    echo "Please wait or contact system administrator."
    exit 1
else
    GPUOSEM=/home/users/gchinn/share/cudarecon/build/projection
fi
echo "============ GPU Status ends ================="
echo

if [ -z "$GPUOSEM" ]; then
    echo "GPUOSEM path not set."
    exit 1
fi

# ======================= USER CONFIG =============================

# Input arguments
LISTMODE_DATA="$1"
NORMALIZATION_LISTMODE_DATA="$2"
USE_NORMALIZATION=1
NORMALIZATION_TH=1e-2
# Output folder
SCRIPT_NAME=$(basename "$0" .sh)
DATE_TIME=$(date "+%Y%m%d_%H%M%S")
OUTDIR="${SCRIPT_NAME}_${DATE_TIME}"
mkdir -p "$OUTDIR"

# Check input files
if [ ! -f "$LISTMODE_DATA" ]; then
    echo "ERROR: List-mode file not found: $LISTMODE_DATA"
    exit 1
fi
if [ "$USE_NORMALIZATION" -eq 1 ] && [ ! -f "$NORMALIZATION_LISTMODE_DATA" ]; then
    echo "ERROR: Normalization LM file not found: $NORMALIZATION_LISTMODE_DATA"
    exit 1
fi

# Verbose level
VERBOSE=3

# Image parameters
VSIZE="350 350 350"
FOV="350 350 350"

# Recon parameters
TOR_SIGMA=0.98
SIGMA_TOF=15.9
# 0.98 for BrainPET
#TOR_HALF_WIDTH=2 for BrainPET
TOR_HALF_WIDTH=3
#PROJECTION_TH=1

PROJECTION_TH=1e-3

ITERATIONS=50
NUMSUBSETS=70
SAVE_INTERVAL=1
OUTPUT_FORMAT=1


# Construct file names safely
BASENAME_NORM=$(basename "$NORMALIZATION_LISTMODE_DATA" .lm)
BASENAME_IMG=$(basename "$LISTMODE_DATA" .lm)
NORMALIZATION_FILE="$OUTDIR/${BASENAME_NORM}_norm_no_tof"
IMAGE_FILENAME="$OUTDIR/${BASENAME_IMG}_normAll_${USE_NORMALIZATION}_tof_itr"
#========================================================================

# Automatically set subset size
FILESIZE=$(stat -c%s "$LISTMODE_DATA")
NUMCOIN=$(expr "$FILESIZE" / 40)
SUBSET_SIZE=$(expr "$NUMCOIN" / "$NUMSUBSETS")
echo "Subset size: $SUBSET_SIZE"

# ==================== NORMALIZATION ==============================
if [ $USE_NORMALIZATION = 1 ]; then
    NORMALIZATION_ARG="-n $NORMALIZATION_FILE.vox \
                      --normalizationThreshold $NORMALIZATION_TH"
    if [ -f $NORMALIZATION_FILE.vox ]; then
          echo "Normalization image $NORMALIZATION_FILE.vox exists."
            echo "No need to re-generate $NORMALIZATION_FILE.vox. Skipped."
    else        echo "Generating normalization file..."
        NORM_COMMAND="$GPUOSEM \
            --fov \"$FOV\" \
            --vsize \"$VSIZE\" \
            -g $TOR_SIGMA \
            -o $NORMALIZATION_FILE \
            -w $TOR_HALF_WIDTH \
            -s $SUBSET_SIZE \
            -l $NORMALIZATION_LISTMODE_DATA \
            -N \
            -v $VERBOSE \
            -d $GPU"
        echo "$NORM_COMMAND"
        eval "$NORM_COMMAND"
    fi
else
    NORMALIZATION_ARG=""
fi

# ==================== IMAGE RECONSTRUCTION =======================

echo
echo "Starting reconstruction..."
RECON_COMMAND="$GPUOSEM \
    --fov \"$FOV\" \
    --vsize \"$VSIZE\" \
    -g $TOR_SIGMA \
    -o $IMAGE_FILENAME \
    -w $TOR_HALF_WIDTH \
    -s $SUBSET_SIZE \
    -i $ITERATIONS \
    -S $SAVE_INTERVAL \
    -l $LISTMODE_DATA \
    -v $VERBOSE \
    $NORMALIZATION_ARG \
    -F $OUTPUT_FORMAT \
    --projectionThreshold $PROJECTION_TH \
    -d $GPU"
    # -T $SIGMA_TOF \

echo "$RECON_COMMAND"
eval "$RECON_COMMAND"