import numpy as np
import os
import argparse
import randoms

# === CONFIG ===
PATH_PREFIX = '/scratch/groups/cslevin/eeganr/scatter/scatter_singles2/output' # should not end in / (filename)
PATH_POSTFIX = 'Singles.dat'
OUT_FOLDER = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr2/' # should end in /
NAME = 'scatter'
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DETECTORS_SIM = 12288
DETECTORS_REAL = 13824
# ===

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start", type=int, default=1, help="start file num")
parser.add_argument("-e", "--end", type=int, default=60, help="end file num")
parser.add_argument("-r", "--real", action="store_true", help="uses real detector indices")
args = parser.parse_args()

FILE_RANGE = range(args.start, args.end + 1)
DETS = DETECTORS_REAL if args.real else DETECTORS_SIM

for i in FILE_RANGE:
    infile = PATH_PREFIX + str(i) + PATH_POSTFIX
    print("Reading file", infile)
    if not os.path.isfile(infile):
        print("Skipped!")
        continue

    randoms.write_lut(
        infile, OUT_FOLDER, NAME, TAU, DETECTORS_SIM
    )
