import numpy as np
import os
import argparse
import randoms

# === CONFIG ===
PATH_PREFIX = '/scratch/users/eeganr/aug6flange/output'
PATH_POSTFIX = 'Singles.dat'
OUT_FOLDER = '/scratch/groups/cslevin/eeganr/flangeless/'
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
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

sc_total = np.zeros(DETS)
pc_total = np.zeros(DETS)
coin_total = np.zeros((DETS, DETS))
dw_total = np.zeros((DETS, DETS))
actuals_total = np.zeros((DETS, DETS))

for i in FILE_RANGE:
    infile = PATH_PREFIX + str(i) + PATH_POSTFIX
    print("Reading file", infile)
    if not os.path.isfile(infile):
        print("Skipped!")
        continue

    singles_count, prompts_count, coin_lor, dw_nums, actuals = randoms.read_file_lm(
        infile, OUT_FOLDER, 'annulus', TAU, DELAY, DETECTORS_SIM
    )
    sc_total += singles_count
    pc_total += prompts_count
    coin_total += coin_lor
    dw_total += dw_nums
    actuals_total += actuals


np.save(OUT_FOLDER + 'singles_count.npy', sc_total)

np.save(OUT_FOLDER + 'prompts_count.npy', pc_total)

np.save(OUT_FOLDER + 'coin_lor.npy', coin_total)

np.save(OUT_FOLDER + 'dw_nums.npy', dw_total)

np.save(OUT_FOLDER + 'actuals.npy', actuals_total)
