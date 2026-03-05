import numpy as np
import os
import argparse
import randoms
from multiprocessing import Pool

# === CONFIG ===
PATH_PREFIX = '/scratch/groups/cslevin/eeganr/scatter/scatter_singles2/output'
PATH_POSTFIX = 'Singles.dat'
OUT_FOLDER = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr4/'
NAME = 'scatter'
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE
DELAY = 10 * CYCLE
DETECTORS_SIM = 12288
DETECTORS_REAL = 13824
N_PROCS = 5   # <---- number of parallel workers
# ===


def process_file(i):
    infile = PATH_PREFIX + str(i) + PATH_POSTFIX
    print("Reading file", infile)

    if not os.path.isfile(infile):
        print("Skipped!")
        return None

    return randoms.read_file_lm(
        infile, OUT_FOLDER, NAME, TAU, DELAY, DETECTORS_SIM
    )


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=int, default=1, help="start file num")
    parser.add_argument("-e", "--end", type=int, default=60, help="end file num")
    parser.add_argument("-r", "--real", action="store_true", help="uses real detector indices")
    args = parser.parse_args()

    FILE_RANGE = range(args.start, args.end + 1)
    DETS = DETECTORS_REAL if args.real else DETECTORS_SIM

    # Initialize totals
    sc_total = np.zeros(DETS)
    pc_total = np.zeros(DETS)
    coin_total = np.zeros((DETS, DETS))
    dw_total = np.zeros((DETS, DETS))
    actuals_total = np.zeros((DETS, DETS))
    scatters_total = np.zeros((DETS, DETS))

    # Parallel processing
    with Pool(processes=N_PROCS) as pool:
        for result in pool.imap_unordered(process_file, FILE_RANGE):
            if result is None:
                continue

            singles_count, prompts_count, coin_lor, dw_nums, actuals, scatters = result

            sc_total += singles_count
            pc_total += prompts_count
            coin_total += coin_lor
            dw_total += dw_nums
            actuals_total += actuals
            scatters_total += scatters

    # Save results
    np.save(OUT_FOLDER + 'singles_count.npy', sc_total)
    np.save(OUT_FOLDER + 'prompts_count.npy', pc_total)
    np.save(OUT_FOLDER + 'coin_lor.npy', coin_total)
    np.save(OUT_FOLDER + 'dw_nums.npy', dw_total)
    np.save(OUT_FOLDER + 'actuals.npy', actuals_total)
    np.save(OUT_FOLDER + 'scatters.npy', scatters_total)