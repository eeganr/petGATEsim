import numpy as np
import os
import argparse
from randomsutils import *
import randoms

# === CONFIG ===
PATH_PRE_PREFIX = '/scratch/users/eeganr/'
PATH_POSTFIX = 'Singles.dat'
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
DETECTORS_SIM = 12288
DETECTORS_REAL = 13824
TIME = 10.0
# ===

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="aug1flange/output", help="path prefix for input files")
parser.add_argument("-o", "--output", type=str, default="estimations.csv", help="output file name")
parser.add_argument("-t", "--time", type=float, default=10.0, help="total simulation time in seconds")
parser.add_argument("-l", "--list", action="store_true", help="store contamination list")
parser.add_argument("-s", "--start", type=int, default=1, help="start file num")
parser.add_argument("-e", "--end", type=int, default=60, help="end file num")
args = parser.parse_args()

# Update Config with Arguments
PATH_PREFIX = PATH_PRE_PREFIX + args.path
OUTPUT_FILE = args.output
CONT_LIST = args.list
TIME = float(args.time)
FILE_RANGE = range(args.start, args.end)


if __name__ == "__main__":
    sp, dw, sr, actual, total = [], [], [], [], []
    for i in FILE_RANGE:
        # Step 1: Read file
        infile = PATH_PREFIX + str(i) + PATH_POSTFIX
        print(f"Reading file {infile}...")

        # Read file and calculate as-read metrics

        singles_count, prompts_count, coin_lor, dw_nums, actuals = randoms.read_file(infile, TAU, TIME, DELAY, DETECTORS_SIM, False)
        
        # Calculate estimation methods

        sp_nums = singles_prompts(singles_count, prompts_count, TIME, DETECTORS_SIM)
        sr_nums = singles_rate(singles_count, TIME, DETECTORS_SIM)

        sp.append(np.sum(sp_nums) / 2.0)
        dw.append(np.sum(dw_nums) / 2.0)
        sr.append(np.sum(sr_nums) / 2.0)
        actual.append(np.sum(actuals) / 2.0)
        total.append(np.sum(coin_lor) / 2.0)

        if CONT_LIST:
            # coin_per_lor = randoms.coincidences_per_lor(coincidences['detector1'], coincidences['detector2'], detectors[-1])
            # coincidences['lor_coins'] = coincidences.apply(lambda x: coin_per_lor[x['detector1']][x['detector2']], axis=1)
            # coincidences['sp_cont'] = coincidences.apply(lambda x: sp_nums[x['detector1']][x['detector2']], axis=1) / coincidences['lor_coins']
            # coincidences['dw_cont'] = coincidences.apply(lambda x: dw_nums[x['detector1']][x['detector2']], axis=1) / coincidences['lor_coins']
            # coincidences['sr_cont'] = coincidences.apply(lambda x: sr_nums[x['detector1']][x['detector2']], axis=1) / coincidences['lor_coins']
            # coincidences['actual_cont'] = coincidences.apply(lambda x: actual_nums[x['detector1']][x['detector2']], axis=1) / coincidences['lor_coins']
            # df = pd.DataFrame({'sp': coincidences['sp_cont'], 'dw': coincidences['dw_cont'], 'sr': coincidences['sr_cont'], 'actual': coincidences['actual_cont']})
            # print("Writing list")
            # with open(f'{LIST_DIRECTORY}list{i}.csv', 'w') as f:
            #     df.to_csv(f)
            pass
        
        # Step 5: Return results
        # actual.append(len(coincidences[~coincidences['true']]))
        print(f"File {str(i)} processed. SP: {sp[-1]}, DW: {dw[-1]}, SR: {sr[-1]}, Actual: {actual[-1]}, Total: {total[-1]}")
        assert False

    if not CONT_LIST:
        # store the collected data
        df = pd.DataFrame({'sp': sp, 'dw': dw, 'sr': sr, 'actual': actual, 'multis': multis, 'total': total})
        with open(OUTPUT_FILE, 'w') as f:
            df.to_csv(f)
    