import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import uproot
import argparse
import randoms
import time
from randomsutils import *


# === CONFIG ===
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
PATH_PRE_PREFIX = "/scratch/users/eeganr/"
PATH_SUFFIX = ".root"
FILE_RANGE = range(1, 100 + 1)
LIST_DIRECTORY = "/scratch/users/eeganr/contamination/july17flange/"
# === END CONFIG ===

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filter", action="store_true", help="whether to filter energy")
parser.add_argument("-p", "--path", type=str, default="june23output/output", help="path prefix for input files")
parser.add_argument("-o", "--output", type=str, default="estimations.csv", help="output file name")
parser.add_argument("-t", "--time", type=float, default=1.0, help="total simulation time in seconds")
parser.add_argument("-l", "--list", action="store_true", help="store contamination list")
args = parser.parse_args()

PATH_PREFIX = PATH_PRE_PREFIX + args.path
FILTER_ENERGY = args.filter
OUTPUT_FILE = args.output
CONT_LIST = args.list
TIME = float(args.time)
if FILTER_ENERGY:
    print("Filtering singles by energy in range 0.450 to 0.750 MeV")

if CONT_LIST:
    print(f"Output will be written to {OUTPUT_FILE}")
else:
    print(f"Outputs will be written to {LIST_DIRECTORY}")


def read_root_file(infile):
    """
        Read in data from singles ROOT file, returns a DataFrame with columns:
        time, detector ID, source
    """    

    # Reads appropriate data field for a given file name
    # (e.g. file name in form Singles;16 for first 16k events)
    def get_all_vals(file, name):
        num = max([int(i.split(';')[1]) for i in file.keys() if i.split(';')[0] == name])
        return file[f'{name};{num}']

    with uproot.open(infile) as file:
        singles_tree = get_all_vals(file, 'Singles')

        singles = pd.DataFrame({
            "time": singles_tree["time"].array(library="np"),
            "detector": singles_tree["crystalID"].array(library="np"),
            "source": list(map(tuple, np.stack((
                singles_tree["sourcePosX"].array(library="np"),
                singles_tree["sourcePosY"].array(library="np"),
                singles_tree["sourcePosZ"].array(library="np")), axis=-1))),
            "energy": singles_tree["energy"].array(library="np"),
        })
        singles = singles.sort_values(by=['time'])

    detectors = np.sort(np.unique(singles['detector']))

    return singles, detectors


# Main function to read files and calculate estimates for many files of form
# output1.root, output2.root, ..., outputN.root
# Range [1, N) defined in FILE_RANGE
# Writes results to estimations.csv
if __name__ == "__main__":
    sp, sp_corr, dw, sr, actual, multis, total = [], [], [], [], [], [], []
    for i in FILE_RANGE:
        # Step 1: Read file
        infile = PATH_PREFIX + str(i) + PATH_SUFFIX
        print(f"Reading file {infile}...")
        singles, detectors = read_root_file(infile)
        print(detectors[-1])

        # Step 1.5: Filter hits by energy if needed
        if FILTER_ENERGY:
            singles = filter_singles(singles)  # Filter singles by energy
            print('Filtered energy')

        # Step 2: Bundle coincidences
        coincidences, multi_coins = bundle_coincidences(singles)  # Bundle singles into coincidences

        # Step 3: Tally stats by detector
        singles_count = randoms.singles_counts(singles['detector'], detectors[-1] + 1)
        prompts_count = randoms.prompts_counts(coincidences['detector1'], coincidences['detector2'], detectors[-1] + 1)

        # Step 4: Calculate estimation methods

        sp_nums = singles_prompts(singles_count, prompts_count, singles, coincidences, detectors, TIME)
        sp_corrected = singles_prompts_multi(singles_count, prompts_count, singles, coincidences, detectors, TIME)
        dw_nums = delayed_window(singles, detectors)
        sr_nums = singles_rate(singles_count, detectors, TIME)

        actuals = coincidences[~coincidences['true']].reset_index()
        actual_nums = randoms.coincidences_per_lor(actuals['detector1'], actuals['detector2'], detectors[-1] + 1)

        sp.append(np.sum(sp_nums) / 2.0)
        sp_corr.append(np.sum(sp_corrected) / 2.0)
        dw.append(np.sum(dw_nums) / 2.0)
        sr.append(np.sum(sr_nums) / 2.0)
        actual.append(len(actuals))
        multis.append(len(multi_coins))
        total.append(len(coincidences))

        if CONT_LIST:
            coin_per_lor = randoms.coincidences_per_lor(coincidences['detector1'], coincidences['detector2'], detectors[-1])
            coincidences['lor_coins'] = coincidences.apply(lambda x: coin_per_lor[x['detector1']][x['detector2']], axis=1)
            coincidences['sp_cont'] = coincidences.apply(lambda x: sp_nums[x['detector1']][x['detector2']], axis=1) / coincidences['lor_coins']
            coincidences['dw_cont'] = coincidences.apply(lambda x: dw_nums[x['detector1']][x['detector2']], axis=1) / coincidences['lor_coins']
            coincidences['sr_cont'] = coincidences.apply(lambda x: sr_nums[x['detector1']][x['detector2']], axis=1) / coincidences['lor_coins']
            coincidences['actual_cont'] = coincidences.apply(lambda x: actual_nums[x['detector1']][x['detector2']], axis=1) / coincidences['lor_coins']
            df = pd.DataFrame({'sp': coincidences['sp_cont'], 'dw': coincidences['dw_cont'], 'sr': coincidences['sr_cont'], 'actual': coincidences['actual_cont']})
            print("Writing list")
            with open(f'{LIST_DIRECTORY}list{i}.csv', 'w') as f:
                df.to_csv(f)
        
        # Step 5: Return results
        # actual.append(len(coincidences[~coincidences['true']]))
        print(f"File {str(i)} processed. SP: {sp[-1]}, SP_CORR: {sp_corr[-1]}, DW: {dw[-1]}, SR: {sr[-1]}, Actual: {actual[-1]}, Multis: {multis[-1]}, Total: {total[-1]}")

    if not CONT_LIST:
        # store the collected data
        df = pd.DataFrame({'sp': sp, 'dw': dw, 'sr': sr, 'actual': actual, 'multis': multis, 'total': total})
        with open(OUTPUT_FILE, 'w') as f:
            df.to_csv(f)
    
