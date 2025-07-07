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
# === END CONFIG ===

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filter", action="store_true", help="whether to filter energy")
parser.add_argument("-p", "--path", type=str, default="june23output/output", help="path prefix for input files")
parser.add_argument("-o", "--output", type=str, default="estimations.csv", help="output file name")
parser.add_argument("-t", "--time", type=float, default=1.0, help="total simulation time in seconds")
args = parser.parse_args()

PATH_PREFIX = PATH_PRE_PREFIX + args.path
FILTER_ENERGY = args.filter
OUTPUT_FILE = args.output
TIME = float(args.time)
if FILTER_ENERGY:
    print("Filtering singles by energy in range 0.450 to 0.750 MeV")
print(f"Output will be written to {OUTPUT_FILE}")


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
        # singles = singles.sort_values(by=['time'])

    detectors = np.sort(np.unique(singles['detector']))

    return singles, detectors


# Main function to read files and calculate estimates for many files of form
# output1.root, output2.root, ..., outputN.root
# Range [1, N) defined in FILE_RANGE
# Writes results to estimations.csv
if __name__ == "__main__":
    sp, dw, sr, actual = [], [], [], []
    for i in FILE_RANGE:
        # Step 1: Read file
        infile = PATH_PREFIX + str(i) + PATH_SUFFIX
        print(f"Reading file {infile}...")
        singles, detectors = read_root_file(infile)

        # Step 1.5: Filter hits by energy if needed
        if FILTER_ENERGY:
            singles = filter_singles(singles)  # Filter singles by energy
            print('Filtered energy')

        # Step 2: Bundle coincidences
        coincidences = bundle_coincidences(singles)  # Bundle singles into coincidences

        # Step 3: Tally stats by detector
        singles_count = randoms.singles_counts(singles['detector'], detectors[-1])
        prompts_count = randoms.prompts_counts(coincidences['detector1'], coincidences['detector2'], detectors[-1])

        # Step 4: Calculate estimation methods
        sp.append(singles_prompts(singles_count, prompts_count, singles, coincidences, detectors, TIME))
        dw.append(delayed_window(singles, detectors))
        sr.append(singles_rate(singles_count, detectors, TIME))
        actual.append(len(coincidences[~coincidences['true']]))
        
        # Step 5: Return results
        # actual.append(len(coincidences[~coincidences['true']]))
        print(f"File {str(i)} processed. SP: {sp[-1]}, DW: {dw[-1]}, SR: {sr[-1]}, Actual: {actual[-1]}")

    df = pd.DataFrame({'sp': sp, 'dw': dw, 'sr': sr, 'actual': actual})
    with open(OUTPUT_FILE, 'w') as f:
        df.to_csv(f)
