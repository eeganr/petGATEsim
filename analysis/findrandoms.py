import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import uproot
import argparse
import randoms
import time


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
parser.add_argument("-t", "--time", type=float, default=60.0, help="total simulation time in seconds")
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

    detectors = np.sort(np.unique(singles['detector']))

    return singles, detectors


def filter_singles(singles, energy_min=0.450, energy_max=0.750):
    """  Filter singles by energy.
        Args:
            singles: DataFrame with columns:
                time, detector, source, energy
            energy_min: minimum energy in MeV (default 0.450)
            energy_max: maximum energy in MeV (default 0.750)
        Returns:
            DataFrame with filtered singles, same columns as input
        Notes:
            Energy-filter singles, parameters are in MeV, defaults 0.450 to 0.750 MeV
            This is the energy range of the 511 keV gamma photons from positron annihilation
            This is the range used in Oliver & Rafecas
    """
    
    singles = singles[singles['energy'] > energy_min].reset_index()
    singles = singles[singles['energy'] < energy_max].reset_index()
    return singles


def bundle_coincidences(singles):
    """ Bundle singles into coincidences, 
        Args:
            singles: DataFrame with columns:
                time, detector, source, energy
            where time is in seconds, detector is the detector ID,
            source is a tuple of (x, y, z) coordinates of the source,
            and energy is the energy of the single event in MeV.
        Returns: 
            DataFrame with columns:
            time1, time2, detector1, detector2, source1, source2, true
            where true is True if the two singles are from the same source (true coincidence)
            False otherwise
    """

    times = np.array(singles['time'])
    energies = np.array(singles['energy'])
    coin_indices = randoms.bundle_coincidences(times, energies, TAU)
    coins = coin_indices.reshape(-1, 2)

    coinci = pd.DataFrame()
    coinci['time1'] = [singles['time'][i[0]] for i in coins]
    coinci['time2'] = [singles['time'][i[1]] for i in coins]
    coinci['detector1'] = [singles['detector'][i[0]] for i in coins]
    coinci['detector2'] = [singles['detector'][i[1]] for i in coins]
    coinci['source1'] = [singles['source'][i[0]] for i in coins]
    coinci['source2'] = [singles['source'][i[1]] for i in coins]
    coinci['true'] = coinci['source1'] == coinci['source2']
    return coinci


def singles_prompts(singles_count, prompts_count, singles, coincidences, detectors):
    """ Calculate the Singles-Prompts rate estimate for the whole scanner
        Args:
            singles_count: array of singles counts per detector
            prompts_count: array of prompts counts per detector pair
            singles: DataFrame with columns:
                time, detector, source, energy
            coincidences: DataFrame with columns:
                time1, time2, detector1, detector2, source1, source2, true
            detectors: array of detector IDs
        Returns:
            Singles-Prompts rate estimate for the whole scanner
    """

    S = len(singles) / TIME  # Rate of singles measured by scanner as a whole
    P = 2 * len(coincidences) / TIME  # Twice the prompts rate

    # Roots of this function are the lambda (L) values needed for the SP estimate.
    def lambda_eq(L):
        return 2 * TAU * L * L - L + S - P * np.exp((L + S)*TAU)
    L = root_scalar(lambda_eq, x0=0)
    if not L.converged:
        raise RuntimeError("Failed to converge on lambda.")
    L = L.root
    
    sp_rates = randoms.sp_rates(singles_count, prompts_count, detectors[-1], L, S, TAU, TIME)
    
    # Calculate the Singles-Prompts rate estimate for the whole scanner
    # summing over all pairs of detectors
    return np.sum(sp_rates) * TIME / 2.0 # 2.0 because summing over the array double counts


def singles_rate(singles_count, detectors):
    """ Calculate the Singles Rate estimate for the whole scanner
        Args:
            singles_count: array of singles counts per detector
            detectors: array of detector IDs
        Returns:
            Singles Rate estimate for the whole scanner
    """

    sr_rates = randoms.sr_rates(singles_count, detectors[-1], TAU, TIME)
    return np.sum(sr_rates) * TIME / 2.0 # 2.0 because summing over the array double counts


# Main function to read files and calculate estimates for many files of form
# output1.root, output2.root, ..., outputN.root
# Range [1, N) defined in FILE_RANGE
# Writes results to estimations.csv
if __name__ == "__main__":
    sp, dw, sr, actual = [], [], [], []
    for i in FILE_RANGE:
        infile = PATH_PREFIX + str(i) + PATH_SUFFIX
        print(f"Reading file {infile}...")
        singles, detectors = read_root_file(infile)

        if FILTER_ENERGY:
            singles = filter_singles(singles)  # Filter singles by energy

        t = time.time()
        coincidences = bundle_coincidences(singles)  # Bundle singles into coincidences
        print("bundling complete: ", time.time() - t)
        t = time.time()
        singles_count = randoms.singles_counts(singles['detector'], detectors[-1])
        prompts_count = randoms.prompts_counts(coincidences['detector1'], coincidences['detector2'], detectors[-1])
        print("counting complete: ", time.time() - t)
        t = time.time()
        sp.append(singles_prompts(singles_count, prompts_count, singles, coincidences, detectors))
        print("Finished SP: ", time.time() - t)

        t = time.time()
        dw.append(randoms.delayed_window(np.array(singles['time']), TAU, DELAY))
        print("Finished DW: ", time.time() - t)

        t = time.time()
        sr.append(singles_rate(singles_count, detectors))
        print("Finished SR: ", time.time() - t)
        
        actual.append(len(coincidences[~coincidences['true']]))
        print(f"File {str(i)} processed. SP: {sp[-1]}, DW: {dw[-1]}, SR: {sr[-1]}, Actual: {actual[-1]}")

    df = pd.DataFrame({'sp': sp, 'dw': dw, 'sr': sr, 'actual': actual})
    with open(OUTPUT_FILE, 'w') as f:
        df.to_csv(f)
