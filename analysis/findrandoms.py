import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import uproot
import argparse
import randoms


# === CONFIG ===
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
DETECTORS = 48 * 48
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
OUTPUT_FILE = args.output + ".csv"
TIME = args.time
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

    return singles


def filter_singles(singles, energy_min=0.450, energy_max=0.750):
    """
        Energy-filter singles, parameters are in MeV, defaults 0.450 to 0.750 MeV
        This is the energy range of the 511 keV gamma photons from positron annihilation
        This is the range used in Oliver & Rafecas
    """
    singles = singles[singles['energy'] > energy_min].reset_index()
    singles = singles[singles['energy'] < energy_max].reset_index()
    return singles


def bundle_coincidences(singles):
    """
        Bundle singles into coincidences, returns a DataFrame with columns:
        time1, time2, detector1, detector2, source1, source2, true
        where true is True if the two singles are from the same source, (true coincidence)
        False otherwise
    """
    import time
    t = time.time()
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
    print("Time taken:", time.time() - t)
    return coinci


def sp_counts(singles, coincidences):
    """
        Calculate Singles-Prompts (SP) counts for singles and coincidences
        Returns a tuple of two dicts: singles_count, prompts_count
        singles_count is a dict of form {detectorID: count} for singles
        prompts_count is a dict of form {detectorID: count} for prompts
        where prompts are the sum of all coincidences involving a given detector
    """
    # Calculate singles/prompts counts
    det1_counts = coincidences['detector1'].value_counts().to_dict()  # det1 coincidences involved
    det2_counts = coincidences['detector2'].value_counts().to_dict()  # det2 coincidences involved

    # Prompts are the sum of all coincidences involving a given detector,
    # regardless of first or second
    # Every detector, regardless of presence, has a key
    prompts = pd.DataFrame({'detector': list(range(DETECTORS))})
    prompts['prompts'] = prompts['detector'].map(
        lambda x: det1_counts.get(x, 0) +
        det2_counts.get(x, 0)
    )

    # dicts of form {detectorID: count}
    prompts_count = prompts.set_index('detector')['prompts'].to_dict()
    singles_count = singles['detector'].value_counts().to_dict()

    return singles_count, prompts_count


def randomsrate(i, j, singles_count, prompts_count, L, S):
    """
        Returns the Singles-Prompts (SP) randoms rate estimate from a pair
        of detectors with crystalIDs i and j per Oliver & Rafecas
    """
    # P_i and P_j are prompts rates for detectors i, j
    P_i = prompts_count.get(i, 0) / TIME  # .get(i, 0) returns 0 if det i not in prompts_count
    P_j = prompts_count.get(j, 0) / TIME
    # S_i and S_j are singles rates for detectors i, j
    S_i = singles_count.get(i, 0) / TIME
    S_j = singles_count.get(j, 0) / TIME
    # Coefficient for the randoms rate equation
    coeff = (2 * TAU * np.exp(-(L + S)*TAU))/((1 - 2 * L * TAU)**2)
    # Further terms in SP equation
    i_term = S_i - np.exp((L + S)*TAU) * P_i
    j_term = S_j - np.exp((L + S)*TAU) * P_j
    return coeff * i_term * j_term


def singles_prompts(singles_count, prompts_count, singles, coincidences):
    """
        Calculate Singles-Prompts (SP) estimate of total randoms rate
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

    # Calculate the Singles-Prompts rate estimate for the whole scanner
    # summing over all pairs of detectors
    sp_rate = 0
    for i in range(DETECTORS):
        for j in range(i, DETECTORS):
            sp_rate += randomsrate(i, j, singles_count, prompts_count, L, S)
    return sp_rate * TIME


# Calculate delayed-window estimate of total randoms rate
def delayed_window(singles):
    dw_estimate = 0
    for t in singles['time']:
        dw_estimate += (
            np.searchsorted(singles['time'], t + DELAY + TAU) -
            np.searchsorted(singles['time'], t + DELAY)
        )  # Num of singles in delayed window
    return dw_estimate


# Calculate singles-rate estimate of total randoms rate
def singles_rate(singles_count):
    sr_rate = 0
    for i in range(DETECTORS):
        for j in range(i, DETECTORS):
            sr_rate += 2 * TAU * singles_count.get(i, 0) / TIME * singles_count.get(j, 0) / TIME
    return sr_rate * TIME


# Main function to read files and calculate estimates for many files of form
# output1.root, output2.root, ..., outputN.root
# Range [1, N) defined in FILE_RANGE
# Writes results to estimations.csv
if __name__ == "__main__":
    sp, dw, sr, actual = [], [], [], []
    for i in FILE_RANGE:
        infile = PATH_PREFIX + str(i) + PATH_SUFFIX
        print(f"Reading file {infile}...")
        singles = read_root_file(infile)

        if FILTER_ENERGY:
            singles = filter_singles(singles)  # Filter singles by energy

        coincidences = bundle_coincidences(singles)  # Bundle singles into coincidences
        assert False
        singles_count, prompts_count = sp_counts(singles, coincidences)

        sp.append(singles_prompts(singles_count, prompts_count, singles, coincidences))
        dw.append(delayed_window(singles))
        sr.append(singles_rate(singles_count))
        actual.append(len(coincidences[~coincidences['true']]))
        print(f"File {str(i)} processed. SP: {sp[-1]}, DW: {dw[-1]}, SR: {sr[-1]}, Actual: {actual[-1]}")

    df = pd.DataFrame({'sp': sp, 'dw': dw, 'sr': sr, 'actual': actual})
    with open(OUTPUT_FILE, 'w') as f:
        df.to_csv(f)
