import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
# import struct
import uproot

# === CONFIG ===
TIME = 60.0  # total sim time (s)
TAU = 1.2e-8  # coincidence window (s)
DELAY = 10 * TAU  # delay for DW estimate (s)
DETECTORS = 48 * 48
PATH_PREFIX = "/scratch/users/eeganr/pastoutput/output"
PATH_SUFFIX = ".root"
FILE_RANGE = range(1, 100 + 1)
# === END CONFIG ===


# Read in data from ROOT files
def read_root_file(infile):

    # Reads appropriate data field for a given file name
    # (e.g. file name in form Singles;16 for first 16k events)
    def get_all_vals(file, name):
        num = max([int(i.split(';')[1]) for i in file.keys() if i.split(';')[0] == name])
        return file[f'{name};{num}']

    with uproot.open(infile) as file:
        singles_tree = get_all_vals(file, 'Singles')
        coincidence_tree = get_all_vals(file, 'Coincidences')

        singles = pd.DataFrame({
            "time": singles_tree["time"].array(library="np"),
            "detector": singles_tree["crystalID"].array(library="np"),
            "source": list(map(tuple, np.stack((
                singles_tree["sourcePosX"].array(library="np"),
                singles_tree["sourcePosY"].array(library="np"),
                singles_tree["sourcePosZ"].array(library="np")), axis=-1))),
        })
        coincidences = pd.DataFrame({
            "time1": coincidence_tree["time1"].array(library="np"),
            "time2": coincidence_tree["time2"].array(library="np"),
            "detector1": coincidence_tree["crystalID1"].array(library="np"),
            "detector2": coincidence_tree["crystalID2"].array(library="np"),
            "source1": list(map(tuple, np.stack((
                coincidence_tree["sourcePosX1"].array(library="np"),
                coincidence_tree["sourcePosY1"].array(library="np"),
                coincidence_tree["sourcePosZ1"].array(library="np")), axis=-1))),
            "source2": list(map(tuple, np.stack((
                coincidence_tree["sourcePosX2"].array(library="np"),
                coincidence_tree["sourcePosY2"].array(library="np"),
                coincidence_tree["sourcePosZ2"].array(library="np")), axis=-1))),
        })
        coincidences['true'] = coincidences['source1'] == coincidences['source2']

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
    return singles, coincidences, singles_count, prompts_count


# Returns the Singles-Prompts (SP) randoms rate estimate from a pair
# of detectors with crystalIDs i and j per Oliver & Rafecas
def randomsrate(i, j, singles_count, prompts_count, L, S):
    # P_i and P_j are prompts rates for detectors i, j
    P_i = prompts_count.get(i, 0) / TIME
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


# Calculate Singles-Prompts (SP) estimate of total randoms rate
def singles_prompts(singles_count, prompts_count, singles, coincidences):

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
            np.searchsorted(singles['time'], t + TAU)
        )
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
        singles, coincidences, singles_count, prompts_count = read_root_file(infile)
        sp.append(singles_prompts(singles_count, prompts_count, singles, coincidences))
        dw.append(delayed_window(singles))
        sr.append(singles_rate(singles_count))
        actual.append(len(coincidences[coincidences['true'] is False]))

    df = pd.DataFrame({'sp': sp, 'dw': dw, 'sr': sr, 'actual': actual})
    with open('estimations.csv', 'w') as f:
        df.to_csv(f)
