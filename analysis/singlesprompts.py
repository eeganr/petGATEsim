import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
# import struct
import uproot

# === CONFIG ===
INFILE = "/scratch/users/eeganr/pastoutput/output1.root"

TIME = 60.0 # total sim time (s)
TAU = 1.2e-8 # coincidence window (s)
DELAY = TAU # delay for DW estimate (s)
num_detectors = 48 * 48


# Read in data from ROOT files

def get_all_vals(file, name):
    num = max([int(i.split(';')[1]) for i in file.keys() if i.split(';')[0] == name])
    return file[f'{name};{num}']

with uproot.open(INFILE) as file:
    singles_tree = get_all_vals(file, 'Singles')
    coincidence_tree = get_all_vals(file, 'Coincidences')

    singles = pd.DataFrame({
        "time": singles_tree["time"].array(library="np"),
        "detector": singles_tree["crystalID"].array(library="np"),
        "source": list(map(tuple, np.stack((singles_tree["sourcePosX"].array(library="np"), 
                            singles_tree["sourcePosY"].array(library="np"), 
                            singles_tree["sourcePosZ"].array(library="np")), axis=-1))),
    })
    coincidences = pd.DataFrame({
        "time1": coincidence_tree["time1"].array(library="np"),
        "time2": coincidence_tree["time2"].array(library="np"),
        "detector1": coincidence_tree["crystalID1"].array(library="np"),
        "detector2": coincidence_tree["crystalID2"].array(library="np"),
        "source1": list(map(tuple, np.stack((coincidence_tree["sourcePosX1"].array(library="np"), 
                            coincidence_tree["sourcePosY1"].array(library="np"), 
                            coincidence_tree["sourcePosZ1"].array(library="np")), axis=-1))),
        "source2": list(map(tuple, np.stack((coincidence_tree["sourcePosX2"].array(library="np"), 
                            coincidence_tree["sourcePosY2"].array(library="np"), 
                            coincidence_tree["sourcePosZ2"].array(library="np")), axis=-1))),
    })
    coincidences['true'] = coincidences['source1'] == coincidences['source2']


# Define Whole-System Equation Constants for SP Method
S = singles_tree.num_entries / TIME # Rate of singles measured by scanner as a whole
P = 2 * coincidence_tree.num_entries / TIME # Twice the prompts rate


# Roots of this function are the lambda (L) values.
def lambda_eq(L):
    return 2 * TAU * L * L - L + S - P * np.exp((L + S)*TAU)
L = root_scalar(lambda_eq, x0=0)
if not L.converged:
    raise RuntimeError("Failed to converge on lambda.")
L = L.root

det1_counts = coincidences['detector1'].value_counts().to_dict() # det1 coincidences involved
det2_counts = coincidences['detector2'].value_counts().to_dict() # det2 coincidences involved
prompts = pd.DataFrame({'detector': list(range(num_detectors))})
prompts['prompts'] = prompts['detector'].map(lambda x: det1_counts.get(x, 0) + det2_counts.get(x, 0))
prompts_count = prompts.set_index('detector')['prompts'].to_dict()
singles_count = singles['detector'].value_counts().to_dict()

# Returns the randoms rate from a pair of detectors with crystalIDs i and j
def randomsrate(i, j, singles_count, prompts_count):
    P_i = prompts_count.get(i, 0) / TIME
    P_j = prompts_count.get(j, 0) / TIME
    S_i = singles_count.get(i, 0) / TIME
    S_j = singles_count.get(j, 0) / TIME
    coeff = (2 * TAU * np.exp(-(L + S)*TAU))/((1 - 2 * L * TAU)**2)
    i_term = S_i - np.exp((L + S)*TAU) * P_i
    j_term = S_j - np.exp((L + S)*TAU) * P_j
    return coeff * i_term * j_term

# Calculate singles-prompts estimate of total randoms rate
def singles_prompts(singles_count, prompts_count):
    sp_rate = 0
    for i in range(num_detectors):
        for j in range(i, num_detectors):
            sp_rate += randomsrate(i, j, singles_count, prompts_count)
    return sp_rate * TIME

# Calculate delayed-window estimate of total randoms rate
def delayed_window(singles):
    dw_estimate = 0
    for t in singles['time']:
        dw_estimate += np.searchsorted(singles['time'], t + DELAY + TAU) - np.searchsorted(singles['time'], t + TAU)
    return dw_estimate

# Calculate singles-rate estimate of total randoms rate
def singles_rate(singles_count):
    sr_rate = 0
    for i in range(num_detectors):
        for j in range(i, num_detectors):
            sr_rate += 2 * TAU * singles_count.get(i, 0) / TIME * singles_count.get(j, 0) / TIME
    return sr_rate * TIME

# Find actual randoms in the data

print("Calculating singles-prompts estimate of total randoms rate...")
print(singles_prompts(singles_count, prompts_count))
print("Calculating delayed-window estimate of total randoms rate...")
print(delayed_window(singles))
print("Calculating singles-rate estimate of total randoms rate...")
print(singles_rate(singles_count))
print("Finding actual randoms in the data...")
actual = coincidences[coincidences['true'] == False]
