import numpy as np
import uproot
import pandas as pd
from scipy.optimize import root_scalar

# === CONFIG ===
input_root_file = "pastoutput/output2.root"
cont_magnitude = 1e-5
num_TOF_bins = 9
TOF_bin_width = 29
sigma_TOF = 60
num_iterations = 2
num_subsets = 8

total_time = 1 # total sim time (s)
TAU = 1.2e-8 # coincidence window (s)

image_shape = (200, 200, 700)  # (x, y, z) voxels # originally (310, 310, 310)
voxel_size = (0.1, 0.1, 0.1)  #mm #originally (1,1,1)
radius_mm = 10 # orinally 130 (mm)
use_tof = True # originally False

# Read in data from ROOT files

def get_all_vals(file, name):
    num = max([int(i.split(';')[1]) for i in file.keys() if i.split(';')[0] == name])
    return file[f'{name};{num}']

with uproot.open(input_root_file) as file:
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

# System-Wide Equation Constants

S = singles_tree.num_entries / total_time # Rate of singles measured by scanner as a whole
P = 2 * coincidence_tree.num_entries / total_time # Twice the prompts rate


# Roots of this function are the lambda (L) values.
def lambda_eq(L):
    return 2 * TAU * L * L - L + S - P * np.exp((L + S)*TAU)

L = root_scalar(lambda_eq, x0=0)
if not L.converged:
    raise RuntimeError("Failed to converge on lambda.")
L = L.root

def randomsrate(i, j):
    P_i = len(coincidences[coincidences['detector1'] == i]) + len(coincidences[coincidences['detector2'] == i])
    P_j = len(coincidences[coincidences['detector1'] == j]) + len(coincidences[coincidences['detector2'] == j])
    S_i = len(singles[singles['detector'] == i])
    S_j = len(singles[singles['detector'] == j])
    coeff = (2 * TAU * np.exp(-(L + S)*TAU))/((1 - 2 * L * TAU)**2)
    i_term = S_i - np.exp((L + S)*TAU) * P_i
    j_term = S_j - np.exp((L + S)*TAU) * P_j
    return coeff * i_term * j_term

total = 0
detectors = singles['detector'].unique()
print(len(detectors))
for i in range(len(detectors)):
    if (i + 1) % 10 == 0:
        print(f"Processing detector {i + 1}/{len(detectors)}")
    for j in range(i, len(detectors)):
        total += randomsrate(detectors[i], detectors[j])

print(total)

