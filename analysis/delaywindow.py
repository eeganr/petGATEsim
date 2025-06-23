import numpy as np
import pandas as pd
# from scipy.optimize import root_scalar
# import struct
import uproot

# === CONFIG ===
TIME = 60.0  # total sim time (s)
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
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


# Calculate delayed-window estimate of total randoms rate
def delayed_window(coincidences, singles):
    dw_estimate = 0
    for t in coincidences['time1']:
        dw_estimate += (
            np.searchsorted(singles['time'], t + DELAY + TAU) -
            np.searchsorted(singles['time'], t + DELAY)
        )  # Num of singles in delayed window
    return dw_estimate


# Calculate delayed-window estimate of total randoms rate (test 2)
def delayed_window2(coincidences, singles):
    dw_estimate = 0
    for t in coincidences['time1']:
        dw_estimate += (
            np.searchsorted(coincidences['time1'], t + DELAY + TAU) -
            np.searchsorted(coincidences['time1'], t + DELAY)
        )  # Num of coincidences in delayed window
    return dw_estimate


# Main function to read files and calculate estimates for many files of form
# output1.root, output2.root, ..., outputN.root
# Range [1, N) defined in FILE_RANGE
# Writes results to estimations.csv
if __name__ == "__main__":
    dw, dw2 = [], [], [], []
    for i in FILE_RANGE:
        infile = PATH_PREFIX + str(i) + PATH_SUFFIX
        print(f"Reading file {infile}...")
        singles, coincidences, singles_count, prompts_count = read_root_file(infile)
        dw.append(delayed_window(coincidences, singles))
        dw2.append(delayed_window2(coincidences, singles))

    df = pd.DataFrame({'dwa': dw, 'dwa2': dw2})
    with open('estimations2.csv', 'w') as f:
        df.to_csv(f)
