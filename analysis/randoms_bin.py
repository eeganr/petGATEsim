import numpy as np
from ClassSingles8Bytes import Singles
import os
import argparse
import randoms
import time
from randomsutils import *
import pandas as pd
import randoms

COARSE_TIME = 1.6e-9
FINE_TIME = 50e-12

DAT_DIR = '/scratch/users/eeganr/realpointsource/'
filename = DAT_DIR + 'Na22_250uCi_Single_06212024_minus5_PETA1_ch0_C_PCIe.dat'

with open(filename, 'rb') as f:
    bytesInFile = os.path.getsize(filename)

    if (not bytesInFile % 8 == 0):
        raise Exception("File not stored properly.")

    num_events = bytesInFile // 8 # separate into bytes
    rawdata = np.fromfile(f, np.uint8)
    rawdata = np.reshape(rawdata, (num_events, 8))

    data = Singles(rawdata, keephighestenergy=False)

print("Finished initial processing.")

t = time.time()

times = randoms.get_times(data.coarse, data.fine, COARSE_TIME, FINE_TIME)

singles = pd.DataFrame()
singles['time'] = times
singles['detector'] = data.crystalID
singles['energy'] = data.energy

print("Done creating df:", time.time() - t)

# singles = singles.sort_values(by=['time'])

TIME = singles['time'].iloc[-1]



t = time.time()

coincidences = bundle_coincidences(singles)

detectors = np.sort(np.unique(singles['detector']))

singles_count = randoms.singles_counts(singles['detector'], detectors[-1])
prompts_count = randoms.prompts_counts(coincidences['detector1'], coincidences['detector2'], detectors[-1])

print("Finished bundling and counting:", time.time() - t)

sp, dw, sr = [], [], []

t = time.time()

sp.append(singles_prompts(singles_count, prompts_count, singles, coincidences, detectors, TIME))
print("Finished sp:", time.time() - t)
t = time.time()
dw.append(delayed_window(singles, detectors))
print("Finished dw:", time.time() - t)
t = time.time()
sr.append(singles_rate(singles_count, detectors, TIME))
print("Finished sr:", time.time() - t)

print(sp, dw, sr)