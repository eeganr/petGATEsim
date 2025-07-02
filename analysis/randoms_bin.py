import numpy as np
from ClassSingles8Bytes import Singles
import os
import argparse
import randoms
import time

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

    singles = Singles(rawdata, keephighestenergy=False)

print("Finished initial processing.")

times = randoms.get_times(singles.coarse, singles.fine, COARSE_TIME, FINE_TIME)
