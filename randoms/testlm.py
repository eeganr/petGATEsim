import numpy as np

filename = '/scratch/groups/cslevin/eeganr/gen2annulus/annulus.lm'

with open(filename, 'rb') as f:
    buffer = np.fromfile(f, dtype=np.float32)
    buffer = buffer.reshape(-1, 10)