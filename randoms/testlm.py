import numpy as np

filename = '/scratch/users/eeganr/flangelm/delay1.lm'

with open(filename, 'rb') as f:
    buffer = np.fromfile(f, dtype=np.float32)
    buffer = buffer.reshape(-1, 10)