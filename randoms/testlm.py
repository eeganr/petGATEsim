import numpy as np

with open('/scratch/users/eeganr/test.lm', 'rb') as f:
    buffer = np.fromfile(f, dtype=np.float32)
    buffer = buffer.reshape(-1, 10)