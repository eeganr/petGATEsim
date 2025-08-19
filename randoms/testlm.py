import numpy as np
import randoms

filename = '/scratch/users/eeganr/0_1.lm'
filenameold = '/scratch/groups/cslevin/eeganr/flangeless_corr/spcorr.lm'

x = np.memmap(filenameold, dtype=np.float32)

print(x)