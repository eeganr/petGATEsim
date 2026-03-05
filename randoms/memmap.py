import randoms
import numpy as np

file1 = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr2/scatter.lut'
f_s = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr2/scatters.npy'
f_r = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr2/actuals.npy'
f_c = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr2/coin_lor.npy'

x = np.memmap(file1, dtype=np.float32).reshape(-1, 4)
s = np.load(f_s)
r = np.load(f_r)
c = np.load(f_c)


# bigfolder = '/scratch/groups/cslevin/eeganr/crc/crc_corr'
#splitfolder = bigfolder + '/split'

# records1 = 0

# name = 'sp'

# for i in range(0, 15):
#     for j in range(i+1, 16):
#         records1 += np.memmap(f'{splitfolder}/{i}_{j}_{name}corr.lm', dtype=np.float32).shape[0] / 10

# records2 = np.memmap(f'{bigfolder}/{name}corr.lm', dtype=np.float32).shape[0] / 10

# print(records1, records2)