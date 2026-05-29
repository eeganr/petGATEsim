import randoms
import numpy as np

lutfile = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr/6/scatter6_tagged.lm'
x = np.memmap(lutfile, dtype=np.float32).reshape(-1, 10)
# lutfile2 = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr/005/scatter005_time.lut'
# y = np.memmap(lutfile2, dtype=np.float64).reshape(-1, 4)
# s = np.load(f_s)
# r = np.load(f_r)
# c = np.load(f_c)

file1 = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr/10/scatter10.lm'
oakfile = '/oak/stanford/groups/cslevin/eeganr/cylwater/cylwat_eval/split/0_10_coin.dat'
f_s = '/scratch/groups/cslevin/eeganr/cylinder/cyl_nocorr/scatters.npy'
f_r = '/scratch/groups/cslevin/eeganr/cylinder/cyl_nocorr/actuals.npy'
f_c = '/scratch/groups/cslevin/eeganr/cylinder/cyl_nocorr/coin_lor.npy'

# bigfolder = '/scratch/groups/cslevin/eeganr/crc/crc_corr'
#splitfolder = bigfolder + '/split'

# records1 = 0

# name = 'sp'

# for i in range(0, 15):
#     for j in range(i+1, 16):
#         records1 += np.memmap(f'{splitfolder}/{i}_{j}_{name}corr.lm', dtype=np.float32).shape[0] / 10

# records2 = np.memmap(f'{bigfolder}/{name}corr.lm', dtype=np.float32).shape[0] / 10

# print(records1, records2)