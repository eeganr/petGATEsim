import randoms
import numpy as np

file1 = '/scratch/groups/cslevin/eeganr/crc/crc_nocorr/split/0_2_coin.lm'
file2 = '/scratch/groups/cslevin/eeganr/crc/crc_corr/split/0_2_delaycorr.lm'

x = np.memmap(file2, dtype=np.float32).reshape(-1, 10)

# bigfolder = '/scratch/groups/cslevin/eeganr/crc/crc_corr'
#splitfolder = bigfolder + '/split'

# records1 = 0

# name = 'sp'

# for i in range(0, 15):
#     for j in range(i+1, 16):
#         records1 += np.memmap(f'{splitfolder}/{i}_{j}_{name}corr.lm', dtype=np.float32).shape[0] / 10

# records2 = np.memmap(f'{bigfolder}/{name}corr.lm', dtype=np.float32).shape[0] / 10

# print(records1, records2)