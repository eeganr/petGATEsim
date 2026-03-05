import randoms
import numpy as np
from multiprocessing import Pool

CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
FOLDER = '/scratch/groups/cslevin/eeganr/scatter/scatter_singles/output'
total = np.zeros((768*16, 768*16))
files = []
for i in range(1, 121):
    file = FOLDER + str(i) + 'Singles.dat'
    files += file

def tally(file):
    return randoms.tally_scatters(file, 768*16, TAU)

print('hi')
with Pool(120) as p:
    print('here')
    totals = p.map(tally, files)

for t in totals:
    total += t

np.save('/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr/scatters.npy', total)