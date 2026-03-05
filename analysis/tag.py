import randoms as rd
import numpy as np

DETS = 768 * 16

FOLDER = '/scratch/groups/cslevin/eeganr/scatter/scatter_nocorr2/'

coins = np.load(FOLDER + 'coin_lor.npy')

coins[coins == 0] = 1  # avoid divide by zero errors, shouldn't matter though

scatters = np.load(FOLDER + 'scatters.npy')
randoms = np.load(FOLDER + 'actuals.npy')

in_lm = FOLDER + 'scatter.lm'
out_lm = FOLDER + 'scatter_tagged.lm'

s_frac = scatters / coins
r_frac = randoms / coins

s_frac = s_frac.flatten()
r_frac = r_frac.flatten()

rd.tag_listmode(in_lm, out_lm, r_frac, s_frac, DETS)