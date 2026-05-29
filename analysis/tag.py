import randoms as rd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, default="")
parser.add_argument("-n", "--name", type=str, default="")

args = parser.parse_args()

name = args.name

DETS = 768 * 16

FOLDER = '/scratch/groups/cslevin/eeganr/' + args.folder

coins = np.load(FOLDER + 'coin_lor.npy')

coins[coins == 0] = 1  # avoid divide by zero errors, shouldn't matter though

scatters = np.load(FOLDER + 'scatters.npy')
randoms = np.load(FOLDER + 'actuals.npy')

in_lm = FOLDER + f'{name}.lm'
out_lm = FOLDER + f'{name}_tagged.lm'

s_frac = scatters / coins
r_frac = randoms / coins

s_frac = s_frac.flatten()
r_frac = r_frac.flatten()

rd.tag_listmode_v2(in_lm, out_lm, r_frac, s_frac, DETS)