import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, default="")

args = parser.parse_args()

FOLDER = "/scratch/groups/cslevin/eeganr/" + args.folder

coins = np.load(FOLDER + 'coin_lor.npy')
# coins = np.ones((12288, 12288))
coins[coins == 0] = 1  # avoid divide by zero errors, shouldn't matter though

scatters = np.load(FOLDER + 'scatters.npy')
randoms = np.load(FOLDER + 'actuals.npy')

r = randoms / coins
s = scatters / coins
# r = np.ones((12288, 12288)).astype('float32')
# s = np.ones((12288, 12288)).astype('float32')

with open("gategeometry.pickle", "rb") as f:
    geo = np.delete(pickle.load(f), 3, axis=1).astype('float32')

print('loaded geo')

result = np.concatenate([np.repeat(geo, len(geo), axis=0), np.tile(geo, (len(geo), 1))], axis=1).astype('float32')
result = np.column_stack((result, r.flatten(), s.flatten()))

np.save(FOLDER + 'rfsf', result)

