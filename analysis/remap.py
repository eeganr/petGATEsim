import randoms
import os
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--a", type=str, default='', help="first path")
parser.add_argument("-b", "--b", type=str, default='', help="second path")
args = parser.parse_args()

with open('geometry.pickle', 'rb') as f:
    geo = pickle.load(f)

geo = geo.flatten()

crystal_map = np.load('convert_768-864.npy')[:, 1]

randoms.remap_lm(args.a, args.b, crystal_map, geo)