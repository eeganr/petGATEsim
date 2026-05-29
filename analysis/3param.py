import randoms
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, default='', help="folder in group scratch")
parser.add_argument("-n", "--name", type=str, default='annulus', help="name of inputs")
parser.add_argument("-r", "--real", action="store_true", help="uses real detector indices")
args = parser.parse_args()

FOLDER = '/scratch/groups/cslevin/' + args.folder + '/'
NAME = args.name

# crystal_map = np.load('convert_768-864.npy')[:, 1]

crystal_map = np.arange(864*16)

randoms.lm_to_threeparam(FOLDER + NAME + '.lm', FOLDER + NAME + '.tp', crystal_map)