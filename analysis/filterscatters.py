import randoms
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, default='', help="folder in group scratch")
parser.add_argument("-n", "--name", type=str, default='annulus', help="name of inputs")
parser.add_argument("-r", "--real", action="store_true", help="uses real detector indices")
args = parser.parse_args()

FOLDER = '/scratch/groups/cslevin/' + args.folder + '/'
NAME = args.name

randoms.remove_scatters(FOLDER + NAME + '.lm', FOLDER + NAME + '_noscatter.lm')