import randoms
import os
import argparse

DETECTORS_SIM = 12288
DETECTORS_REAL = 13824

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, default='eeganr/annulus/annulus_nocorr', help="folder in group scratch")
parser.add_argument("-n", "--name", type=str, default='annulus', help="name of inputs")
parser.add_argument("-r", "--real", action="store_true", help="uses real detector indices")
args = parser.parse_args()


PATH_PREFIX = '/scratch/groups/cslevin/'
FOLDER = args.folder
NAME = args.name

DETS = DETECTORS_REAL if args.real else DETECTORS_SIM

outfolder = f'{PATH_PREFIX}{FOLDER}/split/'

os.makedirs(outfolder[:-1])

infile = f'{PATH_PREFIX}{FOLDER}/{NAME}_delay.lm'
randoms.split_lm(infile, outfolder, 'delay', DETS)

infile = f'{PATH_PREFIX}{FOLDER}/{NAME}_actual.lm'
randoms.split_lm(infile, outfolder, 'actual', DETS)

infile = f'{PATH_PREFIX}{FOLDER}/{NAME}.lm'
randoms.split_lm(infile, outfolder, 'coin', DETS)
