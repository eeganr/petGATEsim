import randoms
import os

DETECTORS_SIM = 12288
DETECTORS_REAL = 13824

PATH_PREFIX = '/scratch/groups/cslevin/eeganr/'
FOLDER = 'crc_raw'
NAME = 'crc'

outfolder = f'{PATH_PREFIX}{FOLDER}/split/'

os.makedirs(outfolder[:-1])

infile = f'{PATH_PREFIX}{FOLDER}/{NAME}_delay.lm'
randoms.split_lm(infile, outfolder, 'delay', DETECTORS_SIM)

infile = f'{PATH_PREFIX}{FOLDER}/{NAME}_actual.lm'
randoms.split_lm(infile, outfolder, 'actual', DETECTORS_SIM)

infile = f'{PATH_PREFIX}{FOLDER}/{NAME}.lm'
randoms.split_lm(infile, outfolder, 'coin', DETECTORS_SIM)
