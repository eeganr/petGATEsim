import numpy as np
import randoms

DETECTORS_SIM = 12288
DETECTORS_REAL = 13824

infile = '/scratch/groups/cslevin/eeganr/gen2annulus3/annulus_delay.lm'

outfolder = '/scratch/groups/cslevin/eeganr/gen2annulus3/split/'

randoms.split_lm(infile, outfolder, 'delay', DETECTORS_SIM)

infile = '/scratch/groups/cslevin/eeganr/gen2annulus3/annulus_actual.lm'

randoms.split_lm(infile, outfolder, 'actual', DETECTORS_SIM)

infile = '/scratch/groups/cslevin/eeganr/gen2annulus3/annulus.lm'

randoms.split_lm(infile, outfolder, 'coin', DETECTORS_SIM)
