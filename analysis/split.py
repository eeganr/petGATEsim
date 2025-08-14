import randoms

DETECTORS_SIM = 12288
DETECTORS_REAL = 13824

PATH_PREFIX = '/scratch/groups/cslevin/eeganr/'
FOLDER = 'gen2annulus3'

outfolder = f'{PATH_PREFIX}{FOLDER}/split/'

infile = f'{PATH_PREFIX}{FOLDER}annulus_delay.lm'
randoms.split_lm(infile, outfolder, 'delay', DETECTORS_SIM)

infile = f'{PATH_PREFIX}{FOLDER}/annulus_actual.lm'
randoms.split_lm(infile, outfolder, 'actual', DETECTORS_SIM)

infile = f'{PATH_PREFIX}{FOLDER}/annulus.lm'
randoms.split_lm(infile, outfolder, 'coin', DETECTORS_SIM)
