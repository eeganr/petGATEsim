"""
Code name:
    analyzeDataSingles.py


Usage:
    python3 analyzeData.py dir_origin(folder that contains the data files and command.txt)


Author:
    Sarah Zou (edited on 2024.12.18 to accommodate triple coincidences)
    Original: Chen-Ming Chang, Qian Dong


Purpose:
    This code is to group coincidences from data acquired by
    PETcoil in Singles aquisition.  It can be used to correct for intercrystal scatter in single aquisition mode. 
    Depending on energy and timing is already calibrated or these files are provided
    The code will output PETA folders with coincidneceDouble and coincidenceTriple files.

Required Python modules:
    1. Included in Python 3.6.5:
        sys, array, pickle

    2. Need to install:
        numpy, matplotlib, peakutils, scipy

Required libraries:
    analyzeTiming.py
"""

import pickle
from array import array
import os
import numpy as np
from analyzeTimingSingles import *
import argparse
import json

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument(dest='dir_origin', help='Path of a folder')          
args = parser.parse_args()

dir_origin = args.dir_origin

with open(args.dir_origin+'/command.txt', 'r') as f:
    print('loading command.txt')
    args.__dict__ = json.load(f)
    args.dir_origin = dir_origin
print(args)    
filelist = os.listdir(args.dir_origin)

filelist = [f for f in filelist if '_PETA' in f]
filelist.sort(key=lambda f: int(f.replace('PETA').split('_PETA')[1].split('_')[0].split('.')[0]))
print(filelist)

if args.Reverse:
    filelist = filelist[::-1]
    

for fname in filelist:
    # process all '.dat' files including 'PETA'
    if 'PETA' in fname and fname[-4:] == '.dat' and args.Denable:
        print('************************************************')
        print(args.dir_origin, fname)
        print('************************************************')
        dir1 = args.dir_origin + '/' + fname.split('_PETA')[0] + '/PETA' + fname[:-4].split('_PETA')[1].split('_')[0]
        analyzetiming(args.dir_origin, fname, args)