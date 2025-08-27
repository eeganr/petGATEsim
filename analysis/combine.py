import pandas as pd
import numpy as np
import randoms
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infolder", type=str, help="in folder for split data, one folder out")
parser.add_argument("-n", "--name", type=str, help="name of estimation method")
args = parser.parse_args()

IN_FOLDER = args.infolder
SPLIT_FOLDER = IN_FOLDER + 'split/'
MODULES = 16
name = args.name

print('Concating dfs')

dfs = []

for i in range(MODULES):
    for j in range(i + 1, MODULES):
        dfs.append(pd.read_pickle(f'{SPLIT_FOLDER}stats/{i}_{j}_stats.pkl'))

df = pd.concat(dfs)

df.to_pickle(f'{IN_FOLDER}total_stats.pkl')

print('Combined')

print(f'Combining {name}s')
paths = []
for i in range(MODULES - 1):
    for j in range(i + 1, MODULES):
        paths.append(f'{SPLIT_FOLDER}{i}_{j}_{name}corr.lm')
        if os.path.abspath(paths[-1]) == os.path.abspath(f'{IN_FOLDER}{name}corr.lm'):
            print("oops!")

print(len(paths))

randoms.combine_lm(paths, f'{IN_FOLDER}{name}corr.lm')