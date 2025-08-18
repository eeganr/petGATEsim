import pandas as pd
import numpy as np
import randoms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infolder", type=str, help="in folder for split data")
parser.add_argument("-n", "--name", type=str, help="name of estimation method")
args = parser.parse_args()

IN_FOLDER = args.infolder
MODULES = 16
name = args.name

print('Concating dfs')

dfs = []

for i in range(MODULES):
    for j in range(i + 1, MODULES):
        dfs.append(pd.read_pickle(f'{IN_FOLDER}stats/{i}_{j}_stats.pkl'))

df = pd.concat(dfs)

df.to_pickle(f'{IN_FOLDER}total_stats.pkl')

print('Combined')

print(f'Combining {name}s')
for i in range(MODULES):
    for j in range(i + 1, MODULES):
        print(f'Combining {i}-{j}')
        randoms.combine_lm(f'{IN_FOLDER}{i}_{j}_{name}corr.lm', f'{IN_FOLDER}{name}corr.lm')