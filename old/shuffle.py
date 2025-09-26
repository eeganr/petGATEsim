import numpy as np
import randoms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infolder", type=str, help="in folder for split data")
parser.add_argument("-n", "--name", type=str, help="name of estimation method")
args = parser.parse_args()

IN_FOLDER = args.infolder
name = args.name

a = f'{IN_FOLDER}{name}corr.lm'
b = f'{IN_FOLDER}{name}corrshuff.lm'

randoms.shuffle_lm(a, b)