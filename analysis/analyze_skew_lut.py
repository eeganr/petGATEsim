'''
Created on 2025-01-29
@author: sarahzou

Want to check skewlut file and print out disribution of the time skews
'''
import numpy as np    
import os
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Analyze skewlut file and print out distribution of the time skews.')
parser.add_argument('directory', type=str, help='Directory containing the .dat files')
args = parser.parse_args()

# Iterate over all .dat files in the directory
for filename in os.listdir(args.directory):
    if filename.endswith('.dat'):
        skewlut_path = os.path.join(args.directory, filename)

with open(skewlut_path, 'rb') as skewfile: # this is an upper triangular matrix
    skewlut = np.fromfile(skewfile, np.int16)
    skewlut = np.reshape(skewlut, (864*16, 864*16)) 

# Flatten the skewlut matrix to get all values
skewlut_values = skewlut.flatten()
skewlut_values = skewlut_values[skewlut_values != 0]

# Print min and max values
print(f"Min skew value: {np.min(skewlut_values)}")
print(f"Max skew value: {np.max(skewlut_values)}")