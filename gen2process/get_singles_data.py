#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:59:58 2024
Coincidence Processor for singles acquisition and does double and triple 
coincidence processing

@author: sarahzou
"""
"""
get_singles_data.py

Coincidence Processor for PETcoil2 singles acquisition. This script processes raw singles data files, extracts double and triple coincidence events, applies energy and timing corrections, and saves results in various formats for further analysis.

Main Features:
--------------
- Reads binary .dat files containing singles events.
- Groups sequential hits into events and identifies double and triple coincidences.
- Applies energy calibration and time-skew corrections using lookup tables and parameter files.
- Saves processed data as .pickle, .npy, .lm, and .txt files.
- Supports filtering by energy and geometry, and can randomize event order for certain analyses.

Key Functions:
--------------
- getSingleArray: Loads and parses singles data from a binary file.
- double_triple_coincidence_indexes: Identifies double and triple coincidences in sorted event data.
- threshold_doubles_by_energy: Filters double coincidences by calibrated energy.
- savelistmode: Converts coincidence pairs to listmode and text output, applying geometry and TOF corrections.
- TripleCoincidencetoTripleMCP: Converts triple coincidence pickles to structured numpy arrays for further analysis.
- get_Multi_Double_Triple_Pickle: Main processing function; extracts and saves double/triple coincidences from raw data.
- writeDoubleTripleCoincidence: Handles the creation and saving of coincidence event files.

Usage:
------
To process all relevant .dat files in a directory:
    python get_singles_data.py <dir_origin>

Arguments:
    dir_origin: Path to the folder containing .dat files and calibration files (paramsPerCrystal.pickle, *.lut).

Dependencies:
-------------
- numpy, pandas, pickle, argparse, pathlib, math, struct, time, array, re, glob
- Custom modules: ClassSingles8Bytes, group_by_submodule, utils

Outputs:
--------
- Pickle files for double and triple coincidences (coincidencePair*.pickle, coincidenceTriple*.pickle)
- Listmode and text files for further analysis (.lm, .txt)
- Numpy arrays for triple coincidences (.npy)
- Binary files of crystal IDs for filtered events

Author:
-------
Sarah Jin Zou (@sjzou)
"""

from array import array
import numpy as np
import os
from ClassSingles8Bytes import Singles
import pickle
import re
import glob
from group_by_submodule import group_by_submodule, fill_zeros_in_columns_with_non_zero_average
from utils import ADCtokeV
import math
import struct
import time
import argparse
import pdb
speedOfLight = 299792458000  # speed of light in mm/s
EVENTARRAYNUM = 28220000
# EVENTARRAYNUM = 50000 # smaller number for debugging
frag_size = 10000
keep_len = 2000
# DATA TYPES

""" Data type for hits or single (high-energy) photons.
    MCP hits or singles files are lists of this data type.
"""
gen2_single_dtype = np.dtype([
    ('time', np.float64),
    ('energy', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32)])


""" Triple coincidence of (high-energy) photons.
    For multi-isotope imaging, there is no current file of this type.
"""
triple_gen2_dtype = np.dtype([
    ('photon0', gen2_single_dtype),
    ('photon1', gen2_single_dtype),
    ('photon2', gen2_single_dtype)])


def retrieve_skew_symmetric_values(matrix, a, b):
    """
    Retrieve values from the matrix based on arrays of indices, considering the skew-symmetric property.
    
    Parameters:
    matrix (np.ndarray): Input matrix to retrieve values from.
    a, b (np.ndarray): Arrays of indices to retrieve values.
    
    Returns:
    np.ndarray: Array of retrieved values.
    """
    # Ensure a and b are arrays
    a = np.asarray(a)
    b = np.asarray(b)

    # Check if a and b have the same length
    if a.shape != b.shape:
        raise ValueError("Arrays a and b must have the same shape")

    # Retrieve values based on the indices
    values = np.empty(a.shape)
    mask = a < b
    values[mask] = matrix[a[mask], b[mask]]
    values[~mask] = -matrix[b[~mask], a[~mask]]

    return values
                
def double_triple_coincidence_indexes(frame_sorted, coarse_sorted, index_sorted, CTR=4):
    # Calculate the differences between consecutive elements
    frame_diff = np.diff(frame_sorted)
    diffs = np.diff(coarse_sorted)
    # Calculate the sum of differences for windows of three elements
    sum_diffs = diffs[:-1] + diffs[1:]
    sum_frame_diff = frame_diff[:-1] + frame_diff[1:]
    # Find the starting indexes where the sum of differences is within X
    triple_starting_indexes = np.where((sum_diffs <= CTR) & (sum_frame_diff == 0))[0]
    double_starting_indexes = np.where((diffs <= CTR) & (frame_diff == 0))[0]
    # Remove triple starting indexes (start and start +1) from double starting indexes
    double_starting_indexes = np.setdiff1d(double_starting_indexes, np.union1d(triple_starting_indexes, triple_starting_indexes + 1))
    double1 = index_sorted[double_starting_indexes]
    double2 = index_sorted[double_starting_indexes + 1]
    triple1 = index_sorted[triple_starting_indexes]
    triple2 = index_sorted[triple_starting_indexes + 1]
    triple3 = index_sorted[triple_starting_indexes + 2]

    double_delay_starting_indexes = np.where((diffs >= 10) & (diffs <= CTR + 10) & (frame_diff == 0))[0]
    double_delay1 = index_sorted[double_delay_starting_indexes]
    double_delay2 = index_sorted[double_delay_starting_indexes + 1]
    return [double1, double2], [triple1, triple2, triple3], [double_delay1, double_delay2]

def threshold_doubles_by_energy(coincidencePair, paramsPerCrystal, energy_range_keV=[450, 600]):
    # Get the energy values for the two crystals in the coincidence pair
    energyADC1 = coincidencePair[6, :]
    energyADC2 = coincidencePair[7, :]

    # Get the crystal IDs for the two crystals in the coincidence pair
    crystalID1 = coincidencePair[4, :]
    crystalID2 = coincidencePair[5, :]

    a_array = paramsPerCrystal[crystalID1, 0]
    b_array = paramsPerCrystal[crystalID1, 1]
    energy1 = ADCtokeV(energyADC1, a_array, b_array)

    a_array = paramsPerCrystal[crystalID2, 0]
    b_array = paramsPerCrystal[crystalID2, 1]
    energy2 = ADCtokeV(energyADC2, a_array, b_array)
    
    # Create a boolean mask where both energies are within the specified range
    energy_valid = (energy1 >= energy_range_keV[0]) & (energy1 <= energy_range_keV[1]) & \
                   (energy2 >= energy_range_keV[0]) & (energy2 <= energy_range_keV[1])
    
    # Apply the mask to the coincidence pair
    coincidencePair = coincidencePair[:, energy_valid]
    
    return coincidencePair

def get_crystalID_from_double_file(double_file, crystalparam_path, threshold_double_energy=False):
    """
    crystalparam_path (str): The path to the crystal parameters pickle file.
    threshold_double_energy (bool, optional): If True, apply energy thresholding to the double coincidences. Defaults to False.
    Returns:
    None
    """
    with open(double_file, 'rb') as file:
        coincidencePair = pickle.load(file)

    with open(crystalparam_path, 'rb') as file:
        paramsPerCrystal = pickle.load(file)
        paramsPerCrystal = fill_zeros_in_columns_with_non_zero_average(paramsPerCrystal) # NOTE: we shouldn't have to do this technically

    if threshold_double_energy:
        coincidencePair = threshold_doubles_by_energy(coincidencePair, paramsPerCrystal)
    
    # Extract crystal IDs
    crystalID1_arr = coincidencePair[4, :]
    crystalID2_arr = coincidencePair[5, :]
    crystal_array = np.array([crystalID1_arr, crystalID2_arr]) # 2 row array, so we save .T of this array to get (N, 2) shape
    if threshold_double_energy:
        crystal_array.T.astype(np.uint16).tofile(f'{double_file.split(".")[0]}_threshold_crystal_IDs.bin')
    else:
        crystal_array.T.astype(np.uint16).tofile(f'{double_file.split(".")[0]}_crystal_IDs.bin')
    return


def savelistmode(double_file, crystalparam_path, geometry_path, skewlut_path=None, tof= True, randomize = False, 
                 threshold_double_energy=False, save_crystal_array=False): 
    with open(double_file, 'rb') as file:
       coincidencePair = pickle.load(file)

    with open(crystalparam_path, 'rb') as file:
        paramsPerCrystal = pickle.load(file)
        paramsPerCrystal = fill_zeros_in_columns_with_non_zero_average(paramsPerCrystal) # NOTE: we shouldn't have to do this technically

    with open(geometry_path, 'rb') as file:
        crystalPositionMap = pickle.load(file)

    if skewlut_path:
        with open(skewlut_path, 'rb') as skewfile: # this is an upper triangular matrix
            skewlut = np.fromfile(skewfile, np.int16)
            skewlut = np.reshape(skewlut, (864*16, 864*16)) 
    else:
        skewlut = np.zeros((864*16, 864*16), dtype=np.int16)
        tof = False

    if randomize:
       rawData = np.transpose(coincidencePair)
       np.random.shuffle(rawData)
       coincidencePair = np.transpose(rawData)

    # filter by invalid crystal IDs
    crystalID1_arr = np.uint16(coincidencePair[4,:])
    tmp = crystalID1_arr
    firstCrystalX_arr = crystalPositionMap[tmp,0]
    crystalID2_arr = np.uint16(coincidencePair[5,:])
    tmp = crystalID2_arr
    SecondCrystalX_arr = crystalPositionMap[tmp,0]
    invalid = ((SecondCrystalX_arr == 0)|(firstCrystalX_arr == 0))
    coincidencePair = coincidencePair[:,~invalid]
    if threshold_double_energy:
        coincidencePair = threshold_doubles_by_energy(coincidencePair, paramsPerCrystal)

    array_path = os.path.splitext(double_file)[0]
    if threshold_double_energy:
        listmodeFile = f'{array_path}_petcoil_threshold.lm'
        textOutputFile = f'{array_path}_petcoil_threshold.txt'
    else:
        listmodeFile = f'{array_path}_petcoil.lm'
        textOutputFile = f'{array_path}_petcoil.txt'
    numberOfCoincidenceEvents = coincidencePair.shape[1]

    val = 0

    crystalID1_arr = np.uint16(coincidencePair[4,:])
    tmp = crystalID1_arr
    firstCrystalX_arr = crystalPositionMap[tmp,0]
    crystalID2_arr = np.uint16(coincidencePair[5,:])
    tmp = crystalID2_arr
    SecondCrystalX_arr = crystalPositionMap[tmp,0]
    invalid = ((SecondCrystalX_arr == 0)|(firstCrystalX_arr == 0))
    coincidencePair = coincidencePair[:,~invalid]
    crystalID1_arr = np.uint16(coincidencePair[4,:])
    tmp = crystalID1_arr
    firstCrystalX_arr = crystalPositionMap[tmp,0]
    firstCrystalY_arr = crystalPositionMap[tmp,1]
    firstCrystalZ_arr = crystalPositionMap[tmp,2]
    crystalID2_arr = np.uint16(coincidencePair[5,:])
    tmp = crystalID2_arr
    SecondCrystalX_arr = crystalPositionMap[tmp,0]
    SecondCrystalY_arr = crystalPositionMap[tmp,1]
    SecondCrystalZ_arr = crystalPositionMap[tmp,2]
    if tof: 
        coarseTime1 = coincidencePair[0,:]
        coarseTime2 = coincidencePair[1,:]
        coarseTimeDiff = coarseTime2 - coarseTime1
        fineTime1 = coincidencePair[2,:]
        fineTime2 = coincidencePair[3,:]
        fineTimeDiff = fineTime2 - fineTime1
        time_diff = coarseTimeDiff * 1600 + fineTimeDiff * 1600/1024 \
                        - retrieve_skew_symmetric_values(skewlut, crystalID2_arr, crystalID1_arr)
        TOF_arr = speedOfLight * (time_diff * math.pow(10, -12))
    else:
        TOF_arr = np.zeros(coincidencePair.shape[1])
    numberOfCoincidenceEvents = coincidencePair.shape[1]
    percent_bar = 0.05
    val = 0
    with open(listmodeFile, 'wb') as lm:        
        with open(textOutputFile, 'a') as f:
            print(numberOfCoincidenceEvents)
            for i in range(numberOfCoincidenceEvents):
                if(i > percent_bar*numberOfCoincidenceEvents):
                    print("Finish ",int(i*100/numberOfCoincidenceEvents),'%')
                    percent_bar += 0.05
                # print(line)
                crystalID1 = crystalID1_arr[i]
                firstCrystalX = firstCrystalX_arr[i]
                firstCrystalY = firstCrystalY_arr[i]
                firstCrystalZ = firstCrystalZ_arr[i]
                crystalID2 = crystalID2_arr[i]
                SecondCrystalX = SecondCrystalX_arr[i]
                SecondCrystalY = SecondCrystalY_arr[i]
                SecondCrystalZ = SecondCrystalZ_arr[i]
                TOF = TOF_arr[i]
                
                eventPair = np.zeros((1, 10))
                eventPair[0, :] = np.array([firstCrystalX, firstCrystalY, firstCrystalZ, TOF, val, SecondCrystalX, SecondCrystalY, SecondCrystalZ, crystalID1, crystalID2])
                # print(eventPair)
                textOutputFmt = ['%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f']
                if(i < 1000):
                    np.savetxt(f, eventPair, fmt=textOutputFmt)
                # eventToWrite = array('f', [firstCrystalX, firstCrystalY, firstCrystalZ, TOF, val, SecondCrystalX, SecondCrystalY, SecondCrystalZ, val, val])
                # eventToWrite.tofile(lm)

                lm.write(struct.pack('f', float(firstCrystalX)))
                lm.write(struct.pack('f', float(firstCrystalY)))
                lm.write(struct.pack('f', float(firstCrystalZ)))
                lm.write(struct.pack('f', float(TOF)))
                lm.write(struct.pack('f', float(val)))
                lm.write(struct.pack('f', float(SecondCrystalX)))
                lm.write(struct.pack('f', float(SecondCrystalY)))
                lm.write(struct.pack('f', float(SecondCrystalZ)))
                lm.write(struct.pack('f', float(val)))
                lm.write(struct.pack('f', float(val)))
    if save_crystal_array:
        crystal_array = np.array([crystalID1_arr, crystalID2_arr])
        if threshold_double_energy:
            crystal_array.T.astype(np.uint16).tofile(f'{array_path}_threshold_crystal_IDs.bin')
        else:
            crystal_array.T.astype(np.uint16).tofile(f'{array_path}_crystal_IDs.bin')


def TripleCoincidencetoTripleMCP(triple_pickle_path, crystalparam_path, geometry_path, skewlut_path=None):

    with open(crystalparam_path, 'rb') as file:
        paramsPerCrystal = pickle.load(file)
        paramsPerCrystal = fill_zeros_in_columns_with_non_zero_average(paramsPerCrystal) # NOTE: we shouldn't have to do this technically

    with open(geometry_path, 'rb') as file:
        crystalPositionMap = pickle.load(file)

    if skewlut_path:
        with open(skewlut_path, 'rb') as skewfile: # this is an upper triangular matrix
            skewlut = np.fromfile(skewfile, np.int16)
            skewlut = np.reshape(skewlut, (864*16, 864*16)) 
    else:
        skewlut = np.zeros((864*16, 864*16), dtype=np.int16)

    with open(triple_pickle_path, 'rb') as file:
        coincidenceTriple = pickle.load(file)

    # LOOK UP ENERGY AND POSITIONS ACCORDING TO CRYSTAL
    # get coincidence crystal ids 
    crystalIDs = (np.uint16(coincidenceTriple[6:9, :])) # shape (3, # of triples)
    energyADC = (coincidenceTriple[9:12, :]) # shape (3, # of triples)

    # want to look up paramsPerCrystal
    energieskeV = np.empty_like(crystalIDs)
    positionsXYZ = np.empty((3 * 3, coincidenceTriple.shape[1]))
    
    for i in range(3):
        crystal = crystalIDs[i]
        ADC = energyADC[i]
        a_array = paramsPerCrystal[crystal, 0]
        b_array = paramsPerCrystal[crystal, 1]
        energieskeV[i] = ADCtokeV(ADC, a_array, b_array)
        
        positionsXYZ[3*i:3*i+3] = (crystalPositionMap[crystal]).T
    
    # now need to figure out the time
    coarsetime = (coincidenceTriple[:3, :])
    finetime = (coincidenceTriple[3:6:, :])

    coarseTimeDiff1 = coarsetime[1, :] - coarsetime[0, :]  # 1.6 ns bins
    coarseTimeDiff2 = coarsetime[2, :] - coarsetime[0, :]

    finepsTimeDiff1 = finetime[1, :] - finetime[0, :] # 1600/1024 ps bins
    finepsTimeDiff2 = finetime[2, :] - finetime[0, :]

    time_diff = np.zeros_like(coarsetime, dtype=np.float64) # time difference is in picoseconds
    time_diff[1, :] = coarseTimeDiff1 * 1600 + finepsTimeDiff1 * 1600/1024 \
                 - retrieve_skew_symmetric_values(skewlut, crystalIDs[1], crystalIDs[0])
    time_diff[2, :] = coarseTimeDiff2 * 1600 + finepsTimeDiff2 * 1600/1024 \
                - retrieve_skew_symmetric_values(skewlut, crystalIDs[2], crystalIDs[0])
                
    # Define the energy range
    lower_bound = 460
    upper_bound = 1500
    
    # Create a boolean mask where all elements in each column are within the range
    energy_valid = np.all((energieskeV >= lower_bound) & (energieskeV <= upper_bound), axis=0)
    # TODO: time_valid should be a paramter also need to make sure the skewlut is correct
    time_valid = np.all(np.abs(time_diff) <= 1000, axis = 0) # filter on 1 ns
    valid_indexes = np.where(energy_valid & time_valid)[0]
    # valid_indexes = np.arange(coincidenceTriple.shape[1])
    
    triple_list = []
    
    for ind in valid_indexes:
        e1 = energieskeV[0, ind] * 0.001 # change to MeV
        e2 = energieskeV[1, ind] * 0.001
        e3 = energieskeV[2, ind] * 0.001
        
        t1 = time_diff[0, ind] * 1e-12 # to seconds
        t2 = time_diff[1, ind] * 1e-12
        t3 = time_diff[2, ind] * 1e-12
        
        p1 = positionsXYZ[:3, ind]
        p2 = positionsXYZ[3:6, ind]
        p3 = positionsXYZ[6:, ind]
        
        # Create instances of gen2_single_dtype and fill in information
        single1 = np.array((t1, e1, p1[0], p1[1], p1[2]), dtype=gen2_single_dtype)
        single2 = np.array((t2, e2, p2[0], p2[1], p2[2]), dtype=gen2_single_dtype)
        single3 = np.array((t3, e3, p3[0], p3[1], p3[2]), dtype=gen2_single_dtype)
        
        # Create an instance of triple_gen2_dtype and fill in information
        triple_list.append((single1, single2, single3))
    
    # Convert triple_list to a numpy array with triple_gen2_dtype
    triple_array = np.array(triple_list, dtype=triple_gen2_dtype)
    
    # Save triple_array as a binary file
    array_path = os.path.splitext(triple_pickle_path)[0]
    np.save(f'{array_path}.npy', triple_array)
    return triple_array

def getSingleArray(file, n_max=None):
    with open(file, 'rb') as f:
        try:
            numOfBytesInFile = os.path.getsize(file)
            bytesToProcess = numOfBytesInFile
            # event array with 8 byte format
            eventsarray = array('B')
            # every time we process EventArrayNum * 8 bytes. If not enough, we process all rest events (np.int(bytesToProcess / 8))
            if n_max:
                eventarraynum = min(n_max, bytesToProcess // 8)
            else:
                eventarraynum = bytesToProcess // 8
            eventsarray.fromfile(f, eventarraynum * 8)
            eventsarray = np.reshape(np.array(eventsarray), (eventarraynum, 8))
            event = Singles(eventsarray, keephighestenergy=False)
            return event
        except EOFError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

def find_peta_and_numbers(string):
    """
    Find the substring "PETA" in the string and return the numbers that directly follow it before an underscore.
    
    Parameters:
    string (str): The input string to search.
    
    Returns:
    list: A list of tuples with the numbers following "PETA".
    """
    # Define the regex pattern to match "PETA" followed by digits directly before an underscore
    pattern = r'PETA(\d+)_'
    
    # Find all matches in the string
    matches = re.findall(pattern, string)
    
    # Convert matches to integers and return as a list of tuples
    return matches[0]

def filter_energy(event, low = 450, high = 600):
    frag_size = 1000
    frag_num = int(event.size / frag_size) + 1
    index_split = np.array_split(event.index, frag_num)
    filename = 'paramsPerCrystal.pickle'
    with open(filename, 'rb') as file:
        paramsPerCrystal = pickle.load(file)
    
    final_indexes = []
    for indexes in index_split:
        crystal = event.crystalID[indexes]
        ADC = event.energy[indexes]
        a_array = paramsPerCrystal[crystal, 0]
        b_array = paramsPerCrystal[crystal, 1]
        energieskeV = ADCtokeV(ADC, a_array, b_array)
        valid_indexes = np.where((energieskeV > low) & (energieskeV < high))[0]
        final_indexes.extend(indexes[valid_indexes])
    
    # Convert final_indexes to a numpy array if needed
    final_indexes = np.array(final_indexes)
    return final_indexes

def process_norm_singles_data(dir_origin, fname, peta_dir):
    get_Multi_Double_Triple_Pickle(dir_origin, fname, peta_dir, combine_submodule=False, diff_submodule=True, triples=False, doubles=True)
       
def get_Multi_Double_Triple_Pickle(dir_origin, fname, peta_dir, combine_submodule=True, diff_submodule=True, triples=True, doubles=True):
    """
    Processes a binary file to extract and store coincidence events in pickle files.

    Parameters:
    dir_origin (str): The directory where the original binary file is located.
    fname (str): The name of the binary file to be processed.
    peta_dir (str): The directory where the output pickle files will be stored.
    combine_submodule (bool, optional): Whether to combine submodules. Defaults to True.
    diff_submodule (bool, optional): Whether to differentiate submodules. Defaults to True.
    triples (bool, optional): Whether to process triple coincidences. Defaults to True.
    doubles (bool, optional): Whether to process double coincidences. Defaults to True.

    Returns:
    None

    Notes:
    - If the output pickle file 'coincidenceTriple0.pickle' already exists in the peta_dir, the function will not run again.
    - The function reads the binary file in chunks, processes the events, and writes the results to pickle files.
    - Progress is printed to the console in increments of 5%.
    - The function uses the `Singles` class and `writeDoubleTripleCoincidence` function, which are assumed to be defined elsewhere in the codebase.
    """
    start = time.time()
    if not os.path.exists(peta_dir):
        os.makedirs(peta_dir)
    file = dir_origin + '/' + fname
    # if os.path.isfile(peta_dir + '/' + 'coincidenceTriple0.pickle'):
        # print("already ran, not running again")
        # return 
    with open(file, 'rb') as f:
        numOfBytesInFile = os.path.getsize(file)
        bytesToProcess = numOfBytesInFile
        bytesRead = 0
        percent_bar = 0.05
        num = 0
        while bytesToProcess > 0:
            # event array with 8 byte format
            eventsarray = array('B')
            # every time we process EventArrayNum * 8 bytes. If not enough, we process all rest events (np.int(bytesToProcess / 8))
            eventarraynum = min(MAXEVENTARRAYNUM, np.int64(bytesToProcess / 8))
            eventsarray.fromfile(f, eventarraynum * 8)
            bytesToProcess -= eventarraynum * 8
            bytesRead += eventarraynum * 8
            current_percent = bytesRead / numOfBytesInFile
            if current_percent > percent_bar:  # print progress for every 800000 bytes
                percent_bar += 0.05
                print(np.int64(current_percent * 10000) / 100, '%', ' completed', ' Elapsed time: ', time.time() - start, 's')
            eventsarray = np.reshape(np.array([eventsarray]), (eventarraynum, 8))
            event = Singles(eventsarray, keephighestenergy=False)
            # TODO: there is no keep length or fragsize
            writeDoubleTripleCoincidence(event, peta_dir, num, dir_origin, combine_submodule=combine_submodule, diff_submodule=diff_submodule, 
                                         triples=triples, doubles=doubles)
            num += 1

def writeDoubleTripleCoincidence(event, peta_dir, num, dir_origin, combine_submodule = True, 
                                 diff_submodule = True, triples = True, doubles = True):
    # Look for .lut files in dir_origin first, then in its parent directory if not found
    dir_origin = Path(dir_origin).resolve()
    skewlut_files = list(dir_origin.glob("*.lut"))
    if not skewlut_files:
        parent_dir = dir_origin.parent
        skewlut_files = list(parent_dir.glob("*.lut"))
    skewlut_path = skewlut_files[0] if skewlut_files else None
    print(f"Using skew_lut_path: {skewlut_path}")
    # Look for paramsPerCrystal.pickle in dir_origin first, then in its parent directory if not found
    crystalparam_files = list(dir_origin.glob("paramsPerCrystal.pickle"))
    if not crystalparam_files and dir_origin.parent != dir_origin:
        crystalparam_files = list(dir_origin.parent.glob("paramsPerCrystal.pickle"))
    crystalparam_path = crystalparam_files[0] if crystalparam_files else 'paramsPerCrystal.pickle'
    print(f"Using crystalparam_path: {crystalparam_path}")
    geometry_path = 'singles_geometry.pickle'

    # sort the event by frame, coarse time, and fine time
    argsort = np.lexsort((event.fine_ps, event.coarse, event.frame))

    frame_sorted = event.frame[argsort]
    index_sorted = event.index[argsort]
    coarse_sorted = event.coarse[argsort]

    if combine_submodule:
        frame_sorted, coarse_sorted, index_sorted = \
            group_by_submodule(event, argsort, dir_origin) # updates the energy information and deletes indexes

    
    [double1, double2], [triple1, triple2, triple3], [double_delay1, double_delay2] = double_triple_coincidence_indexes(frame_sorted, coarse_sorted, index_sorted)
    if triples:
        coincidenceTriple = np.zeros((12, triple1.size), dtype=np.int16)
        coincidenceTriple[0, :triple1.size] = event.coarse[triple1]
        coincidenceTriple[1, :triple1.size] = event.coarse[triple2]
        coincidenceTriple[2, :triple1.size] = event.coarse[triple3]
        # fine time stamp (bin size: 50ps)
        coincidenceTriple[3, :triple1.size] = event.fine[triple1]
        coincidenceTriple[4, :triple1.size] = event.fine[triple2]
        coincidenceTriple[5, :triple1.size] = event.fine[triple3]
        # crystal index
        coincidenceTriple[6, :triple1.size] = event.crystalID[triple1]
        coincidenceTriple[7, :triple1.size] = event.crystalID[triple2]
        coincidenceTriple[8, :triple1.size] = event.crystalID[triple3]
        # energy
        coincidenceTriple[9, :triple1.size] = event.energy[triple1]
        coincidenceTriple[10, :triple1.size] = event.energy[triple2]
        coincidenceTriple[11, :triple1.size] = event.energy[triple3]
    
        if diff_submodule:
        # check for sharing submodules:
        # first check if in the same module:
            cond1 = (event.moduleid[triple1] != event.moduleid[triple2]) & \
                (event.moduleid[triple2] != event.moduleid[triple3]) \
                & (event.moduleid[triple3] != event.moduleid[triple1])
            cond2 = (event.submoduleid[triple1] != event.submoduleid[triple2]) & \
                (event.submoduleid[triple2] != event.submoduleid[triple3]) \
                & (event.submoduleid[triple3] != event.submoduleid[triple1])
            index_valid = cond1 | cond2
            num_diff_sub_modules = np.sum(index_valid)
            coincidenceTriple = coincidenceTriple[:, index_valid]
            print(f"Triple coincidences with one shared submodule: {triple1.size - num_diff_sub_modules} / {triple1.size}")

    
        # # also another way to look at scatter? 
        # scatters += np.sum((np.abs(np.diff(event.crystalID[index])) < 144) & index_valid) 
        
        # save coincidenceTriple and delete it
        triple_pickle_path = f'{peta_dir}/coincidenceTriple_{num}.pickle'
        with open(triple_pickle_path, 'wb') as file:
            pickle.dump(coincidenceTriple, file, pickle.HIGHEST_PROTOCOL)
            
        del coincidenceTriple
    
        TripleCoincidencetoTripleMCP(triple_pickle_path, crystalparam_path, geometry_path, skewlut_path=skewlut_path)
    
    if doubles:
        coincidencePair = makeCoincidencePair(event, double1, double2)
        # check for sharing submodules:
        # first check if in the same module:
        cond1 = (event.moduleid[double1] != event.moduleid[double2])
        cond2 = (event.submoduleid[double1] != event.submoduleid[double2])
        index_valid = cond1 | cond2
        num_diff_sub_modules = np.sum(index_valid)
        coincidencePair = coincidencePair[:, index_valid]

        print(f"Double coincidences with one shared submodule: {double1.size - num_diff_sub_modules} / {double1.size}")

        double_file = f'{peta_dir}/coincidencePair{num}.pickle'
        # save coincidencePair and delete it
        with open(double_file, 'wb') as file:
            pickle.dump(coincidencePair, file, pickle.HIGHEST_PROTOCOL)

        del coincidencePair
        
        savelistmode(double_file, crystalparam_path, geometry_path, threshold_double_energy=True, save_crystal_array=False)

        coincidencePair_delay = makeCoincidencePair(event, double_delay1, double_delay2)
        delay_file = f'{peta_dir}/coincidencePair_delay{num}.pickle'
        # save coincidencePair and delete it
        with open(delay_file, 'wb') as file:
            pickle.dump(coincidencePair_delay, file, pickle.HIGHEST_PROTOCOL)
        del coincidencePair_delay
        savelistmode(delay_file, crystalparam_path, geometry_path, tof=False, threshold_double_energy=True, save_crystal_array=False)


def makeCoincidencePair(event, double1, double2):
    # coarse time stamp
    coincidencePair = np.zeros((8, double1.size), dtype=np.int16)
    coincidencePair[0, :double1.size] = event.coarse[double1]
    coincidencePair[1, :double1.size] = event.coarse[double2]
    # fine time stamp (bin size: 50ps)
    coincidencePair[2, :double1.size] = event.fine[double1]
    coincidencePair[3, :double1.size] = event.fine[double2]
    # crystal index
    coincidencePair[4, :double1.size] = event.crystalID[double1]
    coincidencePair[5, :double1.size] = event.crystalID[double2]
    # energy
    coincidencePair[6, :double1.size] = event.energy[double1]
    coincidencePair[7, :double1.size] = event.energy[double2]
    return coincidencePair


if __name__ == "__main__":
    # want to be given dir_origin
    # NOTE: assume that the crystal parameters (paramsPerCrystal.pickle) are dir_origin
    # for each .dat file, make the peta_dir and save the triple coincidences pickle files there
    # we need the time skew data to process this further.
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument(dest='dir_origin', help='Path of a folder')
    args = parser.parse_args()
    filelist = os.listdir(args.dir_origin)
    filelist = [f for f in filelist if '_PETA' in f or '_TEMP' in f]
    filelist.sort(key=lambda f: int(f.replace('TEMP','PETA').split('_PETA')[1].split('_')[0].split('.')[0]))
    print(filelist)

    for fname in filelist:
        # process all '.dat' files including 'PETA'
        if 'PETA' in fname and fname[-4:] == '.dat':
            print('************************************************')
            print(args.dir_origin, fname)
            print('************************************************')
            peta_dir = args.dir_origin + '/' + fname.split('_PETA')[0] + '/PETA' + fname[:-4].split('_PETA')[1].split('_')[0]
            get_Multi_Double_Triple_Pickle(args.dir_origin, fname, peta_dir)
