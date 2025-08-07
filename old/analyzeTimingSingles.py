"""
Code name:
    analyzeTimingSingles.py


Usage:
   called from analyzeData.py


Author:
    Sarah Zou (edited on 2024.12.18 to accommodate triple coincidences)
    Original: Chen-Ming Chang, Qian Dong
"""

import pickle
import statistics
from array import array
import time
import os
import numpy as np
from ClassSingles8Bytes import Singles
from PETcoilPythonLibrary import reportProgress
from PETcoilPythonLibrary import GetTimeResolution
# from PETcoilPythonLibrary import GetTimeDifference
from PETcoilPythonLibrary import correctCoarseTime
from PETcoilPythonLibrary import GetEnergyResolution
from PETcoilPythonLibrary import savelistmode
import argparse
import matplotlib.pyplot as plt

degree_sign = u'\N{DEGREE SIGN}'
numOfSubmodulesPerModule = 6
numOfHalfchipPerSubmodule = 8
numOfChannelPerHalfchip = 18
numOfBytesPerEvent = 8
# how many events to process every iteration
# EventArrayNum = 1000000
EventArrayNum = 50000000
# The size of each fragment
frag_size = 1000
# frag_size = 500
# The size of events left for next fragment processing
keep_len = 2000

def analyzetiming(dir_origin, fname, args):    
    start = time.time()
    triples=args.triples
    short=args.short 
    # result folder path ../PETA1/..
    print(fname[:-4].split('_PETA')[1])
    dir1 = dir_origin + '/' + fname.split('_PETA')[0] + '/PETA' + fname[:-4].split('_PETA')[1].split('_')[0]
    if fname[-5].isdigit():
        dir1 = dir1 + '_' + fname[:-4].split('_PETA')[1].split('_')[-1]
    dir2 = dir_origin + '/' + fname.split('_PETA')[0]
    if not os.path.exists(dir1):
        os.makedirs(dir1)
    photoPeakBoundaryPerCrystal = ''    
    if os.path.isfile(dir2 + '/' + 'photoPeakBoundaryPerCrystal.pickle'):
        with open(dir2 + '/' + 'photoPeakBoundaryPerCrystal.pickle', 'rb') as file:
            photoPeakBoundaryPerCrystal = pickle.load(file)
    elif os.path.isfile(dir_origin + '/' + 'photoPeakBoundaryPerCrystal.pickle'):
        with open(dir_origin + '/' + 'photoPeakBoundaryPerCrystal.pickle', 'rb') as file:
            photoPeakBoundaryPerCrystal = pickle.load(file)
    # If no energyPerCrystal.pickle, then we need to process the '.dat' instead of processing pickle files.
    coincidencePair_num = 0
    eventarraynum = EventArrayNum
    event_total_number = 0
    singles_total_number = 0
    duplicate_data_num = 0
    delay_data_num = 0
    delay_rate = 0
    singles_mod_counts = np.zeros(16)  
    scatters = 0
    delay_data_num2 = 0
    if not os.path.isfile(dir1 + '/' + 'coincidencePair.pickle') or args.Force_dataread:
        print('start timing analysis')
        file = dir_origin + '/' + fname
        for f_rm in os.listdir(dir1):
            if 'Pair' in f_rm:
                os.remove(dir1 + '/' + f_rm)   
        with open(file, 'rb') as f:
            numOfBytesInFile = os.path.getsize(file)
            # a new {crystal: [energy]} dictionary to store the energy per crystal
            energyPerCrystal = {}
            # a list to store the events unpaired left from the previous event array
            event_keep = []
            # event class with array data for energy, crystalid...
            event = []
            # read the file when remaining bytes > number of bytes per event
            bytesToProcess = numOfBytesInFile
            bytesRead = 0
            # fine time per crystal
            totalValidSingles = 0
            # percent step
            percent_bar = 0.05
            prev_frame = 0
            # loop through the file to get coincidence pairs, energy per crystal ..
            while bytesToProcess > 0:
                # event array with 8 byte format
                eventsarray = array('B')
                # every time we process EventArrayNum * 8 bytes. If not enough, we process all rest events (np.int(bytesToProcess / 8))
                eventarraynum = min(eventarraynum, np.int64(bytesToProcess / 8))
                eventsarray.fromfile(f, eventarraynum * 8)
                bytesToProcess -= eventarraynum * 8
                bytesRead += eventarraynum * 8
                current_percent = bytesRead / numOfBytesInFile
                if current_percent > percent_bar:  # print progress for every 800000 bytes
                    percent_bar += 0.05
                    print(np.int64(current_percent * 10000) / 100, '%', ' completed', ' Elapsed time: ', time.time() - start, 's')
                eventsarray = np.reshape(np.array([eventsarray]), (eventarraynum, 8))
                diff = eventsarray[1:] - eventsarray[:-1]
                duplicate_data_num += np.sum(np.prod(diff == 0, axis = 1)) 

                
                # convert event array with 8-byte format to class with array data
                event = Singles(eventsarray, compact = args.compact, keephighestenergy = args.keephighestenergy)
                event_crystalID = event.crystalID
                event_energy = event.energy
                event_index = event.index
                totalValidSingles += event.size
                # Create a new key (crystal) for crystal that is not in the dict.
                # find energy and fine time per crystal (fine time used for TDC bin width correction)
                if not args.Globalonly:
                    crystalid_argsort = np.argsort(event_crystalID)
                    crystalid_sorted = event_crystalID[crystalid_argsort]
                    crystalid_split_pos = np.where(np.diff(crystalid_sorted))[0] + 1
                    crystalid_unique = crystalid_sorted[np.insert(crystalid_split_pos, 0, 0)]
                    
                    index_sorted = event_index[crystalid_argsort]
                    index_split = np.split(index_sorted, crystalid_split_pos)
                    for i in range(crystalid_unique.size):
                        crystalid = crystalid_unique[i]
                        if crystalid not in energyPerCrystal:
                            energyPerCrystal[crystalid] = np.zeros(512, dtype=np.int64)

                        # calculates energy histogram
                        # how: 1. sort and find the positions where nearby elements are different.
                        # how: 2. The position represents how many events accumulate in each bin.
                        # e.g. energy_sorted [1 1 1 1 1 1 1 1 1 3 3 3 3 6 6 6 ...]
                        # e.g. energy_split_pos [9 13 16 ...]
                        # e.g. energy_sum_perbin [9 4 3 ...]
                        energy_sorted = np.sort(event_energy[index_split[i]])
                        energy_split_pos = np.where(np.diff(energy_sorted))[0] + 1
                        energy_unique_pos = np.insert(energy_split_pos, [0, energy_split_pos.size], [0, energy_sorted.size])
                        energy_sum_perbin = np.diff(energy_unique_pos)
                            
                        energy_value_perbin = energy_sorted[energy_unique_pos[:-1]]
                        energyPerCrystal[crystalid][energy_value_perbin] += energy_sum_perbin
                
                    # we split events to process since too many events may cause wrapping in frame value
                    frag_num = np.int64(event.size / frag_size) + 1
                    index_split = np.array_split(event.index, frag_num)
                    # There are events left unpaired in each fragment and we keep those date to use it for next fragment.
                    index_keep = []
                    index_keep_prev = []
                    # we keep keep_len (2000) unpaired data in each fragment for next fragment coincidence filtering.
                    keep_len = 2000
                    # add previous left events to current fragment 
                    event.AddKeepEvents(event_keep)
                    # loop to process each fragment
                    
                    for i in range(len(index_split)):
                        # concatence index of previous left events with events in current fragment.
                        index = np.concatenate((event.index[index_keep], index_split[i]))
                        if index.size < keep_len:
                            keep_len = index.size
                        # sort event index by fine time stamp, coarse fine time stamp, and frame value
                        argsort = np.lexsort((event.fine_ps[index], event.coarse[index], event.frame[index]))
                        index = index[argsort]
                        frame_frag = event.frame[index]
                        coarse_frag = event.coarse[index]
                        keep_frag = event.keep[index]
                        
                        
                        #If the nearby events have the same frame value and fit the args.coincidenceTimeWindow, they are conincidence events. 
                        # Mark all those singles with index_valid.
                        #index_valid doesn't include the last element in the array.
                        #index_valid = (np.diff(frame_frag) == 0) & (np.diff(coarse_frag) < args.coincidenceTimeWindow)
                        index_valid = (np.diff(coarse_frag) < args.coincidenceTimeWindow)
                        index_valid_pair1 = np.insert(index_valid, index_valid.size, False) # NOTE: add False to the end of the array
                        index_valid_pair2 = np.insert(index_valid, 0, False) # NOTE add False to the start of the array 

                        if triples:
                            diffs = np.diff(coarse_frag)
                            sum_diffs = diffs[:-1] + diffs[1:]
                            triple_index_valid = (sum_diffs < args.coincidenceTimeWindow) # TODO: maybe we want to change this to a different time window for triple coincidences
                            index_valid_triple1 = np.insert(triple_index_valid, 
                                    [triple_index_valid.size, triple_index_valid.size], [False, False]) 
                            index_valid_triple2 = np.insert(triple_index_valid, 
                                    [0, triple_index_valid.size], [False, False]) 
                            index_valid_triple3 = np.insert(triple_index_valid, 
                                    [0, 0], [False, False]) 
                            
                            # TODO: need to take out triples from doubles list
                            # if True in index_valid_triple1 then set to False in index_valid_pair1
                            # also the same for index_valid_triple2 and index_valid_pair2
                        
                        scatters += np.sum((np.abs(np.diff(event.crystalID[index])) < 144) & index_valid) 
                        
                        # calculate expectation for count number per event (how many counts fit in the time window)
                        # only in valid array, 0111011, the number of zeros represents how many events.
                        event_total_number += np.where(index_valid_pair2 == 0)[0].size
                        # e.g. index = [random1, random2, pair1_1,pair1_2,pair1_3,pair2_1,pair2_2, random3, random4 ] 
                        # If three events come together, we will have two coincident pairs: [pair1_1,pair1_2], [pair1_2,pair1_3]
                        # e.g. np.diff = [0,0,1,1,0,1,0,0]
                        # e.g. index_valid_pair1 = [0,0,1,1,0,1,0,0,0]
                        # e.g. index_valid_pair2 = [0,0,0,1,1,0,1,0,0]
                        # e.g. index_pair1 = [pair1_1,pair1_2,pair2_1]
                        # e.g. index_pair2 = [pair1_2,pair1_3,pair2_2]
                        index_pair1 = index[index_valid_pair1, ]
                        index_pair2 = index[index_valid_pair2, ]
                        
                        if triples:
                            index_triple1 = index[index_valid_triple1, ]
                            index_triple2 = index[index_valid_triple2, ]
                            index_triple3 = index[index_valid_triple3, ]
                        
                        # find the coincidences with a gap of 2-10
                        tmp1 = index_pair1             
                        tmp2 = index_pair2      
                        # for index_shift in range(10):
                            # index_valid = (np.diff(tmp1) == 1)
                            # if not index_valid.any():
                                # break
                            # index_valid_pair1 = np.insert(index_valid, index_valid.size, False)
                            # index_valid_pair2 = np.insert(index_valid, 0, False)
                            # tmp1 = tmp1[index_valid_pair1, ]
                            # tmp2 = tmp2[index_valid_pair2, ]
                            # index_pair1 = np.concatenate([index_pair1, tmp1])
                            # index_pair2 = np.concatenate([index_pair2, tmp2])
                            # print(np.concatenate(([index_pair1], [index_pair2]),axis = 0))
                            
                        # keep events unpaired. If more than keep_len, then keep the last keep_len events.
                        index_keep = index[(~(index_valid_pair1|index_valid_pair2))&keep_frag]
                        if len(index_keep) > keep_len:
                            index_keep = index_keep[-keep_len:]
                        # To only keep once unpaired events, mark the keep value to 0   
                        event.keep[index_keep] = 0
                        
                        # filter out coincidences from the same detector
                        # if OnlyDiffDet:
                            # index_valid = (event.moduleid[index_pair1] != event.moduleid[index_pair2])
                            # index_pair1 = index_pair1[index_valid,]
                            # index_pair2 = index_pair2[index_valid,]
                        # # filter out coincidences from the same submodule
                        # elif OnlyDiffSub:
                            # index_valid = (event.submoduleid[index_pair1] != event.submoduleid[index_pair2]) | (event.moduleid[index_pair1] 
                            # != event.moduleid[index_pair2])
                            # index_pair1 = index_pair1[index_valid,]
                            # index_pair2 = index_pair2[index_valid,]
                            
                        # switches index_pair1 and index_pair2 to make index_pair1 have smaller crystal index
                        tmp = np.concatenate(([index_pair1], [index_pair2]), axis=0)
                        argsort = np.argsort(np.concatenate(([event.crystalID[index_pair1]], [event.crystalID[index_pair2]]), axis=0), axis=0)
                        tmp = np.take_along_axis(tmp, argsort, axis=0)
                        index_pair1 = tmp[0,:]
                        index_pair2 = tmp[1,:]
                        # coarse time stamp
                        coincidencePair = np.zeros((10, index_pair1.size), dtype=np.int16)
                        coincidencePair[0, :index_pair1.size] = event.coarse[index_pair1]
                        coincidencePair[1, :index_pair1.size] = event.coarse[index_pair2]
                        # fine time stamp (bin size: 50ps)
                        coincidencePair[2, :index_pair1.size] = event.fine[index_pair1]
                        coincidencePair[3, :index_pair1.size] = event.fine[index_pair2]
                        # crystal index
                        coincidencePair[4, :index_pair1.size] = event.crystalID[index_pair1]
                        coincidencePair[5, :index_pair1.size] = event.crystalID[index_pair2]
                        # energy
                        coincidencePair[6, :index_pair1.size] = event.energy[index_pair1]
                        coincidencePair[7, :index_pair1.size] = event.energy[index_pair2]
                        # fine time stamp ps
                        coincidencePair[8, :index_pair1.size] = event.fine_ps[index_pair1]
                        coincidencePair[9, :index_pair1.size] = event.fine_ps[index_pair2]
                        coincidencePair = coincidencePair[:,0:index_pair1.size]
                        coincidencePair_num += index_pair1.size

                        if triples:
                            tmp = np.concatenate(([index_triple1], [index_triple2], [index_triple3]), axis=0)
                            argsort = np.argsort(np.concatenate((
                                    [event.crystalID[index_triple1]], 
                                    [event.crystalID[index_triple2]], 
                                    [event.crystalID[index_triple3]]), axis=0), axis=0)
                            tmp = np.take_along_axis(tmp, argsort, axis=0)
                            triple1 = tmp[0,:]
                            triple2 = tmp[1,:]
                            triple3 = tmp[2,:]
                            
                            
                            coincidenceTriple = np.zeros((15, triple1.size), dtype=np.int16)
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
                            # fine time stamp ps
                            coincidenceTriple[12, :triple1.size] = event.fine_ps[triple1]
                            coincidenceTriple[13, :triple1.size] = event.fine_ps[triple2]
                            coincidenceTriple[14, :triple1.size] = event.fine_ps[triple3]
                            

                            coincidenceTriple = coincidenceTriple[:,0:triple1.size]

                        coin_filelist = [f for f in os.listdir(dir1) if 'coincidencePair' in f] # TODO: just keep track of str_tmp
                        str_tmp = ''
                        if coin_filelist != []:
                            str_tmp = str(len(coin_filelist) + 1)
                        if short and (len(coin_filelist) > 5):
                            break
                        with open(dir1 + '/' + 'coincidencePair' + str_tmp + '.pickle', 'wb') as file:
                            pickle.dump(coincidencePair, file, pickle.HIGHEST_PROTOCOL)
                            # keep the events unpaired for next iteration
                        if triples:
                            with open(dir1 + '/' + 'coincidenceTriple' + str_tmp + '.pickle', 'wb') as file:
                                pickle.dump(coincidenceTriple, file, pickle.HIGHEST_PROTOCOL)
                            # keep the events unpaired for next iteration
                    event_keep = event.GetKeepEvents(index_keep)
                    
        coincidencePairlen = coincidencePair_num
        singles_total_number = np.sum(singles_mod_counts)
        delay_data_num2 = 0
        for i in range(16):
            for j in range(16):
                if(i != j):
                    delay_data_num2 = delay_data_num2 + singles_mod_counts[i] * singles_mod_counts[j]
        delay_data_num2 = delay_data_num2 / 2      
        if(delay_data_num == 0):
            delay_data_num = delay_data_num2 *4/1e9/3600*2 * args.DurationD * 60
        print("delay", delay_data_num2 *4/1e9/3600*2 * args.DurationD * 60, delay_data_num)
       #pickle the files for future use
        with open(dir1 + '/' + 'energyPerCrystal.pickle', 'wb') as file:
            pickle.dump(energyPerCrystal, file, pickle.HIGHEST_PROTOCOL)
        with open(dir1 + '/' + 'TotalCounts.pickle', 'wb') as file:
            pickle.dump([totalValidSingles, delay_data_num, singles_total_number, event_total_number, coincidencePairlen], file, pickle.HIGHEST_PROTOCOL)
        
        if triples: 
            del coincidenceTriple
        del energyPerCrystal, coincidencePair, event

    
            
            
    print('Elapsed Time: ' + str(time.time()-start) + 's')                  
    with open(dir1 + '/' + 'TotalCounts.pickle', 'rb') as file:
        if not args.old:
             [totalValidSingles, delay_data_num, singles_total_number, event_total_number, coincidencePairlen] = pickle.load(file)
             
        else:
             [totalValidSingles, coincidencePairlen] = pickle.load(file)
             duplicate_data_num = 1
             singles_total_number = 1
             event_total_number = 1 
    print("Total valid single events                  : " + str(totalValidSingles))
    print("Total valid single events (Energy gating)  : " + str(singles_total_number))
    print("Total coincidence pairs                    : " + str(coincidencePairlen) +'\n')
    print("Total random number                        : " + str(delay_data_num) +'\n')
    print("Singles in a event expectation             : " + str(singles_total_number/event_total_number)+'\n')
    print("Duplicate data number (ratio)              : " + str(duplicate_data_num)+' ('+str(float(np.round(duplicate_data_num/totalValidSingles*100,4)))+'%)\n')
    print("Scatters number (ratio)                    : " + str(scatters)+' ('+str(float(np.round(scatters/totalValidSingles*100,4)))+'%)\n')
    
    
#    if args.Globalonly:
#        return


    # EDIT: 12/20/24 SZ: Commented this code because we're only interested in coincidence grouping code
    # if os.path.isfile(dir2 + '/' + 'skew_lut.dat'):
    #     with open(dir2 + '/' + 'skew_lut.dat', 'rb') as file:
    #         skewlut = np.fromfile(file, np.int16)
    #         skewlut = np.reshape(skewlut, (864*16, 864*16)) 
    #         print('##########')
    #         print('skew_lut.dat loaded')
    # elif os.path.isfile(dir_origin + '/' + 'skew_lut.dat'):
    #     with open(dir_origin + '/' + 'skew_lut.dat', 'rb') as file:
    #         skewlut = np.fromfile(file, np.int16)
    #         skewlut = np.reshape(skewlut, (864*16, 864*16)) 
    #         print('##########')
    #         print('skew_lut.dat loaded')
    # else:
    #     skewlut = {}
    
    # # TODO: need to check dir1 instead
    # if os.path.isfile(dir2 + '/' + 'paramsPerCrystal.pickle'):
    #     with open(dir2 + '/' + 'paramsPerCrystal.pickle', 'rb') as file:
    #          paramsPerCrystal = pickle.load(file)
    #          print('##########')
    #          print('paramsPerCrystal.pickle loaded')
    # elif os.path.isfile(dir_origin + '/' + 'paramsPerCrystal.pickle'):
    #     with open(dir_origin + '/' + 'paramsPerCrystal.pickle', 'rb') as file:
    #          paramsPerCrystal = pickle.load(file)
    #          print('##########')
    #          print('paramsPerCrystal.pickle loaded')
    # else:
    #     paramsPerCrystal = ''
        
    # if not os.path.isfile(dir1 + '/' + 'energyGlobal.pickle') or args.Force_coincidence:
    #     with open(dir1 + '/' + 'energyPerCrystal.pickle', 'rb') as file:
    #          energyPerCrystal = pickle.load(file)
    #          print('##########')
    #          print('energyPerCrystal.pickle loaded')
             
    #     coin_filelist = [f for f in os.listdir(dir1) if 'coincidencePair' in f]
    #     coincidencePair = np.zeros((10,0),dtype = np.int16)
    #     for coin_file in coin_filelist:
    #         with open(dir1 + '/' + coin_file, 'rb') as file:
    #              tmp = pickle.load(file)
    #              coincidencePair = np.concatenate([coincidencePair,tmp],axis = 1)
    #     print('##########')
    #     print('coincidencePair.pickle loaded')
                 
            
    #     if args.Fix_energyGate:
    #         filename = dir_origin + '/' + 'energyGlobal.pickle'
    #         print(filename)
    #         if os.path.isfile(filename):
    #             with open(filename, 'rb') as file:
    #                 [energyResolutionPerCrystal, photoPeakBoundaryPerCrystal, global_size_e, global_energyResolution] = pickle.load(file)
    #                 print('##########')
    #                 print("energyGlobal.pickle loaded")
    #     elif args.EnergyGate:
    #         [energyResolutionPerCrystal, photoPeakBoundaryPerCrystal], [global_size_e, global_energyResolution] =\
    #             GetEnergyResolution(energyPerCrystal, coincidencePair, Onlycoincidence = args.Coincidenceonly, plot=args.PlotE, 
    #                 dir = dir1 + '/Energy Plot', energyWindow = args.energyWindow,paramsPerCrystal=paramsPerCrystal, Exponentialfit = args.Exponentialfit, 
    #                 interpret = args.energyinterpret, Globalonly = args.Globalonly, photoPeakBoundaryPerCrystal = photoPeakBoundaryPerCrystal)
    #     else:
    #         energyResolutionPerCrystal = {}
    #         photoPeakBoundaryPerCrystal = {}
    #         global_energyResolution = 0
    #         global_size_e = 1
    #     print('##########')
    #     print('Energy done')
        
        
    #     filename = dir1 + '/' + 'energyGlobal.pickle'
    #     with open(filename, 'wb') as file:
    #         pickle.dump([energyResolutionPerCrystal, photoPeakBoundaryPerCrystal,global_size_e, global_energyResolution], file, pickle.HIGHEST_PROTOCOL)
        
            
    #     if args.savelistmode:
    #         savelistmode(coincidencePair, dir2 + '/Listmode/', dir1, skewlut)   
            
            
    # # set plot=True to plot the timing spectra for checking
    # event_sum = 0
    # stack = []
    # events = 0
    # filename = dir1 + '/' + 'energyGlobal.pickle'
    # with open(filename, 'rb') as file:
    #     [energyResolutionPerCrystal, photoPeakBoundaryPerCrystal, global_size_e, global_energyResolution] = pickle.load(file)
    #     print("energyGlobal.pickle loaded")
    
    # global_size = 0
    # filename = dir1 + '/' + 'timingGlobal.pickle'
    # if not os.path.isfile(filename) or args.Force_coincidence:
             
    #     coin_filelist = [f for f in os.listdir(dir1) if 'coincidencePair' in f]
    #     coincidencePair = np.zeros((10,0),dtype = np.int16)
    #     print('##########')
    #     print('coincidencePair.pickle loaded')
    #     for coin_file in coin_filelist:
    #         with open(dir1 + '/' + coin_file, 'rb') as file:
    #              tmp = pickle.load(file)
    #              coincidencePair = np.concatenate([coincidencePair,tmp],axis = 1)

             
    #     [timeResolutionPerLOR, timeDifference_split, timeDifference_unique_crystal1, 
    #         timeDifference_unique_crystal2], [global_size, global_timeResolution], coinciSum = \
    #         GetTimeResolution(photoPeakBoundaryPerCrystal,coincidencePair, _minCountInLOR=args.minCountInLOR, 
    #                           energyGate=args.EnergyGate, plot=args.PlotT, dir=dir1 + '/Coincidence Plot', 
    #                           skewlut = skewlut, globalonly = args.Globalonly, globaloffset = args.Globaloffset)
        
    #     with open(filename, 'wb') as file:
    #         try:
    #             pickle.dump([timeResolutionPerLOR, timeDifference_split, timeDifference_unique_crystal1, 
    #                          timeDifference_unique_crystal2, coinciSum, global_timeResolution], file, pickle.HIGHEST_PROTOCOL)
    #         except MemoryError:
    #             pickle.dump([timeResolutionPerLOR, [], [], [], coinciSum, global_timeResolution], file, pickle.HIGHEST_PROTOCOL)

    # if not os.path.isfile(dir1 + '/' + 'Delay_timing.png') and args.delay:
    #         delay_filelist = [f for f in os.listdir(dir1) if 'delayPair' in f]
    #         delayPair = np.zeros((10,0),dtype = np.int16)
    #         print('##########')
    #         print('delayPair.pickle loaded')
    #         for delay_file in delay_filelist:
    #             with open(dir1 + '/' + delay_file, 'rb') as file:
    #                  tmp = pickle.load(file)
    #                  delayPair = np.concatenate([delayPair,tmp],axis = 1)
    #         print(delayPair.shape)      
    #         [delaytimeResolutionPerLOR, delaytimeDifference_split, delaytimeDifference_unique_crystal1, delaytimeDifference_unique_crystal2], \
    #             [delay_size, delay_timeResolution], delaySum = GetTimeResolution(photoPeakBoundaryPerCrystal,delayPair, _minCountInLOR=0, 
    #                                                                              energyGate=False, plot=False, dir=dir1 + '/Delay Plot', skewlut = skewlut, 
    #                                                                              globalonly = True, globaloffset = args.Globaloffset, Delay = True, 
    #                                                                              coincidenceTimeWindowOffset = args.coincidenceTimeWindowOffset)
    #         print('##########')

    
    # filename = dir1 + '/' + 'timingGlobal.pickle'
    # with open(filename, 'rb') as file:
    #     [timeResolutionPerLOR, timeDifference_split, timeDifference_unique_crystal1, timeDifference_unique_crystal2, coinciSum, global_timeResolution] = pickle.load(file)
    # if not args.Globalonly:
    #     filename = dir1 + '/' + 'skew_lut.dat'
    #     skewlut_arr = np.zeros((16*6*144,16*6*144), dtype = np.int16)    
    #     for key1 in timeResolutionPerLOR.keys():
    #         for key2 in timeResolutionPerLOR[key1].keys():
    #             skewlut_arr[key1,key2] = timeResolutionPerLOR[key1][key2][1]
    #             skewlut_arr[key2,key1] = -timeResolutionPerLOR[key1][key2][1]
    #     skewfile = open(filename, 'wb')
    #     skewfile.write(skewlut_arr.tobytes())
    #     skewfile.close()
                
    # def value_format(value, digit):
    #     if (value > 1000):
    #         return np.round(value/1000,digit), ' kcps' 
    #     else:
    #         return np.round(value,digit), ' cps' 
            
    # with open(dir1 + '/' + 'summary.txt', 'w') as file:
    #     print('Duration:', args.DurationD)
    #     file.write('Duration                               : ' + str(args.DurationD) + 'min\n')
    #     file.write('Total valid single events              : ' + str(totalValidSingles) + '\n')
    #     file.write("Total valid singles (Energy gating)    : " + str(singles_total_number) + '\n')
    #     file.write('Total coincidence pairs                : ' + str(coincidencePairlen) + '\n')
    #     file.write('Total random number                    : ' + str(delay_data_num2) + '\n')
    #     file.write('Total coincidence pairs (EnergyGating) : ' + str(coinciSum) + '\n')
    #     print('Total coincidence pairs (EnergyGating) : ' + str(coinciSum) + '\n')
    #     file.write("Singles in a event expectation         : " + str(totalValidSingles/event_total_number)+'\n')
    #     file.write("Duplicate data number (ratio)          : " + str(duplicate_data_num)+' ('+str(float(np.round(duplicate_data_num/totalValidSingles*100,4)))+'%)\n')
    #     file.write('Global Timing Resolution: ' + str(np.round(global_timeResolution, 2)) + ' ps ' + str(np.int64(global_size)) + '\n')
    #     print('Global Timing Resolution: ' + str(np.round(global_timeResolution, 2)) + ' ps ' + str(np.int64(global_size)) + '\n')
    #     tmp = global_size_e/60/args.DurationD
    #     file.write('Single Count Rate: ' + str(value_format(tmp, 2)[0])+ value_format(tmp, 2)[1]+ '\n')
    #     print('Single Count Rate: ' + str(value_format(tmp, 2)[0])+ value_format(tmp, 2)[1]+ '\n')
    #     tmp = coinciSum/60/args.DurationD
    #     file.write('Coincience Count Rate: '  + str(value_format(tmp, 2)[0])+ value_format(tmp, 2)[1]+ '\n')
    #     print('Coincience Count Rate: ' + str(value_format(tmp, 2)[0])+ value_format(tmp, 2)[1]+ '\n')
    #     tmp = delay_data_num/60/args.DurationD
    #     file.write('Random Count Rate: ' + str(value_format(tmp, 2)[0])+ value_format(tmp, 2)[1]+ '\n')
    #     print('Random Count Rate: ' + str(value_format(tmp, 2)[0])+ value_format(tmp, 2)[1]+ '\n')
    #     file.write('Events / Singles%: ' + str(np.round(coinciSum/max(1,global_size_e) * 100, 2))+'%\n')
    #     print('Events / Singles%: ' + str(np.round(coinciSum/max(1,global_size_e) * 100,2))+'%\n')
    #     file.write('Global Energy Resolution: ' + str(np.round(global_energyResolution*100, 2)) + ' % ' + str(np.int64(global_size_e)) + '\n')
    #     print('Global Energy Resolution: ' + str(np.round(global_energyResolution*100, 2)) + ' % ' + str(np.int64(global_size_e)) + '\n')

    #     for crystal1 in timeResolutionPerLOR:
    #         for crystal2 in timeResolutionPerLOR[crystal1]:
    #             events += timeResolutionPerLOR[crystal1][crystal2][2]
    #             if(timeResolutionPerLOR[crystal1][crystal2][0] != 0):
    #                 file.write('TR: '+str(crystal1) + ' ' + str(crystal2) + ' ' + str(int(timeResolutionPerLOR[crystal1][crystal2][0])) + ' ' + \
    #                            str(int(timeResolutionPerLOR[crystal1][crystal2][1])) + ' ' + str(int(timeResolutionPerLOR[crystal1][crystal2][2])) + '\n')
    #                 stack.append(timeResolutionPerLOR[crystal1][crystal2][0])
    #     if events > 0:
    #         if (len(stack) == 0):
    #             std = 'Crystal timing resolution average (mean): ' + str(0)
    #         elif (len(stack) == 1):
    #             std = 'Crystal timing resolution average (mean): ' + str(np.round(statistics.mean(stack), 2))
    #         else:
    #             std = 'Crystal timing resolution average (mean+-sd): ' + str(np.round(statistics.mean(stack), 2)) + ' +- ' + \
    #                 str(np.round(statistics.stdev(stack), 2)) + ' ps'
    #         file.write(std)
    #         print(std + str('\n'))
    #     else:
    #         file.write('0 events\n')
    #         print("0 events\n")
    #     event_sum = 0
    #     stack = []
    #     events = 0

    #     for crystal1 in energyResolutionPerCrystal:
    #         file.write('ER: ' + str(crystal1) + ' ' + str(energyResolutionPerCrystal[crystal1][0]) + ' ' + str(energyResolutionPerCrystal[crystal1][1]) + '\n')
    #         events += energyResolutionPerCrystal[crystal1][1]
    #         if(energyResolutionPerCrystal[crystal1][0] != 0):
    #             stack.append(energyResolutionPerCrystal[crystal1][0])
    #     if events > 0:
    #         std = 'Crystal energy resolution average (mean+-sd): ' + str(np.round(statistics.mean(stack)*100, 2)) + ' +- ' + \
    #               str(np.round(statistics.stdev(stack)*100, 2)) + ' %'
    #         file.write(std)
    #         print(std + str('\n'))
    #     else:
    #         file.write('0 events\n')
    #         print("0 events\n")
    #     file.write('Elapsed Time: ' + str(time.time()-start) + 's')
    #     print('Elapsed Time: ' + str(time.time()-start) + 's')
    #     file.close()