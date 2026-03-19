import os
import shutil
import numpy as np
from array import array
import itertools
# Import necessary models from lmfit
from lmfit.models import GaussianModel, ConstantModel
import peakutils
import matplotlib.pyplot as plt
import math
import struct
from array import array
import pickle
import sys
import argparse

pixel_num = 864

#### get params
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument(dest='dir_origin', help='Path of a folder')
parser.add_argument('--r', dest='reverse', action='store_true', default=False,
                    help='Process data in reverse direction(default: False)')
args = parser.parse_args()
print(args)


#### params settings
Listmode = True
RandomEstimation = True # Enable Random Estimation
Skew = True             # Enable Skew Correction
TOF = True              # Enable TOF Windowing
ListmodeCombine = True
Plot = True

# --- NEW FLAG FOR RANDOM CORRECTION MODE ---
# Set to 'Pair' to use only event-pair rejection.
# Set to 'Hybrid' (or anything else) to use pair rejection + histogram correction.
RandomMode = 'Hybrid'  
# -------------------------------------------

# --- NEW HYBRID PARAMETERS ---
TOF_BIN_SIZE_PS = 10 # Finer binning to reduce local variance (e.g., 10 ps)
TOF_RANGE_PS = 10000 # Total range -5000 ps to 5000 ps
NUM_TOF_BINS = int(TOF_RANGE_PS / TOF_BIN_SIZE_PS) + 1 # 1001 bins for 10 ps
# -----------------------------

textOutputFmt = ['%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f']
Det_convert = np.arange(0,16, dtype = int)
speedOfLight = 299792458000  # speed of light in mm
speedOfLight_length_ps = speedOfLight * math.pow(10, -12)

#### split_data files
WDIR=args.dir_origin
Files = {}
for root, dirs, files in os.walk(WDIR):
    for name in files:
        if  '.dat' in name and 'coin' in name:
            if name not in Files.keys():
                Files[name] = []
            Files[name] += [os.path.join(root, name)]
keys = list(Files.keys())
keys.sort(key=lambda f: int(f.split('_')[0])*1000 + int(f.split('_')[1]))
print(Files)
print(keys)

#### result paths
folder = WDIR + '/result'
if not os.path.isdir(folder):
    os.mkdir(folder)
result_path = folder + '/Skew'
# Update path to include RandomMode for differentiation
listmode_path = folder + f'/Listmode_Rcorrect{RandomEstimation*1}_TOF{TOF*1}_{RandomMode}'


#### useful functions
def stack_padding(l):
    return np.column_stack((itertools.zip_longest(*l, fillvalue=np.nan)))

if not os.path.isdir(result_path):
    os.mkdir(result_path)

if not os.path.isdir(listmode_path):
    os.mkdir(listmode_path)

#### load geometry file
if Listmode:
    with open('geometry.pickle', 'rb') as file:
        crystalPositionMap = pickle.load(file)

data_all = np.zeros((864*16, 864*16), dtype = np.int16)

#### iterate all detid pairs
for f in keys:
    print(f)
    sub0 = int(f.split('_')[0])
    sub1 = int(f.split('_')[1])
    skewoffset = np.zeros((pixel_num,pixel_num), dtype = np.int16)

    if os.path.isfile(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin_random_corrected_tof_listmode.jpg'):
        continue
    data = 0
    for file in Files[f]:
        print(file)
        fo=open(file,"rb")
        data_tmp = np.fromfile(fo,dtype = np.int16)
        data_tmp = np.reshape(data_tmp, (int(data_tmp.shape[0]/3), 3)).transpose()
        if isinstance(data, int):
            data = data_tmp
        else:
            data = np.concatenate([data,data_tmp], axis = 1)
        print(data.shape)
        fo.close()
    #### convert TDC bin to ps. 1 TDC bin = 1600/1024 ps
    data[2, :] = data[2, :]*1.5625

    #### plot the time spectrum of that detid pair
    if Plot:
        tmp = data[2,:]
        plt.hist(tmp, np.linspace(-5000,5000,201))
        lim_max = np.max(np.histogram(tmp, np.linspace(-5000,5000,201))[0]) * 1.2
        plt.ylim([0,lim_max])
        plt.xlabel('Time Difference (ps)')
        plt.ylabel('Coincidence Counts')
        plt.title('Raw Time Spectrum')
        plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_raw_coin.jpg')
        plt.clf()

    #### find LOR skew lut table
    if Skew:
        f_skew = result_path + '/' + str(sub0) + '_' + str(sub1) + '_skew_array.dat'
        f_skew2 = 'skew_lut.dat'
        if os.path.isfile(f_skew):
            skewfile = open(f_skew, 'rb')
            skewoffset = np.fromfile(skewfile, np.int16)
            skewoffset = np.reshape(skewoffset, (pixel_num, pixel_num))
        elif os.path.isfile(f_skew2):
            data_all = open(f_skew2, 'rb')
            data_all = np.fromfile(data_all, np.int16)
            data_all = np.reshape(data_all, (864*16, 864*16))
            skewoffset = data_all[sub0*pixel_num:sub0*pixel_num+ pixel_num, sub1*pixel_num:sub1*pixel_num+ pixel_num]
            print('loading skewoffset')
        else:
            #### split data based on crystalid pair (LOR)
            data_argsort = np.lexsort((data[1, :], data[0, :]))
            data_sorted = data[:, data_argsort]
            data_split_pos = np.where(np.diff(data_sorted[1, :]))[0] + 1
            data_unique_crystal1 = np.int16(data_sorted[0, np.insert(data_split_pos, 0, 0)])
            data_unique_crystal2 = np.int16(data_sorted[1, np.insert(data_split_pos, 0, 0)])
            data_split = np.split(data_sorted[2, :], data_split_pos)

            #### find the max count for all LORs and which LOR has the max count
            max_len = np.max([i.shape[0] for i in data_split])
            max_len_i = np.argmax([i.shape[0] for i in data_split])
            frag_size = 2000000 * max((1, int(1000/max_len)))
            num = int(len(data_split)/ frag_size)
            print(len(data_split), max_len, frag_size)
            for i in range(num + 1):
                #### make all LOR have same counts. Less count LORs are filled with NAN
                data_aranged = stack_padding(data_split[i*frag_size: np.minimum(len(data_split),(i+1)*frag_size)])
                #### find the mean value
                offset = np.nanmean(data_aranged, axis = 1)

                #### find the mean value of the right Gaussian [offset, offset + 2000]
                l_b = np.transpose(np.repeat([offset], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset + 2000], data_aranged.shape[1] ,axis = 0))
                data_aranged2 = np.array(data_aranged)
                data_aranged2[(data_aranged < l_b)|(data_aranged > r_b)] = np.nan
                offset2 = np.nanmean(data_aranged2, axis = 1)

                #### find the mean value of the left Gaussian [offset - 2000, offset]
                l_b = np.transpose(np.repeat([offset - 2000], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset], data_aranged.shape[1] ,axis = 0))
                data_aranged3 = np.array(data_aranged)
                data_aranged3[(data_aranged3 < l_b)|(data_aranged3 > r_b)] = np.nan
                offset3 = np.nanmean(data_aranged3, axis = 1)

                #### skew is defined as the average of the left Gaussian center and the right Gaussian center
                offset4 = (offset3 + offset2) * 0.5
                skewoffset[data_unique_crystal1[i*frag_size: (i+1)*frag_size]%pixel_num,data_unique_crystal2[i*frag_size: (i+1)*frag_size]%pixel_num] = offset4

                #### make all LOR have same counts. Less count LORs are filled with NAN
                data_aranged = stack_padding(data_split[i*frag_size: np.minimum(len(data_split),(i+1)*frag_size)])
                #### find the mean value
                offset = np.nanmean(data_aranged, axis = 1)

                #### find the mean value with bound [offset - 2000, offset + 2000]
                l_b = np.transpose(np.repeat([offset - 2000], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset + 2000], data_aranged.shape[1] ,axis = 0))
                data_aranged0 = np.array(data_aranged)
                data_aranged0[(data_aranged0 < l_b)|(data_aranged0 > r_b)] = np.nan
                offset0 = np.nanmean(data_aranged0, axis = 1)

                #### find the mean value of the right Gaussian [offset0, offset0 + 2000]
                l_b = np.transpose(np.repeat([offset0], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset0 + 2000], data_aranged.shape[1] ,axis = 0))
                data_aranged2 = np.array(data_aranged)
                data_aranged2[(data_aranged2 < l_b)|(data_aranged2 > r_b)] = np.nan
                offset2 = np.nanmean(data_aranged2, axis = 1)

                #### find the mean value of the left Gaussian [offset0 - 2000, offset0]
                l_b = np.transpose(np.repeat([offset0 - 2000], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset0], data_aranged.shape[1] ,axis = 0))
                data_aranged3 = np.array(data_aranged)
                data_aranged3[(data_aranged3 < l_b)|(data_aranged3 > r_b)] = np.nan
                offset3 = np.nanmean(data_aranged3, axis = 1)

                #### find the mean value of the right Gaussian [offset2 - 500, offset2 + 500]
                l_b = np.transpose(np.repeat([offset2-500], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset2+500], data_aranged.shape[1] ,axis = 0))
                data_aranged4 = np.array(data_aranged)
                data_aranged4[(data_aranged4 < l_b)|(data_aranged4 > r_b)] = np.nan
                offset4 = np.nanmean(data_aranged4, axis = 1)

                #### find the mean value of the left Gaussian [offset3 - 500, offset3 + 500]
                l_b = np.transpose(np.repeat([offset3-500], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset3+500], data_aranged.shape[1] ,axis = 0))
                data_aranged5 = np.array(data_aranged)
                data_aranged5[(data_aranged5 < l_b)|(data_aranged5 > r_b)] = np.nan
                offset5 = np.nanmean(data_aranged5, axis = 1)

                offset8 = (offset4 + offset5) * 0.5
                skewoffset[data_unique_crystal1[i*frag_size: (i+1)*frag_size]%pixel_num,data_unique_crystal2[i*frag_size: (i+1)*frag_size]%pixel_num] = offset8
             
            skewfile = open(f_skew, 'wb')
            skewfile.write(skewoffset.tobytes())
            skewfile.close()
          
        data_all[sub0*pixel_num:sub0*pixel_num+ pixel_num, sub1*pixel_num:sub1*pixel_num+ pixel_num] = skewoffset
        # Apply skew correction to the data
        data[2,:] = data[2,:] - data_all[np.uint16(data[0,:]),np.uint16(data[1,:])]

    # --- TOF Windowing of Prompt Data is done AFTER Skew Correction ---
    if TOF:
        tof_window_ps = 2000 # Maximum physical TOF difference for a 355mm scanner
        prompt_mask = (data[2,:] >= -tof_window_ps) & (data[2,:] <= tof_window_ps)
        removed_count = np.sum(~prompt_mask)
        data = data[:, prompt_mask]
        print(f"Removed {removed_count} events outside the TOF window from prompts (after skew correction).")
    # ------------------------------------------------------------------

    if Plot:
        tmp = data[2,:]
        plt.hist(tmp, np.linspace(-5000,5000,201))
        plt.ylim([0,lim_max]) 
        plt.xlabel('Skew/TOF-Corrected Time Difference (ps)')
        plt.ylabel('Coincidence Counts')
        plt.title('Time Spectrum after Skew/TOF Correction')
        plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin_skew_corrected.jpg')
        plt.clf()

    #### substract random events
    if RandomEstimation:
        # Include RandomMode in the corrected filename
        f_random_corrected = result_path + f'/{str(sub0)}_{str(sub1)}_random_corrected_{RandomMode}.dat'
        if not os.path.isfile(f_random_corrected):
            delay = 0
            for file in Files[f]:
                print(file.replace('coin','delay'))
                fo=open(file.replace('coin','delay'),"rb")
                data_tmp = np.fromfile(fo,dtype = np.int16)
                data_tmp = np.reshape(data_tmp, (int(data_tmp.shape[0]/3), 3)).transpose()
                if isinstance(delay, int):
                    delay = data_tmp
                else:
                    delay = np.concatenate([delay,data_tmp], axis = 1)
                print(delay.shape)
                fo.close()
            #### shift delay events to the promty window position
            delay[2, :] = delay[2, :]*1.5625 - 10*1600
            fo.close()
             
            # Apply skew correction to the delayed data as well
            if Skew:
                delay[2,:] = delay[2,:] - data_all[np.uint16(delay[0,:]),np.uint16(delay[1,:])]
             
            # --- TOF Window Filtering for DELAYED events (Done AFTER Skew Correction) ---
            if TOF:
                tof_window_ps = 2000 # Maximum physical TOF difference for a 355mm scanner
                delay_mask = (delay[2,:] >= -tof_window_ps) & (delay[2,:] <= tof_window_ps)
                removed_count = np.sum(~delay_mask)
                delay = delay[:, delay_mask]
                print(f"Removed {removed_count} events outside the TOF window from delays.")
             
            # 1. Paired-Event Rejection (Initial Pass, always done)
            print("Performing paired-event random rejection...")
            # Combine data for sorting
            data_combined = np.zeros((4, data.shape[1] + delay.shape[1]), dtype=data.dtype)
            data_combined[0:3, :data.shape[1]] = data
            data_combined[0:3, data.shape[1]:] = delay
            data_combined[3, :data.shape[1]] = 0 # 0 for prompt
            data_combined[3, data.shape[1]:] = 1 # 1 for delayed

            # Sort by LOR ID, then by event type (prompt/delayed), then by time
            lor_combined = np.int64(data_combined[0,:]) * 13824 + data_combined[1,:]
            argsort = np.lexsort((data_combined[2,:], data_combined[3,:], lor_combined))
            sorted_combined = data_combined[:, argsort]
            sorted_lor = lor_combined[argsort]
             
            # Find events that are adjacent and have the same LOR but different types
            is_valid_pair = (np.diff(sorted_lor) == 0) & (np.diff(sorted_combined[3, :]) == 1)
             
            # Remove both events in the pair
            valid_indices = np.where(is_valid_pair)[0]
            indices_to_remove = np.concatenate([valid_indices, valid_indices + 1])
            all_indices = np.arange(sorted_combined.shape[1])
            mask = np.isin(all_indices, indices_to_remove, invert=True)
             
            filtered_data = sorted_combined[:, mask]
            filtered_prompts = filtered_data[0:3, filtered_data[3, :] == 0]
            # Get the delayed events after paired-event rejection for the histogram step
            filtered_delayed = filtered_data[0:3, filtered_data[3, :] == 1]
            print(f"Removed {len(indices_to_remove)/2} pairs.")
             
            # --- CONDITIONAL EXECUTION BLOCK: HYBRID MODE (Optimized for Variance) ---
            if RandomMode == 'Hybrid':
                # Histogram-Based Correction (Second Pass)
                print("Performing **Model-Based** Hybrid Correction (Hybrid Mode)...")
                
                # Use the new, finer binning
                bins = np.linspace(-TOF_RANGE_PS/2, TOF_RANGE_PS/2, NUM_TOF_BINS) 
                bin_centers = (bins[:-1] + bins[1:]) / 2.0
                
                # 1. Create Histograms (Use filtered prompts and *original* delayed for better statistics)
                # Note: Using original delayed data, as it is only used to model the shape/scaling factor (alpha)
                prompt_hist, _ = np.histogram(filtered_prompts[2, :], bins=bins)
                delay_hist, _ = np.histogram(delay[2, :], bins=bins) # Using 'delay' (original delayed events)
                
                # 2. Fit the Delayed Histogram for a Smooth Randoms Estimate
                try:
                    # Model: Gaussian (for prompt remnant) + Constant (for randoms pedestal)
                    # Note: We fit the delayed data, which should be dominated by randoms (constant background)
                    # but may still contain a small Gaussian-like prompt peak if the delay is too small.
                    gmodel = GaussianModel(prefix='g1_')
                    cmodel = ConstantModel(prefix='c1_')
                    model = gmodel + cmodel
                    
                    # Estimate initial parameters
                    max_count = np.max(delay_hist)
                    max_idx = np.argmax(delay_hist)
                    center_ps = bin_centers[max_idx]
                    
                    params = model.make_params(
                        g1_amplitude=max_count * np.sqrt(2 * np.pi * 100**2), # Initial guess for amplitude
                        g1_center=center_ps,
                        g1_sigma=100.0,
                        c1_c=np.mean(delay_hist[(bins[:-1] < -3000) | (bins[:-1] > 3000)]) # Estimate constant from tails
                    )
                    
                    # Constraints (optional but recommended for stability)
                    params['g1_sigma'].min = 50 # Prevent absurdly narrow peaks
                    params['c1_c'].min = 0.0 # Cannot have negative randoms
                    
                    result = model.fit(delay_hist, params, x=bin_centers)
                    
                    # The Randoms estimate is the constant component (c1_c) of the fit
                    # Get the constant part of the fit
                    randoms_pedestal = result.best_values['c1_c']
                    
                    # Calculate alpha using the constant component of the fit:
                    # Alpha = (Mean Prompt Tail) / (Mean Delayed Tail)
                    # Mean Prompt Tail (from data) = Sum(Prompt_hist tails) / N_bins_tails
                    # Mean Delayed Tail (from fit) = c1_c (the constant randoms pedestal)
                    
                    tail_mask = (bin_centers < -3000) | (bin_centers > 3000)
                    if np.sum(tail_mask) > 0:
                        prompt_tail_mean = np.mean(prompt_hist[tail_mask])
                        alpha = prompt_tail_mean / randoms_pedestal if randoms_pedestal > 0 else 1.0
                    else:
                        alpha = 1.0 # Should not happen with current binning
                        
                    # Smooth randoms estimate for subtraction (a constant background is assumed)
                    randoms_hist = np.full_like(prompt_hist, alpha * randoms_pedestal)

                except Exception as e:
                    print(f"LMFIT model fitting failed ({e}). Falling back to simple tail ratio.")
                    tail_range_mask = (bins[:-1] < -3000) | (bins[:-1] > 3000)
                    if np.sum(delay_hist[tail_range_mask]) > 0:
                        alpha = np.sum(prompt_hist[tail_range_mask]) / np.sum(delay_hist[tail_range_mask])
                    else:
                        alpha = 1.0
                    randoms_hist = alpha * delay_hist # Fall back to direct scaling

                # 3. Perform Subtraction and Stochastic Rejection
                corrected_hist = prompt_hist - randoms_hist
                corrected_hist[corrected_hist < 0] = 0 # Ensure no negative counts

                # Stochastic Rejection to create list-mode data
                corrected_data = []
                 
                # Bin the filtered prompts to group events by TOF bin
                binned_prompts = np.digitize(filtered_prompts[2, :], bins) - 1
                 
                # Iterate through each TOF bin
                for i in range(len(bins) - 1):
                    events_in_bin_idx = np.where(binned_prompts == i)[0]
                    num_events_in_bin = len(events_in_bin_idx)
                     
                    if num_events_in_bin > 0:
                        events_to_keep = corrected_hist[i]
                        if events_to_keep > num_events_in_bin:
                            events_to_keep = num_events_in_bin
                         
                        rejection_prob = 1.0 - (events_to_keep / num_events_in_bin)
                         
                        # Keep events: Use a mask, then append to list
                        if rejection_prob > 0:
                            keep_mask = np.random.rand(num_events_in_bin) > rejection_prob
                            corrected_data.append(filtered_prompts[:, events_in_bin_idx[keep_mask]])
                        else:
                            
                            corrected_data.append(filtered_prompts[:, events_in_bin_idx])

                if corrected_data:
                    data = np.concatenate(corrected_data, axis=1) 
                else:
                    data = np.zeros((3, 0), dtype=np.int16) 
             
            else: # RandomMode == 'Pair' or any other value
                print("Using Pair-Only random subtraction mode.")
                data = filtered_prompts # Final result is the pair-rejected data

            # -----------------------------------------------------------------

            f_random = open(f_random_corrected, 'wb')
            f_random.write(data.tobytes())
            f_random.close()
        else:
            fo=open(f_random_corrected,"rb")
            data2 = np.fromfile(fo,dtype = np.int16)
            data2 = np.reshape(data2, (3, int(data2.shape[0]/3)))
            data = data2
            fo.close()
        if Plot:
            tmp = data[2,:]
            # Use a slightly coarser binning for plot visual clarity
            plt.hist(tmp, np.linspace(-5000,5000,100))
            plt.ylim([0,lim_max])
            plt.xlabel('Corrected Time Difference (ps)')
            plt.ylabel('Coincidence Counts')
            plt.title(f'Time Spectrum after Random Correction ({RandomMode} Mode)')
            plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin_skew_random_corrected.jpg')
            plt.clf()


    #### output listmode data
    if Listmode:
        # Use the specific f_random_corrected filename to ensure consistency
        f_listmode = listmode_path + '/' + str(sub0) + '_' + str(sub1) + '.lm'
        if not os.path.isfile(f_listmode):
            numberOfCoincidenceEvents = data.shape[1]
            # ... (Rest of Listmode generation remains the same) ...

            if numberOfCoincidenceEvents == 0:
                print(f"Skipping file generation for {f_listmode} due to zero events.")
                continue

            listmodedata = np.zeros((numberOfCoincidenceEvents, 10), dtype=np.float32)
            print('listmode data shape', listmodedata.shape)

            #### get crystal1 position xyz
            crystalID1_arr = np.uint16(data[0, :])
            tmp = crystalID1_arr
            tmp = Det_convert[np.uint16(tmp / 864)] * 864 + tmp % 864
            crystalID1_arr_convert = tmp
            listmodedata[:, 0] = crystalPositionMap[tmp, 0]  # firstCrystalX
            listmodedata[:, 1] = crystalPositionMap[tmp, 1]  # firstCrystalY
            listmodedata[:, 2] = crystalPositionMap[tmp, 2]  # firstCrystalZ

            #### get crystal2 position xyz
            crystalID2_arr = np.uint16(data[1, :])
            tmp = crystalID2_arr
            tmp = Det_convert[np.uint16(tmp / 864)] * 864 + tmp % 864
            crystalID2_arr_convert = tmp
            listmodedata[:, 5] = crystalPositionMap[tmp, 0]  # SecondCrystalX
            listmodedata[:, 6] = crystalPositionMap[tmp, 1]  # SecondCrystalY
            listmodedata[:, 7] = crystalPositionMap[tmp, 2]  # SecondCrystalZ

            #### get TOF info
            if TOF:
                # The skew-corrected and TOF-windowed time difference is the TOF value
                timediff_ps = -np.float32(data[2, :])
                TOF_arr = speedOfLight_length_ps * timediff_ps
                listmodedata[:, 3] = TOF_arr  # TOF
                if Plot:
                    tmp = TOF_arr
                    plt.hist(tmp, np.linspace(-1000, 1000, 201))
                    lim_max = np.max(np.histogram(tmp, np.linspace(-1000, 1000, 201))[0]) * 1.2
                    plt.ylim([0, lim_max])
                    plt.xlabel('TOF Distance (mm)')
                    plt.ylabel('Coincidence Counts')
                    plt.title('Final TOF Histogram')
                    plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin_random_corrected_tof_listmode.jpg')
                    plt.clf()

            #### store all coin events to listmode data
            l_index = np.arange(numberOfCoincidenceEvents)
            np.random.shuffle(l_index)
            listmodedata = listmodedata[l_index, :]
            with open(f_listmode, 'wb') as lm:
                seg_len = 100000
                for k in range(int(numberOfCoincidenceEvents / seg_len)):
                    lm.write(listmodedata[k * seg_len:(k + 1) * seg_len, :].tobytes())
                k = int(numberOfCoincidenceEvents / seg_len)
                lm.write(listmodedata[k * seg_len:, :].tobytes())
            lm.close()

            #### save a fraction data to txt file for validation
            num_events_to_save = min(100, numberOfCoincidenceEvents)
            eventData = np.zeros((num_events_to_save, 15), dtype=np.float32)

            eventData[:, :10] = listmodedata[:num_events_to_save, :]
            eventData[:, 10] = 0
            eventData[:, 11] = crystalID1_arr[:num_events_to_save]
            eventData[:, 12] = crystalID2_arr[:num_events_to_save]
            eventData[:, 13] = crystalID1_arr_convert[:num_events_to_save]
            eventData[:, 14] = crystalID2_arr_convert[:num_events_to_save]

            with open(f_listmode.replace('lm', 'txt'), 'a') as f:
                np.savetxt(f, eventData, fmt=textOutputFmt)

#### combine all detid pair skew_lut data to a big file
if Skew:
    Files = os.listdir(result_path)
    Files = [i for i in Files if '.dat' in i and 'skew' in i and 'lut' not in i]
    Files.sort(key=lambda f: int(f.split('_')[0])*1000 + int(f.split('_')[1]))
    print(Files)
    for f in Files:
        print(f)
        sub0 = int(f.split('_')[0])
        sub1 = int(f.split('_')[1])
        skewfile = open(result_path + '/' +  f, 'rb')
        data3 = np.fromfile(skewfile, np.int16)
        data3 = np.reshape(data3, (pixel_num, pixel_num))
        data_all[sub0*pixel_num:sub0*pixel_num+ pixel_num, sub1*pixel_num:sub1*pixel_num+ pixel_num] = data3
    f_skew = result_path + '/' +  'skew_lut.dat'
    skewfile = open(f_skew, 'wb')
    skewfile.write(data_all.tobytes())
    skewfile.close()


#### listmode data per detid pair based needs to be resampled and shuffled and then combined.
if ListmodeCombine:
    lstFiles = os.listdir(listmode_path)
    lstFiles = [i for i in lstFiles if '.lm' in i and '_' in i]
    print(lstFiles)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    o_dir = os.path.basename(script_dir)
    f_out = os.path.join(listmode_path, o_dir + '.lm')
    if not (os.path.isfile(f_out)):
        file_dict = {}
        fext=open(f_out,"wb")
        for i in range(1001):
            if i%100 == 0:
                print(i)
            coin_num = 0
            listmodedata = np.zeros((0,10), dtype = np.float32)
            for f in lstFiles:
                if f not in file_dict.keys():
                    file_dict[f]=[open(os.path.join(listmode_path,f),"rb"), os.path.getsize(os.path.join(listmode_path,f)), 0]
                counts = np.int32(file_dict[f][1] / 40 / 1000) * 10
                if i == 1000:
                    counts = np.int32((file_dict[f][1] - file_dict[f][2]) / 4)
                data = np.fromfile(file_dict[f][0],dtype = np.float32, count= counts)
                file_dict[f][2] += data.shape[0]*4
                data = np.reshape(data, (np.int32(data.shape[0]/10),10))
                coin_num += np.int32(data.shape[0])
                listmodedata = np.concatenate([listmodedata,data],axis = 0)
                if i == 1000:
                    file_dict[f][0].close()
            index = np.arange(coin_num)
            np.random.shuffle(index)
            fext.write(listmodedata[index, : ].tobytes())
        fext.close()
 #-------------------------------------------------------------------------------
## NECR Calculation and Plotting
#-------------------------------------------------------------------------------
if RandomEstimation and Plot:
    print("\nCalculating and Plotting NECR...")
    
    # Sort the counts by the detector pair key (f) to ensure order
    sorted_keys = sorted(all_prompt_counts.keys(), key=lambda f: int(f.split('_')[0])*1000 + int(f.split('_')[1]))
    
    prompt_rates = []
    random_rates = []
    necr_values = []
    
    # Use the acquisition time defined globally
    T_acq = TOTAL_ACQUISITION_TIME_SECONDS
    
    if T_acq > 0:
        for f in sorted_keys:
            # 1. Get total counts for this detector pair
            P_counts = all_prompt_counts.get(f, 0)
            R_counts = all_random_counts.get(f, 0)
            
            # 2. Convert counts to rates (counts/s)
            P_rate = P_counts / T_acq
            R_rate = R_counts / T_acq
            
            # 3. Calculate NECR rate (NECR = (P-R)^2 / (P + 2*R))
            # The calculation is often simplified to use the rates directly
            numerator = (P_rate - R_rate)**2
            denominator = P_rate + 2 * R_rate
            
            necr = numerator / denominator if denominator > 0 else 0
            
            prompt_rates.append(P_rate)
            random_rates.append(R_rate)
            necr_values.append(necr)
            
        print(f"Total NECR for all processed pairs: {np.sum(necr_values):.2f} counts/s")
        
        # --- Plotting NECR ---
        plt.figure(figsize=(10, 6))
        
        # For a basic plot, you can use the index as the x-axis
        x_axis = np.arange(len(sorted_keys))
        
        plt.plot(x_axis, prompt_rates, 'b-', label='Prompt Rate ($P$)', linewidth=2)
        plt.plot(x_axis, random_rates, 'r-', label='Random Rate ($R$)', linewidth=2)
        plt.plot(x_axis, necr_values, 'g-', label='NECR', linewidth=3)
        
        plt.xlabel('Detector Pair Index')
        plt.ylabel('Rate (counts/s)')
        plt.title('Noise Equivalent Count Rate (NECR) vs. Detector Pair')
        plt.legend()
        
        # Optional: Set x-ticks to display actual detid pairs less frequently
        step = max(1, len(sorted_keys) // 10)
        plt.xticks(x_axis[::step], sorted_keys[::step], rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save plot to the result folder
        plt.tight_layout()
        plt.savefig(folder + f'/NECR_Plot_{RandomMode}.jpg')
        plt.clf()
    else:
        print("NECR calculation skipped: TOTAL_ACQUISITION_TIME_SECONDS is not set.")       