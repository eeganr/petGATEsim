import os
import shutil
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
import pickle
import sys
import argparse
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d
 
# =============================================================================
# 1. SETUP & PARAMETERS
# =============================================================================
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEOMETRY_FILE = 'geometry.pickle'
SKEW_LUT_FILE = 'skew_lut.dat'
 
parser = argparse.ArgumentParser(description='PET Correction Pipeline')
parser.add_argument(dest='dir_origin', help='Path of data folder')
# Removed Hist, Sinogram, Hybrid. Added KDE-LM.
parser.add_argument('--modes', nargs='+', default=['EventPair', 'KDE-LM'], #['NoRandom', 'EventPair', 'KDE-LM']
                    help='Correction modes to run.')
args = parser.parse_args()
 
print(f"--- Processing: {args.dir_origin} ---")
print(f"--- Modes: {args.modes} ---")
 
# Constants matching process_skew_mod.py
pixel_num = 864
Listmode = True
Skew = False
TOF = True
ListmodeCombine = True
GenerateRandomLUT = True
Plot = True
Det_convert = np.arange(0, 16, dtype=int)
speedOfLight = 299792458000
speedOfLight_length_ps = speedOfLight * math.pow(10, -12)
 
# Specific pairs to save for MAT analysis (sorted tuples for consistency)
SAVE_PAIRS = [tuple(sorted(p)) for p in [(0,7), (0,1), (12,13), (5,12), (2,10), (8,9)]]
mat_data_store = {}
 
# Output Directories
base_res = os.path.join(args.dir_origin, 'result')
skew_res = os.path.join(base_res, 'Skew')
plot_dir = os.path.join(base_res, 'Plots')
plot_svg = os.path.join(plot_dir, 'SVG')
plot_png = os.path.join(plot_dir, 'PNG')
 
for p in [base_res, skew_res, plot_dir, plot_svg, plot_png]:
    os.makedirs(p, exist_ok=True)
 
# Listmode & Randoms LUT Directories
lm_paths = {}
lut_paths = {}
for mode in args.modes:
    p_lm = os.path.join(base_res, f'Listmode_{mode}')
    os.makedirs(p_lm, exist_ok=True)
    lm_paths[mode] = p_lm
   
    if GenerateRandomLUT:
        p_lut = os.path.join(base_res, f'RandomLUT_{mode}')
        os.makedirs(p_lut, exist_ok=True)
        lut_paths[mode] = p_lut
 
# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
 
def stack_padding(l):
    return np.column_stack((itertools.zip_longest(*l, fillvalue=np.nan)))
 
def get_lor_histogram(data_array, p_num):
    if data_array.shape[1] == 0:
        return np.zeros((p_num, p_num), dtype=np.float32)
    c1 = np.uint16(data_array[0, :]) % p_num
    c2 = np.uint16(data_array[1, :]) % p_num
    flat_indices = c1.astype(np.int64) * p_num + c2.astype(np.int64)
    hist = np.bincount(flat_indices, minlength=p_num*p_num)
    return hist.reshape((p_num, p_num)).astype(np.float32)
 
def algo_EventPair_Ref(data, delay):
    """
    Exact implementation of EventPair from process_skew_mod.py
    """
    # 1. Unique LOR ID (process_skew_mod uses 13824 multiplier)
    coin_LOR = np.int64(data[0,:])*13824 + data[1,:]
    delay_LOR = np.int64(delay[0,:])*13824 + delay[1,:]
   
    # 2. Combine
    coin = np.zeros((5, data.shape[1] + delay.shape[1]), dtype=np.int16)
   
    # Fill Coin
    coin[0,:data.shape[1]] = coin_LOR
    coin[1,:data.shape[1]] = 0 # Type 0
    coin[2:5,:data.shape[1]] = data
   
    # Fill Delay
    coin[0,data.shape[1]:] = delay_LOR
    coin[1,data.shape[1]:] = 1 # Type 1
    coin[2:5,data.shape[1]:] = delay
   
    # 3. Sort (Type << Time << LOR)
    argsort = np.lexsort((coin[1,:], coin[4,:], coin[0,:]))
    coin_sorted = coin[:,argsort]
   
    prev = 0
   
    # 4. Iterative Subtraction Loop
    for i in range(1000):
        # valid: diff(LOR)==0 AND diff(Type)==1
        valid = np.insert((np.diff(coin_sorted[0,]) == 0) & (np.diff(coin_sorted[1,]) == 1), 0, False)
       
        valid2 = (coin_sorted[1,:] == 1) # count delays
        curr = np.sum(valid2)
       
        if(curr == prev):
            break
        prev = curr
       
        # Mark both Delay and the Coin before it
        valid = valid | (np.insert(valid[1:], valid.size - 1, False))
       
        coin_sorted = coin_sorted[:, ~valid]
 
    # Return only Prompts (Type 0)
    final_data = coin_sorted[2:5, coin_sorted[1,:] == 0]
    return final_data.astype(np.float32)
 
def algo_KDE_LM(data_prompt, data_delay):
    """
    Renamed from algo_Hybrid to algo_KDE_LM
    """
    # --- DO NOT RUN EVENTPAIR FIRST ---
    # res = algo_EventPair_Ref(data_prompt, data_delay) <--- DELETE THIS
   
    # Work directly on Prompts
    tp = data_prompt[2, :]
   
    # 1. Build the Randoms Model from the DELAY window
    # The delays are the pure measurement of the randoms.
    td = data_delay[2, :]
   
    # Create histograms with matching bins
    bins = np.linspace(-5000, 5000, 401)
    hp, edges = np.histogram(tp, bins=bins)
    hd, _     = np.histogram(td, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
 
    # 2. Smooth the Delay Histogram (Variance Reduction)
    # This replaces the "tail median" logic. We trust the Delay window, but smooth it
    # to remove the "salt and pepper" noise.
    randoms_model = gaussian_filter1d(hd.astype(float), sigma=10.0)
   
    # 3. Smooth Prompts slightly (for stability)
    prompt_model = gaussian_filter1d(hp.astype(float), sigma=1.0)
 
    # 4. Calculate Probability Curve
    # P(Keep) = (Prompts - Smoothed_Randoms) / Prompts
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_keep_curve = (prompt_model - randoms_model) / prompt_model
   
    # Safety clamping
    prob_keep_curve[np.isnan(prob_keep_curve)] = 0.0
    prob_keep_curve = np.clip(prob_keep_curve, 0.0, 1.0)
 
    # 5. Vectorized Filtering (Same as your code)
    event_probs = np.interp(tp, bin_centers, prob_keep_curve, left=0.0, right=0.0)
    rand_vals = np.random.rand(len(tp))
    keep_mask = rand_vals < event_probs
 
    # Return filtered PROMPTS (not EP results)
    return data_prompt[:, keep_mask]
 
# =============================================================================
# 3. MAIN PIPELINE
# =============================================================================
 
Files = {}
for root, dirs, files in os.walk(args.dir_origin):
    for name in files:
        if  '.dat' in name and 'coin' in name:
            if name not in Files.keys(): Files[name] = []
            Files[name] += [os.path.join(root, name)]
keys = list(Files.keys())
keys.sort(key=lambda f: int(f.split('_')[0])*1000 + int(f.split('_')[1]))
 
if os.path.exists(GEOMETRY_FILE):
    with open(GEOMETRY_FILE, 'rb') as f: crystalPositionMap = pickle.load(f)
else:
    sys.exit(f"Geometry missing: {GEOMETRY_FILE}")
 
# Initialize Global Skew
global_skew = np.zeros((864*16, 864*16), dtype=np.int16)
skew_loaded = False
 
# Logic 3: If skew_lut.dat presented, use it
if Skew and os.path.isfile(SKEW_LUT_FILE):
    print("Loaded Global Skew LUT.")
    with open(SKEW_LUT_FILE, 'rb') as f:
        raw = np.fromfile(f, np.int16)
        if raw.size == 864*16*864*16:
            global_skew = np.reshape(raw, (864*16, 864*16))
            skew_loaded = True
 
for f in keys:
    sub0, sub1 = int(f.split('_')[0]), int(f.split('_')[1])
   
    is_save_target = tuple(sorted((sub0, sub1))) in SAVE_PAIRS
   
    all_done = True
    for m in args.modes:
        if not os.path.exists(os.path.join(lm_paths[m], f'{sub0}_{sub1}.lm')):
            all_done = False
            break
   
    if all_done and not is_save_target:
        print(f"Skipping Pair {sub0}_{sub1} (Already Processed)")
        continue
 
    print(f"\nProcessing Pair: {sub0}_{sub1}")
   
    # Logic 1: File Loading (Matched)
    data = 0
    for file in Files[f]:
        fo = open(file, "rb")
        data_tmp = np.fromfile(fo, dtype=np.int16)
        data_tmp = np.reshape(data_tmp, (int(data_tmp.shape[0]/3), 3)).transpose()
        if isinstance(data, int):
            data = data_tmp
        else:
            data = np.concatenate([data, data_tmp], axis=1)
        fo.close()
   
    # Convert TDC bin to ps (1.5625)
    data[2,:] = data[2,:] * 1.5625
   
    # Capture Raw Time for Plotting
    data_raw_time = data[2,:].copy()
   
    # Load Delay
    delay = 0
    has_delay = False
    d_files = [file.replace('coin','delay') for file in Files[f] if os.path.exists(file.replace('coin','delay'))]
    if len(d_files) > 0:
        for file in d_files:
            fo = open(file, "rb")
            data_tmp = np.fromfile(fo, dtype=np.int16)
            data_tmp = np.reshape(data_tmp, (int(data_tmp.shape[0]/3), 3)).transpose()
            if isinstance(delay, int):
                delay = data_tmp
            else:
                delay = np.concatenate([delay, data_tmp], axis=1)
            fo.close()
        delay[2,:] = delay[2,:] * 1.5625 - 16000 # Shift
        has_delay = True
 
    # Logic 2: Skew Calculation
    skewoffset = np.zeros((pixel_num, pixel_num), dtype=np.int16)
    f_skew_local = os.path.join(skew_res, f'{sub0}_{sub1}_skew_array.dat')
 
    if Skew:
        if os.path.isfile(f_skew_local):
            skewfile = open(f_skew_local, 'rb')
            skewoffset = np.fromfile(skewfile, np.int16)
            skewoffset = np.reshape(skewoffset, (pixel_num, pixel_num))
            skewfile.close()
        elif skew_loaded:
             skewoffset = global_skew[sub0*pixel_num:sub0*pixel_num+ pixel_num, sub1*pixel_num:sub1*pixel_num+ pixel_num]
        else:
            # Exact Iterative Logic
            data_argsort = np.lexsort((data[1, :], data[0, :]))
            data_sorted = data[:, data_argsort]
            data_split_pos = np.where(np.diff(data_sorted[1, :]))[0] + 1
            data_unique_crystal1 = np.int16(data_sorted[0, np.insert(data_split_pos, 0, 0)])
            data_unique_crystal2 = np.int16(data_sorted[1, np.insert(data_split_pos, 0, 0)])
            data_split = np.split(data_sorted[2, :], data_split_pos)
           
            frag_size = 2000000
            num = int(len(data_split)/ frag_size) if len(data_split) > 0 else 0
           
            for i in range(num + 1):
                chunk_list = data_split[i*frag_size: np.minimum(len(data_split),(i+1)*frag_size)]
                if not chunk_list: continue
                data_aranged = stack_padding(chunk_list)
               
                offset = np.nanmean(data_aranged, axis = 1)
                l_b = np.transpose(np.repeat([offset - 2000], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset + 2000], data_aranged.shape[1] ,axis = 0))
                data_aranged0 = np.array(data_aranged)
                data_aranged0[(data_aranged0 < l_b)|(data_aranged0 > r_b)] = np.nan
                offset0 = np.nanmean(data_aranged0, axis = 1)
               
                l_b = np.transpose(np.repeat([offset0], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset0 + 2000], data_aranged.shape[1] ,axis = 0))
                data_aranged2 = np.array(data_aranged)
                data_aranged2[(data_aranged2 < l_b)|(data_aranged2 > r_b)] = np.nan
                offset2 = np.nanmean(data_aranged2, axis = 1)
               
                l_b = np.transpose(np.repeat([offset0 - 2000], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset0], data_aranged.shape[1] ,axis = 0))
                data_aranged3 = np.array(data_aranged)
                data_aranged3[(data_aranged3 < l_b)|(data_aranged3 > r_b)] = np.nan
                offset3 = np.nanmean(data_aranged3, axis = 1)
               
                l_b = np.transpose(np.repeat([offset2-500], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset2+500], data_aranged.shape[1] ,axis = 0))
                data_aranged4 = np.array(data_aranged)
                data_aranged4[(data_aranged4 < l_b)|(data_aranged4 > r_b)] = np.nan
                offset4 = np.nanmean(data_aranged4, axis = 1)
               
                l_b = np.transpose(np.repeat([offset3-500], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset3+500], data_aranged.shape[1] ,axis = 0))
                data_aranged5 = np.array(data_aranged)
                data_aranged5[(data_aranged5 < l_b)|(data_aranged5 > r_b)] = np.nan
                offset5 = np.nanmean(data_aranged5, axis = 1)
               
                offset8 = (offset4 + offset5) * 0.5
                c1_idx = data_unique_crystal1[i*frag_size: (i+1)*frag_size] % pixel_num
                c2_idx = data_unique_crystal2[i*frag_size: (i+1)*frag_size] % pixel_num
                skewoffset[c1_idx, c2_idx] = offset8
           
            skewfile = open(f_skew_local, 'wb')
            skewfile.write(skewoffset.tobytes())
            skewfile.close()
 
        global_skew[sub0*pixel_num:sub0*pixel_num+ pixel_num, sub1*pixel_num:sub1*pixel_num+ pixel_num] = skewoffset
       
        # PLOT SKEW CHECK
        if Plot:
            tmp = data[2,:]
            lim_max = np.max(np.histogram(tmp, np.linspace(-5000,5000,201))[0]) * 1.2
            tmp_corrected = data[2,:] - skewoffset[np.uint16(data[0,:])%pixel_num, np.uint16(data[1,:])%pixel_num]
            plt.hist(tmp_corrected, np.linspace(-5000,5000,201))
            plt.ylim([0,lim_max])
            plt.savefig(os.path.join(skew_res, f'{sub0}_{sub1}_coin_random_corrected_tof.jpg'))
            plt.clf()
 
    # ** PLOT SKEW OVERLAY (Before vs After) **
    if Plot:
        pname = f'{sub0}_{sub1}_Skew_BeforeAfter'
        if not os.path.exists(os.path.join(plot_svg, pname + '.svg')):
            plt.figure(figsize=(10,6))
            y_raw, x = np.histogram(data_raw_time, bins=200, range=(-5000, 5000))
            centers = (x[:-1]+x[1:])/2
            plt.plot(centers, y_raw, color='gray', alpha=0.5, label='Before Skew')
           
            c1_arr = np.uint16(data[0,:])
            c2_arr = np.uint16(data[1,:])
            skew_vals = global_skew[sub0*pixel_num + c1_arr%864, sub1*pixel_num + c2_arr%864]
            corrected_time = skew_vals - data[2,:]
           
            y_corr, _ = np.histogram(corrected_time, bins=200, range=(-5000, 5000))
            plt.plot(centers, y_corr, color='blue', label='After Skew')
            plt.title(f'Skew Correction {sub0}-{sub1}')
            plt.legend()
            # Save SVG
            plt.savefig(os.path.join(plot_svg, pname + '.svg'))
            # Save PNG
            plt.savefig(os.path.join(plot_png, pname + '.png'))
            plt.close()
 
    # 3. RUN METHODS
    if GenerateRandomLUT:
        hist_orig = get_lor_histogram(data, pixel_num)
 
    comparison_plot_data = {}
 
    for m in args.modes:
        fname_lm = os.path.join(lm_paths[m], f'{sub0}_{sub1}.lm')
       
        if m == 'NoRandom':
            res = data.astype(np.float32)
        elif m == 'EventPair' and has_delay:
            res = algo_EventPair_Ref(data, delay)
        elif m == 'KDE-LM' and has_delay:
            res = algo_KDE_LM(data, delay)
        else:
            res = data.astype(np.float32)
       
        c1 = np.uint16(res[0,:]); c2 = np.uint16(res[1,:])
        offset_arr = global_skew[sub0*pixel_num + c1%864, sub1*pixel_num + c2%864]
        timediff_ps = np.float32(offset_arr) - np.float32(res[2,:])
       
        comparison_plot_data[m] = timediff_ps
 
        if is_save_target:
            key_name = f'pair_{sub0}_{sub1}_{m}'
            mat_data_store[key_name] = timediff_ps
 
        if GenerateRandomLUT:
            f_lut = os.path.join(lut_paths[m], f'{sub0}_{sub1}_random_lut.dat')
            if not os.path.exists(f_lut):
                hist_corrected = get_lor_histogram(res, pixel_num)
                randoms_map = np.maximum(0, hist_orig - hist_corrected)
                with open(f_lut, 'wb') as f:
                    f.write(randoms_map.tobytes())
 
        if Listmode and not os.path.isfile(fname_lm):
            N = res.shape[1]
            if N > 0:
                lm = np.zeros((N, 10), dtype=np.float32)
                c1g = Det_convert[np.uint16(c1/864)]*864 + c1%864
                c2g = Det_convert[np.uint16(c2/864)]*864 + c2%864
                lm[:,0]=crystalPositionMap[c1g,0]; lm[:,1]=crystalPositionMap[c1g,1]; lm[:,2]=crystalPositionMap[c1g,2]
                lm[:,5]=crystalPositionMap[c2g,0]; lm[:,6]=crystalPositionMap[c2g,1]; lm[:,7]=crystalPositionMap[c2g,2]
                if TOF:
                    lm[:,3] = speedOfLight_length_ps * timediff_ps
               
                idx = np.arange(N); np.random.shuffle(idx)
                lm = lm[idx, :]
                with open(fname_lm, 'wb') as lm_out:
                    seg_len = 100000
                    for k in range(int(N/seg_len)):
                        lm_out.write(lm[k*seg_len:(k+1)*seg_len,:].tobytes())
                    k = int(N/seg_len)
                    lm_out.write(lm[k*seg_len:,:].tobytes())
 
    # ** COMPARISON OVERLAY PLOT **
    if Plot and len(comparison_plot_data) > 0:
        pname = f'{sub0}_{sub1}_Comparison_Overlay'
        if not os.path.exists(os.path.join(plot_svg, pname + '.svg')):
            plt.figure(figsize=(12,8))
            # Updated colors for new mode names
            colors = {'NoRandom':'gray', 'EventPair':'red', 'KDE-LM':'magenta'}
            bins = np.linspace(-5000, 5000, 201)
            centers = (bins[:-1] + bins[1:]) / 2
           
            for m, times in comparison_plot_data.items():
                hist, _ = np.histogram(times, bins=bins)
                plt.plot(centers, hist, label=m, color=colors.get(m,'blue'), alpha=0.8)
           
            plt.title(f'Comparison of All Modes: {sub0}_{sub1}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            # Save SVG
            plt.savefig(os.path.join(plot_svg, pname + '.svg'))
            # Save PNG
            plt.savefig(os.path.join(plot_png, pname + '.png'))
            plt.close()
 
if Skew and not skew_loaded:
    with open(SKEW_LUT_FILE, 'wb') as f: f.write(global_skew.tobytes())
 
if mat_data_store:
    mat_path = os.path.join(base_res, 'Detector_Histograms.mat')
    print(f"Saving analysis data to {mat_path}...")
    sio.savemat(mat_path, mat_data_store)
 
if ListmodeCombine:
    print("\nCombining Listmode files...")
    for mode in args.modes:
        src_dir = lm_paths[mode]
        f_out = os.path.join(src_dir, f'Combined_{mode}.lm')
        lstFiles = [f for f in os.listdir(src_dir) if '.lm' in f and 'Combined' not in f and '_' in f]
        if not os.path.isfile(f_out) and len(lstFiles) > 0:
            print(f"Combining {len(lstFiles)} files for {mode}...")
            file_dict = {}
            fext = open(f_out, "wb")
            for i in range(1001):
                if i % 100 == 0: print(f"  {i/10}%")
                coin_num = 0
                listmodedata = np.zeros((0, 10), dtype=np.float32)
                for f in lstFiles:
                    if f not in file_dict: file_dict[f] = [open(os.path.join(src_dir,f),"rb"), os.path.getsize(os.path.join(src_dir,f)), 0]
                    counts = np.int32(file_dict[f][1] / 40 / 1000) * 10
                    if i == 1000: counts = np.int32((file_dict[f][1] - file_dict[f][2]) / 4)
                    data = np.fromfile(file_dict[f][0], dtype=np.float32, count=counts)
                    file_dict[f][2] += data.shape[0] * 4
                    if data.size > 0:
                        data = np.reshape(data, (np.int32(data.shape[0]/10), 10))
                        coin_num += np.int32(data.shape[0])
                        listmodedata = np.concatenate([listmodedata, data], axis=0)
                    if i == 1000: file_dict[f][0].close()
                if coin_num > 0:
                    index = np.arange(coin_num)
                    np.random.shuffle(index)
                    fext.write(listmodedata[index, :].tobytes())
            fext.close()
 
if GenerateRandomLUT:
    print("\nCombining Randoms LUTs...")
    for mode in args.modes:
        src_dir = lut_paths[mode]
        f_out = os.path.join(base_res, f'RandomLUT_{mode}_Global.dat')
        lstFiles = [f for f in os.listdir(src_dir) if '_random_lut.dat' in f]
        if len(lstFiles) > 0 and not os.path.exists(f_out):
            global_lut = np.zeros((864*16, 864*16), dtype=np.float32)
            for f in lstFiles:
                parts = f.split('_')
                sub0, sub1 = int(parts[0]), int(parts[1])
                with open(os.path.join(src_dir, f), 'rb') as fin:
                    local_lut = np.fromfile(fin, dtype=np.float32)
                    local_lut = local_lut.reshape((pixel_num, pixel_num))
                global_lut[sub0*pixel_num:(sub0+1)*pixel_num, sub1*pixel_num:(sub1+1)*pixel_num] = local_lut
            with open(f_out, 'wb') as fout:
                fout.write(global_lut.tobytes())
            print(f"Saved: {f_out}")
