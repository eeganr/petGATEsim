import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from lmfit.models import GaussianModel, ConstantModel

# BASIC CONFIG
DETECTORS_SIM = 12288
MODULES = 16
crystals_per_det = DETECTORS_SIM // MODULES

RandomMode = 'Hybrid'  
# -------------------------------------------

# --- NEW HYBRID PARAMETERS ---
TOF_BIN_SIZE_PS = 10 # Finer binning to reduce local variance (e.g., 10 ps)
TOF_BIN_SIZE_MM = TOF_BIN_SIZE_PS * 1e-12 * 299792458 * 1e3
TOF_RANGE_PS = 1.6e3 * 2 # Total range -5000 ps to 5000 ps
TOF_RANGE_MM = TOF_RANGE_PS * 1e-12 * 299792458 * 1e3 * 2
NUM_TOF_BINS = int(TOF_RANGE_MM / TOF_BIN_SIZE_MM) + 1 # 1001 bins for 10 ps
TAIL_LIMIT_MM = 800 # mm
# -----------------------------

# ----

# Args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infolder", type=str, help="in folder for split data")
parser.add_argument("-o", "--outfolder", type=str, help="out folder")
parser.add_argument("-l", "--lor", default=None, type=int, help="lor id")
parser.add_argument("-a", "--adetector", type=int, default=None, help="index of det a")
parser.add_argument("-b", "--bdetector", type=int, default=None, help="index of det b")
args = parser.parse_args()
IN_FOLDER = args.infolder
OUT_FOLDER = args.outfolder
if (args.lor == None):
    i = args.adetector
    j = args.bdetector
    if (i == None or j == None):
        raise Exception("not enough LOR info provided")
else:
    lor = args.lor
    i = 0
    j = 1
    while True:
        if lor - (MODULES - j) >= 0:
            lor -= (MODULES - j)
            i += 1
            j += 1
        else:
            j += lor
            break
    print("Processing LOR ", i, j)
# ---

if os.path.isfile(f'{OUT_FOLDER}{i}_{j}_spcorr.lm'):
    print("Exists!")
    quit()

# Make stats directory
print("Making stats dir...")
os.makedirs(f'{OUT_FOLDER}stats', exist_ok=True)


# LOAD DATA
print("Loading data...")
data = np.memmap(f'{IN_FOLDER}split/{i}_{j}_coin.lm', dtype=np.float32, mode='r')
data = data.reshape(-1, 10)
tofs = data[:, 3]

delay = np.memmap(f'{IN_FOLDER}split/{i}_{j}_delay.lm', dtype=np.float32, mode='r')
delay = delay.reshape(-1, 10)
deltofs = delay[:, 3]

act = np.memmap(f'{IN_FOLDER}split/{i}_{j}_actual.lm', dtype=np.float32, mode='r')
act = act.reshape(-1, 10)
acttofs = act[:, 3]


# GENERATE SP DATA

print("Generating SP Data")
def gen_sp_randoms(i, j, filename):
    def slice_sp(i, j):
        start_i = i * crystals_per_det
        stop_i = (i + 1) * crystals_per_det
        start_j = j * crystals_per_det
        stop_j = (j + 1) * crystals_per_det
        return slice(start_i, stop_i), slice(start_j, stop_j)
    
    sp = np.load(filename)
    sec = sp[slice_sp(i, j)]

    sec_int = np.floor(sec) + np.astype(np.random.rand(*sec.shape) < sec % 1, np.int64)
    sec_int = sec_int.astype(np.int64)
    sp_randoms = np.sum(sec_int)
    sp_gen = np.array([
        np.zeros(sp_randoms), np.zeros(sp_randoms), np.zeros(sp_randoms),
        np.random.rand(sp_randoms) * (np.max(tofs) - np.min(tofs)) + np.min(tofs),
        np.zeros(sp_randoms), np.zeros(sp_randoms), np.zeros(sp_randoms), np.zeros(sp_randoms),
        np.concatenate([np.full(np.int64(np.sum(sec_int[row])), row + i * crystals_per_det) for row in range(crystals_per_det)]),
        np.concat([np.repeat(np.arange(j * crystals_per_det, (j + 1) * crystals_per_det), sec_int[col]) for col in range(crystals_per_det)])
    ]).T

    return sp_gen

sps = gen_sp_randoms(i, j, f'{IN_FOLDER}sp.npy')
sptofs = sps[:, 3]


# ACTUALS MATCHING AND SUBTRACTION

print("Actuals Matching & Subtracting Data")

def match_actuals(data, act):
    datasort = np.lexsort((data[:, 3], data[:, -1], data[:, -2]))
    unsort = np.argsort(datasort)
    data = data[datasort]
    act = act[np.lexsort((act[:, 3], act[:, -1], act[:, -2]))]
    israndom = np.zeros(data.shape[0], dtype=bool)
    dat_i = 0
    act_i = 0
    while act_i < act.shape[0]:
        if (data[dat_i][3] == act[act_i][3]):
            israndom[dat_i] = True
            act_i += 1
        dat_i += 1
    return israndom[unsort]


def rm_random(data, delay):
    coin = np.array([
        np.concatenate(( # coin lors followed by delay lors
            data[:, 8].astype(np.int64) * DETECTORS_SIM + data[:, 9].astype(np.int64), 
            delay[:, 8].astype(np.int64) * DETECTORS_SIM + delay[:, 9].astype(np.int64)
        )),
        np.concatenate(( # zeros followed by ones
            np.zeros(data.shape[0]),
            np.ones(delay.shape[0])
        )),
        np.concatenate(( # crystal ID 1s followed by delay crystal ID 1s
            data[:, 8],
            delay[:, 8]
        )),
        np.concatenate(( # crystal ID 2s followed by delay crystal ID 2s
            data[:, 9],
            delay[:, 9]
        )),
        np.concatenate(( # TOFs followed by delay TOFs
            data[:, 3],
            delay[:, 3]
        )),
    ],dtype=np.float64)
    
    #### subtract coin events which have same nearby delay events with same LOR id.

    index = np.linspace(0, data.shape[0] + delay.shape[0] - 1, data.shape[0] + delay.shape[0], dtype = np.int64)
    print(index.size)
    #### sort based on coin/delay id << time << LOR id
    argsort = np.lexsort((coin[1,:], coin[4,:], coin[0,:]))
    coin_sorted = coin[:,argsort]
    index = index[argsort]
    prev = 0

    for i in range(1000):
        #### same LOR crystal pair but one coin and one delay
        valid = np.insert((np.diff(coin_sorted[0,]) == 0)&(np.diff(coin_sorted[1,]) == 1),0,False)
        ###               --- within same LOR        ---  and next is delay but ours is coin ---
        valid2 = (coin_sorted[1,:] == 1) # is a delay
        curr = np.sum(valid2) # number of delays
        print("residual delay: ", curr) 
        if(curr == prev or curr == 0):
            index = index[~valid2] # gets rid of remaining delays
            break
        prev = curr 
        valid = valid | (np.insert(valid[1:],valid.size - 1,False)) # insert false at end
        index = index[~valid]
        coin_sorted = coin_sorted[:,~valid]

    return index


def rm_random_new(data_of, delay_of):
    data = data_of.copy().T
    delay = delay_of.copy().T

    index = np.linspace(0, data.shape[1] + delay.shape[1] - 1, data.shape[1] + delay.shape[1], dtype=np.int64)

    data_combined = np.array([
        np.concatenate(( # crystal ID 1s followed by delay crystal ID 1s
            data[8, :],
            delay[8, :]
        )),
        np.concatenate(( # crystal ID 2s followed by delay crystal ID 2s
            data[9, :],
            delay[9, :]
        )),
        np.concatenate(( # TOFs followed by delay TOFs
            data[3, :],
            delay[3, :]
        )),
        np.concatenate(( # zeros followed by ones
            np.zeros(data.shape[1]),
            np.ones(delay.shape[1])
        )),
    ],dtype=np.float64)

    # Sort by LOR ID, then by event type (prompt/delayed), then by time
    lor_combined = np.concatenate(( # coin lors followed by delay lors
        data[8, :].astype(np.int64) * DETECTORS_SIM + data[9, :].astype(np.int64), 
        delay[8, :].astype(np.int64) * DETECTORS_SIM + delay[9, :].astype(np.int64)
    ))
    argsort = np.lexsort((data_combined[3,:], data_combined[2,:], lor_combined))
    sorted_combined = data_combined[:, argsort]
    sorted_lor = lor_combined[argsort]
    index = index[argsort]
    
    # Find events that are adjacent and have the same LOR but different types
    is_valid_pair = (np.diff(sorted_lor) == 0) & (np.diff(sorted_combined[3, :]) == 1)
    
    # Remove both events in the pair
    valid_indices = np.where(is_valid_pair)[0]
    indices_to_remove = np.concatenate([valid_indices, valid_indices + 1])
    all_indices = np.arange(sorted_combined.shape[1])
    mask = np.isin(all_indices, indices_to_remove, invert=True)
    
    filtered_data = sorted_combined[:, mask]
    onlydatamask = np.where(filtered_data[3, :] == 0)
    filtered_prompts = filtered_data[0:3, filtered_data[3, :] == 0]
    index = index[mask]
    index = index[onlydatamask]
    print(f"Removed {len(indices_to_remove)/2} pairs.")

    # 3. Histogram-Based Correction (Second Pass)
    print("Performing histogram-based correction...")
    bins = np.linspace(-5000, 5000, 201)
    prompt_hist, _ = np.histogram(filtered_prompts[2, :], bins=bins)
    delay_hist, _ = np.histogram(delay_of[:, 3], bins=bins)

    # Find a scaling factor (alpha) from the tails
    tail_range_mask = (bins[:-1] < -TAIL_LIMIT_MM) | (bins[:-1] > TAIL_LIMIT_MM)
    
    if np.sum(delay_hist[tail_range_mask]) > 0:
        alpha = np.sum(prompt_hist[tail_range_mask]) / np.sum(delay_hist[tail_range_mask])
    else:
        alpha = 1.0

    randoms_hist = alpha * delay_hist
    corrected_hist = prompt_hist - randoms_hist
    corrected_hist[corrected_hist < 0] = 0 # Ensure no negative counts

    # 4. Stochastic Rejection to create list-mode data
    corrected_data = []
    
    # Bin the filtered prompts to group events by TOF bin
    binned_prompts = np.digitize(filtered_prompts[2, :], bins) - 1
    
    final_idxs = []

    # Iterate through each TOF bin
    for i in range(len(bins) - 1):
        events_in_bin_idx = np.where(binned_prompts == i)[0]
        bin_idx_in_index = index[events_in_bin_idx]
        num_events_in_bin = len(events_in_bin_idx)
        
        if num_events_in_bin > 0:
            # Calculate rejection probability for this bin
            events_to_keep = corrected_hist[i]
            if events_to_keep > num_events_in_bin:
                events_to_keep = num_events_in_bin
            
            rejection_prob = 1.0 - (events_to_keep / num_events_in_bin)
            
            # Randomly select which events to keep
            keep_mask = np.random.rand(num_events_in_bin) > rejection_prob
            
            # Append events that are kept to the final list
            corrected_data.append(filtered_prompts[:, events_in_bin_idx[keep_mask]])
            final_idxs.append(bin_idx_in_index[keep_mask])
    
    return np.concatenate(final_idxs)


def rm_random_new_v2(data_of, delay_of):

    data = data_of.copy().T
    delay = delay_of.copy().T

    index = np.linspace(0, data.shape[1] + delay.shape[1] - 1, data.shape[1] + delay.shape[1], dtype=np.int64)

    # First remove physically infeasible TOFs
    tof_window_mm = TOF_RANGE_MM / 2 # Maximum physical TOF difference for a 355mm scanner
    prompt_mask = (data[3,:] >= -tof_window_mm) & (data[3,:] <= tof_window_mm)
    data = data[:, prompt_mask]

    delay_mask = (delay[3,:] >= -tof_window_mm) & (delay[3,:] <= tof_window_mm)
    delay = delay[:, delay_mask]

    index = index[np.concat([prompt_mask, delay_mask])]

    print(f"Removed {np.sum(~prompt_mask)} events outside the TOF window from prompts.")
    print(f"Removed {np.sum(~delay_mask)} events outside the TOF window from delays.")

    data_combined = np.array([
        np.concatenate(( # crystal ID 1s followed by delay crystal ID 1s
            data[8, :],
            delay[8, :]
        )),
        np.concatenate(( # crystal ID 2s followed by delay crystal ID 2s
            data[9, :],
            delay[9, :]
        )),
        np.concatenate(( # TOFs followed by delay TOFs
            data[3, :],
            delay[3, :]
        )),
        np.concatenate(( # zeros followed by ones
            np.zeros(data.shape[1]),
            np.ones(delay.shape[1])
        )),
    ],dtype=np.float64)

    # Sort by LOR ID, then by event type (prompt/delayed), then by time
    lor_combined = np.concatenate(( # coin lors followed by delay lors
        data[8, :].astype(np.int64) * DETECTORS_SIM + data[9, :].astype(np.int64), 
        delay[8, :].astype(np.int64) * DETECTORS_SIM + delay[9, :].astype(np.int64)
    ))
    argsort = np.lexsort((data_combined[3,:], data_combined[2,:], lor_combined))
    sorted_combined = data_combined[:, argsort]
    sorted_lor = lor_combined[argsort]
    index = index[argsort]
    
    # Find events that are adjacent and have the same LOR but different types
    is_valid_pair = (np.diff(sorted_lor) == 0) & (np.diff(sorted_combined[3, :]) == 1)
    
    # Remove both events in the pair
    valid_indices = np.where(is_valid_pair)[0]
    indices_to_remove = np.concatenate([valid_indices, valid_indices + 1])
    all_indices = np.arange(sorted_combined.shape[1])
    mask = np.isin(all_indices, indices_to_remove, invert=True)
    
    filtered_data = sorted_combined[:, mask]
    onlydatamask = np.where(filtered_data[3, :] == 0)
    filtered_prompts = filtered_data[0:3, filtered_data[3, :] == 0]
    filtered_delayed = filtered_data[0:3, filtered_data[3, :] == 1]
    index = index[mask]
    index = index[onlydatamask]
    print(f"Removed {len(indices_to_remove)/2} pairs.")

    if RandomMode == 'Hybrid':
        # Histogram-Based Correction (Second Pass)
        print("Performing **Model-Based** Hybrid Correction (Hybrid Mode)...")
        
        # Use the new, finer binning
        bins = np.linspace(-TOF_RANGE_MM/2, TOF_RANGE_MM/2, NUM_TOF_BINS) 
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        
        # 1. Create Histograms (Use filtered prompts and *original* delayed for better statistics)
        # Note: Using original delayed data, as it is only used to model the shape/scaling factor (alpha)
        prompt_hist, _ = np.histogram(filtered_prompts[2, :], bins=bins)
        delay_hist, _ = np.histogram(delay[3, :], bins=bins) # Using 'delay' (original delayed events)
        
        # 2. Fit the Delayed Histogram for a Smooth Randoms Estimate
        try:
            # Model: Gaussian (for prompt remnant) + Constant (for randoms pedestal)
            # Note: We fit the delayed data, which should be dominated by randoms (constant background)
            # but may still contain a small Gaussian-like prompt peak if the delay is too small.
            gmodel = GaussianModel(prefix='g1_')
            cmodel = ConstantModel(prefix='c1_')
            # model = gmodel + cmodel
            model = cmodel
            
            # Estimate initial parameters
            max_count = np.max(delay_hist)
            max_idx = np.argmax(delay_hist)
            center_ps = bin_centers[max_idx]
            
            params = model.make_params(
                # g1_amplitude=max_count * np.sqrt(2 * np.pi * 100**2), # Initial guess for amplitude
                # g1_center=center_ps,
                # g1_sigma=1.0,
                c1_c=np.mean(delay_hist[(bins[:-1] < -TAIL_LIMIT_MM) | (bins[:-1] > TAIL_LIMIT_MM)]) # Estimate constant from tails
            )

            # Constraints (optional but recommended for stability)
            # params['g1_sigma'].min = 5 # Prevent absurdly narrow peaks
            params['c1_c'].min = 0.0 # Cannot have negative randoms
            
            result = model.fit(delay_hist, params, x=bin_centers)

            print(result.best_values)
            
            # The Randoms estimate is the constant component (c1_c) of the fit
            # Get the constant part of the fit
            randoms_pedestal = result.best_values['c1_c']
            
            # Calculate alpha using the constant component of the fit:
            # Alpha = (Mean Prompt Tail) / (Mean Delayed Tail)
            # Mean Prompt Tail (from data) = Sum(Prompt_hist tails) / N_bins_tails
            # Mean Delayed Tail (from fit) = c1_c (the constant randoms pedestal)
            
            tail_mask = (bin_centers < -TAIL_LIMIT_MM) | (bin_centers > TAIL_LIMIT_MM)
            if np.sum(tail_mask) > 0:
                prompt_tail_mean = np.mean(prompt_hist[tail_mask])
                alpha = prompt_tail_mean / randoms_pedestal if randoms_pedestal > 0 else 1.0
                print("Alpha:", alpha)
            else:
                alpha = 1.0 # Should not happen with current binning
                
            # Smooth randoms estimate for subtraction (a constant background is assumed)
            randoms_hist = np.full_like(prompt_hist, alpha * randoms_pedestal)

        except Exception as e:
            print(f"LMFIT model fitting failed ({e}). Falling back to simple tail ratio.")
            tail_range_mask = (bins[:-1] < -TAIL_LIMIT_MM) | (bins[:-1] > TAIL_LIMIT_MM)
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
            
        final_idxs = []
        
        # Iterate through each TOF bin
        for i in range(len(bins) - 1):
            events_in_bin_idx = np.where(binned_prompts == i)[0]
            bin_idx_in_index = index[events_in_bin_idx]
            num_events_in_bin = len(events_in_bin_idx)
            
            if num_events_in_bin > 0:
                # Calculate rejection probability for this bin
                events_to_keep = corrected_hist[i]
                if events_to_keep > num_events_in_bin:
                    events_to_keep = num_events_in_bin
                
                rejection_prob = 1.0 - (events_to_keep / num_events_in_bin)
                
                # Randomly select which events to keep
                keep_mask = np.random.rand(num_events_in_bin) > rejection_prob
                
                # Append events that are kept to the final list
                corrected_data.append(filtered_prompts[:, events_in_bin_idx[keep_mask]])
                final_idxs.append(bin_idx_in_index[keep_mask])

        return np.concatenate(final_idxs) if len(final_idxs) > 0 else np.array([], dtype=np.int64)
        
    else: # RandomMode == 'Pair' or any other value
        print("Using Pair-Only random subtraction mode.")
        return index


def gen_sp_randoms(i, j, filename):
    def slice_sp(i, j):
        start_i = i * crystals_per_det
        stop_i = (i + 1) * crystals_per_det
        start_j = j * crystals_per_det
        stop_j = (j + 1) * crystals_per_det
        return slice(start_i, stop_i), slice(start_j, stop_j)
    
    sp = np.load(filename)
    sec = sp[slice_sp(i, j)]

    sec_int = np.floor(sec) + np.astype(np.random.rand(*sec.shape) < sec % 1, np.int64)
    sec_int = sec_int.astype(np.int64)
    sp_randoms = np.sum(sec_int)
    sp_gen = np.array([
        np.zeros(sp_randoms), np.zeros(sp_randoms), np.zeros(sp_randoms),
        np.random.rand(sp_randoms) * (np.max(tofs) - np.min(tofs)) + np.min(tofs),
        np.zeros(sp_randoms), np.zeros(sp_randoms), np.zeros(sp_randoms), np.zeros(sp_randoms),
        np.concatenate([np.full(np.int64(np.sum(sec_int[row])), row + i * crystals_per_det) for row in range(crystals_per_det)]),
        np.concat([np.repeat(np.arange(j * crystals_per_det, (j + 1) * crystals_per_det), sec_int[col]) for col in range(crystals_per_det)])
    ]).T

    return sp_gen


# delayrm = rm_random(data, delay)
# actualrm = rm_random(data, act)
# sprm = rm_random(data, sps)

delayrm = rm_random_new_v2(data, delay)
actualrm = rm_random_new_v2(data, act)
sprm = rm_random_new_v2(data, sps)

# ANALYSIS OF EFFICACY
print("Analyzing Efficacy")
israndom = match_actuals(data, act)

issort = np.arange(0, data.shape[0])[~israndom]
rsort = np.arange(0, data.shape[0])[israndom]

actualsort = np.sort(actualrm)
act_trues_kept = np.intersect1d(actualsort, issort).shape[0] / issort.shape[0] if issort.shape[0] > 0 else 1
act_randoms_caught = np.setdiff1d(rsort, actualsort).shape[0] / rsort.shape[0]

delsort = np.sort(delayrm)
del_trues_kept = np.intersect1d(delsort, issort).shape[0] / issort.shape[0] if issort.shape[0] > 0 else 1
del_randoms_caught = np.setdiff1d(rsort, delsort).shape[0] / rsort.shape[0]

spsort = np.sort(sprm)
sp_trues_kept = np.intersect1d(spsort, issort).shape[0] / issort.shape[0] if issort.shape[0] > 0 else 1
sp_randoms_caught = np.setdiff1d(rsort, spsort).shape[0] / rsort.shape[0]

stats = pd.DataFrame({
    'i': [i] * 3,
    'j': [j] * 3,
    'method': ['act', 'del', 'sp'],
    'trues_kept': [act_trues_kept, del_trues_kept, sp_trues_kept],
    'randoms_caught': [act_randoms_caught, del_randoms_caught, sp_randoms_caught],
    'num_trues': [issort.shape[0], issort.shape[0], issort.shape[0]],
    'num_randoms': [rsort.shape[0], rsort.shape[0], rsort.shape[0]]
})
stats.to_pickle(f'{OUT_FOLDER}stats/{i}_{j}_stats.pkl')


# FILTERING
print("Filtering Data")
delay_tofs = tofs[delayrm]
actual_tofs = tofs[actualrm]
sp_tofs = tofs[sprm]

delay_data = data[delayrm]
actual_data = data[actualrm]
sp_data = data[sprm]


# Plotting
print("Plotting")
def make_plot(ax, a, b, title, max_y):
    ax.hist(a, bins=np.linspace(-1440, 1440, 100, endpoint=False), alpha=0.4, color='#D60270',)
    ax.hist(b, bins=np.linspace(-1440, 1440, 100, endpoint=False), alpha=0.4, color='#0038A8')
    ax.set_xlabel('Time of Flight (mm)')
    ax.set_ylabel('Counts')
    ax.set_ylim([0, max_y])
    ax.set_title(title)
    ax.grid()

max_height = np.ceil(np.max(np.histogram(tofs, bins=np.linspace(-1440, 1440, 100, endpoint=False))[0]) / 2e5) * 2e5


# Actuals
fig, (reg, ac) = plt.subplots(1, 2, figsize=(15, 5))
make_plot(reg, tofs, acttofs, f'Actuals Pre-Subtraction \u2014 LOR {i}-{j}', max_height)
make_plot(ac, actual_tofs, acttofs, f'Actuals Post-Subtraction \u2014 LOR {i}-{j}', max_height)
plt.savefig(f'{OUT_FOLDER}stats/{i}_{j}_actual.png')

fig, (reg, dw) = plt.subplots(1, 2, figsize=(15, 5))
make_plot(reg, tofs, deltofs, f'Delay Pre-Subtraction \u2014 LOR {i}-{j}', max_height)
make_plot(dw, delay_tofs, deltofs, f'Delay Post-Subtraction \u2014 LOR {i}-{j}', max_height)
plt.savefig(f'{OUT_FOLDER}stats/{i}_{j}_delay.png')

fig, (reg, sp) = plt.subplots(1, 2, figsize=(15, 5))
make_plot(reg, tofs, sptofs, f'SP Pre-Subtraction \u2014 LOR {i}-{j}', max_height)
make_plot(sp, sp_tofs, sptofs, f'SP Post-Subtraction \u2014 LOR {i}-{j}', max_height)
plt.savefig(f'{OUT_FOLDER}stats/{i}_{j}_sp.png')

# Shuffling Data
print("Shuffling data")
np.random.shuffle(delay_data)
np.random.shuffle(actual_data)
np.random.shuffle(sp_data)

# Save Data
print("Saving data")
delay_data.tofile(f'{OUT_FOLDER}{i}_{j}_delaycorr.lm')
actual_data.tofile(f'{OUT_FOLDER}{i}_{j}_actualcorr.lm')
sp_data.tofile(f'{OUT_FOLDER}{i}_{j}_spcorr.lm')