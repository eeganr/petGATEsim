import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

# BASIC CONFIG
DETECTORS_SIM = 12288
MODULES = 16
crystals_per_det = DETECTORS_SIM // MODULES
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


delayrm = rm_random(data, delay)
actualrm = rm_random(data, act)
sprm = rm_random(data, sps)


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
    'randoms_caught': [act_randoms_caught, del_randoms_caught, sp_randoms_caught]
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


# Save Data
print("Saving data")
delay_data.tofile(f'{OUT_FOLDER}{i}_{j}_actualcorr.lm')
actual_data.tofile(f'{OUT_FOLDER}{i}_{j}_delaycorr.lm')
sp_data.tofile(f'{OUT_FOLDER}{i}_{j}_spcorr.lm')