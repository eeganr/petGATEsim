import numpy as np
from randomsutils import singles_prompts, singles_rate

# === CONFIG ===
OUT_FOLDER = '/scratch/groups/cslevin/eeganr/flangeless/'
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
DETECTORS_SIM = 12288
DETECTORS_REAL = 13824
TIME = 600
# ===


print("loading files")
singles_count = np.load(OUT_FOLDER + 'singles_count.npy')
print("loaded singles")

prompts_count = np.load(OUT_FOLDER + 'prompts_count.npy')

print("loaded prompts")

coin_lor = np.load(OUT_FOLDER + 'coin_lor.npy')

print("loaded coin lors")

dw_nums = np.load(OUT_FOLDER + 'dw_nums.npy')

print('loaded dw')

actuals = np.load(OUT_FOLDER + 'actuals.npy')

print('loaded actuals')

sp_nums = singles_prompts(singles_count, prompts_count, TIME, DETECTORS_SIM)
print('calculated sp')
sr_nums = singles_rate(singles_count, TIME, DETECTORS_SIM)
print('calculated sr')

np.save(OUT_FOLDER + 'sp.npy', sp_nums)
np.save(OUT_FOLDER + 'sr.npy', sr_nums)

sp = np.sum(sp_nums) / 2.0
sr = np.sum(sr_nums) / 2.0
dw = np.sum(dw_nums) / 2.0
act = np.sum(actuals) / 2.0

print(f"File processed. SP: {sp}, DW: {dw}, SR: {sr}, Actual: {act}")
