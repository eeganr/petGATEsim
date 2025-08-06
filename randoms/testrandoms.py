import randoms

# === CONFIG ===
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
DETECTORS_SIM = 12288
DETECTORS_REAL = 13824
TIME = 10.0
# ===

scount, pcount, coin_lor, dw, actuals = randoms.read_file("/scratch/users/eeganr/aug1flange/output1Singles.dat", TAU, TIME, DELAY, DETECTORS_SIM, False)


print('done!')