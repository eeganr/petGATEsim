import randoms
import time

# === CONFIG ===
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
DETECTORS_SIM = 12288
DETECTORS_REAL = 13824
TIME = 10.0
# ===

t = time.time()
# scount, pcount, coin_lor, dw, actuals = randoms.read_file(
#     "/scratch/users/eeganr/aug6flange/output1Singles.dat", TAU, TIME, DELAY, DETECTORS_SIM
# )

scount, pcount, coin_lor, dw, actuals = randoms.read_file_lm(
    "/scratch/groups/cslevin/James/output1Singles.dat",
    "/scratch/users/eeganr/flangelm/coins1.lm",
    '/scratch/users/eeganr/flangelm/delay1.lm',
    TAU,
    DELAY,
    DETECTORS_SIM
)
print("Finished processing/writing in:", time.time() - t, "s.")
