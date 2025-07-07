import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import randoms
import time


# === CONFIG ===
CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
# === END CONFIG ===


def filter_singles(singles, energy_min=0.450, energy_max=0.750):
    """  Filter singles by energy.
        Args:
            singles: DataFrame with columns:
                time, detector, source, energy
            energy_min: minimum energy in MeV (default 0.450)
            energy_max: maximum energy in MeV (default 0.750)
        Returns:
            DataFrame with filtered singles, same columns as input
        Notes:
            Energy-filter singles, parameters are in MeV, defaults 0.450 to 0.750 MeV
            This is the energy range of the 511 keV gamma photons from positron annihilation
            This is the range used in Oliver & Rafecas
    """
    
    singles = singles[singles['energy'] > energy_min].reset_index()
    singles = singles[singles['energy'] < energy_max].reset_index()
    return singles


def bundle_coincidences(singles):
    """ Bundle singles into coincidences, 
        Args:
            singles: DataFrame with columns:
                time, detector, energy, (optional: source)
            where time is in seconds, dete ctor is the detector ID,
            and energy is the energy of the single event in MeV.
        Returns: 
            DataFrame with columns:
            time1, time2, detector1, detector2, true
            where true is True if the two singles are from the same source (true coincidence)
            False otherwise
    """
    t = time.time()

    times = np.array(singles['time'])
    detectors = np.array(singles['detector'])
    energies = np.array(singles['energy'])
    coin_indices = randoms.bundle_coincidences(times, TAU)
    coins = coin_indices.reshape(-1, 2)

    coinci = pd.DataFrame()
    # np array casting required to avoid index conflicts /w pandas series
    coinci['time1'] = np.array(singles['time'].iloc[coins[:, 0]])
    coinci['time2'] = np.array(singles['time'].iloc[coins[:, 1]])
    coinci['detector1'] = np.array(singles['detector'].iloc[coins[:, 0]])
    coinci['detector2'] = np.array(singles['detector'].iloc[coins[:, 1]])

    if 'source' in singles.columns:
        coinci['source1'] = np.array(singles['source'].iloc[coins[:, 0]])
        coinci['source2'] = np.array(singles['source'].iloc[coins[:, 1]])
        coinci['true'] = coinci['source1'] == coinci['source2']
    
    return coinci


def singles_prompts(singles_count, prompts_count, singles, coincidences, detectors, TIME):
    """ Calculate the Singles-Prompts rate estimate for the whole scanner
        Args:
            singles_count: array of singles counts per detector
            prompts_count: array of prompts counts per detector pair
            singles: DataFrame with columns:
                time, detector, energy
            coincidences: DataFrame with columns:
                time1, time2, detector1, detector2
            detectors: array of detector IDs
        Returns:
            Singles-Prompts rate estimate for the whole scanner
    """

    S = len(singles) / TIME  # Rate of singles measured by scanner as a whole
    P = 2 * len(coincidences) / TIME  # Twice the prompts rate

    # Roots of this function are the lambda (L) values needed for the SP estimate.
    def lambda_eq(L):
        return 2 * TAU * L * L - L + S - P * np.exp((L + S)*TAU)
    L = root_scalar(lambda_eq, x0=0)
    if not L.converged:
        raise RuntimeError("Failed to converge on lambda.")
    L = L.root
    
    sp_rates = randoms.sp_rates(singles_count, prompts_count, detectors[-1], L, S, TAU, TIME)
    
    # Calculate the Singles-Prompts rate estimate for the whole scanner
    # summing over all pairs of detectors
    return np.sum(sp_rates) * TIME / 2.0 # 2.0 because summing over the array double counts


def delayed_window(singles, detectors):
    return np.sum(randoms.dw_rates(np.array(singles['time']), np.array(singles['detector']), detectors[-1], TAU, DELAY)) / 2.0


def singles_rate(singles_count, detectors, TIME):
    """ Calculate the Singles Rate estimate for the whole scanner
        Args:
            singles_count: array of singles counts per detector
            detectors: array of detector IDs
        Returns:
            Singles Rate estimate for the whole scanner
    """

    sr_rates = randoms.sr_rates(singles_count, detectors[-1], TAU, TIME)
    return np.sum(sr_rates) * TIME / 2.0 # 2.0 because summing over the array double counts


# Main function to read files and calculate estimates for many files of form
# output1.root, output2.root, ..., outputN.root
# Range [1, N) defined in FILE_RANGE
# Writes results to estimations.csv
if __name__ == "__main__":
    sp, dw, sr, actual = [], [], [], []
    for i in FILE_RANGE:
        # Step 1: Read file
        infile = PATH_PREFIX + str(i) + PATH_SUFFIX
        print(f"Reading file {infile}...")
        singles, detectors = read_root_file(infile)

        # Step 1.5: Filter hits by energy if needed
        if FILTER_ENERGY:
            singles = filter_singles(singles)  # Filter singles by energy

        # Step 2: Bundle coincidences
        t = time.time()
        coincidences = bundle_coincidences(singles)  # Bundle singles into coincidences
        print("bundling complete: ", time.time() - t)

        # Step 3: Tally stats by detector
        singles_count = randoms.singles_counts(singles['detector'], detectors[-1])
        prompts_count = randoms.prompts_counts(coincidences['detector1'], coincidences['detector2'], detectors[-1])

        # Step 4: Calculate estimation methods
        sp.append(singles_prompts(singles_count, prompts_count, singles, coincidences, detectors))
        dw.append(randoms.delayed_window(np.array(singles['time']), TAU, DELAY))
        sr.append(singles_rate(singles_count, detectors))
        
        # Step 5: Return results
        actual.append(len(coincidences[~coincidences['true']]))
        print(f"File {str(i)} processed. SP: {sp[-1]}, DW: {dw[-1]}, SR: {sr[-1]}, Actual: {actual[-1]}")

    df = pd.DataFrame({'sp': sp, 'dw': dw, 'sr': sr, 'actual': actual})
    with open(OUTPUT_FILE, 'w') as f:
        df.to_csv(f)
