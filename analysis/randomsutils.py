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


def bundle_coincidences(singles, skew_matrix):
    """ Bundle singles into coincidences with skew correction
        Args:
            singles: DataFrame with columns:
                time, detector, energy
                where time is in seconds, dete ctor is the detector ID,
                and energy is the energy of the single event in MeV.
            skew_matrix: 2d np array of skew times from detX to detY
        Returns: 
            DataFrame with columns:
                time1, time2, detector1, detector2, (source1), (source2), (true)
            np array: list of multiple coincidences by number of hits involved
    """

    # If matrix is only upper triangular
    if (np.all(np.tril(skewlut) == np.zeros(skewlut.shape))):
        skewlut -= skewlut.T # reflects with sign change
    

def bundle_coincidences(singles):
    """ Bundle singles into coincidences, 
        Args:
            singles: DataFrame with columns:
                time, detector, energy, (optional: source)
            where time is in seconds, dete ctor is the detector ID,
            and energy is the energy of the single event in MeV.
        Returns: 
            DataFrame with columns:
                time1, time2, detector1, detector2, (source1), (source2), (true)
            np array: list of multiple coincidences by number of hits involved
    """
    t = time.time()

    times = np.array(singles['time'])
    detectors = np.array(singles['detector'])
    energies = np.array(singles['energy'])

    coin_indices, multis = randoms.bundle_coincidences(times, TAU)
    coins = coin_indices.reshape(-1, 2)

    coinci = pd.DataFrame()
    # np array casting required to avoid index conflicts /w pandas series
    coinci['time1'] = np.array(singles['time'].iloc[coins[:, 0]])
    coinci['time2'] = np.array(singles['time'].iloc[coins[:, 1]])
    coinci['detector1'] = np.array(singles['detector'].iloc[coins[:, 0]])
    coinci['detector2'] = np.array(singles['detector'].iloc[coins[:, 1]])
    coinci['energy1'] = np.array(singles['energy'].iloc[coins[:, 0]])
    coinci['energy2'] = np.array(singles['energy'].iloc[coins[:, 1]])

    if 'source' in singles.columns:
        coinci['source1'] = np.array(singles['source'].iloc[coins[:, 0]])
        coinci['source2'] = np.array(singles['source'].iloc[coins[:, 1]])
        coinci['true'] = coinci['source1'] == coinci['source2']
    if 'globalPosX' in singles.columns:
        coinci['globalPosX1'] = np.array(singles['globalPosX'].iloc[coins[:, 0]])
        coinci['globalPosX2'] = np.array(singles['globalPosX'].iloc[coins[:, 1]])
        coinci['globalPosY1'] = np.array(singles['globalPosY'].iloc[coins[:, 0]])
        coinci['globalPosY2'] = np.array(singles['globalPosY'].iloc[coins[:, 1]])
        coinci['globalPosZ1'] = np.array(singles['globalPosZ'].iloc[coins[:, 0]])
        coinci['globalPosZ2'] = np.array(singles['globalPosZ'].iloc[coins[:, 1]])
    
    return coinci, multis


def bundle_coincidences_multi(singles):
    """ Bundle singles into coincidences, including multiple coincidences
        Args:
            singles: DataFrame with columns:
                time, detector, energy, (optional: source)
            where time is in seconds, dete ctor is the detector ID,
            and energy is the energy of the single event in MeV.
        Returns: 
            DataFrame with columns:
                time1, time2, detector1, detector2, (source1), (source2), (true)
            np array: list of multiple coincidences by number of hits involved
    """



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
            Singles-Prompts randoms estimates for each LOR
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
    return sp_rates * TIME


def singles_prompts_multi(singles_count, prompts_count, singles, coincidences, detectors, TIME):
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
            Singles-Prompts randoms estimates for each LOR
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

    exp_prod = np.prod(np.exp(-(singles_count * (TAU**2) / TIME / TIME)))

    corrections = randoms.sp_correction(singles_count / TIME, detectors[-1], exp_prod, TAU, TIME)
    
    # Calculate the Singles-Prompts rate estimate for the whole scanner
    # summing over all pairs of detectors
    return np.multiply(sp_rates, corrections) * TIME


def delayed_window(singles, detectors):
    """ Calculates the Delayed Window estimate for the whole scanner
    Args:
        singles: DataFrame with columns:
                time, detector, energy, (optional: source)
            where time is in seconds, dete ctor is the detector ID,
            and energy is the energy of the single event in MeV.
        detectors: array of detector IDs
    Returns:
        Delayed Window randoms estimates for each LOR
    """

    return randoms.dw_rates(np.array(singles['time']), np.array(singles['detector']), detectors[-1], TAU, DELAY)


def singles_rate(singles_count, detectors, TIME):
    """ Calculate the Singles Rate estimate for the whole scanner
        Args:
            singles_count: array of singles counts per detector
            detectors: array of detector IDs
        Returns:
            Singles-Rate randoms estimates for each LOR
    """

    sr_rates = randoms.sr_rates(singles_count, detectors[-1], TAU, TIME)
    return sr_rates * TIME

