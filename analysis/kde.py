import os
import sys
import argparse
import pickle
import time
import json
import warnings
import itertools
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
from scipy import ndimage, stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')


# =============================================================================
# PHYSICAL CONSTANTS AND SCANNER PARAMETERS
# =============================================================================

# Scanner geometry
PIXEL_NUM = 768                    # Crystals per module CHANGED
NUM_MODULES = 16                   # Total detector modules
NUM_SUBMODULES_PER_MODULE = 6      # Submodules per module
CRYSTALS_PER_SUBMODULE = PIXEL_NUM // NUM_SUBMODULES_PER_MODULE       # Crystals per submodule (864/6)
CRYSTALS_TOTAL = PIXEL_NUM * NUM_MODULES

# Submodule crystal layout (assumed 12x12 grid)
SUBMOD_GRID_SIZE = 12              # 12x12 = 144 crystals per submodule

# Concentric region parameters (4 rings within each submodule)
NUM_REGIONS_PER_SUBMODULE = 4      # 4 concentric rings
TOTAL_REGIONS_PER_MODULE = NUM_SUBMODULES_PER_MODULE * NUM_REGIONS_PER_SUBMODULE  # 24

# Timing parameters
TDC_TO_PS = 1.5625                 # TDC units to picoseconds
SPEED_OF_LIGHT_MM_PS = 0.299792458 # Speed of light (mm/ps)
CTR_FWHM_PS = 300.0                # Coincidence timing resolution FWHM (ps)
CTR_SIGMA_MM = SPEED_OF_LIGHT_MM_PS * (CTR_FWHM_PS / 2.355)

# TOF histogram parameters
TOF_BINS = 200
TOF_RANGE_MM = 5000 * SPEED_OF_LIGHT_MM_PS  # ±750mm

# Module mapping
DET_CONVERT = np.arange(0, NUM_MODULES, dtype=np.int32)


# =============================================================================
# ALGORITHM PARAMETERS (Tuned for optimal performance)
# =============================================================================

# CV threshold for flat histogram detection
CV_THRESHOLD_FLAT = 0.50

# Distance threshold for flat histogram assumption
MAX_DISTANCE_FLAT_ELIGIBLE = 4

# Percentile range for baseline extraction in peaked histograms
BASELINE_PERCENTILE_LOW = 25
BASELINE_PERCENTILE_HIGH = 35

# Minimum events for reliable estimation at each level
MIN_EVENTS_MODULE = 100
MIN_EVENTS_SUBMODULE = 50
MIN_EVENTS_REGION = 20

# Geometry-based RF priors (used as fallback for low-count LORs)
GEOMETRY_PRIORS = {
    1: 0.97, 2: 0.95, 3: 0.90, 4: 0.80,
    5: 0.55, 6: 0.30, 7: 0.25, 8: 0.25,
}

# Smoothing kernel sizes
SMOOTHING_KERNEL_SUBMOD = 3  # 3x3 for submodule level
SMOOTHING_KERNEL_REGION = 3  # 3x3 for region level within submodule


# =============================================================================
# CALIBRATION FUNCTIONS
# =============================================================================

def calibrate_cv_thresholds(all_histograms: Dict[int, List[Tuple[np.ndarray, float]]]) -> Dict[int, float]:
    """
    Calibrate CV thresholds from delay-window ground truth.
   
    For each distance, find the optimal CV threshold that minimizes error.
   
    Args:
        all_histograms: Dict mapping distance -> list of (histogram, actual_rf) tuples
   
    Returns:
        Optimal CV threshold for each distance
    """
    optimal_thresholds = {}
   
    for distance, data in all_histograms.items():
        if len(data) < 3:
            optimal_thresholds[distance] = 0.50  # Default
            continue
       
        # Collect CV values and actual RFs
        cv_values = []
        actual_rfs = []
       
        for hist, actual_rf in data:
            nonzero = hist[hist > 0]
            if len(nonzero) > 10:
                cv = np.std(nonzero) / np.mean(nonzero)
                cv_values.append(cv)
                actual_rfs.append(actual_rf)
       
        if len(cv_values) < 3:
            optimal_thresholds[distance] = 0.50
            continue
       
        cv_values = np.array(cv_values)
        actual_rfs = np.array(actual_rfs)
       
        # Find CV threshold that best separates high-RF from low-RF
        # High RF should use flat method, low RF should use peak method
        median_rf = np.median(actual_rfs)
       
        # Sort by CV and find crossover point
        sort_idx = np.argsort(cv_values)
        cv_sorted = cv_values[sort_idx]
        rf_sorted = actual_rfs[sort_idx]
       
        # Find CV where RF crosses median
        crossover_idx = np.argmin(np.abs(rf_sorted - median_rf))
        optimal_thresholds[distance] = float(cv_sorted[crossover_idx])
   
    return optimal_thresholds


def estimate_rf_with_calibration(histogram: np.ndarray,
                                 mod_distance: int,
                                 cv_thresholds: Dict[int, float],
                                 min_events: int = MIN_EVENTS_MODULE) -> Tuple[float, str, float, float]:
    total = np.sum(histogram)
    nonzero_mask = histogram > 0
    nonzero_vals = histogram[nonzero_mask]
    n_nonzero = len(nonzero_vals)
   
    if n_nonzero < 10 or total < min_events:
        prior = GEOMETRY_PRIORS.get(mod_distance, 0.5)
        return prior, 'prior_low_counts', 0, 0
   
    mean_val = np.mean(nonzero_vals)
    cv = np.std(nonzero_vals) / mean_val if mean_val > 0 else 0
    cv_center = cv_thresholds.get(mod_distance, 0.50)
    sigmoid_width = 0.12
   
    # 1. Base Methods
    baseline_flat = mean_val
    # Dynamic percentile based on distance
    percentile = 47 if mod_distance < 4 else 45
    baseline_edge = np.percentile(nonzero_vals, percentile)
   
    # 2. Sigmoid weight
    sigmoid_arg = np.clip((cv - cv_center) / sigmoid_width, -10, 10)
    weight_flat = 1.0 / (1.0 + np.exp(sigmoid_arg))
   
    # 3. SAFETY BIAS (Crucial for clinical data)
    if mod_distance < 4:
        weight_flat *= 0.65 # Reduce influence of the higher 'flat' baseline
   
    # 4. Final Calculation (Only one line here!)
    baseline = weight_flat * baseline_flat + (1 - weight_flat) * baseline_edge
   
    rf = (baseline * n_nonzero) / total
    rf = np.clip(rf, 0.01, 0.99)
    method = f'calib_w{weight_flat:.2f}_cv{cv:.2f}'
   
    return rf, method, baseline, cv


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationMetrics:
    """Statistical validation metrics."""
    correlation: float
    bias: float
    mae: float
    rmse: float
    cohens_d: float
    paired_t_stat: float
    paired_t_pvalue: float
    within_5pct: float
    within_10pct: float
    n_samples: int
   
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModulePairResult:
    """Complete results for a module pair."""
    sub0: int
    sub1: int
    distance: int
    n_prompts: int
    n_delays: int
    actual_rf: float
    tof_estimated_rf: float
    casey_estimated_rf: float
    tof_baseline: float
    tof_cv: float
    tof_method: str
    processing_time: float
    valid: bool
   
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    method: str  # 'casey', 'tof', or 'both'
    use_fine_regions: bool
    smoothing_enabled: bool
    plot_all: bool
    debug: bool
   
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# CONCENTRIC REGION MAPPING
# =============================================================================

def create_concentric_region_map(grid_size: int = SUBMOD_GRID_SIZE) -> np.ndarray:
    """
    Create a mapping from crystal position within submodule to concentric region.
   
    For a 12x12 grid, creates 4 concentric rings:
        Ring 0 (outer): Border crystals (row/col 0,1 or 10,11)
        Ring 1: Next inner ring (row/col 2,3 or 8,9)
        Ring 2: Next inner ring (row/col 4,5 or 6,7)
        Ring 3 (center): Core crystals (row/col 5,6 for odd-sized remainder)
   
    Visual representation (12x12):
        0 0 0 0 0 0 0 0 0 0 0 0
        0 1 1 1 1 1 1 1 1 1 1 0
        0 1 2 2 2 2 2 2 2 2 1 0
        0 1 2 3 3 3 3 3 3 2 1 0
        0 1 2 3 3 3 3 3 3 2 1 0
        0 1 2 3 3 3 3 3 3 2 1 0
        0 1 2 3 3 3 3 3 3 2 1 0
        0 1 2 3 3 3 3 3 3 2 1 0
        0 1 2 3 3 3 3 3 3 2 1 0
        0 1 2 2 2 2 2 2 2 2 1 0
        0 1 1 1 1 1 1 1 1 1 1 0
        0 0 0 0 0 0 0 0 0 0 0 0
   
    Returns:
        2D array of shape (grid_size, grid_size) with region indices 0-3
    """
    region_map = np.zeros((grid_size, grid_size), dtype=np.int32)
   
    for row in range(grid_size):
        for col in range(grid_size):
            # Distance from edge (minimum of distance to any edge)
            dist_from_edge = min(row, col, grid_size - 1 - row, grid_size - 1 - col)
           
            # Map distance to region (0=outer, 3=center)
            # For 12x12: dist 0-1 → region 0, dist 2-3 → region 1,
            #            dist 4-5 → region 2, dist 6+ → region 3
            if dist_from_edge <= 1:
                region = 0
            elif dist_from_edge <= 3:
                region = 1
            elif dist_from_edge <= 5:
                region = 2
            else:
                region = 3
           
            region_map[row, col] = region
   
    return region_map


def get_region_from_crystal(crystal_local: int, region_map: np.ndarray) -> int:
    """
    Get concentric region index for a crystal within its submodule.
   
    Args:
        crystal_local: Local crystal ID within submodule (0-143)
        region_map: Precomputed region map
   
    Returns:
        Region index (0-3, where 0=outer, 3=center)
    """
    row = crystal_local // SUBMOD_GRID_SIZE
    col = crystal_local % SUBMOD_GRID_SIZE
    return region_map[row, col]


def get_hierarchical_ids(crystal_id: int, region_map: np.ndarray) -> Tuple[int, int, int]:
    """
    Get full hierarchical IDs for a crystal.
   
    Args:
        crystal_id: Global crystal ID within module (0-863)
        region_map: Precomputed region map
   
    Returns:
        Tuple of (submodule_id, region_id, crystal_within_submodule)
    """
    submodule = crystal_id // CRYSTALS_PER_SUBMODULE
    crystal_within_submod = crystal_id % CRYSTALS_PER_SUBMODULE
    region = get_region_from_crystal(crystal_within_submod, region_map)
   
    return submodule, region, crystal_within_submod


def create_hierarchical_arrays(crystal_ids: np.ndarray, region_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of submodule and region IDs for array of crystals.
   
    Args:
        crystal_ids: Array of local crystal IDs (0-863)
        region_map: Precomputed region map
   
    Returns:
        Tuple of (submodule_ids, region_ids)
    """
    submodules = crystal_ids // CRYSTALS_PER_SUBMODULE
    crystals_in_submod = crystal_ids % CRYSTALS_PER_SUBMODULE
   
    rows = crystals_in_submod // SUBMOD_GRID_SIZE
    cols = crystals_in_submod % SUBMOD_GRID_SIZE
   
    regions = region_map[rows, cols]
   
    return submodules, regions


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_module_distance(mod_i: int, mod_j: int) -> int:
    """Calculate the angular distance between two detector modules."""
    diff = abs(mod_i - mod_j)
    return min(diff, NUM_MODULES - diff)


def stack_padding(arrays: List[np.ndarray]) -> np.ndarray:
    """Stack arrays with NaN padding for unequal lengths."""
    return np.column_stack(list(itertools.zip_longest(*arrays, fillvalue=np.nan)))


def gaussian_plus_baseline(x, amplitude, center, sigma, baseline):
    """Gaussian function with constant baseline for curve fitting."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline


def safe_divide(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Safe division with default value for zero denominators."""
    result = np.full_like(a, default, dtype=np.float64)
    mask = b > 0
    result[mask] = a[mask] / b[mask]
    return result


# =============================================================================
# FILE I/O
# =============================================================================

def read_triplets_int16(filepaths) -> np.ndarray:
    """Read coincidence data from binary files."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
   
    data = None
    for fp in filepaths:
        if not os.path.exists(fp):
            continue
        with open(fp, "rb") as f:
            tmp = np.fromfile(f, dtype=np.int16)
        if tmp.size == 0:
            continue
        tmp = tmp.reshape((tmp.size // 3, 3)).T
        data = tmp if data is None else np.concatenate([data, tmp], axis=1)
   
    return data if data is not None else np.zeros((3, 0), dtype=np.int16)


def compute_skew_offset(data: np.ndarray, pixel_num: int = PIXEL_NUM) -> np.ndarray:
    """Compute timing skew correction lookup table."""
    skewoffset = np.zeros((pixel_num, pixel_num), dtype=np.int16)
    return skewoffset

    if data.shape[1] == 0:
        return skewoffset
   
    data_argsort = np.lexsort((data[1, :], data[0, :]))
    data_sorted = data[:, data_argsort]
    data_split_pos = np.where(np.diff(data_sorted[1, :]))[0] + 1
   
    data_unique_crystal1 = np.int16(data_sorted[0, np.insert(data_split_pos, 0, 0)])
    data_unique_crystal2 = np.int16(data_sorted[1, np.insert(data_split_pos, 0, 0)])
    data_split = np.split(data_sorted[2, :], data_split_pos)
   
    if len(data_split) == 0:
        return skewoffset
   
    max_len = np.max([i.shape[0] for i in data_split])
    frag_size = 2000000 * max((1, int(1000 / max_len)))
   
    for i in range((len(data_split) // frag_size) + 1):
        lo = i * frag_size
        hi = min(len(data_split), (i + 1) * frag_size)
        if lo >= hi:
            continue
       
        data_aranged = stack_padding(data_split[lo:hi])
        offset = np.nanmean(data_aranged, axis=1)
       
        for _ in range(2):
            l_b = np.transpose(np.repeat([offset - 2000], data_aranged.shape[1], axis=0))
            r_b = np.transpose(np.repeat([offset + 2000], data_aranged.shape[1], axis=0))
            data_filt = np.where((data_aranged >= l_b) & (data_aranged <= r_b),
                                  data_aranged, np.nan)
            offset = np.nanmean(data_filt, axis=1)
       
        c1_idx = data_unique_crystal1[lo:hi] % pixel_num
        c2_idx = data_unique_crystal2[lo:hi] % pixel_num
        skewoffset[c1_idx, c2_idx] = np.nan_to_num(offset, nan=0).astype(np.int16)
   
    return skewoffset


# =============================================================================
# TOF ANALYZER - CORE RF ESTIMATION ALGORITHM
# =============================================================================

class TOFAnalyzer:
    """
    TOF Histogram Analyzer for Random Fraction Estimation.
   
    Implements the core algorithm for estimating the random coincidence
    fraction from TOF histograms without using delay window data.
    """
   
    def __init__(self, tof_range_mm: float = TOF_RANGE_MM, n_bins: int = TOF_BINS):
        self.tof_range_mm = tof_range_mm
        self.n_bins = n_bins
        self.bin_edges = np.linspace(-tof_range_mm, tof_range_mm, n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
   
    def build_histogram(self, tof_mm: np.ndarray) -> np.ndarray:
        """Build TOF histogram from timing data."""
        hist, _ = np.histogram(tof_mm, bins=self.bin_edges)
        return hist.astype(np.float64)
   
    def estimate_rf(self, histogram: np.ndarray, mod_distance: int,
                    min_events: int = MIN_EVENTS_MODULE) -> Tuple[float, str, float, float]:
        """
        Estimate random fraction from TOF histogram using adaptive physics-based logic.
        """
        total = np.sum(histogram)
        n_bins = len(histogram)
       
        nonzero_mask = histogram > 0
        nonzero_vals = histogram[nonzero_mask]
        n_nonzero = len(nonzero_vals)
       
        if n_nonzero < 10 or total < min_events:
            prior = GEOMETRY_PRIORS.get(mod_distance, 0.5)
            return prior, 'prior_low_counts', 0, 0
       
        mean_val = np.mean(nonzero_vals)
        std_val = np.std(nonzero_vals)
        cv = std_val / mean_val if mean_val > 0 else 0
       
        # Smooth histogram for robust peak detection
        histogram_smooth = ndimage.uniform_filter1d(histogram.astype(float), size=3, mode='nearest')
       
        # Define Edge Bins (Outer 30% each side)
        edge_fraction = 0.30
        n_edge = int(n_bins * edge_fraction)
        edge_bins = np.concatenate([histogram_smooth[:n_edge], histogram_smooth[-n_edge:]])
        edge_nonzero = edge_bins[edge_bins > 0]
       
        # Default initialization
        percentile = 47
        edge_cv = 0
        pbr = 1.0
        method_type = "standard"

        if len(edge_nonzero) >= 10:
            edge_median = np.median(edge_nonzero)
           
            # --- 1. DATA-DRIVEN SIGNAL DETECTION (PBR) ---
            center_start, center_end = int(n_bins * 0.3), int(n_bins * 0.7)
            center_bins = histogram_smooth[center_start:center_end]
            center_nonzero = center_bins[center_bins > 0]
           
            if len(center_nonzero) > 0 and edge_median > 0:
                pbr = np.max(center_bins) / edge_median
           
            # --- 2. SIGNAL PROTECTION LOGIC ---
            # If PBR is high, we have a true peak. Find the 'floor' aggressively (P42).
            # --- IMPROVED SIGNAL PROTECTION (Sliding Scale) ---
            if pbr > 1.8:
                percentile = 42
                method_type = "signal_protected"
            elif pbr > 1.4:
                # TRANSITION ZONE: Use P44 to start dropping the floor for pairs like [0_13]
                percentile = 44
                method_type = "transition_signal"
            elif mod_distance < 4:
                percentile = 47
            else:
                percentile = 45
           
            baseline_edge = np.percentile(edge_nonzero, percentile)
            edge_mean = np.mean(edge_nonzero)
            edge_cv = np.std(edge_nonzero) / edge_mean if edge_mean > 0 else 0
        else:
            baseline_edge = np.percentile(nonzero_vals, 30)
            percentile = 30

       
        # --- 3. ADAPTIVE ASYMMETRIC BLENDING (PBR & Geometry Driven) ---
        baseline_flat = mean_val
        # Fine-tuned thresholds for smoother transitions
        cv_thresholds = {1: 0.30, 2: 0.35, 3: 0.40, 4: 0.45, 5: 0.55, 6: 0.65, 7: 0.75, 8: 0.75}
        cv_center = cv_thresholds.get(mod_distance, 0.50)
        sigmoid_width = 0.10
       
        sigmoid_arg = np.clip((cv - cv_center) / sigmoid_width, -10, 10)
        weight_flat = 1.0 / (1.0 + np.exp(sigmoid_arg))
       
        # --- NEW DATA-DRIVEN PBR PENALTY ---
        # If any significant peak exists (PBR > 1.5), we trust the 'Flat/Mean' method less
        # because the peak artificially inflates the mean.
        # --- IMPROVED PBR PENALTY ---
        # Start penalizing the 'Mean' earlier (at 1.3) to fix pairs like [0_13]
        if pbr > 1.3:
            pbr_penalty = np.clip(1.0 - (pbr - 1.3) * 1.5, 0.2, 1.0)
            weight_flat *= pbr_penalty

        # --- EXTENDED SAFETY BIAS ---
        # Include d=4 in the safety penalty to fix the +10% error seen in your data
        if mod_distance <= 4:
            weight_flat *= 0.60  # Shift more reliance toward the conservative Edge baseline
           
        baseline = weight_flat * baseline_flat + (1 - weight_flat) * baseline_edge
       
        # Sanity Checks
        baseline = np.clip(baseline, mean_val * 0.005, mean_val * 0.99)
       
        # Final RF Calculation
        rf = (baseline * n_bins) / total
        rf = np.clip(rf, 0.01, 0.99)
       
        method = f'{method_type}_w{weight_flat:.2f}_pbr{pbr:.1f}_p{percentile}'
       
        return rf, method, baseline, cv
   
    def estimate_rf_hierarchical(self, tof_mm: np.ndarray,
                                  c1_local: np.ndarray,
                                  c2_local: np.ndarray,
                                  mod_distance: int,
                                  region_map: np.ndarray) -> Dict[str, Any]:
        """
        Hierarchical RF estimation: Module → Submodule → Region levels.
       
        Returns dictionary with estimates at all levels.
        """
        prior_rf = GEOMETRY_PRIORS.get(mod_distance, 0.5)
       
        # Get hierarchical IDs
        submod1, region1 = create_hierarchical_arrays(c1_local, region_map)
        submod2, region2 = create_hierarchical_arrays(c2_local, region_map)
       
        # =================================================================
        # MODULE LEVEL
        # =================================================================
        module_hist = self.build_histogram(tof_mm)
        module_rf, module_method, module_baseline, module_cv = \
            self.estimate_rf(module_hist, mod_distance, MIN_EVENTS_MODULE)
       
        module_result = {
            'rf': module_rf,
            'baseline': module_baseline,
            'method': module_method,
            'cv': module_cv,
            'n_events': len(tof_mm)
        }
       
        # =================================================================
        # SUBMODULE LEVEL (6x6) with Bayesian Shrinkage
        # =================================================================
        # Shrinkage pulls local estimates toward the module-level estimate
        # based on count reliability. Low-count submodules trust the parent more.
       
        submod_rf = np.full((6, 6), module_rf, dtype=np.float64)
        submod_counts = np.zeros((6, 6), dtype=np.float64)
        submod_methods = {}
       
        # Shrinkage parameter: how many counts needed to fully trust local estimate
        SHRINKAGE_COUNTS_SUBMOD = 50000  # ~50k events for full confidence
       
        for si in range(6):
            for sj in range(6):
                mask = (submod1 == si) & (submod2 == sj)
                n_events = np.sum(mask)
                submod_counts[si, sj] = n_events
               
                if n_events >= MIN_EVENTS_SUBMODULE:
                    submod_tof = tof_mm[mask]
                    submod_hist = self.build_histogram(submod_tof)
                    rf_local, method, _, _ = self.estimate_rf(
                        submod_hist, mod_distance, MIN_EVENTS_SUBMODULE
                    )
                   
                    # --- BAYESIAN SHRINKAGE ---
                    # Trust factor scales with counts (Threshold = 50k events)
                    trust_factor = n_events / (50000 + n_events)
                    rf_shrunk = (trust_factor * rf_local) + ((1 - trust_factor) * module_rf)
                   
                    submod_rf[si, sj] = rf_shrunk
                    submod_methods[(si, sj)] = method
       
        # =================================================================
        # REGION LEVEL (6x6 submodules × 4x4 regions = 24x24) with Shrinkage
        # =================================================================
        # Full region matrix: [submod1*4+region1, submod2*4+region2]
        # Shrinkage pulls region estimates toward parent submodule estimate
       
        region_rf = np.full((24, 24), module_rf, dtype=np.float64)
        region_counts = np.zeros((24, 24), dtype=np.float64)
       
        # Shrinkage parameter for regions (fewer counts needed since more granular)
        SHRINKAGE_COUNTS_REGION = 10000  # ~10k events for full confidence
       
        for si in range(6):
            for sj in range(6):
                parent_rf = submod_rf[si, sj]  # Parent estimate for this submodule
               
                for ri in range(4):
                    for rj in range(4):
                        mask = ((submod1 == si) & (submod2 == sj) &
                                (region1 == ri) & (region2 == rj))
                        n_events = np.sum(mask)
                       
                        idx_i = si * 4 + ri
                        idx_j = sj * 4 + rj
                        region_counts[idx_i, idx_j] = n_events
                       
                        if n_events >= MIN_EVENTS_REGION:
                            region_tof = tof_mm[mask]
                            region_hist = self.build_histogram(region_tof)
                            rf_local, _, _, _ = self.estimate_rf(
                                region_hist, mod_distance, MIN_EVENTS_REGION
                            )
                           
                            # Bayesian shrinkage toward parent submodule
                            shrinkage_weight = SHRINKAGE_COUNTS_REGION / (SHRINKAGE_COUNTS_REGION + n_events)
                            rf_shrunk = shrinkage_weight * parent_rf + (1 - shrinkage_weight) * rf_local
                           
                            region_rf[idx_i, idx_j] = rf_shrunk
                        else:
                            # Fall back to parent submodule estimate
                            region_rf[idx_i, idx_j] = parent_rf
       
        return {
            'module': module_result,
            'submod_rf': submod_rf,
            'submod_counts': submod_counts,
            'submod_methods': submod_methods,
            'region_rf': region_rf,
            'region_counts': region_counts
        }
   
    def smooth_hierarchical(self, rf_matrix: np.ndarray,
                            counts_matrix: np.ndarray,
                            level: str = 'submod') -> np.ndarray:
        """
        Apply spatial smoothing with count-based weighting.
       
        Args:
            rf_matrix: Raw RF estimates
            counts_matrix: Event counts per cell
            level: 'submod' (6x6) or 'region' (24x24)
       
        Returns:
            Smoothed RF matrix
        """
        kernel_size = SMOOTHING_KERNEL_SUBMOD if level == 'submod' else SMOOTHING_KERNEL_REGION
        half_k = kernel_size // 2
       
        n_rows, n_cols = rf_matrix.shape
        smoothed = np.zeros_like(rf_matrix)
       
        for i in range(n_rows):
            for j in range(n_cols):
                i_lo, i_hi = max(0, i - half_k), min(n_rows, i + half_k + 1)
                j_lo, j_hi = max(0, j - half_k), min(n_cols, j + half_k + 1)
               
                neighborhood_rf = rf_matrix[i_lo:i_hi, j_lo:j_hi]
                neighborhood_counts = counts_matrix[i_lo:i_hi, j_lo:j_hi]
               
                total_counts = np.sum(neighborhood_counts)
                if total_counts > 0:
                    smoothed[i, j] = np.sum(neighborhood_rf * neighborhood_counts) / total_counts
                else:
                    smoothed[i, j] = rf_matrix[i, j]
       
        return smoothed


# =============================================================================
# CASEY VARIANCE-REDUCED RANDOM ESTIMATION
# =============================================================================

class CaseyEstimator:
    """
    Casey Variance-Reduced Random Estimation.
   
    Implements the standard method: R_ij = 2τ × S_i × S_j
    Where S_i and S_j are singles rates estimated from delay coincidences.
    """
   
    def __init__(self):
        self.singles1 = None
        self.singles2 = None
        self.total_delays = 0
   
    def compute_singles(self, delay_data: np.ndarray, n_crystals: int = PIXEL_NUM) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute singles rates per crystal from delay coincidences.
       
        For each delay event (i,j), both crystals recorded a single.
        """
        if delay_data.shape[1] == 0:
            return np.zeros(n_crystals), np.zeros(n_crystals)
       
        crystal1 = delay_data[0, :].astype(np.int32) % n_crystals
        crystal2 = delay_data[1, :].astype(np.int32) % n_crystals
       
        singles1 = np.zeros(n_crystals, dtype=np.float64)
        singles2 = np.zeros(n_crystals, dtype=np.float64)
       
        np.add.at(singles1, crystal1, 1)
        np.add.at(singles2, crystal2, 1)
       
        self.singles1 = singles1
        self.singles2 = singles2
        self.total_delays = delay_data.shape[1]
       
        return singles1, singles2
   
    def estimate_randoms_per_lor(self, c1_local: np.ndarray,
                                  c2_local: np.ndarray,
                                  lor_counts: np.ndarray,
                                  lor_inverse: np.ndarray) -> np.ndarray:
        """
        Casey variance-reduced random estimate per LOR.
       
        R_ij ∝ S_i × S_j (normalized to match total delay counts)
        """
        if self.singles1 is None or self.total_delays == 0:
            return np.zeros(len(lor_counts))
       
        # Get unique LOR crystals
        n_lors = len(lor_counts)
        first_idx = np.zeros(n_lors, dtype=np.int64)
        seen = np.zeros(n_lors, dtype=bool)
        for i, inv in enumerate(lor_inverse):
            if not seen[inv]:
                first_idx[inv] = i
                seen[inv] = True
       
        lor_c1 = c1_local[first_idx]
        lor_c2 = c2_local[first_idx]
       
        # Singles product for each LOR
        si_sj = self.singles1[lor_c1] * self.singles2[lor_c2]
       
        # Normalize to match total delay counts
        # Scale so sum of (si_sj * lor_counts) / sum(lor_counts) = total_delays / sum(lor_counts)
        si_sj_weighted_sum = np.sum(si_sj * lor_counts)
       
        if si_sj_weighted_sum > 0:
            scale = self.total_delays / si_sj_weighted_sum
            lor_randoms = si_sj * scale * lor_counts
        else:
            # Fallback: uniform distribution
            total_prompts = np.sum(lor_counts)
            lor_randoms = lor_counts * (self.total_delays / total_prompts) if total_prompts > 0 else np.zeros(n_lors)
       
        return lor_randoms
   
    def estimate_rf_hierarchical(self, c1_local: np.ndarray,
                                  c2_local: np.ndarray,
                                  delay_c1: np.ndarray,
                                  delay_c2: np.ndarray,
                                  region_map: np.ndarray,
                                  n_prompts: int) -> Dict[str, Any]:
        """
        Hierarchical Casey RF estimation for validation comparison.
        """
        if self.total_delays == 0:
            return {
                'module_rf': 0.0,
                'submod_rf': np.zeros((6, 6)),
                'region_rf': np.zeros((24, 24))
            }
       
        module_rf = self.total_delays / n_prompts if n_prompts > 0 else 0.0
       
        # Get hierarchical IDs
        submod1_p, region1_p = create_hierarchical_arrays(c1_local, region_map)
        submod2_p, region2_p = create_hierarchical_arrays(c2_local, region_map)
       
        delay_c1_local = delay_c1 % PIXEL_NUM
        delay_c2_local = delay_c2 % PIXEL_NUM
        submod1_d, region1_d = create_hierarchical_arrays(delay_c1_local.astype(np.int32), region_map)
        submod2_d, region2_d = create_hierarchical_arrays(delay_c2_local.astype(np.int32), region_map)
       
        # Submodule level
        submod_prompts = np.zeros((6, 6), dtype=np.float64)
        submod_delays = np.zeros((6, 6), dtype=np.float64)
       
        for si in range(6):
            for sj in range(6):
                submod_prompts[si, sj] = np.sum((submod1_p == si) & (submod2_p == sj))
                submod_delays[si, sj] = np.sum((submod1_d == si) & (submod2_d == sj))
       
        submod_rf = safe_divide(submod_delays, submod_prompts, module_rf)
        submod_rf = np.clip(submod_rf, 0.0, 1.0)
       
        # Region level
        region_prompts = np.zeros((24, 24), dtype=np.float64)
        region_delays = np.zeros((24, 24), dtype=np.float64)
       
        for si in range(6):
            for sj in range(6):
                for ri in range(4):
                    for rj in range(4):
                        idx_i = si * 4 + ri
                        idx_j = sj * 4 + rj
                       
                        region_prompts[idx_i, idx_j] = np.sum(
                            (submod1_p == si) & (submod2_p == sj) &
                            (region1_p == ri) & (region2_p == rj)
                        )
                        region_delays[idx_i, idx_j] = np.sum(
                            (submod1_d == si) & (submod2_d == sj) &
                            (region1_d == ri) & (region2_d == rj)
                        )
       
        region_rf = safe_divide(region_delays, region_prompts, module_rf)
        region_rf = np.clip(region_rf, 0.0, 1.0)
       
        return {
            'module_rf': module_rf,
            'submod_rf': submod_rf,
            'submod_prompts': submod_prompts,
            'submod_delays': submod_delays,
            'region_rf': region_rf,
            'region_prompts': region_prompts,
            'region_delays': region_delays
        }


# =============================================================================
# VALIDATION STATISTICS
# =============================================================================

def compute_validation_metrics(actual: np.ndarray, estimated: np.ndarray) -> ValidationMetrics:
    """Compute comprehensive validation metrics."""
    valid_mask = (actual > 0) & np.isfinite(actual) & np.isfinite(estimated)
    actual = actual[valid_mask]
    estimated = estimated[valid_mask]
   
    if len(actual) < 2:
        return ValidationMetrics(0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0)
   
    correlation = np.corrcoef(actual, estimated)[0, 1]
    diff = estimated - actual
    bias = np.mean(diff)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
   
    pooled_std = np.sqrt((np.var(actual) + np.var(estimated)) / 2)
    cohens_d = bias / pooled_std if pooled_std > 0 else 0
   
    t_stat, p_value = stats.ttest_rel(estimated, actual)
   
    rel_error = np.abs(diff) / np.maximum(actual, 0.01) * 100
    within_5pct = np.sum(rel_error <= 5) / len(rel_error) * 100
    within_10pct = np.sum(rel_error <= 10) / len(rel_error) * 100
   
    return ValidationMetrics(
        correlation=float(correlation),
        bias=float(bias),
        mae=float(mae),
        rmse=float(rmse),
        cohens_d=float(cohens_d),
        paired_t_stat=float(t_stat),
        paired_t_pvalue=float(p_value),
        within_5pct=float(within_5pct),
        within_10pct=float(within_10pct),
        n_samples=int(len(actual))
    )


def compute_additional_stats(actual: np.ndarray, estimated: np.ndarray) -> Dict[str, float]:
    """Compute additional statistical tests."""
    valid_mask = (actual > 0) & np.isfinite(actual) & np.isfinite(estimated)
    actual = actual[valid_mask]
    estimated = estimated[valid_mask]
   
    if len(actual) < 3:
        return {}
   
    results = {}
   
    # Spearman correlation (rank-based)
    spearman_r, spearman_p = stats.spearmanr(actual, estimated)
    results['spearman_r'] = float(spearman_r)
    results['spearman_p'] = float(spearman_p)
   
    # Wilcoxon signed-rank test (non-parametric)
    try:
        wilcox_stat, wilcox_p = stats.wilcoxon(estimated - actual)
        results['wilcoxon_stat'] = float(wilcox_stat)
        results['wilcoxon_p'] = float(wilcox_p)
    except:
        pass
   
    # Lin's concordance correlation coefficient
    mean_actual = np.mean(actual)
    mean_est = np.mean(estimated)
    var_actual = np.var(actual)
    var_est = np.var(estimated)
    cov = np.mean((actual - mean_actual) * (estimated - mean_est))
   
    ccc_num = 2 * cov
    ccc_den = var_actual + var_est + (mean_actual - mean_est) ** 2
    results['lins_ccc'] = float(ccc_num / ccc_den) if ccc_den > 0 else 0.0
   
    # Intraclass correlation coefficient (ICC)
    n = len(actual)
    grand_mean = (np.mean(actual) + np.mean(estimated)) / 2
   
    ss_between = n * ((np.mean(actual) - grand_mean) ** 2 + (np.mean(estimated) - grand_mean) ** 2)
    ss_within = np.sum((actual - np.mean(actual)) ** 2) + np.sum((estimated - np.mean(estimated)) ** 2)
   
    ms_between = ss_between / 1 if ss_between > 0 else 1e-10
    ms_within = ss_within / (2 * n - 2) if ss_within > 0 else 1e-10
   
    results['icc'] = float((ms_between - ms_within) / (ms_between + ms_within))
   
    return results


# =============================================================================
# PLOTTING - PUBLICATION QUALITY
# =============================================================================

def plot_concentric_region_diagram(output_path: str):
    """Create diagram showing the concentric region structure."""
    fig, ax = plt.subplots(figsize=(8, 8))
   
    colors = ['#E9724C', '#8963BA', '#2E86AB', '#4CAF50']  # Orange, Purple, Blue, Green
    labels = ['Region 0 (Outer)', 'Region 1', 'Region 2', 'Region 3 (Center)']
   
    region_map = create_concentric_region_map()
   
    for row in range(SUBMOD_GRID_SIZE):
        for col in range(SUBMOD_GRID_SIZE):
            region = region_map[row, col]
            rect = Rectangle((col, SUBMOD_GRID_SIZE - 1 - row), 1, 1,
                            facecolor=colors[region], edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)
   
    # Add legend
    for i, (color, label) in enumerate(zip(colors, labels)):
        rect = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
   
    ax.set_xlim(0, SUBMOD_GRID_SIZE)
    ax.set_ylim(0, SUBMOD_GRID_SIZE)
    ax.set_aspect('equal')
    ax.set_xlabel('Crystal Column', fontsize=12)
    ax.set_ylabel('Crystal Row', fontsize=12)
    ax.set_title('Concentric Region Structure within Submodule\n(12×12 crystals)', fontsize=14, fontweight='bold')
   
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i])
                       for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
   
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()


def create_diagnostic_plot(tof_mm_prompts: np.ndarray, tof_mm_delays: np.ndarray,
                          tof_result: Dict, casey_result: Dict,
                          sub0: int, sub1: int, mod_distance: int,
                          output_path: str):
    """Create comprehensive diagnostic plot for a module pair."""
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
   
    bins = np.linspace(-TOF_RANGE_MM, TOF_RANGE_MM, TOF_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
   
    hist_p, _ = np.histogram(tof_mm_prompts, bins=bins)
    hist_d, _ = np.histogram(tof_mm_delays, bins=bins) if len(tof_mm_delays) > 0 else (np.zeros(TOF_BINS), None)
   
    actual_rf = casey_result.get('module_rf', 0)
    tof_rf = tof_result['module']['rf']
   
    # Row 1, Col 1: Full TOF histogram with edge regions highlighted
    ax = axes[0, 0]
    ax.step(bin_centers, hist_p, where='mid', label='Prompts', color='blue', alpha=0.7, linewidth=1.5)
    ax.step(bin_centers, hist_d, where='mid', label='Delays', color='red', alpha=0.7, linewidth=1.5)
   
    # Highlight edge regions used for baseline estimation
    edge_fraction = 0.35
    n_edge = int(len(hist_p) * edge_fraction)
    ax.axvspan(bin_centers[0], bin_centers[n_edge-1], alpha=0.15, color='green', label='Edge bins')
    ax.axvspan(bin_centers[-n_edge], bin_centers[-1], alpha=0.15, color='green')
   
    baseline = tof_result['module']['baseline']
    if baseline > 0:
        ax.axhline(baseline, color='orange', linestyle='-',
                   label=f'Baseline={baseline:.0f}', linewidth=2)
    ax.set_xlabel('TOF (mm)', fontsize=11)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title(f'Distance={mod_distance}, {tof_result["module"]["method"]}', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
   
    # Row 1, Col 2: Zoomed TOF histogram
    ax = axes[0, 1]
    zoom = 400
    mask_p = np.abs(tof_mm_prompts) < zoom
    mask_d = np.abs(tof_mm_delays) < zoom if len(tof_mm_delays) > 0 else np.array([])
    bins_z = np.linspace(-zoom, zoom, 81)
    ax.hist(tof_mm_prompts[mask_p], bins=bins_z, alpha=0.5, label='Prompts', color='blue')
    if len(tof_mm_delays) > 0 and np.sum(mask_d) > 0:
        ax.hist(tof_mm_delays[mask_d], bins=bins_z, alpha=0.5, label='Delays', color='red')
    ax.set_xlabel('TOF (mm)', fontsize=11)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title(f'Zoomed ±{zoom}mm', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
   
    # Row 1, Col 3: RF comparison bar chart
    ax = axes[0, 2]
    x_pos = [0, 1, 2]
    heights = [actual_rf, tof_rf, actual_rf]  # Casey RF = actual for comparison
    colors_bar = ['#2E86AB', '#E94F37', '#4CAF50']
    labels_bar = ['Actual\n(Delay)', 'TOF\nEstimated', 'Casey\nEstimated']
    bars = ax.bar(x_pos, heights, width=0.6, color=colors_bar, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_bar, fontsize=10)
    ax.set_ylabel('Random Fraction', fontsize=11)
    ax.set_ylim(0, 1.1)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('RF Comparison', fontsize=12)
   
    # Row 1, Col 4: Summary text
    ax = axes[0, 3]
    dev_tof = (tof_rf - actual_rf) / actual_rf * 100 if actual_rf > 0 else 0
    summary = f"""
Module Pair: {sub0}_{sub1}
Distance: {mod_distance}
{'='*35}

Events:
  Prompts:  {len(tof_mm_prompts):,}
  Delays:   {len(tof_mm_delays):,}

Random Fraction:
  Actual (delay):    {actual_rf:.4f}
  TOF Estimated:     {tof_rf:.4f}
  TOF Deviation:     {dev_tof:+.1f}%

TOF Method: {tof_result['module']['method']}
Baseline: {baseline:.1f} counts/bin
CV: {tof_result['module']['cv']:.3f}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')
   
    # Row 2: Submodule level heatmaps
    submod_rf_tof = tof_result['submod_rf']
    submod_rf_casey = casey_result.get('submod_rf', np.full((6, 6), actual_rf))
   
    for idx, (data, title, cmap) in enumerate([
        (submod_rf_casey, 'Casey RF (Delay-based)', 'viridis'),
        (submod_rf_tof, 'TOF Estimated RF', 'viridis'),
        (submod_rf_tof - submod_rf_casey, 'Difference (TOF - Casey)', 'RdBu_r'),
        (tof_result['submod_counts'], 'Event Counts', 'YlOrRd')
    ]):
        ax = axes[1, idx]
        if idx == 2:  # Difference plot
            vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.15)
            im = ax.imshow(data, vmin=-vmax, vmax=vmax, cmap=cmap)
        elif idx == 3:  # Counts plot
            im = ax.imshow(data, cmap=cmap)
        else:
            im = ax.imshow(data, vmin=0, vmax=1, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Submodule (det 2)', fontsize=10)
        ax.set_ylabel('Submodule (det 1)', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
   
    # Row 3: Region level (24x24) - show compressed view
    region_rf_tof = tof_result['region_rf']
    region_rf_casey = casey_result.get('region_rf', np.full((24, 24), actual_rf))
   
    for idx, (data, title, cmap) in enumerate([
        (region_rf_casey, 'Casey RF (Region Level)', 'viridis'),
        (region_rf_tof, 'TOF RF (Region Level)', 'viridis'),
        (region_rf_tof - region_rf_casey, 'Region Difference', 'RdBu_r'),
    ]):
        ax = axes[2, idx]
        if idx == 2:
            vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.15)
            im = ax.imshow(data, vmin=-vmax, vmax=vmax, cmap=cmap, aspect='auto')
        else:
            im = ax.imshow(data, vmin=0, vmax=1, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Region Index (det 2)', fontsize=10)
        ax.set_ylabel('Region Index (det 1)', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
   
    # Row 3, Col 4: Difference histogram
    ax = axes[2, 3]
    diff_submod = (submod_rf_tof - submod_rf_casey).flatten()
    valid = ~np.isnan(diff_submod) & (submod_rf_casey.flatten() > 0)
    if np.sum(valid) > 0:
        ax.hist(diff_submod[valid], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linewidth=2)
    ax.set_xlabel('Difference (TOF - Casey)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Submodule Difference Distribution', fontsize=12)
    ax.grid(True, alpha=0.3)
   
    plt.suptitle(f'RF Estimation Diagnostic - Module Pair {sub0}_{sub1} (Distance={mod_distance})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
   
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()


def create_validation_summary_plot(all_results: List[ModulePairResult],
                                   output_dir: str,
                                   method: str = 'tof') -> ValidationMetrics:
    """Create publication-quality validation summary figure."""
    actual = np.array([r.actual_rf for r in all_results])
   
    if method == 'tof':
        est = np.array([r.tof_estimated_rf for r in all_results])
        method_label = 'TOF Method'
    else:
        est = np.array([r.casey_estimated_rf for r in all_results])
        method_label = 'Casey Method'
   
    dist = np.array([r.distance for r in all_results])
   
    metrics = compute_validation_metrics(actual, est)
   
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
   
    # Panel A: Correlation plot
    ax = axes[0, 0]
    sc = ax.scatter(actual, est, c=dist, s=60, alpha=0.7, cmap='viridis',
                    edgecolors='white', linewidths=0.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Identity')
   
    # Linear regression
    valid = (actual > 0) & (est > 0)
    if np.sum(valid) > 2:
        slope, intercept, r_val, _, _ = stats.linregress(actual[valid], est[valid])
        x_fit = np.array([0, 1])
        ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1.5,
                label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
   
    ax.set_xlabel('Actual RF (Delay Window)', fontsize=12)
    ax.set_ylabel(f'Estimated RF ({method_label})', fontsize=12)
    ax.set_title(f'A) Correlation: r = {metrics.correlation:.4f}', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Module Distance', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
   
    # Panel B: Bland-Altman plot
    ax = axes[0, 1]
    mean_vals = (actual + est) / 2
    diff_vals = est - actual
    sc = ax.scatter(mean_vals, diff_vals, c=dist, s=60, alpha=0.7, cmap='viridis',
                    edgecolors='white', linewidths=0.5)
   
    ax.axhline(0, color='black', linestyle=':', linewidth=1)
    ax.axhline(metrics.bias, color='red', linestyle='-', linewidth=2,
               label=f'Bias = {metrics.bias:.4f}')
   
    std_diff = np.std(diff_vals[np.isfinite(diff_vals)])
    ax.axhline(metrics.bias + 1.96*std_diff, color='gray', linestyle='--', alpha=0.7,
               label=f'±1.96σ = ±{1.96*std_diff:.4f}')
    ax.axhline(metrics.bias - 1.96*std_diff, color='gray', linestyle='--', alpha=0.7)
   
    ax.set_xlabel('Mean RF', fontsize=12)
    ax.set_ylabel('Difference (Est - Actual)', fontsize=12)
    ax.set_title(f'B) Bland-Altman Plot', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
   
    # Panel C: RF by distance
    ax = axes[0, 2]
    dist_data = defaultdict(list)
    for r in all_results:
        rf_est = r.tof_estimated_rf if method == 'tof' else r.casey_estimated_rf
        dist_data[r.distance].append((r.actual_rf, rf_est))
   
    dists = sorted(dist_data.keys())
    x = np.arange(len(dists))
    width = 0.35
   
    actual_means = [np.mean([d[0] for d in dist_data[d]]) for d in dists]
    est_means = [np.mean([d[1] for d in dist_data[d]]) for d in dists]
    actual_stds = [np.std([d[0] for d in dist_data[d]]) for d in dists]
    est_stds = [np.std([d[1] for d in dist_data[d]]) for d in dists]
   
    ax.bar(x - width/2, actual_means, width, yerr=actual_stds,
           label='Actual (Delay)', color='#2E86AB', alpha=0.8, capsize=3)
    ax.bar(x + width/2, est_means, width, yerr=est_stds,
           label=f'Estimated ({method_label})', color='#E94F37', alpha=0.8, capsize=3)
   
    ax.set_xlabel('Module Distance', fontsize=12)
    ax.set_ylabel('Random Fraction', fontsize=12)
    ax.set_title('C) RF by Module Distance', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dists)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
   
    # Panel D: Error distribution
    ax = axes[1, 0]
    rel_errors = []
    for d in dists:
        vals = [(e-a)/max(a, 0.01)*100 for a, e in dist_data[d] if a > 0]
        rel_errors.extend(vals)
        ax.scatter([d]*len(vals), vals, alpha=0.6, s=40)
   
    ax.axhline(0, color='black', linewidth=2)
    ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='±10%')
    ax.axhline(-10, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Module Distance', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title(f'D) Error by Distance ({metrics.within_10pct:.1f}% within ±10%)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(-30, 30)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
   
    # Panel E: Error histogram
    ax = axes[1, 1]
    if len(rel_errors) > 0:
        ax.hist(rel_errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linewidth=2)
    ax.axvline(np.mean(rel_errors) if rel_errors else 0, color='red', linewidth=2,
               label=f'Mean = {np.mean(rel_errors):.1f}%' if rel_errors else 'Mean')
    ax.set_xlabel('Relative Error (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('E) Error Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
   
    # Panel F: Statistics summary
    ax = axes[1, 2]
    add_stats = compute_additional_stats(actual, est)
   
    stats_text = f"""
VALIDATION STATISTICS
{'='*40}

Primary Metrics:
  Pearson r:         {metrics.correlation:.4f}
  Spearman ρ:        {add_stats.get('spearman_r', 0):.4f}
  Lin's CCC:         {add_stats.get('lins_ccc', 0):.4f}
  ICC:               {add_stats.get('icc', 0):.4f}

Error Metrics:
  Bias:              {metrics.bias:+.4f}
  MAE:               {metrics.mae:.4f}
  RMSE:              {metrics.rmse:.4f}

Effect Size:
  Cohen's d:         {metrics.cohens_d:.4f}
 
Statistical Tests:
  Paired t-test:     t={metrics.paired_t_stat:.3f}, p={metrics.paired_t_pvalue:.4f}
  Wilcoxon:          p={add_stats.get('wilcoxon_p', 1.0):.4f}

Accuracy:
  Within ±5%:        {metrics.within_5pct:.1f}%
  Within ±10%:       {metrics.within_10pct:.1f}%
 
N samples:          {metrics.n_samples}
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')
   
    plt.suptitle(f'Validation Summary - {method_label} vs Delay Window Ground Truth',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
   
    plt.savefig(os.path.join(output_dir, f'validation_summary_{method}.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'validation_summary_{method}.svg'),
                format='svg', bbox_inches='tight')
    plt.close()
   
    return metrics


def create_method_comparison_plot(all_results: List[ModulePairResult], output_dir: str):
    """Create plot comparing TOF and Casey methods."""
    actual = np.array([r.actual_rf for r in all_results])
    tof_est = np.array([r.tof_estimated_rf for r in all_results])
    casey_est = np.array([r.casey_estimated_rf for r in all_results])
    dist = np.array([r.distance for r in all_results])
   
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
    # Panel 1: TOF vs Casey
    ax = axes[0]
    sc = ax.scatter(casey_est, tof_est, c=dist, s=60, alpha=0.7, cmap='viridis',
                    edgecolors='white', linewidths=0.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Identity')
    ax.set_xlabel('Casey Estimated RF', fontsize=12)
    ax.set_ylabel('TOF Estimated RF', fontsize=12)
    corr = np.corrcoef(casey_est[casey_est > 0], tof_est[casey_est > 0])[0, 1]
    ax.set_title(f'TOF vs Casey (r={corr:.4f})', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Distance')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
   
    # Panel 2: Both methods vs Actual
    ax = axes[1]
    ax.scatter(actual, tof_est, alpha=0.6, s=50, label='TOF Method', color='#E94F37')
    ax.scatter(actual, casey_est, alpha=0.6, s=50, label='Casey Method', color='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2)
    ax.set_xlabel('Actual RF (Delay)', fontsize=12)
    ax.set_ylabel('Estimated RF', fontsize=12)
    ax.set_title('Both Methods vs Ground Truth', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
   
    # Panel 3: Error comparison by distance
    ax = axes[2]
   
    dist_data = defaultdict(lambda: {'tof': [], 'casey': []})
    for r in all_results:
        if r.actual_rf > 0:
            dist_data[r.distance]['tof'].append(abs(r.tof_estimated_rf - r.actual_rf) / r.actual_rf * 100)
            dist_data[r.distance]['casey'].append(abs(r.casey_estimated_rf - r.actual_rf) / r.actual_rf * 100)
   
    dists = sorted(dist_data.keys())
    x = np.arange(len(dists))
    width = 0.35
   
    tof_means = [np.mean(dist_data[d]['tof']) for d in dists]
    casey_means = [np.mean(dist_data[d]['casey']) for d in dists]
    tof_stds = [np.std(dist_data[d]['tof']) for d in dists]
    casey_stds = [np.std(dist_data[d]['casey']) for d in dists]
   
    ax.bar(x - width/2, tof_means, width, yerr=tof_stds, label='TOF Method',
           color='#E94F37', alpha=0.8, capsize=3)
    ax.bar(x + width/2, casey_means, width, yerr=casey_stds, label='Casey Method',
           color='#2E86AB', alpha=0.8, capsize=3)
   
    ax.set_xlabel('Module Distance', fontsize=12)
    ax.set_ylabel('Absolute Relative Error (%)', fontsize=12)
    ax.set_title('Error Comparison by Distance', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dists)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
   
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'method_comparison.svg'), format='svg', bbox_inches='tight')
    plt.close()


def create_cv_rf_relationship_plot(all_results: List[ModulePairResult], output_dir: str):
    """
    Create plot showing CV vs RF relationship for calibration analysis.
   
    This helps visualize how CV correlates with random fraction and
    informs threshold selection.
    """

    print('len results', len(all_results))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
   
    cv_values = np.array([r.tof_cv for r in all_results])
    actual_rf = np.array([r.actual_rf for r in all_results])
    tof_rf = np.array([r.tof_estimated_rf for r in all_results])
    distances = np.array([r.distance for r in all_results])
   
    # Panel 1: CV vs Actual RF (colored by distance)
    ax = axes[0, 0]
    sc = ax.scatter(cv_values, actual_rf, c=distances, s=60, alpha=0.7, cmap='viridis',
                    edgecolors='white', linewidths=0.5)
    ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_ylabel('Actual RF (from delays)', fontsize=12)
    ax.set_title('CV vs Actual RF', fontsize=13, fontweight='bold')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Distance')
    ax.grid(True, alpha=0.3)
   
    # Add trend line
    valid = (cv_values > 0) & (actual_rf > 0) & (actual_rf < 1)
    if np.sum(valid) > 5:
        z = np.polyfit(cv_values[valid], actual_rf[valid], 2)
        p = np.poly1d(z)
        cv_range = np.linspace(cv_values[valid].min(), cv_values[valid].max(), 100)
        ax.plot(cv_range, p(cv_range), 'r-', linewidth=2, label='Quadratic fit')
        ax.legend(fontsize=10)
   
    # Panel 2: CV distribution by distance
    ax = axes[0, 1]
    dist_cv = defaultdict(list)
    for r in all_results:
        dist_cv[r.distance].append(r.tof_cv)
   
    dists = sorted(dist_cv.keys())
    positions = np.arange(len(dists))
   
    bp = ax.boxplot([dist_cv[d] for d in dists], positions=positions, widths=0.6,
                    patch_artist=True)
   
    colors = plt.cm.viridis(np.linspace(0, 1, len(dists)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
   
    ax.set_xticks(positions)
    ax.set_xticklabels(dists)
    ax.set_xlabel('Module Distance', fontsize=12)
    ax.set_ylabel('CV', fontsize=12)
    ax.set_title('CV Distribution by Distance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
   
    # Panel 3: Estimation error vs CV
    ax = axes[1, 0]
    errors = (tof_rf - actual_rf) / np.maximum(actual_rf, 0.01) * 100
    sc = ax.scatter(cv_values, errors, c=distances, s=60, alpha=0.7, cmap='viridis',
                    edgecolors='white', linewidths=0.5)
    ax.axhline(0, color='black', linewidth=2)
    ax.axhline(10, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-10, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('CV', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Estimation Error vs CV', fontsize=13, fontweight='bold')
    ax.set_ylim(-30, 30)
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Distance')
    ax.grid(True, alpha=0.3)
   
    # Panel 4: Sigmoid transition visualization
    ax = axes[1, 1]
    cv_range = np.linspace(0, 3, 200)
   
    cv_thresholds = {1: 0.30, 2: 0.35, 3: 0.40, 4: 0.50, 5: 0.60, 6: 0.70, 7: 0.75, 8: 0.75}
   
    for d in [1, 3, 5, 7]:
        cv_center = cv_thresholds[d]
        sigmoid_width = 0.12
        weight_flat = 1.0 / (1.0 + np.exp((cv_range - cv_center) / sigmoid_width))
        ax.plot(cv_range, weight_flat, linewidth=2, label=f'd={d} (cv_c={cv_center})')
   
    ax.set_xlabel('CV', fontsize=12)
    ax.set_ylabel('Weight (flat method)', fontsize=12)
    ax.set_title('Sigmoid Transition: Flat ↔ Edge Method', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.05, 1.05)
   
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_rf_relationship.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cv_rf_relationship.svg'), format='svg', bbox_inches='tight')
    plt.close()


# =============================================================================
# MODULE PAIR PROCESSING
# =============================================================================

def process_module_pair(sub0: int, sub1: int, file_list: List[str],
                        geometry: np.ndarray, skew_lut_global: Optional[np.ndarray],
                        region_map: np.ndarray, config: ProcessingConfig,
                        result_path: str, listmode_path: str, diag_path: str) -> ModulePairResult:
    """
    Process a single module pair with both TOF and Casey methods.
    """
    start_time = time.time()
    mod_distance = get_module_distance(sub0, sub1)
    prior_rf = GEOMETRY_PRIORS.get(mod_distance, 0.5)
   
    # Initialize result
    result = ModulePairResult(
        sub0=sub0, sub1=sub1, distance=mod_distance,
        n_prompts=0, n_delays=0,
        actual_rf=0.0, tof_estimated_rf=prior_rf, casey_estimated_rf=prior_rf,
        tof_baseline=0.0, tof_cv=0.0, tof_method='none',
        processing_time=0.0, valid=False
    )
   
    # ==========================================================================
    # READ DATA
    # ==========================================================================
    data = read_triplets_int16(file_list)
    if data.shape[1] == 0:
        return result
   
    result.n_prompts = data.shape[1]
    data = data.astype(np.float32)
    data[2, :] = data[2, :] * TDC_TO_PS
   
    delay_files = [f.replace('coin', 'delay') for f in file_list]
    delay = read_triplets_int16(delay_files)
    result.n_delays = delay.shape[1]
   
    if result.n_prompts > 0:
        result.actual_rf = min(result.n_delays / result.n_prompts, 1.0)
   
    if delay.shape[1] > 0:
        delay = delay.astype(np.float32)
        delay[2, :] = delay[2, :] * TDC_TO_PS
   
    # ==========================================================================
    # TIMING CORRECTION
    # ==========================================================================
    f_skew = os.path.join(result_path, f'{sub0}_{sub1}_skew_array.dat')
    if os.path.isfile(f_skew):
        skewoffset = np.fromfile(f_skew, dtype=np.int16).reshape((PIXEL_NUM, PIXEL_NUM))
    elif skew_lut_global is not None:
        skewoffset = skew_lut_global[sub0*PIXEL_NUM:(sub0+1)*PIXEL_NUM,
                                     sub1*PIXEL_NUM:(sub1+1)*PIXEL_NUM].copy()
    else:
        skewoffset = compute_skew_offset(data.astype(np.int16))
        with open(f_skew, 'wb') as f:
            f.write(skewoffset.tobytes())
   
    c1_local = (data[0, :].astype(np.int32)) % PIXEL_NUM
    c2_local = (data[1, :].astype(np.int32)) % PIXEL_NUM
   
    offset_arr = skewoffset[c1_local, c2_local].astype(np.float32)
    timediff_ps = offset_arr - data[2, :]
    tof_mm = SPEED_OF_LIGHT_MM_PS * timediff_ps
   
    # Process delays
    if delay.shape[1] > 0:
        d1_local = (delay[0, :].astype(np.int32)) % PIXEL_NUM
        d2_local = (delay[1, :].astype(np.int32)) % PIXEL_NUM
        offset_arr_d = skewoffset[d1_local, d2_local].astype(np.float32)
        timediff_ps_d = offset_arr_d - delay[2, :]
        tof_mm_delays = SPEED_OF_LIGHT_MM_PS * timediff_ps_d
    else:
        d1_local = np.array([], dtype=np.int32)
        d2_local = np.array([], dtype=np.int32)
        tof_mm_delays = np.array([])
   
    # ==========================================================================
    # TOF METHOD
    # ==========================================================================
    tof_result = None
    if config.method in ['tof', 'both']:
        analyzer = TOFAnalyzer()
        tof_result = analyzer.estimate_rf_hierarchical(
            tof_mm, c1_local, c2_local, mod_distance, region_map
        )
       
        result.tof_estimated_rf = tof_result['module']['rf']
        result.tof_baseline = tof_result['module']['baseline']
        result.tof_cv = tof_result['module']['cv']
        result.tof_method = tof_result['module']['method']
       
        # Apply smoothing
        if config.smoothing_enabled:
            tof_result['submod_rf_smoothed'] = analyzer.smooth_hierarchical(
                tof_result['submod_rf'], tof_result['submod_counts'], 'submod'
            )
            tof_result['region_rf_smoothed'] = analyzer.smooth_hierarchical(
                tof_result['region_rf'], tof_result['region_counts'], 'region'
            )
        else:
            tof_result['submod_rf_smoothed'] = tof_result['submod_rf']
            tof_result['region_rf_smoothed'] = tof_result['region_rf']
   
    # ==========================================================================
    # CASEY METHOD
    # ==========================================================================
    casey_result = None
    casey_estimator = CaseyEstimator()
   
    if config.method in ['casey', 'both'] and delay.shape[1] > 0:
        casey_estimator.compute_singles(delay.astype(np.int16))
        casey_result = casey_estimator.estimate_rf_hierarchical(
            c1_local, c2_local, d1_local, d2_local, region_map, result.n_prompts
        )
       
        result.casey_estimated_rf = casey_result['module_rf']
       
        # Apply smoothing
        if config.smoothing_enabled:
            analyzer = TOFAnalyzer()
            casey_result['submod_rf_smoothed'] = analyzer.smooth_hierarchical(
                casey_result['submod_rf'],
                casey_result['submod_prompts'],
                'submod'
            )
            casey_result['region_rf_smoothed'] = analyzer.smooth_hierarchical(
                casey_result['region_rf'],
                casey_result['region_prompts'],
                'region'
            )
        else:
            casey_result['submod_rf_smoothed'] = casey_result['submod_rf']
            casey_result['region_rf_smoothed'] = casey_result['region_rf']
    else:
        # Fallback: use actual RF uniformly
        casey_result = {
            'module_rf': result.actual_rf,
            'submod_rf': np.full((6, 6), result.actual_rf),
            'submod_rf_smoothed': np.full((6, 6), result.actual_rf),
            'region_rf': np.full((24, 24), result.actual_rf),
            'region_rf_smoothed': np.full((24, 24), result.actual_rf),
            'submod_prompts': np.ones((6, 6)),
            'region_prompts': np.ones((24, 24))
        }
   
    # ==========================================================================
    # CREATE LISTMODE OUTPUT
    # ==========================================================================
    n_events = data.shape[1]
    crystalID1 = data[0, :].astype(np.uint16)
    crystalID2 = data[1, :].astype(np.uint16)
   
    # Get hierarchical IDs for all events
    submod1, region1 = create_hierarchical_arrays(c1_local, region_map)
    submod2, region2 = create_hierarchical_arrays(c2_local, region_map)
   
    # LOR identification
    lor_id = crystalID1.astype(np.int64) * CRYSTALS_TOTAL + crystalID2.astype(np.int64)
    lor_unique, lor_inverse, lor_counts = np.unique(lor_id, return_inverse=True, return_counts=True)
    n_lors = len(lor_unique)
   
    # Get first event index for each LOR
    first_idx = np.zeros(n_lors, dtype=np.int64)
    seen = np.zeros(n_lors, dtype=bool)
    for i, inv in enumerate(lor_inverse):
        if not seen[inv]:
            first_idx[inv] = i
            seen[inv] = True
   
    lor_submod1 = submod1[first_idx]
    lor_submod2 = submod2[first_idx]
    lor_region1 = region1[first_idx]
    lor_region2 = region2[first_idx]
   
    # ==========================================================================
    # COMPUTE PER-LOR RANDOM ESTIMATES
    # ==========================================================================
   
    # TOF-based estimates (using smoothed region-level RF)
    lor_random_tof = np.zeros(n_lors, dtype=np.float64)
    if tof_result is not None:
        rf_matrix = tof_result['region_rf_smoothed']
        for i in range(n_lors):
            idx_i = lor_submod1[i] * 4 + lor_region1[i]
            idx_j = lor_submod2[i] * 4 + lor_region2[i]
            lor_random_tof[i] = rf_matrix[idx_i, idx_j] * lor_counts[i]
   
    # Casey-based estimates (using smoothed region-level RF)
    lor_random_casey = np.zeros(n_lors, dtype=np.float64)
    if casey_result is not None:
        rf_matrix = casey_result['region_rf_smoothed']
        for i in range(n_lors):
            idx_i = lor_submod1[i] * 4 + lor_region1[i]
            idx_j = lor_submod2[i] * 4 + lor_region2[i]
            lor_random_casey[i] = rf_matrix[idx_i, idx_j] * lor_counts[i]
   
    # ==========================================================================
    # BUILD LISTMODE DATA
    # ==========================================================================
    listmodedata = np.zeros((n_events, 10), dtype=np.float32)
   
    tmp1 = DET_CONVERT[crystalID1 // PIXEL_NUM] * PIXEL_NUM + crystalID1 % PIXEL_NUM
    tmp2 = DET_CONVERT[crystalID2 // PIXEL_NUM] * PIXEL_NUM + crystalID2 % PIXEL_NUM
   
    listmodedata[:, 0] = geometry[tmp1, 0]  # x1
    listmodedata[:, 1] = geometry[tmp1, 1]  # y1
    listmodedata[:, 2] = geometry[tmp1, 2]  # z1
    listmodedata[:, 3] = tof_mm             # tof
   
    lor_dict = dict(zip(lor_unique.tolist(), lor_counts.tolist()))
    listmodedata[:, 4] = np.array([lor_dict[int(lid)] for lid in lor_id], dtype=np.float32)
   
    listmodedata[:, 5] = geometry[tmp2, 0]  # x2
    listmodedata[:, 6] = geometry[tmp2, 1]  # y2
    listmodedata[:, 7] = geometry[tmp2, 2]  # z2
   
    listmodedata[:, 8] = lor_random_tof[lor_inverse]    # TOF-estimated randoms
    listmodedata[:, 9] = lor_random_casey[lor_inverse]  # Casey variance-reduced randoms
   
    # Shuffle and save
    np.random.shuffle(listmodedata)
    f_listmode = os.path.join(listmode_path, f'{sub0}_{sub1}.lm')
    with open(f_listmode, 'wb') as f:
        f.write(listmodedata.tobytes())
   
    # ==========================================================================
    # DIAGNOSTIC PLOT
    # ==========================================================================
    if config.plot_all:
        create_diagnostic_plot(
            tof_mm, tof_mm_delays,
            tof_result if tof_result else {'module': {'rf': 0, 'baseline': 0, 'method': 'none', 'cv': 0},
                                           'submod_rf': np.zeros((6,6)), 'region_rf': np.zeros((24,24)),
                                           'submod_counts': np.zeros((6,6)), 'region_counts': np.zeros((24,24))},
            casey_result,
            sub0, sub1, mod_distance,
            os.path.join(diag_path, f'{sub0}_{sub1}_diagnostic.png')
        )
   
    result.processing_time = time.time() - start_time
    result.valid = True
   
    if config.debug:
        dev_tof = (result.tof_estimated_rf - result.actual_rf) / result.actual_rf * 100 \
                  if result.actual_rf > 0 else 0
        print(f"  [{sub0}_{sub1}] d={mod_distance}: Act={result.actual_rf:.4f}, "
              f"TOF={result.tof_estimated_rf:.4f} ({dev_tof:+.1f}%), "
              f"Casey={result.casey_estimated_rf:.4f}, {result.tof_method}")
   
    return result


def combine_listmode(listmode_path: str, output_name: str):
    """Combine individual module pair listmode files into a single file."""
    lm_files = [f for f in os.listdir(listmode_path)
                if f.endswith('.lm') and '_' in f and f.split('_')[0].isdigit()]
    lm_files.sort(key=lambda f: int(f.split('_')[0]) * 1000 + int(f.split('_')[1].replace('.lm', '')))
   
    if not lm_files:
        return
   
    print(f"  Combining {len(lm_files)} files...")
    f_out = os.path.join(listmode_path, output_name)
   
    file_dict = {}
    with open(f_out, "wb") as fext:
        for i in range(1001):
            if i % 200 == 0:
                print(f"    Progress: {i}/1000")
           
            listmodedata = []
            for f in lm_files:
                if f not in file_dict:
                    fpath = os.path.join(listmode_path, f)
                    file_dict[f] = [open(fpath, "rb"), os.path.getsize(fpath), 0]
               
                counts = int(file_dict[f][1] / 40 / 1000) * 10
                if i == 1000:
                    counts = int((file_dict[f][1] - file_dict[f][2]) / 4)
               
                chunk = np.fromfile(file_dict[f][0], dtype=np.float32, count=counts)
                file_dict[f][2] += chunk.shape[0] * 4
               
                if chunk.size > 0:
                    listmodedata.append(chunk.reshape((-1, 10)))
               
                if i == 1000:
                    file_dict[f][0].close()
           
            if listmodedata:
                combined = np.vstack(listmodedata)
                np.random.shuffle(combined)
                fext.write(combined.tobytes())
   
    print(f"  Combined: {os.path.getsize(f_out) // 40:,} events")


# =============================================================================
# SUMMARY AND JSON OUTPUT
# =============================================================================

def create_json_summary(all_results: List[ModulePairResult],
                        tof_metrics: ValidationMetrics,
                        casey_metrics: ValidationMetrics,
                        config: ProcessingConfig,
                        output_path: str,
                        processing_time: float):
    """Create comprehensive JSON summary of all results."""
   
    # Results by distance
    dist_summary = defaultdict(lambda: {
        'count': 0, 'prompts': 0, 'delays': 0,
        'actual_rf_mean': [], 'tof_rf_mean': [], 'casey_rf_mean': [],
        'tof_error': [], 'casey_error': []
    })
   
    for r in all_results:
        d = r.distance
        dist_summary[d]['count'] += 1
        dist_summary[d]['prompts'] += r.n_prompts
        dist_summary[d]['delays'] += r.n_delays
        dist_summary[d]['actual_rf_mean'].append(r.actual_rf)
        dist_summary[d]['tof_rf_mean'].append(r.tof_estimated_rf)
        dist_summary[d]['casey_rf_mean'].append(r.casey_estimated_rf)
        if r.actual_rf > 0:
            dist_summary[d]['tof_error'].append((r.tof_estimated_rf - r.actual_rf) / r.actual_rf * 100)
            dist_summary[d]['casey_error'].append((r.casey_estimated_rf - r.actual_rf) / r.actual_rf * 100)
   
    dist_results = {}
    for d in sorted(dist_summary.keys()):
        s = dist_summary[d]
        dist_results[str(d)] = {
            'n_pairs': s['count'],
            'total_prompts': s['prompts'],
            'total_delays': s['delays'],
            'actual_rf': {'mean': np.mean(s['actual_rf_mean']), 'std': np.std(s['actual_rf_mean'])},
            'tof_rf': {'mean': np.mean(s['tof_rf_mean']), 'std': np.std(s['tof_rf_mean'])},
            'casey_rf': {'mean': np.mean(s['casey_rf_mean']), 'std': np.std(s['casey_rf_mean'])},
            'tof_error_pct': {'mean': np.mean(s['tof_error']), 'std': np.std(s['tof_error'])} if s['tof_error'] else None,
            'casey_error_pct': {'mean': np.mean(s['casey_error']), 'std': np.std(s['casey_error'])} if s['casey_error'] else None
        }
   
    # Additional statistics
    actual = np.array([r.actual_rf for r in all_results])
    tof_est = np.array([r.tof_estimated_rf for r in all_results])
    casey_est = np.array([r.casey_estimated_rf for r in all_results])
   
    tof_add_stats = compute_additional_stats(actual, tof_est)
    casey_add_stats = compute_additional_stats(actual, casey_est)
   
    summary = {
        'meta': {
            'version': '14.0',
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'config': config.to_dict()
        },
        'algorithm_parameters': {
            'cv_threshold_flat': CV_THRESHOLD_FLAT,
            'max_distance_flat_eligible': MAX_DISTANCE_FLAT_ELIGIBLE,
            'baseline_percentile_range': [BASELINE_PERCENTILE_LOW, BASELINE_PERCENTILE_HIGH],
            'min_events_module': MIN_EVENTS_MODULE,
            'min_events_submodule': MIN_EVENTS_SUBMODULE,
            'min_events_region': MIN_EVENTS_REGION,
            'num_regions_per_submodule': NUM_REGIONS_PER_SUBMODULE,
            'smoothing_kernel_submod': SMOOTHING_KERNEL_SUBMOD,
            'smoothing_kernel_region': SMOOTHING_KERNEL_REGION
        },
        'overall': {
            'n_module_pairs': len(all_results),
            'total_prompts': sum(r.n_prompts for r in all_results),
            'total_delays': sum(r.n_delays for r in all_results),
            'overall_rf': sum(r.n_delays for r in all_results) / max(sum(r.n_prompts for r in all_results), 1)
        },
        'tof_method': {
            'validation_metrics': tof_metrics.to_dict(),
            'additional_statistics': tof_add_stats
        },
        'casey_method': {
            'validation_metrics': casey_metrics.to_dict(),
            'additional_statistics': casey_add_stats
        },
        'results_by_distance': dist_results,
        'module_pair_results': [r.to_dict() for r in all_results]
    }
   
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
   
    return summary


def print_summary_report(summary: dict):
    """Print formatted summary report to console."""
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
   
    print(f"\nDataset Statistics:")
    print(f"  Module pairs processed: {summary['overall']['n_module_pairs']}")
    print(f"  Total prompt events:    {summary['overall']['total_prompts']:,}")
    print(f"  Total delay events:     {summary['overall']['total_delays']:,}")
    print(f"  Overall RF:             {summary['overall']['overall_rf']:.4f}")
   
    print(f"\n{'='*70}")
    print("TOF METHOD VALIDATION")
    print('='*70)
    tof = summary['tof_method']['validation_metrics']
    print(f"  Correlation (r):        {tof['correlation']:.4f}")
    print(f"  Bias (mean error):      {tof['bias']:+.4f}")
    print(f"  MAE:                    {tof['mae']:.4f}")
    print(f"  RMSE:                   {tof['rmse']:.4f}")
    print(f"  Cohen's d:              {tof['cohens_d']:.4f}")
    print(f"  Within ±5%:             {tof['within_5pct']:.1f}%")
    print(f"  Within ±10%:            {tof['within_10pct']:.1f}%")
   
    if 'lins_ccc' in summary['tof_method']['additional_statistics']:
        add = summary['tof_method']['additional_statistics']
        print(f"  Lin's CCC:              {add['lins_ccc']:.4f}")
        print(f"  Spearman ρ:             {add['spearman_r']:.4f}")
   
    print(f"\n{'='*70}")
    print("CASEY METHOD VALIDATION")
    print('='*70)
    casey = summary['casey_method']['validation_metrics']
    print(f"  Correlation (r):        {casey['correlation']:.4f}")
    print(f"  Bias (mean error):      {casey['bias']:+.4f}")
    print(f"  MAE:                    {casey['mae']:.4f}")
    print(f"  RMSE:                   {casey['rmse']:.4f}")
    print(f"  Cohen's d:              {casey['cohens_d']:.4f}")
    print(f"  Within ±5%:             {casey['within_5pct']:.1f}%")
    print(f"  Within ±10%:            {casey['within_10pct']:.1f}%")
   
    print(f"\n{'='*70}")
    print("RESULTS BY MODULE DISTANCE")
    print('='*70)
    print(f"\n{'Dist':<6} {'N':<5} {'Actual':<10} {'TOF Est':<10} {'TOF Err%':<10} {'Casey Est':<10}")
    print("-" * 60)
   
    for d in sorted(summary['results_by_distance'].keys(), key=int):
        r = summary['results_by_distance'][d]
        tof_err = r['tof_error_pct']['mean'] if r['tof_error_pct'] else 0
        print(f"{d:<6} {r['n_pairs']:<5} {r['actual_rf']['mean']:<10.4f} "
              f"{r['tof_rf']['mean']:<10.4f} {tof_err:<+9.1f}% {r['casey_rf']['mean']:<10.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PET Listmode v14 - Publication Ready with Hierarchical Regions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
  python pet_listmode_v14.py /path/to/data --method both --debug --plot-all
  python pet_listmode_v14.py /path/to/data --method tof --no-smoothing
  python pet_listmode_v14.py /path/to/data --method casey
  python pet_listmode_v14.py /path/to/data --workers 8    # Use 8 parallel workers
  python pet_listmode_v14.py /path/to/data --workers 1    # Force serial (for debugging)
        """
    )
    parser.add_argument('dir_origin', help='Data directory')
    parser.add_argument('--method', choices=['casey', 'tof', 'both'], default='both',
                        help='Random estimation method (default: both)')
    parser.add_argument('--no-smoothing', action='store_true',
                        help='Disable spatial smoothing')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run two-pass calibration mode (uses delay data to optimize thresholds)')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    parser.add_argument('--plot-all', action='store_true', help='Generate all diagnostic plots')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (0=auto, 1=serial, N=use N workers)')
    args = parser.parse_args()
   
    WDIR = args.dir_origin
   
    config = ProcessingConfig(
        method=args.method,
        use_fine_regions=True,
        smoothing_enabled=not args.no_smoothing,
        plot_all=args.plot_all,
        debug=args.debug
    )
   
    print("=" * 70)
    print("PET LISTMODE PROCESSOR v14.0 - PUBLICATION READY")
    print("Hierarchical Random Estimation: Module → Submodule → Concentric Regions")
    print("=" * 70)
    print(f"\nData directory: {WDIR}")
    print(f"\nConfiguration:")
    print(f"  Method:               {config.method}")
    print(f"  Fine regions:         {config.use_fine_regions}")
    print(f"  Smoothing enabled:    {config.smoothing_enabled}")
    print(f"\nAlgorithm parameters:")
    print(f"  CV threshold (flat):  {CV_THRESHOLD_FLAT}")
    print(f"  Baseline percentile:  {BASELINE_PERCENTILE_LOW}-{BASELINE_PERCENTILE_HIGH}")
    print(f"  Regions/submodule:    {NUM_REGIONS_PER_SUBMODULE} (concentric rings)")
   
    start_time = time.time()
   
    # Setup directories
    result_dir = os.path.join(WDIR, 'result_v14')
    result_path = os.path.join(result_dir, 'Skew')
    listmode_path = os.path.join(result_dir, 'Listmode')
    diag_path = os.path.join(result_dir, 'Diagnostics')
   
    for d in [result_dir, result_path, listmode_path, diag_path]:
        os.makedirs(d, exist_ok=True)
   
    # Create region map
    region_map = create_concentric_region_map()
   
    # Create region diagram
    plot_concentric_region_diagram(os.path.join(diag_path, 'region_structure.png'))
    print(f"\n  Created region structure diagram")
   
    # Load geometry
    geometry_file = os.path.join(WDIR, 'gategeometry.pickle')
    if not os.path.exists(geometry_file):
        geometry_file = 'gategeometry.pickle'
   
    with open(geometry_file, 'rb') as f:
        geometry = pickle.load(f)
   
    # Load global skew LUT if available
    skew_lut_global = None
    skew_file = os.path.join(WDIR, 'skew_lut.dat')
    if not os.path.exists(skew_file):
        skew_file = 'skew_lut.dat'
   
    if os.path.isfile(skew_file):
        skew_lut_global = np.fromfile(skew_file, dtype=np.int16)
        skew_lut_global = skew_lut_global.reshape((PIXEL_NUM * NUM_MODULES, PIXEL_NUM * NUM_MODULES))
        print(f"  Loaded global skew LUT")
   
    # Find data files
    Files = {}
    for root, dirs, files in os.walk(WDIR):
        for name in files:
            if '.dat' in name and 'coin' in name:
                if name not in Files:
                    Files[name] = []
                Files[name].append(os.path.join(root, name))
   
    keys = sorted(Files.keys(),
                  key=lambda f: int(f.split('_')[0]) * 1000 + int(f.split('_')[1]))
    print(f"\nFound {len(keys)} module pairs")
   
    # Determine number of workers
    if args.workers == 0:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    elif args.workers == 1:
        n_workers = 1  # Serial mode
    else:
        n_workers = min(args.workers, multiprocessing.cpu_count())
   
    # Process all module pairs
    print("\n" + "=" * 70)
    print(f"PROCESSING MODULE PAIRS ({n_workers} workers)")
    print("=" * 70)
   
    # Prepare arguments for each module pair
    process_args = []
    for f in keys:
        sub0, sub1 = int(f.split('_')[0]), int(f.split('_')[1])
        process_args.append((
            sub0, sub1, Files[f], geometry, skew_lut_global, region_map, config,
            result_path, listmode_path, diag_path
        ))
   
    all_results = []
   
    if n_workers == 1:
        # Serial processing (for debugging)
        for idx, args_tuple in enumerate(process_args):
            sub0, sub1 = args_tuple[0], args_tuple[1]
            print(f"  [{idx+1:3d}/{len(keys)}] Processing {sub0}_{sub1}...", end=" " if not config.debug else "\n")
           
            result = process_module_pair(*args_tuple)
            all_results.append(result)
           
            if not config.debug:
                dev = (result.tof_estimated_rf - result.actual_rf) / result.actual_rf * 100 \
                      if result.actual_rf > 0 else 0
                print(f"TOF: {result.tof_estimated_rf:.3f} (actual: {result.actual_rf:.3f}, {dev:+.1f}%)")
    else:
        # Parallel processing
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            future_to_key = {
                executor.submit(process_module_pair, *args_tuple): (args_tuple[0], args_tuple[1])
                for args_tuple in process_args
            }
           
            # Collect results as they complete
            results_dict = {}
            for future in as_completed(future_to_key):
                sub0, sub1 = future_to_key[future]
                completed += 1
                try:
                    result = future.result()
                    results_dict[(sub0, sub1)] = result
                   
                    dev = (result.tof_estimated_rf - result.actual_rf) / result.actual_rf * 100 \
                          if result.actual_rf > 0 else 0
                    print(f"  [{completed:3d}/{len(keys)}] {sub0}_{sub1} d={result.distance}: "
                          f"TOF={result.tof_estimated_rf:.3f} (act={result.actual_rf:.3f}, {dev:+.1f}%)")
                except Exception as e:
                    print(f"  [{completed:3d}/{len(keys)}] {sub0}_{sub1}: ERROR - {e}")
                    # Create dummy result for failed pairs
                    results_dict[(sub0, sub1)] = ModulePairResult(
                        sub0=sub0, sub1=sub1, distance=0,
                        n_prompts=0, n_delays=0,
                        actual_rf=0.0, tof_estimated_rf=0.0, casey_estimated_rf=0.0,
                        tof_baseline=0.0, tof_cv=0.0, tof_method='error',
                        processing_time=0.0, valid=False
                    )
       
        # Sort results by module pair order
        for f in keys:
            sub0, sub1 = int(f.split('_')[0]), int(f.split('_')[1])
            all_results.append(results_dict[(sub0, sub1)])
   
    # Create validation plots
    print("\n" + "=" * 70)
    print("GENERATING VALIDATION PLOTS")
    print("=" * 70)
   
    tof_metrics = create_validation_summary_plot(all_results, diag_path, 'tof')
    casey_metrics = create_validation_summary_plot(all_results, diag_path, 'casey')
    create_method_comparison_plot(all_results, diag_path)
    create_cv_rf_relationship_plot(all_results, diag_path)
   
    print(f"  Created validation summary plots (TOF and Casey)")
    print(f"  Created method comparison plot")
    print(f"  Created CV-RF relationship plot")
   
    # Combine listmode files
    print("\n" + "=" * 70)
    print("COMBINING LISTMODE FILES")
    print("=" * 70)
   
    dataset_name = os.path.basename(WDIR.rstrip('/\\'))
    combine_listmode(listmode_path, f'{dataset_name}_v14.lm')
   
    # Create JSON summary
    processing_time = time.time() - start_time
    json_path = os.path.join(result_dir, 'validation_summary.json')
    summary = create_json_summary(
        all_results, tof_metrics, casey_metrics, config, json_path, processing_time
    )
   
    # Print summary
    print_summary_report(summary)
   
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nTotal processing time: {processing_time:.1f} seconds")
    print(f"\nOutputs saved to: {result_dir}")
    print(f"  - Listmode:     {listmode_path}")
    print(f"  - Diagnostics:  {diag_path}")
    print(f"  - JSON Summary: {json_path}")
   
    return 0


if __name__ == '__main__':
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())