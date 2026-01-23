import os
import sys
import scipy.io
import shutil
import numpy as np
import itertools
import math
import pickle
import argparse
import multiprocessing
import signal
from scipy.ndimage import gaussian_filter1d

# --- Force non-interactive backend for parallel plotting ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# -----------------------------------------------------------

# ==========================================
# 1. Constants & Parameters
# ==========================================
PIXEL_NUM = 864
SPEED_OF_LIGHT = 299792458000
SPEED_OF_LIGHT_LENGTH_PS = SPEED_OF_LIGHT * math.pow(10, -12)
TDC_BIN_TO_PS = 1.5625

ASIC_STRIDE_BLOCK = 864
ASIC_STRIDE_SUBMOD = 144

# Global variables for worker processes
global_crystal_map = None

# ==========================================
# 2. KDE & Local Conservation Logic
# ==========================================
def adaptive_sigma(n_prompts, sigma_high=2.0, sigma_low=8.0, pivot=200_000):
    if n_prompts <= 0: return sigma_low
    w = min(n_prompts / pivot, 1.0)
    return sigma_low * (1 - w) + sigma_high * w

def compute_hierarchical_correction(prompt, delay, p_evt, total_stride,
                                    strides=[ASIC_STRIDE_SUBMOD, ASIC_STRIDE_BLOCK],
                                    cv_target=0.10):
    n_pivot = (1.0 / cv_target) ** 2

    def get_layer(stride_val):
        b1p = (prompt[0].astype(np.int32) // stride_val)
        b2p = (prompt[1].astype(np.int32) // stride_val)
        b1d = (delay[0].astype(np.int32) // stride_val)
        b2d = (delay[1].astype(np.int32) // stride_val)

        max_idx = (total_stride // stride_val) - 1
        b1p, b2p = np.clip(b1p, 0, max_idx), np.clip(b2p, 0, max_idx)
        b1d, b2d = np.clip(b1d, 0, max_idx), np.clip(b2d, 0, max_idx)
       
        b1p, b2p = np.minimum(b1p, b2p), np.maximum(b1p, b2p)
        b1d, b2d = np.minimum(b1d, b2d), np.maximum(b1d, b2d)

        multiplier = max_idx + 1
        hp = b1p * multiplier + b2p
        hd = b1d * multiplier + b2d
        max_hash = multiplier * multiplier

        pred = np.bincount(hp, weights=p_evt, minlength=max_hash)
        obs  = np.bincount(hd, minlength=max_hash)
        total_counts = np.bincount(hp, minlength=max_hash)

        with np.errstate(divide="ignore", invalid="ignore"):
            raw_factor = obs / (pred + 1e-9)
        raw_factor = np.nan_to_num(raw_factor, nan=1.0)
       
        weights = total_counts / (total_counts + n_pivot)
        return raw_factor[hp], weights[hp]

    f_sub, w_sub = get_layer(strides[0])
    f_blk, w_blk = get_layer(strides[1])

    fallback_correction = w_blk * f_blk + (1.0 - w_blk) * 1.0
    final_correction = w_sub * f_sub + (1.0 - w_sub) * fallback_correction

    return np.clip(final_correction, 0.7, 1.3)

def compute_local_conservation(prompt, delay, p_evt, stride_val, total_stride, N0=100.0):
    if delay.shape[1] == 0:
        return np.ones(prompt.shape[1], dtype=np.float32)

    max_idx = (total_stride // stride_val) - 1
   
    def get_group_ids(data_arr):
        b1 = (data_arr[0].astype(np.int32) // stride_val)
        b2 = (data_arr[1].astype(np.int32) // stride_val)
        b1 = np.clip(b1, 0, max_idx)
        b2 = np.clip(b2, 0, max_idx)
        return np.minimum(b1, b2) * (max_idx + 1) + np.maximum(b1, b2)

    hp = get_group_ids(prompt)
    hd = get_group_ids(delay)
   
    max_hash = (max_idx + 1)**2

    D_g = np.bincount(hd, minlength=max_hash)
    E_g = np.bincount(hp, weights=p_evt, minlength=max_hash)
    N_g = np.bincount(hp, minlength=max_hash)

    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_g = D_g / (E_g + 1e-9)
    alpha_g = np.nan_to_num(alpha_g, nan=1.0)

    w_g = N_g / (N_g + N0)
    alpha_g_reg = 1.0 + w_g * (alpha_g - 1.0)
   
    alpha_g_reg = np.clip(alpha_g_reg, 0.8, 1.25)

    return alpha_g_reg[hp]

def algo_KDE_LM_Cascade(prompt_sk, delay_sk, rng, stride,
                        sigma_d=4.0, sigma_p=2.0, gamma=0.8, p_max=0.95, norm_mode="unity"):
    n_p = prompt_sk.shape[1]
    n_d = delay_sk.shape[1]

    if n_p < 100 or n_d < 100:
        return np.ones(n_p, dtype=bool)

    # Inputs are PS
    tp = prompt_sk[2]
    td = delay_sk[2]

    # --- 1. Temporal Smoothing ---
    BIN_PS = 25.0
    bins = np.arange(-6000, 6000 + BIN_PS, BIN_PS)
    centers = 0.5 * (bins[:-1] + bins[1:])

    hp, _ = np.histogram(tp, bins=bins)
    hd, _ = np.histogram(td, bins=bins)

    curr_sigma_p = adaptive_sigma(n_p, sigma_high=sigma_p, sigma_low=max(8.0, sigma_p*4))
    curr_sigma_d = adaptive_sigma(n_d, sigma_high=sigma_d, sigma_low=max(8.0, sigma_d*4))

    hp_s = gaussian_filter1d(hp.astype(np.float32), sigma=curr_sigma_p)
    hd_s = gaussian_filter1d(hd.astype(np.float32), sigma=curr_sigma_d)

    p_random = hd_s / (hp_s + 1e-9)
    p_random = np.clip(p_random, 0.0, 1.0)
    p_evt = np.interp(tp, centers, p_random)

    p_evt *= gamma

    # --- 2. Hierarchical Structural Refinement ---
    block_factor = compute_hierarchical_correction(
        prompt_sk, delay_sk, p_evt,
        total_stride=stride,
        strides=[ASIC_STRIDE_SUBMOD, ASIC_STRIDE_BLOCK],
        cv_target=0.10
    )
    p_evt *= block_factor
    p_evt = np.clip(p_evt, 0.0, 1.0)

    # --- 3. Tail-Only Local Conservation ---
    peak_idx = np.argmax(hp_s)
    peak_t = centers[peak_idx]
    peak_val = hp_s[peak_idx]
   
    half_max = peak_val * 0.5
    left_idx = peak_idx
    while left_idx > 0 and hp_s[left_idx] > half_max: left_idx -= 1
    right_idx = peak_idx
    while right_idx < len(hp_s) - 1 and hp_s[right_idx] > half_max: right_idx += 1
       
    fwhm = centers[right_idx] - centers[left_idx]
    sigma_prompt_robust = fwhm / 2.355
    if sigma_prompt_robust < 50.0: sigma_prompt_robust = 300.0
   
    tail_mask = np.abs(tp - peak_t) > 3.0 * sigma_prompt_robust
   
    alpha_sub = compute_local_conservation(
        prompt_sk, delay_sk, p_evt,
        stride_val=ASIC_STRIDE_SUBMOD, total_stride=stride, N0=300.0
    )
    p_evt[tail_mask] *= alpha_sub[tail_mask]
    p_evt = np.clip(p_evt, 0.0, 1.0)

    alpha_blk = compute_local_conservation(
        prompt_sk, delay_sk, p_evt,
        stride_val=ASIC_STRIDE_BLOCK, total_stride=stride, N0=100.0
    )
    p_evt[tail_mask] *= alpha_blk[tail_mask]
   
    # --- 4. Final Safety Caps ---
    p_evt = np.clip(p_evt, 0.0, p_max)

    # --- 5. Global Conservation ---
    expected_randoms = np.sum(p_evt)
    measured_randoms = float(n_d)
   
    if expected_randoms > 0:
        conservation_factor = measured_randoms / expected_randoms
        conservation_factor = np.clip(conservation_factor, 0.9, 1.1)
        p_evt *= conservation_factor
        p_evt = np.clip(p_evt, 0.0, p_max)

    # --- 6. Thinning ---
    keep = rng.random(tp.size) > p_evt
    return keep

# ==========================================
# 3. Worker Logic
# ==========================================
def init_worker(geo_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global global_crystal_map
    try:
        with open(geo_path, 'rb') as f:
            global_crystal_map = pickle.load(f)
    except Exception as e:
        print(f"[Worker Error] Failed to load geometry: {e}")
        global_crystal_map = None

def process_pair(task_args):
    (key, skew_folder, final_output_folder, args_tuple) = task_args
    (do_plot, do_tof, fraction, max_events, sigma_d, sigma_p, kde_norm, save_mat, gamma, p_max) = args_tuple
   
    sub0, sub1 = map(int, key.split('_'))
    seed_val = (sub0 * 1000 + sub1) & 0xFFFFFFFF
    rng = np.random.default_rng(seed_val)

    try:
        f_listmode_out = os.path.join(final_output_folder, f"{key}.lm")
        f_mat_out = os.path.join(final_output_folder, f"{key}_kde_analysis.mat")
        f_png_out = os.path.join(final_output_folder, f"{key}_kde_subtraction.png")

        if os.path.isfile(f_listmode_out) and os.path.getsize(f_listmode_out) > 1024:
            return f"Skipped {key}"

        f_coin_in = os.path.join(skew_folder, f"{key}_coin_corrected.lm")
        f_delay_in = os.path.join(skew_folder, f"{key}_delay_corrected.lm")
       
        if not os.path.isfile(f_coin_in):
            return f"Skipped {key}: Input missing"

        def read_lm_safe(path):
            with open(path, 'rb') as f:
                raw = np.fromfile(f, dtype=np.int16)
                if raw.size % 3 != 0:
                    valid_size = (raw.size // 3) * 3
                    if valid_size == 0: return np.zeros((3, 0), dtype=np.int16)
                    raw = raw[:valid_size]
                return np.reshape(raw, (int(raw.size/3), 3)).transpose()

        data_coin = read_lm_safe(f_coin_in)
        if data_coin.shape[1] == 0: return f"Skipped {key}: No events"

        data_delay = np.zeros((3, 0), dtype=np.int16)
        if os.path.isfile(f_delay_in):
            try:
                data_delay = read_lm_safe(f_delay_in)
            except: pass

        # ==========================================
        # [NEW] SIMULATE DYNAMIC (SHORT SCAN) BEHAVIOR
        # ==========================================
        if fraction < 1.0 or max_events is not None:
            n_c = data_coin.shape[1]
           
            # 1. Determine Target Count based on Prompts
            if fraction < 1.0:
                target_c = int(n_c * fraction)
            elif max_events is not None and n_c > max_events:
                target_c = max_events
            else:
                target_c = n_c
               
            # 2. Calculate Effective Fraction (f_eff)
            # This ensures Delays are cut at the same relative point in their stream
            f_eff = target_c / n_c if n_c > 0 else 1.0
           
            # 3. Slice Prompts (Sequential - First T seconds)
            data_coin = data_coin[:, :target_c]

            # 4. Slice Delays (Coupled by Fraction)
            if data_delay.shape[1] > 0:
                n_d = data_delay.shape[1]
                target_d = int(n_d * f_eff)
                data_delay = data_delay[:, :target_d]
        # ==========================================

        # PS Conversion
        prompt_sk_ps = data_coin.copy().astype(np.float32)
        prompt_sk_ps[2,:] *= TDC_BIN_TO_PS
       
        delay_sk_ps = data_delay.copy().astype(np.float32)
        if data_delay.shape[1] > 0:
            delay_sk_ps[2,:] *= TDC_BIN_TO_PS

        # --- B. KDE Subtraction (On truncated data) ---
        final_data_idx = np.ones(data_coin.shape[1], dtype=bool)
       
        if data_delay.shape[1] > 0:
            global global_crystal_map
            if global_crystal_map is None: raise ValueError("Worker geometry not loaded")
            current_stride = int(global_crystal_map.shape[0])

            final_data_idx = algo_KDE_LM_Cascade(
                prompt_sk_ps, delay_sk_ps, rng,
                stride=current_stride,
                sigma_d=sigma_d, sigma_p=sigma_p, gamma=gamma, p_max=p_max, norm_mode=kde_norm
            )
       
        final_data = data_coin[:, final_data_idx]

        # --- C. Save Analysis ---
        if save_mat or do_plot:
            bins = np.linspace(-5000, 5000, 201)
            hist_coin_before, _ = np.histogram(prompt_sk_ps[2,:], bins=bins)
            hist_delay_before = np.zeros(len(bins)-1)
            if data_delay.shape[1] > 0:
                hist_delay_before, _ = np.histogram(delay_sk_ps[2,:], bins=bins)
           
            final_time_ps = final_data[2, :].astype(np.float32) * TDC_BIN_TO_PS
            hist_coin_after, _ = np.histogram(final_time_ps, bins=bins)

            if save_mat:
                mat_dict = {
                    'bins': bins, 'hist_coin_before': hist_coin_before,
                    'hist_delay_before': hist_delay_before, 'hist_coin_after': hist_coin_after,
                    'count_before': data_coin.shape[1], 'count_after': final_data.shape[1],
                    'fraction_used': fraction
                }
                scipy.io.savemat(f_mat_out, mat_dict)

            if do_plot:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1); plt.title(f"{key} KDE Before")
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                plt.step(bin_centers, hist_coin_before, label='Coin', color='gray')
                plt.step(bin_centers, hist_delay_before, label='Delay', color='orange')
                plt.legend()
                plt.subplot(1, 2, 2); plt.title(f"{key} KDE After")
                plt.step(bin_centers, hist_coin_after, label='KDE Robust', color='red')
                plt.legend()
                plt.savefig(f_png_out); plt.close()

        # --- D. Write Listmode ---
        num_events = final_data.shape[1]
        listmodedata = np.zeros((num_events, 10), dtype=np.float32)
        Det_convert = np.arange(0, 16, dtype=int)
       
        if global_crystal_map is None: raise ValueError("Worker geometry not loaded")

        c1_arr = np.uint16(final_data[0, :])
        c1_mapped = Det_convert[np.uint16(c1_arr/864)] * 864 + c1_arr%864
        listmodedata[:, 0] = global_crystal_map[c1_mapped, 0]
        listmodedata[:, 1] = global_crystal_map[c1_mapped, 1]
        listmodedata[:, 2] = global_crystal_map[c1_mapped, 2]
       
        c2_arr = np.uint16(final_data[1, :])
        c2_mapped = Det_convert[np.uint16(c2_arr/864)] * 864 + c2_arr%864
        listmodedata[:, 5] = global_crystal_map[c2_mapped, 0]
        listmodedata[:, 6] = global_crystal_map[c2_mapped, 1]
        listmodedata[:, 7] = global_crystal_map[c2_mapped, 2]
       
        if do_tof:
            timediff_ps = -1.0 * np.float32(final_data[2, :]) * TDC_BIN_TO_PS
            listmodedata[:, 3] = timediff_ps * SPEED_OF_LIGHT_LENGTH_PS
           
        rng.shuffle(listmodedata)
        with open(f_listmode_out, 'wb') as lm:
            lm.write(listmodedata.tobytes())
           
        return f"Done {key}: {num_events} events (KDE Dynamic Sim)"

    except Exception as e:
        import traceback
        return f"Error {key}: {str(e)}\n{traceback.format_exc()}"

# ==========================================
# 4. Main Entry Point
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KDE Random Correction (Dynamic Sim)')
    parser.add_argument(dest='dir_origin', help='Path of the data folder')    
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of data to keep')
    parser.add_argument('--max_events', type=int, default=None, help='Max events per pair limit')
   
    # KDE Config
    parser.add_argument('--sigma_d', type=float, default=4.0, help='Base Delay Sigma')
    parser.add_argument('--sigma_p', type=float, default=2.0, help='Base Prompt Sigma')
    parser.add_argument('--gamma', type=float, default=0.8, help='Global Random Scaling')
    parser.add_argument('--p_max', type=float, default=0.95, help='Max Probability Cap')
    parser.add_argument('--kde_norm', choices=['unity', 'tail'], default='unity', help='Normalization mode')

    # Flags
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    parser.add_argument('--no-mat', dest='save_mat', action='store_false', help='Disable .mat saving')
    parser.add_argument('--no-tof', dest='tof', action='store_false', help='Disable TOF calculation')
    parser.add_argument('--no-combine', dest='combine', action='store_false', help='Disable final merge')
   
    parser.set_defaults(plot=True, save_mat=True, tof=True, combine=True)
    args = parser.parse_args()

    if args.fraction <= 0 or args.fraction > 1:
        print(f"Error: --fraction must be > 0 and <= 1.")
        sys.exit(1)

    WDIR = os.path.abspath(args.dir_origin)
    # skew_folder = os.path.join(WDIR, 'result', 'Skew')
    skew_folder = os.path.join(WDIR, 'split')
    out_suffix = "_KDE"
    if args.fraction < 1.0: out_suffix += f"_frac{args.fraction:.2f}"
   
    final_output_folder = os.path.join(WDIR, 'result', f'Final_Listmode{out_suffix}')

    if not os.path.isdir(skew_folder):
        print(f"Error: Skew folder not found at {skew_folder}")
        sys.exit()
    if not os.path.isdir(final_output_folder):
        os.makedirs(final_output_folder)

    geo_file = 'geometry.pickle'
    if not os.path.isfile(geo_file):
        print("Error: geometry.pickle not found.")
        sys.exit()

    files = os.listdir(skew_folder)
    pair_keys = []
    for f in files:
        if f.endswith('_coin_corrected.lm'):
            parts = f.split('_')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                pair_keys.append(f"{parts[0]}_{parts[1]}")

    pair_keys = sorted(list(set(pair_keys)), key=lambda x: int(x.split('_')[0])*1000 + int(x.split('_')[1]))
   
    print(f"Found {len(pair_keys)} pairs.")
    print(f"Output: {final_output_folder}")
    print(f"Settings: SigmaD={args.sigma_d}, Gamma={args.gamma}, P_Max={args.p_max}")

    # Parallel Execution
    num_workers = args.workers if args.workers else max(1, multiprocessing.cpu_count() // 2)
    print(f"Starting pool with {num_workers} workers...")

    config_args = (args.plot, args.tof, args.fraction, args.max_events,
                   args.sigma_d, args.sigma_p, args.kde_norm, args.save_mat,
                   args.gamma, args.p_max)

    pool = multiprocessing.Pool(num_workers, initializer=init_worker, initargs=(geo_file,))
   
    tasks = [(k, skew_folder, final_output_folder, config_args) for k in pair_keys]

    try:
        for i, res in enumerate(pool.imap_unordered(process_pair, tasks)):
            if i % 10 == 0: print(f"  > Processed {i}/{len(tasks)} pairs...")
            if "Error" in res: print(res)
       
        pool.close()
        pool.join()
        print("Parallel processing finished.")

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Stopping...")
        pool.terminate()
        sys.exit(1)

    # Combine Step
    if args.combine:
        print("\nCombining listmode files...")
        dir_name = WDIR.replace('\\', '/').split('/')[-1]
        if not dir_name: dir_name = "output"
       
        f_final_out = os.path.join(final_output_folder, f"{dir_name}_combined_KDE.lm")
        lm_files = [f for f in os.listdir(final_output_folder) if f.endswith('.lm') and '_' in f and 'combined' not in f]
       
        if not os.path.isfile(f_final_out):
            file_handles = {}
            for fname in lm_files:
                path = os.path.join(final_output_folder, fname)
                size = os.path.getsize(path)
                file_handles[fname] = [open(path, "rb"), size, 0]
               
            total_events_count = 0
            with open(f_final_out, "wb") as fext:
                for i in range(1001):
                    if i % 100 == 0: print(f"  > Merge Chunk {i}/1000")
                    chunk_buffer = np.zeros((0, 10), dtype=np.float32)
                   
                    for fname in lm_files:
                        f_obj, f_size, f_read = file_handles[fname]
                        events_per_chunk = int((f_size // 40) // 1000)
                        if i == 1000:
                            events_this_chunk = int((f_size - f_read) // 40)
                        else:
                            events_this_chunk = events_per_chunk
                           
                        if events_this_chunk > 0:
                            data = np.fromfile(f_obj, dtype=np.float32, count=events_this_chunk * 10)
                            file_handles[fname][2] += data.size * 4
                            data = data.reshape((-1, 10))
                            chunk_buffer = np.concatenate([chunk_buffer, data], axis=0)
                           
                        if i == 1000: f_obj.close()
                   
                    if chunk_buffer.shape[0] > 0:
                        np.random.shuffle(chunk_buffer)
                        fext.write(chunk_buffer.tobytes())
                        total_events_count += chunk_buffer.shape[0]

            print(f"Final Combined File: {f_final_out}")
            print(f"Total Events: {total_events_count}")

    print("KDE Pipeline Complete.")

import os
import sys
import scipy.io
import shutil
import numpy as np
import itertools
import math
import pickle
import argparse
import multiprocessing
import signal

# --- Force non-interactive backend for parallel plotting ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# -----------------------------------------------------------

# ==========================================
# 1. Constants
# ==========================================
SPEED_OF_LIGHT = 299792458000
SPEED_OF_LIGHT_LENGTH_PS = SPEED_OF_LIGHT * math.pow(10, -12)
TDC_BIN_TO_PS = 1.5625

# Global variables for worker processes
global_crystal_map = None
global_stride = 0

# ==========================================
# 2. Worker Logic (Exact Subtraction)
# ==========================================
def init_worker(geo_path):
    """
    Initialize worker: Ignore SIGINT and load geometry once.
    Determines stride dynamically from geometry file.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global global_crystal_map, global_stride
    try:
        with open(geo_path, 'rb') as f:
            global_crystal_map = pickle.load(f)
            # Dynamic Stride: Number of rows in geometry map = Max Crystal ID + 1
            global_stride = int(global_crystal_map.shape[0])
    except Exception as e:
        print(f"[Worker Error] Failed to load geometry: {e}")
        global_crystal_map = None
        global_stride = 0

def process_pair(task_args):
    """
    Worker function for Exact Event-Pair Random Subtraction.
    """
    (key, skew_folder, final_output_folder,
     do_subtraction, do_plot, do_tof, fraction, max_events) = task_args
   
    # 1. Deterministic Seed
    sub0, sub1 = map(int, key.split('_'))
    seed_val = (sub0 * 1000 + sub1) & 0xFFFFFFFF
    rng = np.random.default_rng(seed_val)

    try:
        # Paths
        f_listmode_out = os.path.join(final_output_folder, f"{key}.lm")
        f_mat_out = os.path.join(final_output_folder, f"{key}_analysis.mat")
        f_png_out = os.path.join(final_output_folder, f"{key}_random_subtraction.png")

        # Resume Logic
        if os.path.isfile(f_listmode_out) and os.path.getsize(f_listmode_out) > 1024:
            return f"Skipped {key}"

        f_coin_in = os.path.join(skew_folder, f"{key}_coin_corrected.lm")
        f_delay_in = os.path.join(skew_folder, f"{key}_delay_corrected.lm")
       
        if not os.path.isfile(f_coin_in):
            return f"Skipped {key}: Input missing"

        # --- A. Read Data (Safe Truncation) ---
        def read_lm_safe(path):
            with open(path, 'rb') as f:
                raw = np.fromfile(f, dtype=np.int16)
                if raw.size % 3 != 0:
                    valid_size = (raw.size // 3) * 3
                    if valid_size == 0: return np.zeros((3, 0), dtype=np.int16)
                    raw = raw[:valid_size]
                return np.reshape(raw, (int(raw.size/3), 3)).transpose()

        data_coin = read_lm_safe(f_coin_in)
        if data_coin.shape[1] == 0: return f"Skipped {key}: No events"

        data_delay = np.zeros((3, 0), dtype=np.int16)
        if do_subtraction and os.path.isfile(f_delay_in):
            try:
                data_delay = read_lm_safe(f_delay_in)
            except: pass

        # ==========================================
        # [NEW] SIMULATE DYNAMIC SCAN (PRE-SUBTRACTION)
        # ==========================================
        # We slice the data BEFORE subtraction to simulate acquiring fewer counts.
        if fraction < 1.0 or max_events is not None:
            n_c = data_coin.shape[1]
           
            # 1. Determine Target Prompt Count
            if fraction < 1.0:
                target_c = int(n_c * fraction)
            elif max_events is not None and n_c > max_events:
                target_c = max_events
            else:
                target_c = n_c
               
            # 2. Calculate Effective Fraction (f_eff)
            # This ensures we take the corresponding chunk of the delay stream
            f_eff = target_c / n_c if n_c > 0 else 1.0
           
            # 3. Slice Prompts (Sequential - First T seconds)
            data_coin = data_coin[:, :target_c]

            # 4. Slice Delays (Coupled Fraction)
            if data_delay.shape[1] > 0:
                n_d = data_delay.shape[1]
                target_d = int(n_d * f_eff)
                data_delay = data_delay[:, :target_d]
        # ==========================================

        # --- B. Histogram Before (On Reduced Data) ---
        bins = np.linspace(-5000, 5000, 201)
        hist_coin_before, _ = np.histogram(data_coin[2, :].astype(np.float32) * TDC_BIN_TO_PS, bins=bins)
       
        hist_delay_before = np.zeros(len(bins)-1)
        if data_delay.shape[1] > 0:
            hist_delay_before, _ = np.histogram(data_delay[2, :].astype(np.float32) * TDC_BIN_TO_PS, bins=bins)

        # --- C. Exact Random Subtraction Logic ---
        final_data = data_coin
       
        if do_subtraction and data_delay.shape[1] > 0:
            global global_stride
            if global_stride == 0: raise ValueError("Worker stride not initialized")

            # 1. Calculate LOR IDs separately (Memory Efficient)
            # Formula: min(c1,c2) * stride + max(c1,c2)
           
            def get_global_lor_ids(data_arr, stride):
                c1 = data_arr[0, :].astype(np.int64)
                c2 = data_arr[1, :].astype(np.int64)
                return np.minimum(c1, c2) * stride + np.maximum(c1, c2)

            coin_raw_ids = get_global_lor_ids(data_coin, global_stride)
            delay_raw_ids = get_global_lor_ids(data_delay, global_stride)
           
            # 2. Map sparse IDs to compact integers 0..N
            # We must map both prompt & delay IDs to the same compact space
            all_ids = np.concatenate([coin_raw_ids, delay_raw_ids])
            unique_ids, inverse_indices = np.unique(all_ids, return_inverse=True)
           
            n_p = len(coin_raw_ids)
            coin_indices = inverse_indices[:n_p]
            delay_indices = inverse_indices[n_p:]
           
            # 3. Count Delays per LOR
            delay_counts = np.bincount(delay_indices, minlength=len(unique_ids))
           
            # 4. Randomize Prompts (Unbiased Selection)
            # Shuffle indices so taking the "first K" is random
            perm = rng.permutation(n_p)
            data_coin_shuffled = data_coin[:, perm]
            coin_indices_shuffled = coin_indices[perm]
           
            # 5. Sort Prompts by Compact LOR ID
            # Groups all prompts of the same LOR together
            sort_idx = np.argsort(coin_indices_shuffled)
            sorted_indices = coin_indices_shuffled[sort_idx]
            sorted_data = data_coin_shuffled[:, sort_idx]
           
            # 6. Determine Keep Counts
            unique_p, starts_p, counts_p = np.unique(sorted_indices, return_index=True, return_counts=True)
           
            # Lookup delay counts for these LORs
            counts_d = delay_counts[unique_p]
           
            # Calculate how many to keep: max(0, P - D)
            counts_keep = np.maximum(0, counts_p - counts_d)
           
            # 7. Build Selection Mask
            keep_mask = np.zeros(n_p, dtype=bool)
           
            for i in range(len(unique_p)):
                n_keep = counts_keep[i]
                if n_keep > 0:
                    s = starts_p[i]
                    keep_mask[s : s + n_keep] = True
           
            # 8. Apply Mask
            final_data = sorted_data[:, keep_mask]

        # --- D. Save Analysis ---
        final_time_ps = final_data[2, :].astype(np.float32) * TDC_BIN_TO_PS
        hist_coin_after, _ = np.histogram(final_time_ps, bins=bins)

        if True: # Always save MAT
            mat_dict = {
                'bins': bins,
                'hist_coin_before': hist_coin_before,
                'hist_delay_before': hist_delay_before,
                'hist_coin_after': hist_coin_after,
                'count_before': data_coin.shape[1],
                'count_after': final_data.shape[1],
                'fraction_used': fraction
            }
            scipy.io.savemat(f_mat_out, mat_dict)

        if do_plot:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1); plt.title(f"{key} Before")
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            plt.step(bin_centers, hist_coin_before, label='Coin', color='gray')
            plt.step(bin_centers, hist_delay_before, label='Delay', color='orange')
            plt.legend()
            plt.subplot(1, 2, 2); plt.title(f"{key} After")
            plt.step(bin_centers, hist_coin_after, label='Corrected', color='green')
            plt.legend()
            plt.savefig(f_png_out); plt.close()

        # --- E. Write Listmode ---
        num_events = final_data.shape[1]
        listmodedata = np.zeros((num_events, 10), dtype=np.float32)
        Det_convert = np.arange(0, 16, dtype=int)
       
        # Use Global Map
        global global_crystal_map
        if global_crystal_map is None: raise ValueError("Worker map not loaded")

        c1_arr = np.uint16(final_data[0, :])
        c1_mapped = Det_convert[np.uint16(c1_arr/864)] * 864 + c1_arr%864
        listmodedata[:, 0] = global_crystal_map[c1_mapped, 0]
        listmodedata[:, 1] = global_crystal_map[c1_mapped, 1]
        listmodedata[:, 2] = global_crystal_map[c1_mapped, 2]
       
        c2_arr = np.uint16(final_data[1, :])
        c2_mapped = Det_convert[np.uint16(c2_arr/864)] * 864 + c2_arr%864
        listmodedata[:, 5] = global_crystal_map[c2_mapped, 0]
        listmodedata[:, 6] = global_crystal_map[c2_mapped, 1]
        listmodedata[:, 7] = global_crystal_map[c2_mapped, 2]
       
        if do_tof:
            timediff_ps = -1.0 * np.float32(final_data[2, :]) * TDC_BIN_TO_PS
            listmodedata[:, 3] = timediff_ps * SPEED_OF_LIGHT_LENGTH_PS
           
        rng.shuffle(listmodedata)
        with open(f_listmode_out, 'wb') as lm:
            lm.write(listmodedata.tobytes())
           
        return f"Done {key}: {num_events} events"

    except Exception as e:
        import traceback
        return f"Error {key}: {str(e)}\n{traceback.format_exc()}"

# ==========================================
# 3. Main Entry Point
# ==========================================
if __name__ == '__main__':

    # Paths
    WDIR = os.path.abspath(args.dir_origin)
    skew_folder = os.path.join(WDIR, 'result', 'Skew')
   
    out_suffix = "_Sub"
    if args.fraction < 1.0: out_suffix += f"_frac{args.fraction:.2f}"
   
    final_output_folder = os.path.join(WDIR, 'result', f'Final_Listmode{out_suffix}')

    if not os.path.isdir(skew_folder):
        print(f"Error: Skew folder not found at {skew_folder}")
        sys.exit()
    if not os.path.isdir(final_output_folder):
        os.makedirs(final_output_folder)

    # Geometry Path (Passed to initializer)
    geo_file = 'geometry.pickle'
    if not os.path.isfile(geo_file):
        print("Error: geometry.pickle not found.")
        sys.exit()

    # File Discovery
    files = os.listdir(skew_folder)
    pair_keys = []
    for f in files:
        if f.endswith('_coin_corrected.lm'):
            parts = f.split('_')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                pair_keys.append(f"{parts[0]}_{parts[1]}")

    pair_keys = sorted(list(set(pair_keys)), key=lambda x: int(x.split('_')[0])*1000 + int(x.split('_')[1]))
   
    print(f"Found {len(pair_keys)} pairs.")
    print(f"Output: {final_output_folder}")

    # Parallel Execution
    num_workers = args.workers if args.workers else max(1, multiprocessing.cpu_count() // 2)
    print(f"Starting pool with {num_workers} workers...")

    # Pack args (Correct 8-item tuple; Geo is in initializer)
    tasks = [
        (k, skew_folder, final_output_folder,
         args.random_subtraction, args.plot, args.tof, args.fraction, args.max_events)
        for k in pair_keys
    ]

    # Initialize workers with geometry
    pool = multiprocessing.Pool(num_workers, initializer=init_worker, initargs=(geo_file,))
    try:
        for i, res in enumerate(pool.imap_unordered(process_pair, tasks)):
            if i % 10 == 0: print(f"  > Processed {i}/{len(tasks)} pairs...")
            if "Error" in res: print(res)
       
        pool.close()
        pool.join()
        print("Parallel processing finished.")

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Stopping...")
        pool.terminate()
        sys.exit(1)

    # Combine Step (FIXED CHUNKING)
    if args.combine:
        print("\nCombining listmode files...")
        dir_name = WDIR.replace('\\', '/').split('/')[-1]
        f_final_out = os.path.join(final_output_folder, f"{dir_name}_combined_Sub.lm")
       
        lm_files = [f for f in os.listdir(final_output_folder) if f.endswith('.lm') and '_' in f and 'combined' not in f]
       
        if not os.path.isfile(f_final_out):
            file_handles = {}
            for fname in lm_files:
                path = os.path.join(final_output_folder, fname)
                size = os.path.getsize(path)
                file_handles[fname] = [open(path, "rb"), size, 0]
               
            total_events_count = 0
            with open(f_final_out, "wb") as fext:
                for i in range(1001):
                    if i % 100 == 0: print(f"  > Merge Chunk {i}/1000")
                    chunk_buffer = np.zeros((0, 10), dtype=np.float32)
                   
                    for fname in lm_files:
                        f_obj, f_size, f_read = file_handles[fname]
                       
                        # [FIX] Correct Chunk Logic: Events = Bytes / 40
                        events_per_chunk = int((f_size // 40) // 1000)
                        if i == 1000:
                            events_this_chunk = int((f_size - f_read) // 40)
                        else:
                            events_this_chunk = events_per_chunk
                           
                        if events_this_chunk > 0:
                            data = np.fromfile(f_obj, dtype=np.float32, count=events_this_chunk * 10)
                            file_handles[fname][2] += data.size * 4
                            data = data.reshape((-1, 10))
                            chunk_buffer = np.concatenate([chunk_buffer, data], axis=0)
                           
                        if i == 1000: f_obj.close()
                   
                    if chunk_buffer.shape[0] > 0:
                        np.random.shuffle(chunk_buffer)
                        fext.write(chunk_buffer.tobytes())
                        total_events_count += chunk_buffer.shape[0]

            print(f"Final Combined File: {f_final_out}")
            print(f"Total Events: {total_events_count}")

    print("Subtraction Pipeline Complete.")