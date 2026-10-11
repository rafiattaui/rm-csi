import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import zscore

# --- Preprocessing Pipeline Functions ---

def remove_outliers_zscore(amp: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    clean_amp = amp.copy()
    z_scores = np.abs(zscore(clean_amp, axis=0, nan_policy='omit'))
    z_scores = np.nan_to_num(z_scores)
    outliers = z_scores > threshold
    medians = np.median(clean_amp, axis=0)
    for sc in range(clean_amp.shape[1]):
        clean_amp[outliers[:, sc], sc] = medians[sc]
    return clean_amp

def butter_lowpass_filter(data: np.ndarray, cutoff: float = 10.0, fs: float = 100.0, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def sanitize_phase(complex_csi: np.ndarray) -> np.ndarray:
    raw_phase = np.angle(complex_csi)
    unwrapped_phase = np.unwrap(raw_phase, axis=1)
    N_frames, N_subcarriers = unwrapped_phase.shape
    subcarrier_indices = np.arange(N_subcarriers)
    sanitized_phase = np.zeros_like(unwrapped_phase)
    
    for k in range(N_frames):
        slope, intercept = np.polyfit(subcarrier_indices, unwrapped_phase[k, :], 1)
        linear_offset = slope * subcarrier_indices + intercept
        sanitized_phase[k, :] = unwrapped_phase[k, :] - linear_offset
        
    return sanitized_phase

def minmax_normalize(matrix: np.ndarray) -> np.ndarray:
    min_val = np.min(matrix, axis=0, keepdims=True)
    max_val = np.max(matrix, axis=0, keepdims=True)
    denom = max_val - min_val
    denom[denom == 0] = 1.0
    return (matrix - min_val) / denom

def create_sliding_windows(data: np.ndarray, window_size: int = 500, step_size: int = 250):
    n_frames = data.shape[0]
    windows = []
    for start in range(0, n_frames - window_size + 1, step_size):
        windows.append(data[start:start + window_size])
    return np.array(windows, dtype=np.float32)

# --- Worker Function ---

def process_file_worker(args):
    raw_path, output_dir, window_size, step_size, fs, cutoff = args
    file_stem = Path(raw_path).stem.replace("_csi", "")
    out_path = os.path.join(output_dir, f"{file_stem}_preprocessed.npz")
    
    try:
        raw_csi = np.load(raw_path)['csi']  # (100000, 242)
        
        # 1. Amplitude processing
        raw_amp = np.abs(raw_csi)
        clean_amp = remove_outliers_zscore(raw_amp)
        filtered_amp = butter_lowpass_filter(clean_amp, cutoff=cutoff, fs=fs)
        norm_amp = minmax_normalize(filtered_amp)
        
        # 2. Phase processing
        clean_phase = sanitize_phase(raw_csi)
        norm_phase = minmax_normalize(clean_phase)
        
        # 3. Combine into Dual Channel: (N_frames, 2, 242)
        dual_channel = np.stack([norm_amp, norm_phase], axis=1)
        
        # 4. Windowing
        windows = create_sliding_windows(dual_channel, window_size=window_size, step_size=step_size)
        
        # Save preprocessed windows
        np.savez_compressed(out_path, windows=windows)
        return raw_path, True, f"Generated {windows.shape[0]} windows of shape {windows.shape[1:]}"
        
    except Exception as e:
        return raw_path, False, str(e)

# --- Main Batch Runner ---

def batch_preprocess_dataset(input_dir: str, output_dir: str, num_workers: int = 4):
    os.makedirs(output_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(input_dir, "**", "*_csi.npz"), recursive=True)
    
    if not raw_files:
        print(f"No *_csi.npz files found in {input_dir}")
        return

    print(f"Preprocessing {len(raw_files)} files offline...")
    tasks = [(f, output_dir, 500, 250, 100.0, 10.0) for f in raw_files]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file_worker, task): task[0] for task in tasks}
        for future in as_completed(futures):
            pcap_path, success, msg = future.result()
            filename = os.path.basename(pcap_path)
            if success:
                print(f"[SUCCESS] {filename} -> {msg}")
            else:
                print(f"[FAILED]  {filename} -> {msg}")

if __name__ == "__main__":
    RAW_DIR = "data/HAR-3/CSI_processed/M1"
    PREPROCESSED_DIR = ".data/HAR-3/CSI_preprocessed/M1"
    
    batch_preprocess_dataset(RAW_DIR, PREPROCESSED_DIR, num_workers=4)