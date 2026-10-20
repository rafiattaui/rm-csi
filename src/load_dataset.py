import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset

class CSIDatasetFast(Dataset):
    def __init__(self, preprocessed_dir: str, target_subjects: list):
        self.samples = []
        self.labels = []
        
        preprocessed_dir = str(preprocessed_dir)
        activity_map = {chr(65 + i): i for i in range(20)}
        
        # Recursively scan for all .npz files in the target folder
        files = glob.glob(os.path.join(preprocessed_dir, "**", "*.npz"), recursive=True)
        
        if not files:
            raise FileNotFoundError(
                f"No '.npz' files found in directory: '{os.path.abspath(preprocessed_dir)}'."
            )

        matched_files = 0
        for fp in files:
            filename = os.path.basename(fp)
            
            # Matches X_D_YY_ZZ regardless of trailing _csi or _preprocessed tags
            match = re.match(r"([A-T])_(\d+)_([A-Za-z0-9]+)_(P\d+)", filename)
            if not match:
                continue
                
            activity, day, device, subject = match.groups()
            
            if subject in target_subjects:
                matched_files += 1
                label = activity_map[activity]
                
                with np.load(fp) as data:
                    # Accepts 'windows' or 'csi' key
                    key = 'windows' if 'windows' in data else 'csi'
                    windows = data[key]
                
                # Ensure 4D shape: (N_windows, 2, 500, 242)
                if windows.ndim == 3:
                    # If saved as (N_windows, 500, 242) -> transpose to (N_windows, 2, 500, 242)
                    windows = np.transpose(windows, (0, 2, 1, 3)) if windows.shape[1] == 2 else np.transpose(windows, (0, 2, 1, 3))
                elif windows.ndim == 4 and windows.shape[1] != 2:
                    # Handle (N_windows, 500, 2, 242) -> transpose to (N_windows, 2, 500, 242)
                    windows = np.transpose(windows, (0, 2, 1, 3))

                self.samples.append(windows)
                self.labels.append(np.full(len(windows), label, dtype=np.int64))

        if matched_files == 0 or len(self.samples) == 0:
            raise ValueError(
                f"Found files in '{preprocessed_dir}', but NONE matched target subjects: {target_subjects}."
            )
                
        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), torch.tensor(self.labels[idx])