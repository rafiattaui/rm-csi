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
        
        activity_map = {chr(65 + i): i for i in range(20)}
        files = glob.glob(os.path.join(preprocessed_dir, "*_preprocessed.npz"))
        
        for fp in files:
            filename = os.path.basename(fp)
            # Parse X_D_YY_ZZ (e.g., A_3_M1_P1_preprocessed.npz)
            match = re.match(r"([A-T])_(\d+)_([A-Za-z0-9]+)_(P\d+)_preprocessed\.npz", filename)
            if not match:
                continue
                
            activity, day, device, subject = match.groups()
            
            if subject in target_subjects:
                label = activity_map[activity]
                windows = np.load(fp)['windows'] # Shape: (N_windows, 500, 2, 242)
                
                # Permute to PyTorch Conv2D layout: (N_windows, Channels, Time_Steps, Subcarriers)
                windows = np.transpose(windows, (0, 2, 1, 3)) # (N_windows, 2, 500, 242)
                
                self.samples.append(windows)
                self.labels.append(np.full(len(windows), label, dtype=np.int64))
                
        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), torch.tensor(self.labels[idx])