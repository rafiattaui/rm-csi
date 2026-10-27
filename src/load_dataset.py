"""PyTorch Dataset for preprocessed CSI HAR windows.

Expected files: <activity>_<day>_<device>_P<subject>_preprocessed.npz
produced by preprocess_csi.py. Each file contains windows with shape
(N, 2, T, F), where channel 0 is amplitude and channel 1 is phase.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CSIDataset(Dataset):
    """Load preprocessed CSI windows for one or more devices.

    Args:
        root: Directory containing *_preprocessed.npz files (searched recursively).
        device: Device ID to include, e.g. "M1". Use None to include all devices.
        activities: Optional iterable of activity IDs (0=A, ..., 19=T).
        subjects: Optional iterable of subject IDs, as integers (P2 -> 2).
        days: Optional iterable of day IDs (1..6).
        return_metadata: If True, return (x, y, metadata); otherwise return (x, y).
        cache_size: Number of recordings' window arrays to keep in memory per Dataset
            instance. Increase for speed if RAM permits; set to 0 to disable caching.
        transform: Optional callable applied to each input tensor x.

    Returns:
        x: float32 tensor with shape (2, T, F).
        y: scalar long tensor with activity_id in [0, 19].
        metadata (optional): dict with subject, day, stream, t_start, device, and path.
    """

    def __init__(
        self,
        root,
        device="M1",
        activities=None,
        subjects=None,
        days=None,
        return_metadata=False,
        cache_size=2,
        transform=None,
    ):
        self.root = Path(root)
        self.device = device
        self.activities = None if activities is None else {int(v) for v in activities}
        self.subjects = None if subjects is None else {int(v) for v in subjects}
        self.days = None if days is None else {int(v) for v in days}
        self.return_metadata = return_metadata
        self.cache_size = max(0, int(cache_size))
        self.transform = transform
        self._cache = OrderedDict()

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")

        self.records = []
        self.cumulative_sizes = []
        total = 0
        paths = sorted(self.root.rglob("*_preprocessed.npz"))
        for path in paths:
            # Metadata is stored in each NPZ by preprocess_csi.py. Filename parsing
            # is a fallback for files missing those fields.
            with np.load(path, allow_pickle=False) as data:
                stem = path.name.removesuffix("_preprocessed.npz")
                parts = stem.split("_")
                try:
                    filename_activity = ord(parts[0][0]) - ord("A")
                    filename_day = int(parts[1])
                    filename_device = parts[2]
                    filename_subject = int(parts[3].lstrip("P"))
                except (IndexError, ValueError):
                    filename_activity = filename_day = filename_subject = None
                    filename_device = None

                activity_id = int(data["activity_id"]) if "activity_id" in data.files else filename_activity
                day = int(data["day"]) if "day" in data.files else filename_day
                subject = int(data["subject"]) if "subject" in data.files else filename_subject
                file_device = str(data["device"].item()) if "device" in data.files else filename_device
                if activity_id is None or day is None or subject is None:
                    raise ValueError(f"Could not determine labels from {path}")

                if self.device is not None and file_device != self.device:
                    continue
                if self.activities is not None and activity_id not in self.activities:
                    continue
                if self.subjects is not None and subject not in self.subjects:
                    continue
                if self.days is not None and day not in self.days:
                    continue

                n_windows = len(data["windows"])
                if n_windows == 0:
                    continue
                self.records.append({
                    "path": path,
                    "n_windows": n_windows,
                    "activity_id": activity_id,
                    "subject": subject,
                    "day": day,
                    "device": file_device,
                    "has_stream": "stream" in data.files,
                    "has_t_start": "t_start" in data.files,
                })
                total += n_windows
                self.cumulative_sizes.append(total)

        self.total_windows = total
        if not self.records:
            raise ValueError(
                f"No matching preprocessed CSI windows found in {self.root} "
                f"(device={self.device!r}, activities={self.activities}, "
                f"subjects={self.subjects}, days={self.days})"
            )

    def __len__(self):
        return self.total_windows

    def _load_record(self, record_index):
        path = self.records[record_index]["path"]
        if self.cache_size > 0 and path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        with np.load(path, allow_pickle=False) as data:
            record_data = {
                "windows": data["windows"].astype(np.float32, copy=False),
                "stream": data["stream"].copy() if "stream" in data.files else None,
                "t_start": data["t_start"].copy() if "t_start" in data.files else None,
            }

        if self.cache_size > 0:
            self._cache[path] = record_data
            self._cache.move_to_end(path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return record_data

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        record_index = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        previous_total = 0 if record_index == 0 else self.cumulative_sizes[record_index - 1]
        local_index = index - previous_total
        record = self.records[record_index]
        loaded = self._load_record(record_index)

        x = torch.from_numpy(np.array(loaded["windows"][local_index], dtype=np.float32, copy=True))
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(record["activity_id"], dtype=torch.long)

        if not self.return_metadata:
            return x, y

        metadata = {
            "subject": record["subject"],
            "day": record["day"],
            "device": record["device"],
            "path": str(record["path"]),
            "window_index_in_file": int(local_index),
            "stream": int(loaded["stream"][local_index]) if loaded["stream"] is not None else None,
            "t_start": float(loaded["t_start"][local_index]) if loaded["t_start"] is not None else None,
        }
        return x, y, metadata


if __name__ == "__main__":
    # Example smoke test; update this path to your preprocessed data directory.
    ds = CSIDataset("data/HAR-3/CSI_preprocessed/M1", device="M1", return_metadata=True)
    x, y, meta = ds[0]
    print(f"Samples: {len(ds)} | x: {tuple(x.shape)} | y: {y.item()} | metadata: {meta}")
