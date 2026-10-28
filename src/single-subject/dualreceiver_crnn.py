"""
Train a CRNN (CNN-LSTM Hybrid) for CSI-based activity recognition.
Expected input: *_preprocessed.npz files produced by preprocess_csi.py.
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class DualReceiverCSIDataset(Dataset):
    """Loads and pairs preprocessed CSI windows from both M1 and M2 receivers."""

    def __init__(self, m1_root, m2_root, subjects=None, activities=None, samples=None, cache_recordings=2, is_train=False):
        self.m1_root = Path(m1_root)
        self.m2_root = Path(m2_root)
        self.cache_recordings = max(0, int(cache_recordings))
        self.normalization = None
        self.is_train = is_train

        if samples is not None:
            self.samples = samples
        else:
            self.subject_filter = None if subjects is None else {int(s) for s in subjects}
            self.activity_filter = None if activities is None else {int(a) for a in activities}
            self.samples = self._scan_paired_files()

        if not self.samples:
            raise ValueError(f"No matching paired windows found between {self.m1_root} and {self.m2_root}")

        self.cumulative = np.cumsum([s["n_windows"] for s in self.samples])
        self.total_windows = int(self.cumulative[-1])

    def _scan_paired_files(self):
        m1_files = sorted(self.m1_root.rglob("*_preprocessed.npz"))
        paired_samples = []
        
        for m1_path in m1_files:
            # 1. Swap the directory root from M1 to M2
            m2_path_str = str(m1_path).replace(str(self.m1_root), str(self.m2_root))
            # 2. Swap the device tag in the filename itself (e.g., A_1_M1_P1 -> A_1_M2_P1)
            m2_path_str = m2_path_str.replace("_M1_", "_M2_")
            m2_path = Path(m2_path_str)
            
            if not m2_path.exists():
                print(f"Skipping (Missing M2 Pair): {m2_path.name}")
                continue # Skip if M2 recording is missing for this file

            with np.load(m1_path, allow_pickle=False) as d1, np.load(m2_path, allow_pickle=False) as d2:
                subject = int(d1["subject"])
                activity_id = int(d1["activity_id"])
                
                if self.subject_filter is not None and subject not in self.subject_filter: continue
                if self.activity_filter is not None and activity_id not in self.activity_filter: continue
                
                # Ensure both files have the exact same number of windows
                n_windows = min(len(d1["windows"]), len(d2["windows"]))
                if n_windows == 0: continue

                paired_samples.append({
                    "m1_path": m1_path,
                    "m2_path": m2_path,
                    "subject": subject,
                    "activity_id": activity_id,
                    "n_windows": n_windows,
                })
        return paired_samples

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        file_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        prev = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        local_idx = index - prev
        sample = self.samples[file_idx]

        cache = getattr(self, "_cache", {})
        self._cache = cache
        
        # Load M1 window
        m1_key = str(sample["m1_path"])
        if m1_key not in cache:
            with np.load(sample["m1_path"], allow_pickle=False) as d:
                cache[m1_key] = d["windows"].astype(np.float32, copy=True)
        
        # Load M2 window
        m2_key = str(sample["m2_path"])
        if m2_key not in cache:
            with np.load(sample["m2_path"], allow_pickle=False) as d:
                cache[m2_key] = d["windows"].astype(np.float32, copy=True)

        if self.cache_recordings > 0 and len(cache) > self.cache_recordings * 2:
            for _ in range(2): cache.pop(next(iter(cache)))

        # Extract Amplitude channel (Channel 0) from both receivers and stack them
        x_m1 = cache[m1_key][local_idx, 0:1, :, :] # M1 Amplitude
        x_m2 = cache[m2_key][local_idx, 0:1, :, :] # M2 Amplitude
        
        x = np.concatenate([x_m1, x_m2], axis=0)   # Resulting shape: (2, T, F) -> Channel 0: M1, Channel 1: M2
        x = torch.from_numpy(x.copy())

        if self.normalization is not None:
            mean, std = self.normalization
            x = (x - mean) / std

        # Time-Series Data Augmentation (Training only)
        if self.is_train:
            if random.random() < 0.5: x += torch.randn_like(x) * 0.05
            if random.random() < 0.5:
                t_max = x.shape[1]
                t_idx = random.randint(0, max(0, t_max - 15))
                x[:, t_idx:t_idx+15, :] = 0
            if random.random() < 0.5:
                f_max = x.shape[2]
                f_idx = random.randint(0, max(0, f_max - 20))
                x[:, :, f_idx:f_idx+20] = 0

        y = torch.tensor(sample["activity_id"], dtype=torch.long)
        return x, y


class CSICRNN(nn.Module):
    """CNN-LSTM Hybrid for CSI Amplitude data."""

    def __init__(self, num_classes=20, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        
        # Spatial Feature Extractor (Compresses Frequencies, keeps Time intact)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)), 

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((None, 1)) 
        )
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c = self.cnn(x)              # (Batch, 128, Time, 1)
        c = c.squeeze(3)             # (Batch, 128, Time)
        c = c.permute(0, 2, 1)       # (Batch, Time, 128)
        
        lstm_out, _ = self.lstm(c)   
        final_time_step = lstm_out[:, -1, :] 
        return self.classifier(final_time_step)


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",")] if value else None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

def split_train_val_recordings(samples, train_subjects, val_fraction=0.15, seed=42):
    rng = random.Random(seed)
    candidates = [s for s in samples if s["subject"] in set(train_subjects)]
    by_activity = {}
    for sample in candidates: 
        by_activity.setdefault(sample["activity_id"], []).append(sample)
        
    val_paths = set()
    for group in by_activity.values():
        if len(group) < 2: continue
        rng.shuffle(group)
        n_val = max(1, min(int(round(len(group) * val_fraction)), len(group) - 1))
        # Updated to use 'm1_path' as the unique ID for the pair
        val_paths.update(str(s["m1_path"]) for s in group[:n_val])
        
    train_paths = [s["m1_path"] for s in candidates if str(s["m1_path"]) not in val_paths]
    val_paths = [s["m1_path"] for s in candidates if str(s["m1_path"]) in val_paths]
    return train_paths, val_paths

def compute_train_normalization(dataset):
    sums, sums_sq, count = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64), 0
    for sample in dataset.samples:
        with np.load(sample["m1_path"], allow_pickle=False) as d1, np.load(sample["m2_path"], allow_pickle=False) as d2:
            x1 = np.asarray(d1["windows"][:, 0:1, :, :], dtype=np.float32)
            x2 = np.asarray(d2["windows"][:, 0:1, :, :], dtype=np.float32)
            x = np.concatenate([x1, x2], axis=1)
        if x.size == 0: continue
        sums += x.sum(axis=(0, 2, 3), dtype=np.float64)
        sums_sq += np.square(x, dtype=np.float64).sum(axis=(0, 2, 3), dtype=np.float64)
        count += x.shape[0] * x.shape[2] * x.shape[3]
        
    mean = sums / count
    std = np.sqrt(np.maximum(sums_sq / count - mean * mean, 1e-12))
    return (torch.tensor(mean, dtype=torch.float32).view(2, 1, 1),
            torch.tensor(std, dtype=torch.float32).view(2, 1, 1))

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total, class_count = 0.0, 0, 0, 20
    confusion = torch.zeros((class_count, class_count), dtype=torch.int64, device=device)
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            
            batch_size = y.size(0)
            loss_sum += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += batch_size
            
            indices = y * class_count + pred
            confusion += torch.bincount(indices, minlength=class_count**2).view(class_count, class_count)
            
    confusion = confusion.cpu().numpy()
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    
    present = (tp + fp + fn) > 0 
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    
    return loss_sum / max(total, 1), correct / max(total, 1), macro_f1, confusion, precision, recall, f1

def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    epoch_bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=True) if tqdm else loader

    for x, y in epoch_bar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
            
        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size

        if tqdm:
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.3f}")

    return running_loss / total, correct / total

def split_intra_subject_recordings(samples, train_frac=0.70, val_frac=0.15, seed=42):
    rng = random.Random(seed)
    
    # Group by (Activity, Subject) to ensure stratified splitting across all people and actions
    grouped = {}
    for sample in samples: 
        key = (sample["activity_id"], sample["subject"])
        grouped.setdefault(key, []).append(sample)
        
    train_paths, val_paths, test_paths = set(), set(), set()
    
    for group in grouped.values():
        rng.shuffle(group)
        n_total = len(group)
        n_train = max(1, int(round(n_total * train_frac)))
        n_val = max(1, int(round(n_total * val_frac)))
        
        # Ensure we don't overflow the list boundaries
        if n_train + n_val >= n_total:
            n_val = max(1, (n_total - n_train) // 2)
            
        train_paths.update(str(s["m1_path"]) for s in group[:n_train])
        val_paths.update(str(s["m1_path"]) for s in group[n_train:n_train + n_val])
        test_paths.update(str(s["m1_path"]) for s in group[n_train + n_val:])
        
    return train_paths, val_paths, test_paths

def prepare_datasets(args, device):
    all_data = DualReceiverCSIDataset(args.m1_dir, args.m2_dir, activities=parse_int_list(args.activities))
    
    # Generate the 3-way intra-subject split
    train_paths, val_paths, test_paths = split_intra_subject_recordings(all_data.samples, seed=args.seed)
    
    def filter_samples(paths_set): 
        return [s for s in all_data.samples if str(s["m1_path"]) in paths_set]
    
    train_ds = DualReceiverCSIDataset(args.m1_dir, args.m2_dir, samples=filter_samples(train_paths), cache_recordings=args.cache_recordings, is_train=True)
    val_ds = DualReceiverCSIDataset(args.m1_dir, args.m2_dir, samples=filter_samples(val_paths), cache_recordings=args.cache_recordings, is_train=False)
    test_ds = DualReceiverCSIDataset(args.m1_dir, args.m2_dir, samples=filter_samples(test_paths), cache_recordings=args.cache_recordings, is_train=False)
    
    # We return "All Subjects" for the logging printouts
    all_subjects = sorted({s["subject"] for s in all_data.samples})
    return train_ds, val_ds, test_ds, all_subjects, all_subjects, all_subjects, "Intra-subject (Same-Subject) split"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-dir", required=True)
    parser.add_argument("--m2-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--csi-device", default="M1")
    # parser.add_argument("--test-subjects", default=None) not needed.
    parser.add_argument("--val-subjects", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--cache-recordings", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--activities", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3) # Increased regularisation
    parser.add_argument("--dropout", type=float, default=0.5)       # Increased dropout
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/csi_crnn.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu")

    train_ds, val_ds, test_ds, train_subs, val_subs, test_subs, val_mode = prepare_datasets(args, device)
    
    use_pin = device.type == "cuda"
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, use_pin)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, use_pin)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, use_pin)

    norm_mean, norm_std = compute_train_normalization(train_ds)
    for ds in (train_ds, val_ds, test_ds):
        ds.normalization = (norm_mean, norm_std)

    model = CSICRNN(num_classes=20, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    best_val_f1, best_state, epochs_without_improvement = -1.0, None, 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, epochs_without_improvement = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_state}, output_path)
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nEvaluating best model on test set...")
    if best_state: model.load_state_dict(best_state)
    t_loss, t_acc, t_f1, confusion, prec, rec, f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test | Loss {t_loss:.4f} Acc {t_acc:.4f} F1 {t_f1:.4f}")

if __name__ == "__main__":
    main()