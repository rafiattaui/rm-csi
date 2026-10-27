"""
Train a small 2D CNN for CSI-based activity recognition.

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


class CSIDataset(Dataset):
    """Load preprocessed CSI windows, filtered by device and subject IDs."""

    def __init__(self, root, device="M1", subjects=None, activities=None, samples=None, cache_recordings=2):
        self.root = Path(root)
        self.cache_recordings = max(0, int(cache_recordings))
        self.normalization = None

        # Optimization: Accept pre-parsed samples to avoid redundant file I/O
        if samples is not None:
            self.samples = samples
        else:
            self.device_filter = device
            self.subject_filter = None if subjects is None else {int(s) for s in subjects}
            self.activity_filter = None if activities is None else {int(a) for a in activities}
            self.samples = self._scan_files()

        if not self.samples:
            raise ValueError(f"No matching windows found in {self.root}")

        self.cumulative = np.cumsum([s["n_windows"] for s in self.samples])
        self.total_windows = int(self.cumulative[-1])

    def _scan_files(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.root}")

        files = sorted(self.root.rglob("*_preprocessed.npz"))
        if not files:
            raise FileNotFoundError(f"No *_preprocessed.npz files found under {self.root}.")

        parsed_samples = []
        for path in files:
            with np.load(path, allow_pickle=False) as d:
                file_device = str(d["device"].item()) if "device" in d.files else None
                subject = int(d["subject"])
                activity_id = int(d["activity_id"])
                
                if self.device_filter and file_device != self.device_filter: continue
                if self.subject_filter is not None and subject not in self.subject_filter: continue
                if self.activity_filter is not None and activity_id not in self.activity_filter: continue
                
                parsed_samples.append({
                    "path": path,
                    "subject": subject,
                    "activity_id": activity_id,
                    "n_windows": len(d["windows"]),
                })
        return parsed_samples

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        file_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        prev = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        local_idx = index - prev
        sample = self.samples[file_idx]

        cache = getattr(self, "_cache", {})
        self._cache = cache
        path_key = str(sample["path"])
        
        if path_key not in cache:
            with np.load(sample["path"], allow_pickle=False) as d:
                loaded_windows = d["windows"].astype(np.float32, copy=True)
            if self.cache_recordings > 0:
                cache[path_key] = loaded_windows
                while len(cache) > self.cache_recordings:
                    cache.pop(next(iter(cache)))
            else:
                cache[path_key] = loaded_windows

        x = torch.from_numpy(cache[path_key][local_idx].copy())
        if self.cache_recordings == 0:
            cache.pop(path_key, None)
            
        if self.normalization is not None:
            mean, std = self.normalization
            x = (x - mean) / std
            
        y = torch.tensor(sample["activity_id"], dtype=torch.long)
        return x, y


class SmallCSI2DCNN(nn.Module):
    """Compact CNN with Kaiming initialization."""

    def __init__(self, num_classes=20, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",")] if value else None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
    )


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
        val_paths.update(str(s["path"]) for s in group[:n_val])
        
    train_paths = [s["path"] for s in candidates if str(s["path"]) not in val_paths]
    val_paths = [s["path"] for s in candidates if str(s["path"]) in val_paths]
    return train_paths, val_paths


def compute_train_normalization(dataset):
    sums, sums_sq, count = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64), 0
    for sample in dataset.samples:
        with np.load(sample["path"], allow_pickle=False) as d:
            x = np.asarray(d["windows"], dtype=np.float32)
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


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    epoch_bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=True) if tqdm else loader

    for x, y in epoch_bar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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


def prepare_datasets(args, device):
    print(f"Scanning preprocessed files under: {args.data_dir}")
    all_data = CSIDataset(args.data_dir, device=args.csi_device, activities=parse_int_list(args.activities))
    all_subjects = sorted({s["subject"] for s in all_data.samples})
    
    test_subjects = parse_int_list(args.test_subjects) or [all_subjects[-1]]
    remaining_subjects = [s for s in all_subjects if s not in test_subjects]
    train_subjects = remaining_subjects
    
    train_paths, val_paths = split_train_val_recordings(all_data.samples, train_subjects, val_fraction=args.val_fraction, seed=args.seed)
    
    # Instantiate optimized subsets without rescanning
    def filter_samples(paths): return [s for s in all_data.samples if str(s["path"]) in paths]
    test_paths = [str(s["path"]) for s in all_data.samples if s["subject"] in test_subjects]
    
    train_ds = CSIDataset(args.data_dir, samples=filter_samples(train_paths), cache_recordings=args.cache_recordings)
    val_ds = CSIDataset(args.data_dir, samples=filter_samples(val_paths), cache_recordings=args.cache_recordings)
    test_ds = CSIDataset(args.data_dir, samples=filter_samples(test_paths), cache_recordings=args.cache_recordings)
    
    return train_ds, val_ds, test_ds, train_subjects, None, test_subjects, "recording-level split within training subjects"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--csi-device", default="M1")
    parser.add_argument("--test-subjects", default=None)
    parser.add_argument("--val-subjects", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--cache-recordings", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--activities", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/csi_cnn.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu")
    print(f"Training device: {device}")

    train_ds, val_ds, test_ds, train_subs, val_subs, test_subs, val_mode = prepare_datasets(args, device)
    
    use_pin = device.type == "cuda"
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, use_pin)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, use_pin)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, use_pin)

    norm_mean, norm_std = compute_train_normalization(train_ds)
    for ds in (train_ds, val_ds, test_ds):
        ds.normalization = (norm_mean, norm_std)

    model = SmallCSI2DCNN(num_classes=20, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler decays learning rate when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_val_f1, best_state, epochs_without_improvement = -1.0, None, 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs)
        val_loss, val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, epochs_without_improvement = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_state}, output_path)
            print(f"  --> New best F1: {best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nEvaluating best model on test set...")
    if best_state: model.load_state_dict(best_state)
    t_loss, t_acc, t_f1, confusion, prec, rec, f1 = evaluate(model, test_loader, criterion, device)
    
    print(f"Test | Loss {t_loss:.4f} Acc {t_acc:.4f} F1 {t_f1:.4f}")
    for c in range(20):
        if confusion[c].sum() > 0:
            print(f" Class {c:02d}: Precision {prec[c]:.4f}, Recall {rec[c]:.4f}, F1 {f1[c]:.4f}")


if __name__ == "__main__":
    main()