"""
Train a single-receiver CNN for CSI-based activity recognition.
Expected input: *_preprocessed.npz files produced by preprocess_csi.py
(keys: windows [N, C, T, F], subject, activity_id) for ONE receiver.

Split modes:
  - default: random recording-level 70/15/15 split (within-subject)
  - --test-subjects 5 [--val-subjects 4]: subject-held-out split (cross-subject)
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

NUM_CLASSES = 20


class SingleReceiverCSIDataset(Dataset):
    """Loads preprocessed CSI windows from a single receiver."""

    def __init__(self, root, activities=None, samples=None, cache_recordings=2, is_train=False, channel=0, use_both_channels=True):
        self.root = Path(root)
        self.cache_recordings = max(0, int(cache_recordings))
        self.normalization = None
        self.is_train = is_train
        self.channel = int(channel)
        self.use_both_channels = use_both_channels

        if samples is not None:
            self.samples = samples
        else:
            self.activity_filter = None if activities is None else {int(a) for a in activities}
            self.samples = self._scan_files()

        if not self.samples:
            raise ValueError(f"No matching windows found in {self.root}")

        self.cumulative = np.cumsum([s["n_windows"] for s in self.samples])
        self.total_windows = int(self.cumulative[-1])

    def _scan_files(self):
        found = []
        for path in sorted(self.root.rglob("*_preprocessed.npz")):
            with np.load(path, allow_pickle=False) as d:
                subject = int(d["subject"])
                activity_id = int(d["activity_id"])
                if self.activity_filter is not None and activity_id not in self.activity_filter:
                    continue
                n_windows = len(d["windows"])
                if n_windows == 0:
                    continue

            found.append({
                "path": path,
                "subject": subject,
                "activity_id": activity_id,
                "n_windows": n_windows,
            })
        return found

    def __len__(self):
        return self.total_windows

    def _load(self, path):
        key = str(path)
        cache = self._cache
        if key not in cache:
            with np.load(path, allow_pickle=False) as d:
                cache[key] = d["windows"].astype(np.float32, copy=True)
        return cache[key]

    def __getitem__(self, index):
        file_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        prev = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        local_idx = index - prev
        sample = self.samples[file_idx]

        if not hasattr(self, "_cache"):
            self._cache = {}

        windows = self._load(sample["path"])

        if self.use_both_channels:
            x = torch.from_numpy(windows[local_idx, :2].copy())  # Shape: (2, T, F)
        else:
            x = torch.from_numpy(windows[local_idx, self.channel:self.channel + 1].copy())  # Shape: (1, T, F)

        if self.cache_recordings > 0 and len(self._cache) > self.cache_recordings:
            self._cache.pop(next(iter(self._cache)))

        if self.normalization is not None:
            mean, std = self.normalization  # Shapes: (1, C, 1, F)
            # Squeezing dim 0 gives (C, 1, F), which broadcasts over (C, T, F)
            x = (x - mean.squeeze(0)) / std.squeeze(0)

        if self.is_train:
            # Data augmentations
            if random.random() < 0.5:
                x += torch.randn_like(x) * 0.05
            if random.random() < 0.5:
                t_idx = random.randint(0, max(0, x.shape[1] - 15))
                x[:, t_idx:t_idx + 15, :] = 0
            if random.random() < 0.5:
                f_idx = random.randint(0, max(0, x.shape[2] - 20))
                x[:, :, f_idx:f_idx + 20] = 0

        y = torch.tensor(sample["activity_id"], dtype=torch.long)
        return x, y


def conv_block(in_ch, out_ch, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SingleReceiverCNN(nn.Module):
    """CNN for one receiver: three conv blocks, global pooling, linear classifier."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, width=32):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, width),                      # (B, w,  T/2, F/2)
            conv_block(width, width * 4),              # (B, 4w, T/4, F/4)
            conv_block(width * 4, width * 8, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.temporal_classifier = nn.Sequential(
            nn.Conv1d(width * 8, width * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Pool time only AFTER temporal feature extraction
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):                              # x: (B, 1, T, F)
        f = self.features(x)
        f = self.pool(f).flatten(1)                    # (B, 8*width)
        return self.classifier(f)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def parse_int_list(value):
    return [int(p.strip()) for p in value.split(",")] if value else None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)


def compute_train_normalization(dataset):
    """
    Computes per-channel, per-subcarrier mean and std across training windows.
    Returns tensors of shape (1, C, 1, F) for broadcasting across (B, C, T, F).
    """
    total = None      # Accumulator for sums, shape (1, C, 1, F)
    total_sq = None   # Accumulator for squared sums, shape (1, C, 1, F)
    total_frames = 0  # Accumulator for total frames (N * T)

    # Determine channel slice based on dataset settings (e.g., both channels :2 or single channel)
    ch_slice = slice(0, 2) if getattr(dataset, "use_both_channels", False) else slice(dataset.channel, dataset.channel + 1)

    for sample in dataset.samples:
        with np.load(sample["path"], allow_pickle=False) as d:
            # d["windows"] shape: (N_windows, 2, T, F)
            x = d["windows"][:, ch_slice].astype(np.float64)

        if x.size == 0:
            continue

        # Sum across sample windows (axis 0) and time frames (axis 2) -> leaves shape (1, C, 1, F)
        sum_x = x.sum(axis=(0, 2), keepdims=True)
        sum_x2 = np.square(x).sum(axis=(0, 2), keepdims=True)

        if total is None:
            total = sum_x
            total_sq = sum_x2
        else:
            total += sum_x
            total_sq += sum_x2

        total_frames += x.shape[0] * x.shape[2]  # N_windows * T

    if total_frames == 0 or total is None:
        raise ValueError("No valid window data found to compute normalization statistics.")

    # Calculate per-subcarrier mean and standard deviation
    mean = total / total_frames
    variance = (total_sq / total_frames) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    std = np.maximum(std, 1e-6)  # Guard against division by zero for silent/flat subcarriers

    # Convert to PyTorch tensors with shape (1, C, 1, F)
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)

    return mean_tensor, std_tensor


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            bs = y.size(0)
            loss_sum += loss.item() * bs
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += bs

            idx = y * NUM_CLASSES + pred
            confusion += torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)

    confusion = confusion.cpu().numpy()
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = (tp + fn) > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    return loss_sum / max(total, 1), correct / max(total, 1), macro_f1, confusion


def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=True) if tqdm else loader

    for x, y in bar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs
        if tqdm:
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.3f}")

    return running_loss / total, correct / total


def prepare_datasets(args):
    all_data = SingleReceiverCSIDataset(args.data_dir, activities=parse_int_list(args.activities),
                                        channel=args.channel)
    samples = list(all_data.samples)

    test_subs = parse_int_list(args.test_subjects)
    if test_subs:
        # Cross-subject: hold out whole subjects
        val_subs = parse_int_list(args.val_subjects)
        subjects = sorted({s["subject"] for s in samples})
        remaining = [s for s in subjects if s not in test_subs]
        if not val_subs:
            val_subs = remaining[-1:]
        train_subs = [s for s in remaining if s not in val_subs]
        if not train_subs:
            raise ValueError("No training subjects left after holding out val/test subjects.")

        train_samples = [s for s in samples if s["subject"] in train_subs]
        val_samples = [s for s in samples if s["subject"] in val_subs]
        test_samples = [s for s in samples if s["subject"] in test_subs]
        mode = f"Cross-subject (train {train_subs} | val {val_subs} | test {test_subs})"
    else:
        # Within-subject: random recording-level split
        rng = random.Random(args.seed)
        rng.shuffle(samples)
        n_train = max(1, int(len(samples) * 0.70))
        n_val = max(1, int(len(samples) * 0.15))
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]
        if not val_samples and len(train_samples) > 1:
            val_samples.append(train_samples.pop())
        if not test_samples and len(train_samples) > 1:
            test_samples.append(train_samples.pop())
        mode = "Within-subject (random recording-level split)"

    def build(s, train):
        return SingleReceiverCSIDataset(args.data_dir, samples=s, cache_recordings=args.cache_recordings,
                                        is_train=train, channel=args.channel)

    return build(train_samples, True), build(val_samples, False), build(test_samples, False), mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory of *_preprocessed.npz for the receiver.")
    parser.add_argument("--channel", type=int, default=0, help="Channel index in 'windows' (0 = amplitude).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--test-subjects", default=None, help="Comma-separated subject IDs to hold out for testing (cross-subject).")
    parser.add_argument("--val-subjects", default=None, help="Comma-separated subject IDs for validation (cross-subject mode).")
    parser.add_argument("--cache-recordings", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--activities", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=32, help="Base channel width of the CNN.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/csi_single_cnn.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu")

    train_ds, val_ds, test_ds, mode = prepare_datasets(args)
    print(f"Split mode: {mode}")
    print(f"Recordings  train/val/test: {len(train_ds.samples)}/{len(val_ds.samples)}/{len(test_ds.samples)}")
    print(f"Windows     train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    pin = device.type == "cuda"
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, pin)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, pin)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, pin)

    norm = compute_train_normalization(train_ds)
    for ds in (train_ds, val_ds, test_ds):
        ds.normalization = norm

    model = SingleReceiverCNN(num_classes=NUM_CLASSES, dropout=args.dropout, width=args.width).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_val_f1, best_state, bad_epochs = -1.0, None, 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        print(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, bad_epochs = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_state, "args": vars(args)}, output_path)
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nEvaluating best model on test set...")
    if best_state:
        model.load_state_dict(best_state)
    t_loss, t_acc, t_f1, confusion = evaluate(model, test_loader, criterion, device)
    print(f"Test | Loss {t_loss:.4f} Acc {t_acc:.4f} F1 {t_f1:.4f}")

    per_class_recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    print("Per-class recall:", np.round(per_class_recall, 3).tolist())


if __name__ == "__main__":
    main()