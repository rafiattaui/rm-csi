"""
Train a dual-receiver CNN for CSI-based activity recognition.
Expected input: *_preprocessed.npz files produced by preprocess_csi.py.

Each receiver (M1, M2) is processed by its own convolutional stem, the two
feature maps are concatenated, and a shared CNN trunk + global pooling classifies
the window. No recurrent layers.

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


class DualReceiverCSIDataset(Dataset):
    """Loads and pairs preprocessed CSI windows from both M1 and M2 receivers."""

    def __init__(self, m1_root, m2_root, activities=None, samples=None, cache_recordings=2, is_train=False):
        self.m1_root = Path(m1_root)
        self.m2_root = Path(m2_root)
        self.cache_recordings = max(0, int(cache_recordings))
        self.normalization = None
        self.is_train = is_train

        if samples is not None:
            self.samples = samples
        else:
            self.activity_filter = None if activities is None else {int(a) for a in activities}
            self.samples = self._scan_paired_files()

        if not self.samples:
            raise ValueError(f"No matching paired windows found between {self.m1_root} and {self.m2_root}")

        self.cumulative = np.cumsum([s["n_windows"] for s in self.samples])
        self.total_windows = int(self.cumulative[-1])

    def _scan_paired_files(self):
        paired = []
        for m1_path in sorted(self.m1_root.rglob("*_preprocessed.npz")):
            try:
                rel_path = m1_path.relative_to(self.m1_root)
            except ValueError:
                continue
            m2_path = self.m2_root / rel_path
            if "_M1_" in m2_path.name:
                m2_path = m2_path.with_name(m2_path.name.replace("_M1_", "_M2_"))
            if not m2_path.exists():
                continue

            with np.load(m1_path, allow_pickle=False) as d1, np.load(m2_path, allow_pickle=False) as d2:
                subject = int(d1["subject"])
                activity_id = int(d1["activity_id"])
                if self.activity_filter is not None and activity_id not in self.activity_filter:
                    continue
                n_windows = min(len(d1["windows"]), len(d2["windows"]))
                if n_windows == 0:
                    continue

            paired.append({
                "m1_path": m1_path,
                "m2_path": m2_path,
                "subject": subject,
                "activity_id": activity_id,
                "n_windows": n_windows,
            })
        return paired

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

        w1 = self._load(sample["m1_path"])
        w2 = self._load(sample["m2_path"])

        if self.cache_recordings > 0 and len(self._cache) > self.cache_recordings * 2:
            for _ in range(2):
                self._cache.pop(next(iter(self._cache)))

        # Amplitude channel (0) from each receiver -> (2, T, F)
        x = np.concatenate([w1[local_idx, 0:1], w2[local_idx, 0:1]], axis=0)
        x = torch.from_numpy(x.copy())

        if self.normalization is not None:
            mean, std = self.normalization
            x = (x - mean) / std

        if self.is_train:
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


class DualReceiverCNN(nn.Module):
    """Two-branch CNN: one stem per receiver, fused, then a shared trunk."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, width=32):
        super().__init__()
        # Receiver-specific stems (each sees one amplitude channel)
        self.stem_m1 = conv_block(1, width)
        self.stem_m2 = conv_block(1, width)

        # Shared trunk on the fused feature map
        self.trunk = nn.Sequential(
            conv_block(width * 2, width * 4),
            conv_block(width * 4, width * 8, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width * 8, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):                       # x: (B, 2, T, F)
        f1 = self.stem_m1(x[:, 0:1])
        f2 = self.stem_m2(x[:, 1:2])
        f = torch.cat([f1, f2], dim=1)          # (B, 2*width, T/2, F/2)
        f = self.trunk(f)
        f = self.pool(f).flatten(1)             # (B, 8*width)
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
    sums = np.zeros(2, dtype=np.float64)
    sums_sq = np.zeros(2, dtype=np.float64)
    count = 0
    for sample in dataset.samples:
        n = sample["n_windows"]
        with np.load(sample["m1_path"], allow_pickle=False) as d1, \
             np.load(sample["m2_path"], allow_pickle=False) as d2:
            x1 = np.asarray(d1["windows"][:n, 0:1], dtype=np.float32)
            x2 = np.asarray(d2["windows"][:n, 0:1], dtype=np.float32)
        x = np.concatenate([x1, x2], axis=1)
        if x.size == 0:
            continue
        sums += x.sum(axis=(0, 2, 3), dtype=np.float64)
        sums_sq += np.square(x, dtype=np.float64).sum(axis=(0, 2, 3), dtype=np.float64)
        count += x.shape[0] * x.shape[2] * x.shape[3]

    mean = sums / count
    std = np.sqrt(np.maximum(sums_sq / count - mean * mean, 1e-12))
    return (torch.tensor(mean, dtype=torch.float32).view(2, 1, 1),
            torch.tensor(std, dtype=torch.float32).view(2, 1, 1))


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
    all_data = DualReceiverCSIDataset(args.m1_dir, args.m2_dir, activities=parse_int_list(args.activities))
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
        return DualReceiverCSIDataset(args.m1_dir, args.m2_dir, samples=s,
                                      cache_recordings=args.cache_recordings, is_train=train)

    return build(train_samples, True), build(val_samples, False), build(test_samples, False), mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-dir", required=True)
    parser.add_argument("--m2-dir", required=True)
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
    parser.add_argument("--output", default="checkpoints/csi_dual_cnn.pt")
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

    model = DualReceiverCNN(num_classes=NUM_CLASSES, dropout=args.dropout, width=args.width).to(device)
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