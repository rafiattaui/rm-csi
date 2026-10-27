"""
Train a small 2D CNN for CSI-based activity recognition.

Expected input: *_preprocessed.npz files produced by preprocess_csi.py.
Each file contains:
    windows: (N, 2, T, F)  # amplitude and phase
    activity_id, subject, day, device, stream, t_start

Example:
    python train_csi_cnn.py --data-dir data/HAR-3/CSI_preprocessed/M1
    python train_csi_cnn.py --data-dir data/HAR-3/CSI_preprocessed/M1 --device cuda
    python train_csi_cnn.py --data-dir data/HAR-3/CSI_preprocessed/M1 --test-subjects 5 --val-subjects 4

CUDA is used automatically when available unless --device cpu is specified.
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

    def __init__(self, root, device="M1", subjects=None, activities=None):
        self.root = Path(root)
        self.device_filter = device
        self.subject_filter = None if subjects is None else {int(s) for s in subjects}
        self.activity_filter = None if activities is None else {int(a) for a in activities}
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.root}")

        files = sorted(self.root.rglob("*_preprocessed.npz"))
        if not files:
            raise FileNotFoundError(
                f"No *_preprocessed.npz files found under {self.root}. "
                "Point --data-dir to the preprocessed folder."
            )

        for path in files:
            # Read only scalar metadata at indexing time; windows are loaded lazily.
            with np.load(path, allow_pickle=False) as d:
                file_device = str(d["device"].item()) if "device" in d.files else None
                subject = int(d["subject"])
                activity_id = int(d["activity_id"])
                n_windows = len(d["windows"])
                if self.device_filter and file_device != self.device_filter:
                    continue
                if self.subject_filter is not None and subject not in self.subject_filter:
                    continue
                if self.activity_filter is not None and activity_id not in self.activity_filter:
                    continue
                self.samples.append({
                    "path": path,
                    "subject": subject,
                    "activity_id": activity_id,
                    "n_windows": n_windows,
                })

        if not self.samples:
            raise ValueError(
                f"No matching windows found for device={device!r}, "
                f"subjects={subjects}, activities={activities} in {self.root}"
            )

        self.cumulative = np.cumsum([s["n_windows"] for s in self.samples])
        self.total_windows = int(self.cumulative[-1])

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        file_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        prev = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        local_idx = index - prev
        sample = self.samples[file_idx]

        # Per-worker cache keeps recently used recordings open in memory.
        cache = getattr(self, "_cache", None)
        if cache is None:
            self._cache = {}
            cache = self._cache
        path_key = str(sample["path"])
        if path_key not in cache:
            with np.load(sample["path"], allow_pickle=False) as d:
                cache[path_key] = d["windows"].astype(np.float32, copy=True)

        x = torch.from_numpy(cache[path_key][local_idx])
        y = torch.tensor(sample["activity_id"], dtype=torch.long)
        return x, y


class SmallCSI2DCNN(nn.Module):
    """Compact CNN; input shape (B, 2, T, F)."""

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

    def forward(self, x):
        return self.classifier(self.features(x))


def parse_int_list(value):
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    class_count = 20
    tp = np.zeros(class_count, dtype=np.int64)
    fp = np.zeros(class_count, dtype=np.int64)
    fn = np.zeros(class_count, dtype=np.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            batch_size = y.size(0)
            loss_sum += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += batch_size
            for c in range(class_count):
                tp[c] += ((pred == c) & (y == c)).sum().item()
                fp[c] += ((pred == c) & (y != c)).sum().item()
                fn[c] += ((pred != c) & (y == c)).sum().item()

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = (tp + fn) > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    return loss_sum / max(total, 1), correct / max(total, 1), macro_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing *_preprocessed.npz files")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="'auto' uses CUDA if available, otherwise CPU")
    parser.add_argument("--csi-device", default="M1", help="Sensing device filter, e.g. M1")
    parser.add_argument("--test-subjects", default=None,
                        help="Comma-separated held-out test subject IDs, e.g. 5 or 5,6")
    parser.add_argument("--val-subjects", default=None,
                        help="Comma-separated validation subject IDs, e.g. 4")
    parser.add_argument("--activities", default=None,
                        help="Optional comma-separated activity IDs 0-19 (A=0,...,T=19)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/csi_cnn.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False. "
                               "Install a CUDA-enabled PyTorch build and check your NVIDIA driver.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Training device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
        print(f"CUDA/PyTorch: {torch.version.cuda} / {torch.__version__}", flush=True)

    test_subjects = parse_int_list(args.test_subjects)
    val_subjects = parse_int_list(args.val_subjects)
    activities = parse_int_list(args.activities)

    print(f"Scanning preprocessed files under: {args.data_dir}", flush=True)
    all_data = CSIDataset(args.data_dir, device=args.csi_device, activities=activities)
    print(f"Found {len(all_data.samples)} matching recording files across "
          f"{len({s['subject'] for s in all_data.samples})} subjects.", flush=True)
    all_subjects = sorted({s["subject"] for s in all_data.samples})
    if len(all_subjects) < 3 and (test_subjects is None or val_subjects is None):
        raise ValueError(
            f"Found only {len(all_subjects)} subjects {all_subjects}. "
            "Need at least 3 subjects for train/validation/test; provide valid "
            "--test-subjects and --val-subjects or collect more subjects."
        )

    if test_subjects is None:
        test_subjects = [all_subjects[-1]]
    if val_subjects is None:
        val_subjects = [s for s in all_subjects if s not in test_subjects][-1:]
    train_subjects = [s for s in all_subjects
                      if s not in set(test_subjects) | set(val_subjects)]

    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError(
            f"Invalid split. Available subjects={all_subjects}, train={train_subjects}, "
            f"val={val_subjects}, test={test_subjects}. Need non-empty sets."
        )
    if set(train_subjects) & set(val_subjects) or set(train_subjects) & set(test_subjects) or set(val_subjects) & set(test_subjects):
        raise ValueError("Train, validation, and test subject IDs must not overlap.")
    if not set(test_subjects + val_subjects).issubset(all_subjects):
        raise ValueError(f"Unknown held-out subject. Available subject IDs: {all_subjects}")

    train_ds = CSIDataset(args.data_dir, args.csi_device, train_subjects, activities)
    val_ds = CSIDataset(args.data_dir, args.csi_device, val_subjects, activities)
    test_ds = CSIDataset(args.data_dir, args.csi_device, test_subjects, activities)
    print(f"Subjects: train={train_subjects}, val={val_subjects}, test={test_subjects}", flush=True)
    print(f"Windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)
    print(f"Batches per epoch: train={int(np.ceil(len(train_ds) / args.batch_size))}, "
          f"val={int(np.ceil(len(val_ds) / args.batch_size))}, "
          f"test={int(np.ceil(len(test_ds) / args.batch_size))}", flush=True)
    if tqdm is None:
        print("Tip: install tqdm for live progress bars: pip install tqdm", flush=True)

    use_pin_memory = device.type == "cuda"
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, use_pin_memory)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, use_pin_memory)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, use_pin_memory)

    # Infer class count from selected activity IDs, but preserve original IDs in labels.
    # For full A-T classification this is 20 output classes. If --activities is used,
    # output stays 20 classes so labels remain their original activity IDs.
    model = SmallCSI2DCNN(num_classes=20, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting training loop...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch}/{args.epochs} [train]",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        ) if tqdm is not None else train_loader

        for batch_idx, (x, y) in enumerate(epoch_bar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_size_now = y.size(0)
            running_loss += loss.item() * batch_size_now
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch_size_now

            if tqdm is not None:
                epoch_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg_loss=f"{running_loss / max(total, 1):.4f}",
                    acc=f"{correct / max(total, 1):.3f}",
                )
            elif batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(f"  Epoch {epoch}/{args.epochs}: batch {batch_idx}/{len(train_loader)} "
                      f"({100 * batch_idx / len(train_loader):.1f}%), "
                      f"loss={loss.item():.4f}, avg_loss={running_loss / max(total, 1):.4f}, "
                      f"acc={correct / max(total, 1):.3f}", flush=True)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        print(f"Epoch {epoch}/{args.epochs}: training complete; running validation...", flush=True)
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} macro-F1 {val_macro_f1:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "model_name": "SmallCSI2DCNN",
                "num_classes": 20,
                "input_shape": None,
                "csi_device": args.csi_device,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "args": vars(args),
            }, output_path)
            print(f"  New best validation loss: {best_val_loss:.4f}; checkpoint saved to {output_path}", flush=True)

    print("Training complete; evaluating best checkpoint on test set...", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, test_macro_f1 = evaluate(model, test_loader, criterion, device)
    print(
        f"Test | loss {test_loss:.4f} acc {test_acc:.4f} macro-F1 {test_macro_f1:.4f}",
        flush=True,
    )
    print(f"Best checkpoint saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
