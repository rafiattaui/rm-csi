import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# Import dataset loader and model
from load_dataset import CSIDatasetFast
from cnn import DualChannelCSI_CRNN

# =====================================================================
# Configuration
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "HAR-3" / "CSI_preprocessed"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Leave-One-Subject-Out (LOSO) Subject Allocation
TRAIN_SUBJECTS = ['P1', 'P2']  # Used for Training and Validation
TEST_SUBJECTS  = ['P3']        # Held-out Test Set

ACTIVITY_MAP = [chr(65 + i) for i in range(20)]  # ['A', 'B', ..., 'T']

# =====================================================================
# Helper Functions
# =====================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    eval_loss = running_loss / total
    eval_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return eval_loss, eval_acc, np.array(all_preds), np.array(all_labels)


# =====================================================================
# Main Pipeline
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}\n")

    # 1. Load Datasets
    print("Loading Preprocessed Datasets...")
    full_train_ds = CSIDatasetFast(PREPROCESSED_DIR, target_subjects=TRAIN_SUBJECTS)
    test_ds       = CSIDatasetFast(PREPROCESSED_DIR, target_subjects=TEST_SUBJECTS)

    # 2. Split Training Data into 80% Train / 20% Validation
    val_size = int(0.2 * len(full_train_ds))
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(
        full_train_ds, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset Split | Train: {len(train_ds)} | Val: {len(val_ds)} | Test (P3): {len(test_ds)}")

    # 3. Create DataLoaders
    # Keep num_workers=0 on Windows to avoid IPC pipe limit crashes
    num_workers = 0 if os.name == 'nt' else 2
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # 4. Instantiate Model, Optimizer, Loss, and LR Scheduler
    model = DualChannelCSI_CRNN(num_classes=20).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0.0
    best_model_path = "best_csi_model.pt"

    # 5. Training Loop
    print("\nStarting Model Training...")
    print("-" * 70)
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Time':^8}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - start_time

        print(f"{epoch:^6d} | {train_loss:^10.4f} | {train_acc*100:^9.2f}% | {val_loss:^10.4f} | {val_acc*100:^9.2f}% | {elapsed:^7.1f}s")

        # Save Best Model Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print("-" * 70)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%\n")

    # -----------------------------------------------------------------
    # 6. Evaluation on Unseen Test Subject (P3)
    # -----------------------------------------------------------------
    print("=" * 70)
    print("RUNNING EVALUATION ON UNSEEN TEST SUBJECT (P3)")
    print("=" * 70)

    # Load best weights
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    # Printed Classification Report (Precision, Recall, F1-Score per Activity)
    print("Detailed Classification Report per Activity Class:")
    print(classification_report(test_targets, test_preds, target_names=ACTIVITY_MAP, digits=4))


if __name__ == "__main__":
    main()