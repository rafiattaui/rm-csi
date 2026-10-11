import os
import time
import torch
from torch.utils.data import DataLoader
from load_dataset import CSIDatasetFast 

def test_csi_dataloader(preprocessed_dir: str):
    print("=" * 60)
    print("1. INITIALIZING DATASETS (Leave-One-Subject-Out)")
    print("=" * 60)
    
    train_ds = CSIDatasetFast(preprocessed_dir, target_subjects=['P1', 'P2'])
    test_ds  = CSIDatasetFast(preprocessed_dir, target_subjects=['P3'])

    print(f"✓ Train Dataset loaded: {len(train_ds)} total window samples")
    print(f"✓ Test Dataset loaded:  {len(test_ds)} total window samples")

    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty! Check preprocessed directory path.")

    # -----------------------------------------------------------------
    # 2. SINGLE ITEM TEST (__getitem__)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. SINGLE SAMPLE INTEGRITY CHECK")
    print("=" * 60)
    
    sample_x, sample_y = train_ds[0]
    
    print(f"✓ Feature Tensor Shape: {sample_x.shape}  (Expected: [2, 500, 242])")
    print(f"✓ Label Value:          {sample_y.item()} (Activity Class Integer)")
    print(f"✓ Feature Data Type:    {sample_x.dtype} (Expected: torch.float32)")
    print(f"✓ Label Data Type:      {sample_y.dtype} (Expected: torch.int64)")
    print(f"✓ Min / Max Value Range: {sample_x.min().item():.3f} / {sample_x.max().item():.3f}")

    assert sample_x.shape == torch.Size([2, 500, 242]), f"Incorrect tensor shape: {sample_x.shape}"
    assert not torch.isnan(sample_x).any(), "ERROR: Tensor contains NaN values!"
    assert not torch.isinf(sample_x).any(), "ERROR: Tensor contains Inf values!"
    print("✓ All single-sample assertions PASSED!")

    # -----------------------------------------------------------------
    # 3. DATALOADER BENCHMARK & BATCHING TEST
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. DATALOADER BENCHMARK & BATCHING TEST")
    print("=" * 60)
    
    BATCH_SIZE = 32
    
    # Use num_workers=0 on Windows for in-memory datasets to avoid Win32 IPC pickle pipe limits
    num_workers = 0 if os.name == 'nt' else 2
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    print(f"DataLoader workers:   {num_workers}")
    print(f"Pin Memory:           {use_pin_memory}")

    start_time = time.time()
    num_batches = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if batch_idx == 0:
            print(f"✓ Batch 0 Feature Shape: {images.shape}")
            print(f"✓ Batch 0 Label Shape:   {labels.shape}")
            print(f"✓ Unique Labels in Batch 0: {torch.unique(labels).tolist()}")

        num_batches += 1
        total_samples += len(images)

    elapsed_time = time.time() - start_time
    throughput = total_samples / elapsed_time

    print(f"\n✓ Iterated through {num_batches} batches ({total_samples} samples) in {elapsed_time:.2f}s")
    print(f"✓ Throughput: {throughput:.1f} samples/sec")
    print("=" * 60)
    print("✓ ALL TESTS PASSED SUCCESSFULLY! Ready for PyTorch model training.")
    print("=" * 60)

if __name__ == "__main__":
    PREPROCESSED_DIRECTORY = "data/HAR-3/CSI_preprocessed/M1"
    test_csi_dataloader(PREPROCESSED_DIRECTORY)