import os
import glob
import csv
import numpy as np

def validate_single_npz(npz_path: str, expected_subcarriers: int = 242):
    filename = os.path.basename(npz_path)
    report = {
        "filename": filename,
        "status": "PASS",
        "num_frames": 0,
        "num_subcarriers": 0,
        "nan_or_inf": False,
        "zero_frames": False,
        "mean_amplitude": 0.0,
        "max_amplitude": 0.0,
        "issues": []
    }

    try:
        data = np.load(npz_path)
        
        if 'csi' not in data:
            report["status"] = "FAIL"
            report["issues"].append("Missing 'csi' key")
            return report
            
        csi = data['csi']
        
        if csi.size == 0 or csi.ndim != 2:
            report["status"] = "FAIL"
            report["issues"].append(f"Invalid shape: {csi.shape}")
            return report
            
        frames, subcarriers = csi.shape
        report["num_frames"] = frames
        report["num_subcarriers"] = subcarriers
        
        # Check subcarrier count against 242
        if subcarriers != expected_subcarriers:
            report["status"] = "WARN"
            report["issues"].append(f"Subcarriers ({subcarriers}) != expected ({expected_subcarriers})")

        # Corruption check
        if np.isnan(csi).any() or np.isinf(csi).any():
            report["nan_or_inf"] = True
            report["status"] = "FAIL"
            report["issues"].append("Contains NaN or Inf values")

        amp = np.abs(csi)
        report["mean_amplitude"] = round(float(np.mean(amp)), 2)
        report["max_amplitude"] = round(float(np.max(amp)), 2)

        if report["mean_amplitude"] == 0:
            report["zero_frames"] = True
            report["status"] = "FAIL"
            report["issues"].append("Dead signal (all zero)")

    except Exception as e:
        report["status"] = "FAIL"
        report["issues"].append(f"File load error: {str(e)}")

    if not report["issues"]:
        report["issues"].append("None")

    return report


def batch_validate_dataset(dataset_dir: str, expected_subcarriers: int = 242):
    npz_files = glob.glob(os.path.join(dataset_dir, "**", "*_csi.npz"), recursive=True)
    
    if not npz_files:
        print(f"No converted '*_csi.npz' files found in '{dataset_dir}'")
        return

    print(f"Starting batch validation on {len(npz_files)} files...\n")
    
    passed = 0
    for idx, npz_path in enumerate(npz_files, 1):
        res = validate_single_npz(npz_path, expected_subcarriers=expected_subcarriers)
        status = res["status"]
        filename = res["filename"]
        
        if status == "PASS":
            passed += 1
            print(f"[{idx}/{len(npz_files)}] [PASS] {filename} | Frames: {res['num_frames']} | Subcarriers: {res['num_subcarriers']} | Mean Amp: {res['mean_amplitude']}")
        else:
            print(f"[{idx}/{len(npz_files)}] [{status}] {filename} | Issues: {'; '.join(res['issues'])}")

    print(f"\nValidation Complete: {passed}/{len(npz_files)} passed validation.")


if __name__ == "__main__":
    batch_validate_dataset("data/HAR-3/CSI_processed/M1", expected_subcarriers=242)