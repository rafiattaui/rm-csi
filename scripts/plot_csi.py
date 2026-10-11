import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_csi_file(npz_path: str, max_frames_to_plot: int = 2000):
    """
    Plots CSI Amplitude Heatmap, Subcarrier Spectrum, Single-subcarrier time-series,
    and Phase profile for a converted .npz dataset.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    print(f"Loading CSI data from: {npz_path}")
    data = np.load(npz_path)
    csi = data['csi']  # Shape: (N_frames, N_subcarriers)
    
    total_frames, num_subcarriers = csi.shape
    print(f"Dataset shape: {total_frames} frames x {num_subcarriers} subcarriers")

    # Limit frames for clear visualization if dataset is huge (e.g. 100,000 frames)
    plot_frames = min(total_frames, max_frames_to_plot)
    csi_subset = csi[:plot_frames, :]

    # Calculate magnitude and phase
    amplitude = np.abs(csi_subset)
    phase = np.angle(csi_subset)

    # Setup 2x2 Plot Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"CSI Analysis: {os.path.basename(npz_path)}\n(Showing first {plot_frames} of {total_frames} frames)", fontsize=14, fontweight='bold')

    # -----------------------------------------------------------------
    # Plot 1: Amplitude Heatmap over Time
    # -----------------------------------------------------------------
    im1 = axes[0, 0].imshow(
        amplitude.T, 
        aspect='auto', 
        cmap='viridis', 
        origin='lower',
        extent=[0, plot_frames, 0, num_subcarriers]
    )
    axes[0, 0].set_title("CSI Amplitude Heatmap (Subcarriers vs Time)")
    axes[0, 0].set_xlabel("Frame Index")
    axes[0, 0].set_ylabel("Subcarrier Index")
    fig.colorbar(im1, ax=axes[0, 0], label="Amplitude")

    # -----------------------------------------------------------------
    # Plot 2: Subcarrier Power Spectrum Snapshots
    # -----------------------------------------------------------------
    sample_indices = [0, plot_frames // 4, plot_frames // 2, (3 * plot_frames) // 4]
    for frame_idx in sample_indices:
        axes[0, 1].plot(amplitude[frame_idx, :], label=f"Frame {frame_idx}", alpha=0.85, linewidth=1.2)
    
    axes[0, 1].set_title("Subcarrier Power Spectrum Snapshots")
    axes[0, 1].set_xlabel("Subcarrier Index")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper right")

    # -----------------------------------------------------------------
    # Plot 3: Time Series of Selected Subcarriers
    # -----------------------------------------------------------------
    subcarrier_targets = [num_subcarriers // 4, num_subcarriers // 2, (3 * num_subcarriers) // 4]
    for sc in subcarrier_targets:
        axes[1, 0].plot(amplitude[:, sc], label=f"Subcarrier {sc}", alpha=0.85, linewidth=1.0)
    
    axes[1, 0].set_title("CSI Amplitude Time-Series (Specific Subcarriers)")
    axes[1, 0].set_xlabel("Frame Index")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right")

    # -----------------------------------------------------------------
    # Plot 4: Unwrapped Phase Spectrum Across Subcarriers
    # -----------------------------------------------------------------
    # Phase unwrapping removes 2*pi phase jumps across adjacent subcarriers
    unwrapped_phase = np.unwrap(phase[0, :])
    axes[1, 1].plot(phase[0, :], label="Raw Phase (rad)", color="tab:gray", alpha=0.6, linestyle=":")
    axes[1, 1].plot(unwrapped_phase, label="Unwrapped Phase (rad)", color="tab:red", linewidth=1.5)
    
    axes[1, 1].set_title("CSI Phase Profile across Subcarriers (Frame 0)")
    axes[1, 1].set_xlabel("Subcarrier Index")
    axes[1, 1].set_ylabel("Phase (radians)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    
    # Save high-res image and display window
    out_img = os.path.splitext(npz_path)[0] + "_plot.png"
    plt.savefig(out_img, dpi=300)
    print(f"Plot saved to: {out_img}")
    plt.show()


if __name__ == "__main__":
    DATASET_DIR = "data/HAR-3/CSI_processed/M1"
    
    # Auto-find the first .npz file in the processed directory
    npz_files = glob.glob(os.path.join(DATASET_DIR, "**", "*_csi.npz"), recursive=True)
    
    if not npz_files:
        print(f"No converted *_csi.npz files found in {DATASET_DIR}")
    else:
        # Plot the first file found (or pass a specific filepath)
        target_file = npz_files[9]
        plot_csi_file(target_file, max_frames_to_plot=2000)