
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gzip
import nrrd

DATA_DIR = "/home/sarah/Dev/MCGPU-PET/sample_simulation"
RESULTS_DIR = os.path.join(DATA_DIR, "scripts/results_energy")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Constants (Must match NEMA_IEC.in)
NUM_ANGLES = 252
NUM_RADIAL = 256
IMG_SHAPE = (128, 128)

def load_sinogram(path):
    print(f"Loading {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
        
    with gzip.open(path, 'rb') as f:
        data = f.read()
    
    # Read as int32, convert to float
    arr = np.frombuffer(data, dtype=np.int32).astype(np.float32)
    
    # Determine number of sinograms
    n_sinos = arr.size // (NUM_ANGLES * NUM_RADIAL)
    if n_sinos == 0:
        return None
        
    arr_3d = arr.reshape((n_sinos, NUM_ANGLES, NUM_RADIAL))
    
    # FIX: Use Peak Detection like reconstruct_nema.py
    sino_counts = arr_3d.sum(axis=(1, 2))
    peak_idx = np.argmax(sino_counts)
    print(f"  Peak Index: {peak_idx}, Counts: {sino_counts[peak_idx]:.0f}")
    
    start = max(0, peak_idx - 10)
    end = min(n_sinos, peak_idx + 10)
    arr_2d = arr_3d[start:end, :, :].sum(axis=0)
    return arr_2d

def main():
    # Paths
    std_dir = os.path.join(DATA_DIR, "output_std")
    narrow_dir = os.path.join(DATA_DIR, "output_narrow")
    
    print("--- Loading Standard Window Data (350-600 keV) ---")
    std_trues = load_sinogram(os.path.join(std_dir, "sinogram_Trues.raw.gz"))
    std_scatter = load_sinogram(os.path.join(std_dir, "sinogram_Scatter.raw.gz"))
    
    print("--- Loading Narrow Window Data (450-550 keV) ---")
    narrow_trues = load_sinogram(os.path.join(narrow_dir, "sinogram_Trues.raw.gz"))
    narrow_scatter = load_sinogram(os.path.join(narrow_dir, "sinogram_Scatter.raw.gz"))
    
    if any(x is None for x in [std_trues, std_scatter, narrow_trues, narrow_scatter]):
        print("Error: Missing data.")
        return

    # Calculate Scatter Fractions
    sf_std = std_scatter.sum() / (std_trues.sum() + std_scatter.sum())
    sf_narrow = narrow_scatter.sum() / (narrow_trues.sum() + narrow_scatter.sum())
    
    print(f"Scatter Fraction (Standard): {sf_std:.2%}")
    print(f"Scatter Fraction (Narrow):   {sf_narrow:.2%}")
    
    # Scaling check
    # Since we might have different acquisition times (if I failed to re-run Std at 10s), 
    # we should normalize by acquisition time?
    # Or just normalize to max?
    # User request: "effects of different energy window sizes"
    # Ideally, we show raw counts to show sensitivity loss.
    # Assuming both match in time (10s), we plot raw.
    
    # Profiles
    mid_row = NUM_ANGLES // 2 # ~90 degrees?
    
    # Plot Scatter Profile Comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(std_scatter[mid_row, :], 'r-', label=f'Standard (SF={sf_std:.1%})', linewidth=2)
    plt.plot(narrow_scatter[mid_row, :], 'b--', label=f'Narrow (SF={sf_narrow:.1%})', linewidth=2)
    plt.title("Scatter Comparison (Sinogram Profile)")
    plt.xlabel("Radial Bin")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Trues Profile Comparison (Sensitivity)
    plt.subplot(1, 2, 2)
    plt.plot(std_trues[mid_row, :], 'g-', label='Standard Trues', linewidth=2)
    plt.plot(narrow_trues[mid_row, :], 'm--', label='Narrow Trues', linewidth=2)
    plt.title("Sensitivity Comparison (Trues)")
    plt.xlabel("Radial Bin")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "energy_comparison.png"))
    print(f"Saved {os.path.join(RESULTS_DIR, 'energy_comparison.png')}")
    
    # Also save SF comparison as text or just print
    with open(os.path.join(RESULTS_DIR, "sf_stats.txt"), "w") as f:
        f.write(f"Standard (350-600): SF = {sf_std:.4f}\n")
        f.write(f"Narrow   (450-550): SF = {sf_narrow:.4f}\n")

if __name__ == "__main__":
    main()
