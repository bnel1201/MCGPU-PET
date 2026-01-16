
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import gzip
import nrrd

# Add legacy recon folder to path
LEGACY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../recon_demo_legacy/PET-Recon/recon"))
sys.path.append(LEGACY_PATH)

import algorithms
import system

DATA_DIR = "/home/sarah/Dev/MCGPU-PET/sample_simulation"
RESULTS_DIR = os.path.join(DATA_DIR, "scripts/results_recon_vol")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Constants
NUM_ANGLES = 252
NUM_RADIAL = 256
IMG_SHAPE = (128, 128)
VOXEL_SIZE_CM = 0.25 # 2.5 mm

def main():
    print("--- Volume Reconstruction (OSEM, 5mm Slice Thickness) ---")
    
    # 1. Load Sinogram Data
    trues_path = os.path.join(DATA_DIR, "sinogram_Trues.raw.gz")
    print(f"Reading {trues_path}...")
    
    with gzip.open(trues_path, 'rb') as f:
        data = f.read()
    raw = np.frombuffer(data, dtype=np.int32).astype(np.float32)
    
    n_sinos = raw.size // (NUM_ANGLES * NUM_RADIAL)
    raw_3d = raw.reshape((n_sinos, NUM_ANGLES, NUM_RADIAL))
    
    # 2. Identify Direct Segment
    # We found the peak at index ~808. In standard Michelograms, direct planes are often contiguous.
    # We will trust that the peak represents the center of the axial FOV.
    # The Phantom is 128 slices (32cm).
    # We want to reconstruct ~25cm of FOV? NEMA input says AXIAL FOV = 25.0 cm.
    # 25cm / 0.5cm (5mm) = 50 slices.
    # 25cm in native bins (2.5mm) = 100 bins.
    # So we need a range of 100 sinograms centered on the peak.
    
    sino_counts = raw_3d.sum(axis=(1, 2))
    peak_idx = np.argmax(sino_counts)
    print(f"Peak Counts Index: {peak_idx}")
    
    # Define Range (Expanded to cover whole volume as requested)
    # User asked for "whole volume".
    # Phantom is 32cm approx.
    # 200 * 2.5mm = 50cm. This should cover everything available.
    half_width = 200
    start_idx = max(0, peak_idx - half_width)
    end_idx = min(n_sinos, peak_idx + half_width)
    
    print(f"Reconstruction Range: {start_idx} to {end_idx} (Native Slices)")
    
    # 3. Binning (2.5mm -> 5mm)
    # Combine every 2 slices.
    # If odd number of slices, drop last or handle.
    
    native_slices = raw_3d[start_idx:end_idx, :, :]
    n_native = native_slices.shape[0]
    
    # Ensure even number for 2:1 binning
    if n_native % 2 != 0:
        n_native -= 1
        native_slices = native_slices[:n_native]
        
    n_output = n_native // 2
    print(f"Binning: {n_native} native slices -> {n_output} output slices (5mm thickness)")
    
    binned_sinos = np.zeros((n_output, NUM_ANGLES, NUM_RADIAL), dtype=np.float32)
    for i in range(n_output):
        # Sum pairs
        binned_sinos[i] = native_slices[2*i] + native_slices[2*i+1]
        
    # 4. Generate System Matrix (2D)
    print("Generating System Matrix...")
    # Geometric Parameters from NEMA_IEC.in / MCGPU-PET
    DETECTOR_RADIUS_CM = 15.0
    NCRYSTALS = 504
    
    # Calculate geometric Bin Size (Transaxial)
    fov_diameter_cm = 2 * DETECTOR_RADIUS_CM * np.sin((np.pi * NUM_RADIAL) / NCRYSTALS)
    bin_size_cm = fov_diameter_cm / NUM_RADIAL
    print(f"Geometric Calibration: Voxel={VOXEL_SIZE_CM}cm, Bin={bin_size_cm:.4f}cm (Ratio: {VOXEL_SIZE_CM/bin_size_cm:.2f})")
    
    A = system.generate_toy_system_matrix(IMG_SHAPE, NUM_ANGLES, NUM_RADIAL, pixel_size_cm=VOXEL_SIZE_CM, bin_size_cm=bin_size_cm)
    
    # 5. Generate AC Factors (Mu-Map)
    print("Generating AC Factors (Mu-Map)...")
    # Read Voxel File (NEMA phantom)
    vox_file = os.path.join(DATA_DIR, "nema_iec_128.vox")
    
    # We assume standard NEMA logic: Material 2 = Water, Mat 1 = Air.
    # NEMA phantom is 128x128x128.
    # We need the Transaxial Mu-Map (approximated as constant for the volume or just 2D)
    # Since the phantom is a cylinder, the 2D cross-section is valid for most Z.
    
    try:
        with open(vox_file, 'r') as f:
             lines = f.readlines()
             # Skip header (approx 12 lines, look for SECTION)
             start = 0
             for i, line in enumerate(lines):
                  if "[END OF VXH SECTION]" in line:
                       start = i + 1
                       break
             raw_vox = np.loadtxt(lines[start:])
             
        # Reshape: (Z, Y, X)
        mat_map = raw_vox[:, 0].reshape((128,128,128))
        
        # Take central slice (Slice 64) as representative Mu-Map
        mat_slice = mat_map[64, :, :]
        
        mu_map = np.zeros_like(mat_slice, dtype=np.float32)
        mu_map[mat_slice == 2] = 0.096 # cm^-1 for Water
        # Air is 0
        
        # Forward Project
        pixel_size_cm = VOXEL_SIZE_CM
        attenuation_proj = A.dot(mu_map.flatten()) * pixel_size_cm
        ac_factors_2d = np.exp(-attenuation_proj).reshape(NUM_ANGLES, NUM_RADIAL)
        print(f"AC Factors generated. Range: {ac_factors_2d.min():.4f} to {ac_factors_2d.max():.4f}")
        
    except Exception as e:
        print(f"Error generating Mu-Map: {e}. Proceeding without AC.")
        ac_factors_2d = None

    # 6. OSEM Reconstruction Loop
    subsets = system.get_subsets(NUM_ANGLES, NUM_RADIAL, n_subsets=12)
    volume = np.zeros((n_output, IMG_SHAPE[0], IMG_SHAPE[1]), dtype=np.float32)
    
    print("Running OSEM on Volume...")
    for z in range(n_output):
        sino = binned_sinos[z].flatten()
        
        # Prepare AC factors (flat)
        ac_flat = ac_factors_2d.flatten() if ac_factors_2d is not None else None
        
        # Initialize
        rec, _ = algorithms.run_osem(sino, A, np.zeros_like(sino), subsets, n_iters=4, img_shape=IMG_SHAPE, ac_factors=ac_flat)
        volume[z, :, :] = rec.reshape(IMG_SHAPE)
        
        if z % 5 == 0:
            print(f"  Processed Slice {z+1}/{n_output}")
            
    print("Reconstruction Complete.")
    
    # 6. Save as NRRD
    output_filename = os.path.join(OUTPUT_DIR, "recon_volume_osem_5mm.nrrd")
    print(f"Saving to {output_filename}...")
    nrrd.write(output_filename, volume)
    
    # 7. Generate MIP for quick check
    mip = np.max(volume, axis=0)
    plt.figure(figsize=(6,6))
    plt.imshow(mip, cmap='hot', origin='lower')
    plt.title("Recon Volume MIP (OSEM 5mm)")
    plt.colorbar()
    plt.savefig(os.path.join(RESULTS_DIR, "recon_vol_mip.png"))
    print("Saved MIP check.")

if __name__ == "__main__":
    main()
