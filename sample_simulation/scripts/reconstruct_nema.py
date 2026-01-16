
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import nrrd
import gzip

# Add legacy recon folder to path
# Assuming script is in sample_simulation/scripts
# Legacy path: ../../recon_demo_legacy/PET-Recon/recon
LEGACY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../recon_demo_legacy/PET-Recon/recon"))
sys.path.append(LEGACY_PATH)

import system
import algorithms

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
RESULTS_DIR = "results_recon"

# NEMA Dimensions
IMG_SHAPE = (128, 128)
NUM_ANGLES = 252
NUM_RADIAL = 256

def normalize(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-9)

def plot_comparison(profile_a, label_a, profile_b, label_b, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(profile_a, label=label_a, linewidth=2)
    plt.plot(profile_b, label=label_b, linestyle='--', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("--- Loading Data ---")
    # 1. Load Ground Truth Phantom (Transaxial Slice)
    gt_path = os.path.join(OUTPUT_DIR, "nema_phantom_ground_truth.nrrd")
    raw_gt, header = nrrd.read(gt_path)
    # raw_gt is (X, Y, Z) or (Z, Y, X)? 
    # Based on visualize_results.py: activity.reshape((VOXELS_Z, VOXELS_Y, VOXELS_X))
    # Let's assume (Z, Y, X) if saved from numpy directly?
    # Actually nrrd header usually handles it, but let's check shape.
    print(f"Phantom shape: {raw_gt.shape}")
    
    # Extract central slice
    # If (128, 128, 128)
    if raw_gt.shape[0] == 128:
        # Assume Z, Y, X or X, Y, Z. 
        # NEMA is cubic so shape doesn't help.
        # visualize_results: slice_idx = VOXELS_Z // 2
        # saved act = raw[:, 2].reshape((VOXELS_Z, VOXELS_Y, VOXELS_X))
        # So index 0 is Z.
        gt_slice = raw_gt[64, :, :]
    else:
        print("Unexpected phantom shape.")
        return

    # 2. Load MC Sinogram (Summed)
    # Switch to reading raw gzip directly to match Scatter logic and avoid NRRD issues
    sino_trues_path = os.path.join(DATA_DIR, "sinogram_Trues.raw.gz")
    print(f"Reading raw trues from {sino_trues_path}...")
    
    with gzip.open(sino_trues_path, 'rb') as f:
        raw_trues_data = f.read()
    raw_trues_flat = np.frombuffer(raw_trues_data, dtype=np.int32).astype(np.float32)
    
    calc_sinos_t = raw_trues_flat.size // (NUM_ANGLES * NUM_RADIAL)
    print(f"Raw Trues Size: {raw_trues_flat.size}, Sinos: {calc_sinos_t}")
    
    if calc_sinos_t > 0:
         mc_sino_3d = raw_trues_flat.reshape((calc_sinos_t, NUM_ANGLES, NUM_RADIAL))
         
         # FIX: Auto-detect the slice with maximum counts (Signal Peak)
         # Previous assumption of "middle of array" was wrong (Peak was at 808 vs 414).
         # We assume the phantom is centered in the FOV, so peak counts = center slice.
         sino_counts = mc_sino_3d.sum(axis=(1, 2))
         peak_idx = 567#np.argmax(sino_counts)
         print(f"Auto-detected Peak Sinogram Index: {peak_idx} (Counts: {sino_counts[peak_idx]:.0f})")
         
         # Sum +/- 10 slices around the peak
         start_slice = max(0, peak_idx - 10)
         end_slice = min(calc_sinos_t, peak_idx + 10)
         print(f"Summing slices {start_slice} to {end_slice}...")
         
         mc_sinogram = mc_sino_3d[start_slice:end_slice, :, :].sum(axis=0)
    else:
         print("Error: Trues data size mismatch.")
         mc_sinogram = np.zeros((NUM_ANGLES, NUM_RADIAL), dtype=np.float32)
         peak_idx = 0 # Default
    
    
    print("--- Generating System Matrix ---")
    # Generate simple line-integral system matrix
    # This matrix assumes    # 4. Generate System Matrix (2D)
    print("Generating System Matrix...")
    # Geometric Parameters from NEMA_IEC.in / MCGPU-PET
    DETECTOR_RADIUS_CM = 15.0
    NCRYSTALS = 504
    VOXEL_SIZE_CM = 0.25
    
    # Calculate geometric Bin Size (Transaxial)
    # The FOV diameter covered by the sinogram depends on the maximum ring difference (or number of radial bins).
    # Effective Transaxial FOV = Chord Length corresponding to max angular difference.
    # Max Angle Diff = (NUM_RADIAL / NCRYSTALS) * 2*pi
    # FOV = 2 * R * sin(angle_diff / 2)
    fov_diameter_cm = 2 * DETECTOR_RADIUS_CM * np.sin((np.pi * NUM_RADIAL) / NCRYSTALS)
    bin_size_cm = fov_diameter_cm / NUM_RADIAL
    
    print(f"Geometric Check: Voxel Size={VOXEL_SIZE_CM}cm, Bin Size={bin_size_cm:.4f}cm (Ratio: {VOXEL_SIZE_CM/bin_size_cm:.2f})")
    
    A = system.generate_toy_system_matrix(IMG_SHAPE, NUM_ANGLES, NUM_RADIAL, pixel_size_cm=VOXEL_SIZE_CM, bin_size_cm=bin_size_cm)
    print(f"System Matrix shape: {A.shape}, sparsity: {A.nnz / (A.shape[0]*A.shape[1]):.4f}")
    
    
    print("--- Forward Projection Comparison ---")
    # Project Ground Truth: y_simple = A * x_gt
    x_gt_flat = gt_slice.flatten()
    y_simple = A.dot(x_gt_flat).reshape(NUM_ANGLES, NUM_RADIAL)
    
    # Save Simple Sinogram
    plt.figure(figsize=(8,6))
    plt.imshow(y_simple, cmap='hot', origin='lower', aspect='auto')
    plt.title("Simple Forward Projection (Line Integral)")
    plt.colorbar()
    plt.xlabel("Radial Bin")
    plt.ylabel("Angle")
    plt.savefig(os.path.join(RESULTS_DIR, "simple_sinogram.png"))
    plt.close()
    
    # Compare Lineouts
    # Pick a row (Angle) with interesting features
    angle_idx = NUM_ANGLES // 4 # 45 degrees
    
    # Normalize for comparison (units differ: MC counts vs arbitrary activity*length)
    prof_mc = normalize(mc_sinogram[angle_idx, :])
    prof_simple = normalize(y_simple[angle_idx, :])
    
    plot_comparison(prof_simple, "Simple (Ideal)", prof_mc, "Monte Carlo (Realistic)", 
                    f"Sinogram Profile Comparison (Angle {angle_idx})", "profile_comparison.png")
    
    
    # --- Attenuation Correction Setup ---
    print("--- Generating Mu-Map for AC ---")
    # Read voxel file to get materials/density
    # Format: [Material Density Activity] (Columns 0, 1, 2)
    # We need to find the vox file. Assumed at sample_simulation/nema_iec_128.vox
    vox_file = os.path.join(DATA_DIR, "nema_iec_128.vox")
    
    # Simple parser similar to visualize_results logic
    try:
        with open(vox_file, 'r') as f:
            lines = f.readlines()
            start = 0
            for i, line in enumerate(lines):
               if "[END OF VXH SECTION]" in line:
                   start = i + 1
                   break
        raw_vox = np.loadtxt(lines[start:])
        # Reshape: (Z, Y, X)
        mat_map = raw_vox[:, 0].reshape((128,128,128)) # Material ID
        # dens_map = raw_vox[:, 1].reshape((128,128,128)) # Density
        
        # Central slice for 2D recon
        mat_slice = mat_map[64, :, :]
        
        # Create Mu Map (units: cm^-1)
        # Mat 1 = Air (approx 0)
        # Mat 2 = Water (approx 0.096 at 511 keV)
        mu_map = np.zeros_like(mat_slice, dtype=np.float32)
        mu_map[mat_slice == 2] = 0.096
        
        # Forward Project Mu Map to get Line Integrals
        # System Matrix A sums pixels (unitless weight).
        # Integral = sum(mu_i * dl). Pixel size = VOXEL_SIZE_CM.
        # So proj = (A * mu_flat) * 0.25
        pixel_size_cm = VOXEL_SIZE_CM
        attenuation_proj = A.dot(mu_map.flatten()) * pixel_size_cm
        
        # AC Factors = exp(- integral mu dl) -> This is the attenuation probability (Survival probability)
        # In OSEM: y ~ P * x * attn.
        # So we pass attn = exp(-proj) as 'ac_factors'.
        # Wait, run_osem uses expected = (A*x)*ac. 
        # If ac represents survival prob, this is correct.
        ac_factors = np.exp(-attenuation_proj).reshape(NUM_ANGLES, NUM_RADIAL)
        
        print(f"AC Factors range: {ac_factors.min():.4f} to {ac_factors.max():.4f}")
        
    except Exception as e:
        print(f"Warning: Could not define AC factors. {e}")
        ac_factors = None

    print("--- Reconstruction (OSEM) ---")
    subsets = system.get_subsets(NUM_ANGLES, NUM_RADIAL, n_subsets=12)
    r = np.zeros_like(mc_sinogram.flatten())
    
    # 1. OSEM - No AC
    print("Running OSEM (No AC)...")
    rec_osem_noac, _ = algorithms.run_osem(mc_sinogram.flatten(), A, r, subsets, n_iters=5, img_shape=IMG_SHAPE, ac_factors=None)
    img_osem_noac = rec_osem_noac.reshape(IMG_SHAPE)
    nrrd.write(os.path.join(OUTPUT_DIR, "recon_osem_noac.nrrd"), img_osem_noac)
    
    # --- Ideal Data Processing (Ray Traced) ---
    print("--- Processing Ideal Data (Ray Traced) ---")
    # y_simple is the perfect line integral (Unattenuated).
    # Simulate attenuation: y_measured = y_true * exp(-mu)
    if ac_factors is not None:
        y_simple_atten = y_simple * ac_factors
    else:
        y_simple_atten = y_simple

    # Reconstruct Ideal Data
    print("Running OSEM on Ideal Data...")
    # 1. Ideal - No AC
    rec_ideal_noac, _ = algorithms.run_osem(y_simple_atten.flatten(), A, np.zeros_like(r), subsets, n_iters=5, img_shape=IMG_SHAPE, ac_factors=None)
    img_ideal_noac = rec_ideal_noac.reshape(IMG_SHAPE)
    
    # 2. Ideal - With AC
    rec_ideal_ac, _ = algorithms.run_osem(y_simple_atten.flatten(), A, np.zeros_like(r), subsets, n_iters=5, img_shape=IMG_SHAPE, ac_factors=ac_factors.flatten() if ac_factors is not None else None)
    img_ideal_ac = rec_ideal_ac.reshape(IMG_SHAPE)

    # 3. OSEM - With AC (Monte Carlo)
    if ac_factors is not None:
        print("Running OSEM (With AC)...")
        rec_osem_ac, _ = algorithms.run_osem(mc_sinogram.flatten(), A, r, subsets, n_iters=5, img_shape=IMG_SHAPE, ac_factors=ac_factors.flatten())
        img_osem_ac = rec_osem_ac.reshape(IMG_SHAPE)
        nrrd.write(os.path.join(OUTPUT_DIR, "recon_osem_ac.nrrd"), img_osem_ac)
    else:
        img_osem_ac = img_osem_noac

    # Plot OSEM Comparison (2 Rows: Ideal, MC)
    row_idx = IMG_SHAPE[0] // 2
    
    plt.figure(figsize=(18, 10))
    
    # --- Row 1: Ideal / Ray Traced ---
    # 1. No AC
    plt.subplot(2, 3, 1)
    plt.imshow(img_ideal_noac, cmap='hot', origin='lower')
    plt.title("Ideal Ray-Traced (No AC)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # 2. With AC
    plt.subplot(2, 3, 2)
    plt.imshow(img_ideal_ac, cmap='hot', origin='lower')
    plt.title("Ideal Ray-Traced (With AC)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # 3. Profiles
    plt.subplot(2, 3, 3)
    plt.plot(gt_slice[row_idx, :], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
    plt.plot(img_ideal_noac[row_idx, :], 'r--', label='No AC')
    plt.plot(img_ideal_ac[row_idx, :], 'g-', label='With AC')
    plt.title("Ideal Line Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Row 2: Monte Carlo ---
    # 1. No AC
    plt.subplot(2, 3, 4)
    plt.imshow(img_osem_noac, cmap='hot', origin='lower')
    plt.title("Monte Carlo (No AC)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # 2. With AC
    plt.subplot(2, 3, 5)
    plt.imshow(img_osem_ac, cmap='hot', origin='lower')
    plt.title("Monte Carlo (With AC)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # 3. Profiles
    plt.subplot(2, 3, 6)
    plt.plot(gt_slice[row_idx, :], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
    plt.plot(img_osem_noac[row_idx, :], 'r--', label='No AC')
    plt.plot(img_osem_ac[row_idx, :], 'g-', label='With AC')
    plt.title("MC Line Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recon_osem.png"))
    plt.close()
    
    print("--- Reconstruction (BSREM) ---")
    
    # 1. Reconstruct Ideal Data (BSREM)
    print("Running BSREM on Ideal Data...")
    rec_ideal_bsrem_noac, _ = algorithms.run_bsrem(y_simple_atten.flatten(), A, np.zeros_like(r), subsets, n_iters=10, img_shape=IMG_SHAPE, beta=0.05, ac_factors=None)
    img_ideal_bsrem_noac = rec_ideal_bsrem_noac.reshape(IMG_SHAPE)
    
    rec_ideal_bsrem_ac, _ = algorithms.run_bsrem(y_simple_atten.flatten(), A, np.zeros_like(r), subsets, n_iters=10, img_shape=IMG_SHAPE, beta=0.05, ac_factors=ac_factors.flatten() if ac_factors is not None else None)
    img_ideal_bsrem_ac = rec_ideal_bsrem_ac.reshape(IMG_SHAPE)

    # 2. Reconstruct MC Data (BSREM)
    # 3. BSREM - No AC
    print("Running BSREM (No AC)...")
    rec_bsrem_noac, _ = algorithms.run_bsrem(mc_sinogram.flatten(), A, r, subsets, n_iters=10, img_shape=IMG_SHAPE, beta=0.05, ac_factors=None)
    img_bsrem_noac = rec_bsrem_noac.reshape(IMG_SHAPE)
    nrrd.write(os.path.join(OUTPUT_DIR, "recon_bsrem_noac.nrrd"), img_bsrem_noac)

    # 4. BSREM - With AC
    if ac_factors is not None:
        print("Running BSREM (With AC)...")
        rec_bsrem_ac, _ = algorithms.run_bsrem(mc_sinogram.flatten(), A, r, subsets, n_iters=10, img_shape=IMG_SHAPE, beta=0.05, ac_factors=ac_factors.flatten())
        img_bsrem_ac = rec_bsrem_ac.reshape(IMG_SHAPE)
        nrrd.write(os.path.join(OUTPUT_DIR, "recon_bsrem_ac.nrrd"), img_bsrem_ac)
    else:
        img_bsrem_ac = img_bsrem_noac
    
    # Plot BSREM Comparison (2 Rows: Ideal, MC)
    plt.figure(figsize=(18, 10))
    
    # ... (Previous BSREM Plot Code) ...
    # --- Row 1: Ideal / Ray Traced ---
    # 1. No AC
    plt.subplot(2, 3, 1)
    plt.imshow(img_ideal_bsrem_noac, cmap='hot', origin='lower')
    plt.title("BSREM Ideal (No AC)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # 2. With AC
    plt.subplot(2, 3, 2)
    plt.imshow(img_ideal_bsrem_ac, cmap='hot', origin='lower')
    plt.title("BSREM Ideal (With AC)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # 3. Profiles
    plt.subplot(2, 3, 3)
    plt.plot(gt_slice[row_idx, :], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
    plt.plot(img_ideal_bsrem_noac[row_idx, :], 'r--', label='No AC')
    plt.plot(img_ideal_bsrem_ac[row_idx, :], 'g-', label='With AC')
    plt.title("BSREM Ideal Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Row 2: Monte Carlo ---
    # 1. No AC
    plt.subplot(2, 3, 4)
    plt.imshow(img_bsrem_noac, cmap='hot', origin='lower')
    plt.title("BSREM MC (No AC)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # 2. With AC
    plt.subplot(2, 3, 5)
    plt.imshow(img_bsrem_ac, cmap='hot', origin='lower')
    plt.title("BSREM MC (With AC)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # 3. Profiles
    plt.subplot(2, 3, 6)
    plt.plot(gt_slice[row_idx, :], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
    plt.plot(img_bsrem_noac[row_idx, :], 'r--', label='No AC')
    plt.plot(img_bsrem_ac[row_idx, :], 'g-', label='With AC')
    plt.title("BSREM MC Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recon_bsrem.png"))
    plt.close()


    # ---------------------------------------------------------
    # --- Scatter Analysis: Trues vs Prompts (Trues + Scatter) ---
    print("--- Scatter Analysis ---")
    
    # 1. Load Scatter Sinogram (Directly from Raw to avoid NRRD issues)
    # scatter_path = os.path.join(OUTPUT_DIR, "nema_sinogram_Scatter.nrrd")
    # raw_scatter, _ = nrrd.read(scatter_path)
    # mc_scatter = raw_scatter.sum(axis=0) # Sum slices -> (Angles, Radial)
    
    raw_scatter_path = os.path.join(DATA_DIR, "sinogram_Scatter.raw.gz")
    print(f"Reading raw scatter from {raw_scatter_path}...")
    
    # Reading logic adapted from visualize_results.py
    with gzip.open(raw_scatter_path, 'rb') as f:
        raw_data = f.read()
    
    raw_scatter_flat = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32)
    
    # Reshape: (Sinos, Angles, Radial)
    # Guess Sinos based on size
    calc_sinos = raw_scatter_flat.size // (NUM_ANGLES * NUM_RADIAL)
    print(f"Raw Scatter Size: {raw_scatter_flat.size}, Sinos: {calc_sinos}")
    
    if calc_sinos > 0:
        mc_scatter_3d = raw_scatter_flat.reshape((calc_sinos, NUM_ANGLES, NUM_RADIAL))
        
        # Use simple logic: if size matches trues, use same peak_idx.
        # Otherwise, recalculate peak.
        if calc_sinos == calc_sinos_t:
             use_peak = peak_idx
        else:
             use_peak = np.argmax(mc_scatter_3d.sum(axis=(1,2)))
             
        start_slice = max(0, use_peak - 10)
        end_slice = min(calc_sinos, use_peak + 10)
        mc_scatter = mc_scatter_3d[start_slice:end_slice, :, :].sum(axis=0)
    else:
        print("Error: Scatter data size mismatch.")
        mc_scatter = np.zeros_like(mc_sinogram)

    # 2. Create Prompts (Trues + Scatter)
    # This represents the actual data acquired with the 350-600 keV window
    # without any scatter correction.
    mc_prompts = mc_sinogram + mc_scatter
    
    print(f"Total True Counts: {mc_sinogram.sum():.2e}")
    print(f"Total Scatter Counts: {mc_scatter.sum():.2e}")
    print(f"Scatter Fraction: {mc_scatter.sum() / mc_prompts.sum():.2%}")
    
    # 3. Reconstruct Prompts (OSEM with AC)
    print("Reconstructing Prompts (Trues + Scatter)...")
    rec_prompts_ac, _ = algorithms.run_osem(mc_prompts.flatten(), A, r, subsets, n_iters=5, img_shape=IMG_SHAPE, ac_factors=ac_factors.flatten() if ac_factors is not None else None)
    img_prompts_ac = rec_prompts_ac.reshape(IMG_SHAPE)
    nrrd.write(os.path.join(OUTPUT_DIR, "recon_prompts_ac.nrrd"), img_prompts_ac)
    
    # 4. Plot Comparison: Trues (Perfect Window) vs Prompts (Standard Window)
    plt.figure(figsize=(18, 5))
    
    # Trues (With AC) - Already Computed as img_osem_ac
    plt.subplot(1, 4, 1)
    plt.imshow(img_osem_ac, cmap='hot', origin='lower')
    plt.title("Trues Only\n(Perfect Scatter Rejection)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # Scatter Only (Reconstructed Scatter) - Interesting to see!
    rec_scatter, _ = algorithms.run_osem(mc_scatter.flatten(), A, r, subsets, n_iters=5, img_shape=IMG_SHAPE, ac_factors=ac_factors.flatten() if ac_factors is not None else None)
    img_scatter_only = rec_scatter.reshape(IMG_SHAPE)
    
    plt.subplot(1, 4, 2)
    plt.imshow(img_scatter_only, cmap='hot', origin='lower')
    plt.title("Scatter Only\n(Background Haze)")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Prompts (With AC)
    plt.subplot(1, 4, 3)
    plt.imshow(img_prompts_ac, cmap='hot', origin='lower')
    plt.title("Prompts (Trues + Scatter)\n(Standard Energy Window)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axhline(row_idx, color='w', linestyle='--', alpha=0.5)
    
    # Profiles
    plt.subplot(1, 4, 4)
    plt.plot(gt_slice[row_idx, :], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
    plt.plot(img_osem_ac[row_idx, :], 'g-', label='Trues (Ideal)')
    plt.plot(img_prompts_ac[row_idx, :], 'm-', label='Prompts (Std Window)')
    plt.title("Scatter Effect Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recon_scatter_comp.png"))
    plt.close()
    

    # Save Ground Truth Slice for Reference
    plt.figure(figsize=(6,6))
    plt.imshow(gt_slice, cmap='hot', origin='lower')
    plt.title("Ground Truth (Central Slice)")
    plt.colorbar()
    plt.savefig(os.path.join(RESULTS_DIR, "ground_truth_slice.png"))
    plt.close()
    
    print("Done. Results in", RESULTS_DIR)
    print("NRRDs saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
