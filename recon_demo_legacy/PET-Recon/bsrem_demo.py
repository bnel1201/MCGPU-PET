# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.data import shepp_logan_phantom

from recon.system import generate_toy_system_matrix, get_subsets
from recon.algorithms import run_osem, run_bsrem
from recon.analysis import get_phantom_rois, compute_metrics, create_circular_mask

# ==========================================
# 1. System Setup
# ==========================================

# Parameters
img_size = 128
n_angles = 120
n_bins = 180
n_subsets = 4
n_iters = 30
beta_reg = 0.05

print(f"Generating System Matrix for {img_size}x{img_size} image...")
A = generate_toy_system_matrix((img_size, img_size), n_angles, n_bins)
subsets = get_subsets(n_angles, n_bins, n_subsets)
print("System Matrix Generated.")

# ==========================================
# 2. Data Generation (Phantom, Mu-Map, Sinogram)
# ==========================================

# Create Ground Truth Activity (Lambda)
phantom = resize(shepp_logan_phantom(), (img_size, img_size))
true_lambda = phantom.flatten() * 100
true_lambda[true_lambda < 0] = 0

# Create Attenuation Map (Mu-Map)
# Simulating water/tissue ~ 0.096 cm^-1, bone ~ 0.17 cm^-1
# For simplicity, we scale the phantom to be our mu-map
mu_map_img = phantom * 0.1 
mu_map_img[mu_map_img < 0] = 0
mu_map = mu_map_img.flatten()

# Calculate Attenuation Factors
proj_mu = A.dot(mu_map)
attn_factors = np.exp(-proj_mu)

# Forward Project with Attenuation: E[y] = (A * lambda) * attn + r
background_counts = 1.0 
r = np.full(A.shape[0], background_counts)
clean_sinogram = (A.dot(true_lambda) * attn_factors) + r

# Add Poisson Noise
np.random.seed(42)
y = np.random.poisson(clean_sinogram).astype(float)

# ==========================================
# 3. Run Reconstructions
# ==========================================

print("Running OSEM (AC ON)...")
recon_osem_ac, obj_osem = run_osem(y, A, r, subsets, n_iters, (img_size, img_size), ac_factors=attn_factors)

print("Running BSREM (AC ON)...")
recon_bsrem_ac, obj_bsrem = run_bsrem(y, A, r, subsets, n_iters, (img_size, img_size), beta_reg, ac_factors=attn_factors)

print("Running BSREM (AC OFF)...")
recon_bsrem_no_ac, _ = run_bsrem(y, A, r, subsets, n_iters, (img_size, img_size), beta_reg, ac_factors=None)

# ==========================================
# 4. Analysis & Visualization (Separated Figures)
# ==========================================

rois = get_phantom_rois(img_size)
mask_high, mask_low, mask_bg = rois
vmax = np.max(true_lambda)
mid_row = img_size // 2

def plot_rois(axis):
    for mask, color in zip([mask_high, mask_low, mask_bg], ['r', 'b', 'g']):
        axis.contour(mask, colors=color, levels=[0.5], linewidths=1)

def plot_attenuation_comparison():
    """Generates Figure 1: Attenuation Effects (AC On vs AC Off)"""
    
    # Calculate Metrics for this comparison
    metrics = [
        compute_metrics(true_lambda.reshape(img_size, img_size), "True", rois),
        compute_metrics(recon_bsrem_no_ac.reshape(img_size, img_size), "AC Off", rois),
        compute_metrics(recon_bsrem_ac.reshape(img_size, img_size), "AC On", rois)
    ]
    
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Images
    ax[0, 0].imshow(true_lambda.reshape(img_size, img_size), cmap='gray', vmax=vmax)
    plot_rois(ax[0, 0])
    ax[0, 0].set_title("True Phantom (w/ ROIs)")
    
    ax[0, 1].imshow(recon_bsrem_no_ac.reshape(img_size, img_size), cmap='gray', vmax=vmax)
    ax[0, 1].set_title("AC OFF (BSREM)\nCupping Artifact")
    
    ax[0, 2].imshow(recon_bsrem_ac.reshape(img_size, img_size), cmap='gray', vmax=vmax)
    ax[0, 2].set_title("AC ON (BSREM)\nCorrected")
    
    # Row 2: Profiles & Mu-Map
    ax[1, 0].imshow(mu_map_img, cmap='bone')
    ax[1, 0].set_title("Attenuation Map (Mu)")
    
    ax[1, 1].plot(true_lambda.reshape(img_size, img_size)[mid_row, :], 'k-', label='True')
    ax[1, 1].plot(recon_bsrem_no_ac.reshape(img_size, img_size)[mid_row, :], 'r--', alpha=0.7, label='AC OFF')
    ax[1, 1].plot(recon_bsrem_ac.reshape(img_size, img_size)[mid_row, :], 'b-', alpha=0.9, label='AC ON')
    ax[1, 1].set_title("Horizontal Line Profile")
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    
    ax[1, 2].imshow(attn_factors.reshape(n_angles, n_bins), cmap='gray', aspect='auto')
    ax[1, 2].set_title("Attenuation Factors")
    
    # Row 3: Metrics Bar Plots
    methods = [m['name'] for m in metrics]
    x = np.arange(len(methods))
    bar_width = 0.35
    
    # Contrast
    ax[2, 0].bar(x - bar_width/2, [m['contrast_high'] for m in metrics], bar_width, label='High', color='r', alpha=0.7)
    ax[2, 0].bar(x + bar_width/2, [m['contrast_low'] for m in metrics], bar_width, label='Low', color='b', alpha=0.7)
    ax[2, 0].set_xticks(x)
    ax[2, 0].set_xticklabels(methods)
    ax[2, 0].set_title("Contrast")
    ax[2, 0].legend()
    
    # CNR
    ax[2, 1].bar(x - bar_width/2, [m['cnr_high'] for m in metrics], bar_width, label='High', color='r', alpha=0.7)
    ax[2, 1].bar(x + bar_width/2, [m['cnr_low'] for m in metrics], bar_width, label='Low', color='b', alpha=0.7)
    ax[2, 1].set_xticks(x)
    ax[2, 1].set_xticklabels(methods)
    ax[2, 1].set_title("CNR")
    
    ax[2, 2].axis('off') # Empty plot for layout balance
    
    plt.tight_layout()
    print("Saving demo_attenuation_comparison.png...")
    plt.savefig("demo_attenuation_comparison.png")
    # plt.show() # Optional if running interactively

def plot_recon_comparison():
    """Generates Figure 2: Algorithm Comparison (OSEM vs BSREM, both AC On)"""
    
    metrics_all = [
        compute_metrics(true_lambda.reshape(img_size, img_size), "True", rois),
        compute_metrics(recon_osem_ac.reshape(img_size, img_size), "OSEM", rois),
        compute_metrics(recon_bsrem_ac.reshape(img_size, img_size), "BSREM", rois)
    ]
    
    # For Noise and CNR, we essentially only want to compare the reconstruction methods
    # against each other, as "True" has 0 noise and Infinite CNR.
    metrics_recons = [
        compute_metrics(recon_osem_ac.reshape(img_size, img_size), "OSEM", rois),
        compute_metrics(recon_bsrem_ac.reshape(img_size, img_size), "BSREM", rois)
    ]
    
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Images
    ax[0, 0].imshow(true_lambda.reshape(img_size, img_size), cmap='gray', vmax=vmax)
    plot_rois(ax[0, 0])
    ax[0, 0].set_title("True Phantom")
    
    ax[0, 1].imshow(recon_osem_ac.reshape(img_size, img_size), cmap='gray', vmax=vmax)
    ax[0, 1].set_title(f"OSEM ({n_iters} iters)")
    
    ax[0, 2].imshow(recon_bsrem_ac.reshape(img_size, img_size), cmap='gray', vmax=vmax)
    ax[0, 2].set_title(f"BSREM ({n_iters} iters)\nSmoother")
    
    # Row 2: Sinogram & Interface
    ax[1, 0].imshow(y.reshape(n_angles, n_bins), cmap='viridis', aspect='auto')
    ax[1, 0].set_title("Noisy Input Sinogram")
    
    ax[1, 1].plot(true_lambda.reshape(img_size, img_size)[mid_row, :], 'k-', label='True')
    ax[1, 1].plot(recon_osem_ac.reshape(img_size, img_size)[mid_row, :], 'r--', alpha=0.7, label='OSEM')
    ax[1, 1].plot(recon_bsrem_ac.reshape(img_size, img_size)[mid_row, :], 'b-', alpha=0.9, label='BSREM')
    ax[1, 1].set_title("Horizontal Line Profile")
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    
    ax[1, 2].plot(obj_osem, label='OSEM LL')
    ax[1, 2].plot(obj_bsrem, label='BSREM PL')
    ax[1, 2].set_title("Objective Convergence")
    ax[1, 2].legend()
    ax[1, 2].grid(True)
    
    # Row 3: Metrics
    
    # 3.0: Noise (Recons Only)
    methods = [m['name'] for m in metrics_recons]
    x = np.arange(len(methods))
    bar_width = 0.35
    
    ax[2, 0].bar(x - bar_width/2, [m['std_high'] for m in metrics_recons], bar_width, label='High', color='r', alpha=0.7)
    ax[2, 0].bar(x + bar_width/2, [m['std_low'] for m in metrics_recons], bar_width, label='Low', color='b', alpha=0.7)
    ax[2, 0].set_xticks(x)
    ax[2, 0].set_xticklabels(methods)
    ax[2, 0].set_title("Noise (Std Dev)")
    ax[2, 0].legend()
    
    # 3.1: CNR (Recons Only)
    ax[2, 1].bar(x - bar_width/2, [m['cnr_high'] for m in metrics_recons], bar_width, label='High', color='r', alpha=0.7)
    ax[2, 1].bar(x + bar_width/2, [m['cnr_low'] for m in metrics_recons], bar_width, label='Low', color='b', alpha=0.7)
    ax[2, 1].set_xticks(x)
    ax[2, 1].set_xticklabels(methods)
    ax[2, 1].set_title("CNR")
    
    # 3.2: Contrast (All, including True)
    methods_all = [m['name'] for m in metrics_all]
    x_all = np.arange(len(methods_all))
    
    ax[2, 2].bar(x_all - bar_width/2, [m['contrast_high'] for m in metrics_all], bar_width, label='High', color='r', alpha=0.7)
    ax[2, 2].bar(x_all + bar_width/2, [m['contrast_low'] for m in metrics_all], bar_width, label='Low', color='b', alpha=0.7)
    ax[2, 2].set_xticks(x_all)
    ax[2, 2].set_xticklabels(methods_all)
    ax[2, 2].set_title("Contrast Accurracy (vs True)")
    
    plt.tight_layout()
    print("Saving demo_recon_comparison.png...")
    plt.savefig("demo_recon_comparison.png")

# Run Plots
plot_attenuation_comparison()
plot_recon_comparison()
print("Done. Images saved.")
