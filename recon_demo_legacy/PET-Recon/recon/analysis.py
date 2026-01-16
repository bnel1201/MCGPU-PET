import numpy as np

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    return dist_from_center <= radius

def get_phantom_rois(img_size):
    """
    Returns dictionary of standard ROIs for the Shepp-Logan phantom
    scaled to img_size.
    """
    # High Contrast: Top bright spot
    mask_high = create_circular_mask(img_size, img_size, center=(64, 48), radius=5) 
    # Low Contrast: Bottom-ish grey area
    mask_low = create_circular_mask(img_size, img_size, center=(64, 80), radius=5)
    # Background: Top left separate from phantom
    mask_bg = create_circular_mask(img_size, img_size, center=(20, 20), radius=10)
    
    return mask_high, mask_low, mask_bg

def compute_metrics(image, name, rois):
    """
    Computes Noise, Contrast, and CNR given an image and ROIs.
    rois: tuple (mask_high, mask_low, mask_bg)
    """
    mask_high, mask_low, mask_bg = rois
    
    # Noise (Std Dev)
    std_high = np.std(image[mask_high])
    std_low = np.std(image[mask_low])
    std_bg = np.std(image[mask_bg])
    
    # Contrast: Mean ROI - Mean BG
    mean_high = np.mean(image[mask_high])
    mean_low = np.mean(image[mask_low])
    mean_bg = np.mean(image[mask_bg])
    
    contrast_high = mean_high - mean_bg
    contrast_low = mean_low - mean_bg
    
    # CNR: Contrast / Std_BG
    # Avoid div by zero
    s_bg_safe = std_bg if std_bg > 1e-6 else 1.0
    cnr_high = contrast_high / s_bg_safe
    cnr_low = contrast_low / s_bg_safe
    
    return {
        "name": name,
        "std_high": std_high, "std_low": std_low, "std_bg": std_bg,
        "contrast_high": contrast_high, "contrast_low": contrast_low,
        "cnr_high": cnr_high, "cnr_low": cnr_low
    }
