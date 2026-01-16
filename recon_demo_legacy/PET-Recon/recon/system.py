import numpy as np
from scipy.sparse import coo_matrix

def generate_toy_system_matrix(img_shape, num_angles, num_bins, pixel_size_cm=1.0, bin_size_cm=1.0):
    """
    Generates a simplified sparse system matrix (A) for a 2D image.
    This replaces a complex ray-tracer for demonstration purposes.
    """
    H, W = img_shape
    n_pixels = H * W
    n_projections = num_angles * num_bins
    
    # Coordinates of pixels
    y, x = np.indices(img_shape)
    x_c = x.flatten() - W / 2.0
    y_c = y.flatten() - H / 2.0
    
    angles = np.linspace(0, 180, num_angles, endpoint=False)
    rows = []
    cols = []
    data = []
    
    # Simple line integral approximation
    # For each angle, project pixel locations onto the detector line
    for i, ang in enumerate(angles):
        theta = np.deg2rad(ang)
        # Rotate coordinates
        # p_pixels = x * cos(theta) + y * sin(theta)
        p_pixels = x_c * np.cos(theta) + y_c * np.sin(theta)
        
        # Convert to physical distance
        p_cm = p_pixels * pixel_size_cm
        
        # Map p to bin index
        # Center of detector is 0.
        bin_idx = np.round(p_cm / bin_size_cm + num_bins / 2.0).astype(int)
        
        # Valid bins only
        valid_mask = (bin_idx >= 0) & (bin_idx < num_bins)
        
        if np.any(valid_mask):
             current_rows = (i * num_bins) + bin_idx[valid_mask]
             current_cols = np.flatnonzero(valid_mask)
             
             rows.extend(current_rows)
             cols.extend(current_cols)
             data.extend(np.ones(len(current_rows))) # Simplified weight=1
        
    # Create Sparse Matrix
    A = coo_matrix((data, (rows, cols)), shape=(n_projections, n_pixels)).tocsr()
    return A

def get_subsets(n_angles, n_bins, n_subsets):
    """
    Partitions projection indices into M subsets based on angles.
    """
    total_proj = n_angles * n_bins
    indices = np.arange(total_proj)
    # Reshape to (angles, bins) to slice by angle
    indices_reshaped = indices.reshape((n_angles, n_bins))
    
    subsets = []
    for m in range(n_subsets):
        # Take every M-th angle
        subset_indices = indices_reshaped[m::n_subsets, :].flatten()
        subsets.append(subset_indices)
    return subsets
