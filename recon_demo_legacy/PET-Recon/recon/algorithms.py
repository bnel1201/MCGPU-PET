import numpy as np

def compute_penalty_gradient_quadratic(x, beta, img_shape):
    """
    Computes gradient of Quadratic Roughness Penalty R(lambda).
    R(lambda) = beta/2 * sum(w_jk * (lambda_j - lambda_k)^2)
    Grad_j R = beta * sum(w_jk * (lambda_j - lambda_k))
    """
    img = x.reshape(img_shape)
    grad = np.zeros_like(img)
    
    # Standard 4-neighbor differences
    diff_u = img - np.roll(img, 1, axis=0) # Up neighbor difference
    diff_d = img - np.roll(img, -1, axis=0)
    diff_l = img - np.roll(img, 1, axis=1)
    diff_r = img - np.roll(img, -1, axis=1)
    
    # Sum of differences (assuming w_jk = 1 for nearest neighbors)
    grad = diff_u + diff_d + diff_l + diff_r
    
    return beta * grad.flatten()

def run_osem(y, A, r, subsets, n_iters, img_shape, ac_factors=None):
    """
    OSEM with optional Attenuation Correction.
    ac_factors: Array of size (n_projections,), containing exp(-int mu).
                If None/1.0, NO correction is performed (AC Off).
    """
    x = np.full(img_shape[0]*img_shape[1], 1.0) # Initial guess
    objective_history = []
    
    if ac_factors is None:
        ac_factors = np.ones_like(y)
    
    for it in range(n_iters):
        for subset_idx in subsets:
            # Get subset matrices
            A_sub = A[subset_idx, :]
            y_sub = y[subset_idx]
            r_sub = r[subset_idx]
            ac_sub = ac_factors[subset_idx]
            
            # Forward projection on subset: (A*x)*ac + r
            expected = (A_sub.dot(x) * ac_sub) + r_sub
            
            # Sensitivity: A^T * ac
            sensitivity = A_sub.T.dot(ac_sub)
            # Avoid division by zero
            sensitivity[sensitivity == 0] = 1e-9
            
            ratio = y_sub / (expected + 1e-9)
            # Backproj: A^T * (ratio * ac)
            backproj = A_sub.T.dot(ratio * ac_sub)
            
            x = (x / sensitivity) * backproj
            
        # Log likelihood (approx)
        full_expected = (A.dot(x) * ac_factors) + r
        ll = np.sum(y * np.log(full_expected + 1e-9) - full_expected)
        objective_history.append(ll)
        
    return x, objective_history

def run_bsrem(y, A, r, subsets, n_iters, img_shape, beta, U_bound=None, ac_factors=None):
    """
    Modified BSREM (Block Sequential Regularized EM) with Attenuation Correction.
    Based on Ahn & Fessler 2003.
    """
    x = np.full(img_shape[0]*img_shape[1], 1.0)
    # n_pixels = len(x)
    M = len(subsets)
    
    if ac_factors is None:
        ac_factors = np.ones_like(y)
    
    # Calculate U (Upper Bound) using Eq 32 in Appendix A
    # U = max_i (y_i / min_j a_ij). Simplified here to a safe large value.
    if U_bound is None:
        min_nonzero_a = A.data.min() if A.nnz > 0 else 1.0
        U_bound = np.max(y) / min_nonzero_a * 2.0 
    
    # Precompute sensitivities for scaling
    # p_j = sum(a_ij * ac_i) / M (Eq 24)
    sensitivity_full = A.T.dot(ac_factors)
    p = sensitivity_full / M
    p[p == 0] = 1e-9 # Avoid div by zero
    
    objective_history = []
    
    for n in range(n_iters):
        
        # Relaxation Parameter alpha_n (Eq 31) 
        # Using alpha_0 = 1, gamma = 0.1 (tuned for this toy problem)
        alpha_n = 1.0 / (0.1 * n + 1)
        
        for m, subset_idx in enumerate(subsets):
            A_sub = A[subset_idx, :]
            y_sub = y[subset_idx]
            r_sub = r[subset_idx]
            ac_sub = ac_factors[subset_idx]
            
            # 1. Gradient of Likelihood for Subset m
            # grad_L = A_m^T * ac * (y_m / (A_m x * ac + r_m) - 1)
            l_proj = (A_sub.dot(x) * ac_sub) + r_sub
            ratio = y_sub / (l_proj + 1e-9) - 1.0
            grad_likelihood = A_sub.T.dot(ratio * ac_sub)
            
            # 2. Gradient of Penalty
            # Scaled by 1/M because we include regularization in every subset (Eq 27) 
            grad_penalty = compute_penalty_gradient_quadratic(x, beta, img_shape) / M
            
            # Total Gradient for this subset
            # Note: The paper defines f_m = L_m - R_m.
            grad_f = grad_likelihood - grad_penalty
            
            # 3. Scaling Function D(lambda) (Eq 22) [cite: 125, 134]
            # If x < U/2: d_j = x_j / p_j
            # If x >= U/2: d_j = (U - x_j) / p_j
            d = np.where(x < U_bound / 2.0, 
                         x / p, 
                         (U_bound - x) / p)
            
            # 4. Update Step (Eq 21) 
            x_update = alpha_n * d * grad_f
            x = x + x_update
            
            # 5. Enforce Bounds (Modified BSREM-II projection) [cite: 142]
            # Clip to [epsilon, U] to ensure strict positivity and boundedness
            x = np.clip(x, 1e-6, U_bound)
            
        # Compute Objective: LogLikelihood - Beta * R(x)
        full_proj = (A.dot(x) * ac_factors) + r
        ll = np.sum(y * np.log(full_proj + 1e-9) - full_proj)
        # R(x) calculation
        img = x.reshape(img_shape)
        # Simple R calc
        diff_sq = (img - np.roll(img, 1, axis=0))**2 + (img - np.roll(img, 1, axis=1))**2
        R_val = 0.5 * beta * np.sum(diff_sq)
        
        objective_history.append(ll - R_val)
        
    return x, objective_history
