import gzip
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
RESULTS_DIR = "results_vis"
DATA_DIR = ".."  # dataset is in the parent directory of scripts/

# Dimensions from MCGPU-PET.out / MCGPU-PET.in
# Image: 9x9x9
# Dimensions
# Default (Simple Demo): 9x9x9
VOXELS_X, VOXELS_Y, VOXELS_Z = 9, 9, 9 
PREFIX = ""

# Auto-detect based on file size
trues_path = os.path.join(DATA_DIR, "image_Trues.raw.gz")
if os.path.exists(trues_path):
    with gzip.open(trues_path, 'rb') as f:
        # Read a bit to check size? Or seek? Gzip seek is slow.
        # Ideally we read standard size.
        # But wait, we can just check if sys.argv is NOT "nema" but size indicates NEMA.
        # 128^3 * 4 bytes = 8,388,608 bytes. 
        # 9^3 * 4 bytes = 2,916 bytes.
        # We can read the whole buffer since it's small for simple demo, large for NEMA.
        # Let's peek.
        pass
    
    # Check if 'nema' argument provided OR size implies NEMA
    # Getting gzip uncompressed size is hard without reading.
    # But usually we know NEMA is the main use case now.
    
    # Helper to check size
    def get_uncompressed_size(path):
        with gzip.open(path, 'rb') as f:
            f.seek(0, 2) # Seek to end
            return f.tell()
            
    # Actually seeking in Gzip is not always supported or efficient.
    # Let's just USE THE ARGUMENT check as primary, but also check if data read fails.
    # OR better: The "read_gzip_binary" function can return the size mismatch.
    pass

import sys
# Logic: If argument is present OR if we fail to match 9x9x9 in read_gzip_binary, we switch?
# No, easier to just check file size if possible or assume NEMA if 9x9x9 is too small.

# Explicit check for NEMA_IEC.in or similar?
if os.path.exists(os.path.join(DATA_DIR, "NEMA_IEC.in")) or (len(sys.argv) > 1 and sys.argv[1] == "nema"):
    # If NEMA input exists, assume NEMA mode unless specified otherwise
    print("Detected NEMA configuration (NEMA_IEC.in exists). Using NEMA dimensions.")
    PREFIX = "nema_"
    VOXELS_X, VOXELS_Y, VOXELS_Z = 128, 128, 128
    NRAD = 256
    NANGLES = 252
else:
    print("Using Default (Simple Demo) dimensions.")
    PREFIX = ""


def read_gzip_binary(filepath, shape, dtype=np.int32):
    """Reads a gzipped binary file into a numpy array."""
    print(f"Reading {filepath}...")
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    
    if data.size != np.prod(shape):
        print(f"Warning: Data size {data.size} does not match expected shape {shape} (prod={np.prod(shape)}).")
        # Attempt to reshape to linear if mismatch, or just return flattened
        return data
    
    return data.reshape(shape)

def plot_image_slice(data, title, filename):
    """Plots the central slice of a 3D image."""
    # Central slice index
    z_center = data.shape[2] // 2
    slice_data = data[:, :, z_center]
    
    plt.figure(figsize=(6, 5))
    plt.imshow(slice_data, cmap='hot', origin='lower', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title(f"{title} (Z-Slice {z_center})")
    plt.xlabel("X Voxel")
    plt.ylabel("Y Voxel")
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved {filename}")

def plot_sinogram_slice(data, title, filename):
    """Plots a sinogram slice (summed or central)."""
    # Reshape: The data is typically [Sinograms, Angles, Radial] or [Slices, Angles, Radial]
    # The output log says: Size of the 3D sinogram = 147 X 168 X 1293 (Rad x Ang x Sinos)
    # Fortran order is common in some scientific codes, but C is default for numpy.
    # Let's try to interpret as [NSINOS, NANGLES, NRAD] first, or [NRAD, NANGLES, NSINOS].
    # Based on standard C flattening where the last index varies fastest:
    # If the log says "147 X 168 X 1293", and it's C-style, it might mean shape (147, 168, 1293) or (1293, 168, 147).
    # Usually sinograms are (Angles, Radial) per slice.
    # Given the large number 1293 (likely Z slices * Segments), it's probably the slowest dimension.
    # Let's try reshaping to (NSINOS, NANGLES, NRAD) assuming Z is the outer dimension?
    # Or (NRAD, NANGLES, NSINOS) if X,Y,Z order.
    # Let's verify with the file size. 147*168*1293 * 4 bytes = ~127 MB. Matches sinogram_Trues.raw.gz uncompressed size.
    
    # We will assume [NSINOS, NANGLES, NRAD] for plotting a single sinogram.
    # Actually, if the code iterates x, y, z then Z might be last.
    # Let's just sum over the Z/Sino dimension to get a composite sinogram for visualization
    # if we are unsure of the order.
    
    # However, let's look at the read order.
    # If we reshape to (1293, 168, 147), we can plot one slice (e.g. middle sinogram).
    
    if data.ndim == 1:
        # Fallback reshape if read_gzip_binary failed to reshape correctly
        # Trying (1293, 168, 147)
        try:
            data = data.reshape((NSINOS, NANGLES, NRAD))
        except:
             try:
                 data = data.reshape((NRAD, NANGLES, NSINOS))
             except:
                 print("Could not reshape sinogram data.")
                 return

    # Slice index
    slice_idx = data.shape[0] // 2
    
    # If shape is (1293, 168, 147)
    if data.shape[0] == NSINOS:
        slice_data = data[slice_idx, :, :]
        xlabel, ylabel = "Radial Bin", "Angle"
    else:
        # Assuming last dim is NSINOS
        slice_idx = data.shape[2] // 2
        slice_data = data[:, :, slice_idx].T # Transpose to get Angle vs Radial
        xlabel, ylabel = "Radial Bin", "Angle"

    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data, cmap='gray', aspect='auto', origin='lower')
    plt.colorbar(label='Counts')
    plt.title(f"{title} (Slice {slice_idx})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved {filename}")

def plot_spectrum(filepath, title, filename):
    """Reads and plots the energy spectrum."""
    print(f"Reading {filepath}...")
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    # Skip header lines
    try:
        data = np.loadtxt(filepath, comments='#')
        # Expecting [Bin, Count]
        energies = data[:, 0]
        counts = data[:, 1]
        
        plt.figure(figsize=(8, 5))
        plt.plot(energies, counts)
        plt.title(title)
        plt.xlabel("Energy Bin (keV)")
        plt.ylabel("Counts")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(os.path.join(RESULTS_DIR, filename))
        plt.close()
        print(f"Saved {filename}")
        
    except Exception as e:
        print(f"Error plotting spectrum: {e}")

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def plot_phantom(filepath, shape, title, filename):
    """Reads ASCII voxel file and plots activity slice."""
    print(f"Reading phantom {filepath}...")
    try:
        # Skip header lines until [END OF VXH SECTION]
        # Then read data. 
        # But numpy loadtxt with `comments` might be hard if headers vary.
        # Let's read lines manually to find start of data
        data_start_line = 0
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if "[END OF VXH SECTION]" in line:
                    data_start_line = i + 1
                    break
        
        # Load data: Mat Dens Act
        # shape is (Z, Y, X) flat list of rows
        raw_data = np.loadtxt(filepath, skiprows=data_start_line)
        # Columns: [Material, Density, Activity]
        activity = raw_data[:, 2]
        

        # Reshape to 3D
        # Order is Z, Y, X based on generator loops
        activity_3d = activity.reshape(shape) # (Z, Y, X)
        
        # Plot central Z slice (Transaxial)
        z_center = shape[0] // 2
        plot_2d_slice(activity_3d[z_center, :, :], title, filename, xlabel="X", ylabel="Y")
        
    except Exception as e:
        print(f"Error plotting phantom: {e}")


def plot_2d_slice(data, title, filename, xlabel="X", ylabel="Y"):
    """Plots a 2D numpy array."""
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='hot', origin='lower', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Counts')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved {filename}")


import nrrd

# ... (Existing imports)

def save_nrrd(data, filename, spacing=None):
    """Saves data to NRRD format."""
    output_path = os.path.join(DATA_DIR, "outputs", filename)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    # Optional: Set spacing in header (e.g., specific voxel size)
    # For now, just saving raw data.
    # Note: nrrd.write expects (X, Y, Z) or (Z, Y, X)? 
    # It usually respects the numpy array order.
    # Slicer prefers (X, Y, Z) usually but handles headers.
    
    print(f"Saving {output_path}...")
    nrrd.write(output_path, data)

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # 1. 3D Images
    # Use PREFIX for reading files
    if PREFIX == "nema_":
         # Plot Ground Truth
         phantom_path = os.path.join(DATA_DIR, "nema_iec_128.vox")
         # We need to read it again to get the data for saving
         # Re-implementing simplified read for saving
         try:
            # Quick hack: call plot_phantom but it doesn't return data.
            # Let's just read it inline or duplicate logic.
            # Better: Make plot_phantom return data? 
            # Or just use the standard read logic here for the ground truth.
             with open(phantom_path, 'r') as f:
                lines = f.readlines()
                start = 0
                for i, line in enumerate(lines):
                    if "[END OF VXH SECTION]" in line:
                        start = i + 1
                        break
             raw = np.loadtxt(lines[start:])
             act = raw[:, 2].reshape((VOXELS_Z, VOXELS_Y, VOXELS_X))
             
             plot_phantom(phantom_path, (VOXELS_Z, VOXELS_Y, VOXELS_X), "Ground Truth Activity", "nema_phantom_activity.png")
             
             # Save NRRD
             save_nrrd(act, "nema_phantom_ground_truth.nrrd")
         except Exception as e:
             print(f"Error handling phantom: {e}")
         pass

    img_trues_path = os.path.join(DATA_DIR, "image_Trues.raw.gz")
    # MC-GPU memory order is Z-slowest, then Y, then X-fastest?
    # Kernel: ivox = x + y*NX + z*NX*NY.
    # So flattened array is [Z=0..NZ][Y=0..NY][X=0..NX].
    # Reshape should be (Z, Y, X).
    img_trues = read_gzip_binary(img_trues_path, (VOXELS_Z, VOXELS_Y, VOXELS_X))
    if img_trues is not None:
        # Transaxial Slice (XY plane at mid Z)
        mid_z = VOXELS_Z // 2
        plot_2d_slice(img_trues[mid_z, :, :], f"{PREFIX}True Coincidences (Transaxial Slice)", f"{PREFIX}image_trues_slice.png", xlabel="X", ylabel="Y")
        
        # MIP (Maximum Intensity Projection along Z)
        # Projects onto XY plane
        mip_z = np.max(img_trues, axis=0)
        plot_2d_slice(mip_z, f"{PREFIX}True Coincidences (MIP)", f"{PREFIX}image_trues_mip.png", xlabel="X", ylabel="Y")
        
        # Save NRRD
        save_nrrd(img_trues, f"{PREFIX}image_Trues.nrrd")

    img_scatter_path = os.path.join(DATA_DIR, "image_Scatter.raw.gz")
    img_scatter = read_gzip_binary(img_scatter_path, (VOXELS_Z, VOXELS_Y, VOXELS_X))
    if img_scatter is not None:
         mid_z = VOXELS_Z // 2
         plot_2d_slice(img_scatter[mid_z, :, :], f"{PREFIX}Scatter Coincidences (Transaxial Slice)", f"{PREFIX}image_scatter_slice.png", xlabel="X", ylabel="Y")
         save_nrrd(img_scatter, f"{PREFIX}image_Scatter.nrrd")

    # 2. Sinograms
    # Note: Sinograms are huge, might take memory.
    sino_trues_path = os.path.join(DATA_DIR, "sinogram_Trues.raw.gz")
    # For NEMA, we don't know exact NSINOS yet.
    # Try to read all and reshape if possible, or just skip if we can't guess.
    if PREFIX == "nema_":
        # Just read flat and try to guess or just plot middle slice from linear array
        sino_trues = read_gzip_binary(sino_trues_path, ( -1, )) 
        # approximate plotting
        # total size / (NRAD * NANGLES) = nsinos

        # Reshape to (Sinos, Angles, Radial)
        # Memory layout is [Sino][Angle][Radial] based on kernel: 
        # ibin = izm*NANGLES*NRAD + ith*NRAD + ir;
        calc_sinos = sino_trues.size // (NRAD * NANGLES)
        if calc_sinos > 0:
            sino_trues = sino_trues.reshape((calc_sinos, NANGLES, NRAD))
            
            # Save Full Sinogram NRRD
            save_nrrd(sino_trues, f"{PREFIX}sinogram_Trues.nrrd")

            # Sum over all sinograms (SSR - Single Slice Rebinning approximation for visualization)
            # This aggregates statistics from all 800+ planes (Direct and Oblique)
            print(f"Summing {calc_sinos} sinogram planes for visualization...")
            sino_sum = sino_trues.sum(axis=0) # Shape (NANGLES, NRAD)
            
            plot_2d_slice(sino_sum, "Sinogram (Trues) - Summed Slices", f"{PREFIX}sinogram_trues_slice.png", xlabel="Radial Bin", ylabel="Angle")
        
        # Load and Save Scatter Sinogram (Added for Scatter Analysis)
        sino_scatter_path = os.path.join(DATA_DIR, "sinogram_Scatter.raw.gz")
        if os.path.exists(sino_scatter_path):
             sino_scatter = read_gzip_binary(sino_scatter_path, ( -1, ))
             calc_sinos_s = sino_scatter.size // (NRAD * NANGLES)
             if calc_sinos_s > 0:
                 sino_scatter = sino_scatter.reshape((calc_sinos_s, NANGLES, NRAD))
                 save_nrrd(sino_scatter, f"{PREFIX}sinogram_Scatter.nrrd")
                 
                 sino_scatter_sum = sino_scatter.sum(axis=0)
                 plot_2d_slice(sino_scatter_sum, "Sinogram (Scatter) - Summed Slices", f"{PREFIX}sinogram_scatter_slice.png", xlabel="Radial Bin", ylabel="Angle")

    # 3. Energy Spectrum
    spectrum_path = os.path.join(DATA_DIR, "Energy_Sinogram_Spectrum.dat")
    plot_spectrum(spectrum_path, f"{PREFIX}Detected Energy Spectrum", f"{PREFIX}energy_spectrum.png")

if __name__ == "__main__":
    main()
