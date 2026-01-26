import os
import numpy as np
import matplotlib.pyplot as plt
from mcgpu import MCGPUWrapper

def main():
    # Initialize wrapper
    # We use sample_simulation as working dir so it finds the voxel file 'nema_iec_128.vox'
    # The executable and material files are now provided by the installed package!
    wrapper = MCGPUWrapper(
        working_dir='sample_simulation',
        verbose=True
    )

    # Define parameters (short run for demo)
    # Using defaults mostly, but specifying explicit activities
    params = {
        'time_sec': 5.0,  # Short time
        'random_seed': 12345,
        'activities': [
            (1, 1000.0), # Material 1 (Air?) - Low activity
            (2, 50000.0) # Material 2 (Water?) - Higher activity
        ],

        'voxel_file': 'nema_iec_128.vox',
        
        # NEMA Parameters (matching NEMA_IEC.in)
        'detector_height': 25.0,
        'detector_radius': -15.0,
        'fov_z': 25.0,
        'num_rows': 128,
        'num_crystals': 504,
        'num_angles': 252,
        'num_rad_bins': 256,
        'num_z_slices': 128,
        'image_res': 128
    }

    try:
        print("Starting simulation...")
        results = wrapper.run(params)
        print("Simulation finished.")
        
        # Analyze results
        if 'image_Trues' in results:
            img = results['image_Trues']
            print(f"True Image Shape: {img.shape}")
            print(f"Total True Counts: {np.sum(img)}")
            
            # Simple visualization of a central slice
            # Assuming Z, Y, X order
            center_slice = img.shape[0] // 2
            plt.figure(figsize=(10, 5))
            plt.imshow(img[center_slice, :, :], cmap='hot', interpolation='nearest')
            plt.colorbar(label='Counts')
            plt.title(f'True Coincidences (Slice {center_slice})')
            # Save in current dir, not working dir
            plt.savefig('demo_trues_slice.png')
            print("Saved visualization to demo_trues_slice.png")
            
        if 'sinogram_Trues' in results:
            sino = results['sinogram_Trues']
            print(f"True Sinogram Shape: {sino.shape}")
            
            # Visualize a sinogram plane (e.g. middle sinogram)
            center_sino = sino.shape[0] // 2
            plt.figure(figsize=(10, 5))
            plt.imshow(sino[center_sino, :, :], cmap='gray', aspect='auto')
            plt.colorbar(label='Counts')
            plt.title(f'True Sinogram (Plane {center_sino})')
            plt.xlabel('Radial Bin')
            plt.ylabel('Angle')
            plt.savefig('demo_trues_sinogram.png')
            print("Saved visualization to demo_trues_sinogram.png")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
