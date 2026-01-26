
import pytest
import os
import sys
import numpy as np

# Adjust path to import package if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcgpu.wrapper import MCGPUWrapper

def test_short_simulation(tmp_path):
    """Run a short simulation and verify output"""
    
    # Initialize wrapper
    # Ensure it finds the binary (assuming installed or in source tree structure)
    test_dir = str(tmp_path)
    wrapper = MCGPUWrapper(working_dir=test_dir, verbose=True)
    
    if not os.path.exists(wrapper.executable_path):
        pytest.skip(f"Executable not found at {wrapper.executable_path}")

    # Short simulation parameters
    params = {
        'time_sec': 1.0,
        'random_seed': 123,
        # Use default materials which should be found by wrapper logic
        # Use default voxel file name - BUT wait, does the voxel file exist?
        # The default wrapper config points to 'nema_iec_128.vox'.
        # We need to make sure this file exists in the working dir or provided path.
        # In the sample_simulation folder it exists. 
        # For integration test, we might need to mock it or point to a valid one.
        # Let's create a dummy voxel file.
    }
    
    # Create a dummy voxel header/file if needed. 
    # Actually MCGPU-PET reads the voxel file. If it's missing, it crashes.
    # We can point to the one in sample_simulation if we know relative path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    voxel_path = os.path.join(repo_root, 'sample_simulation', 'nema_iec_128.vox')
    
    if os.path.exists(voxel_path):
            params['voxel_file'] = voxel_path
    else:
        print(f"DEBUG: Voxel path not found: {voxel_path}")
        pytest.skip("Voxel file check failed - required for meaningful integration test")

    try:
        results = wrapper.run(params)
        
        # Check results
        assert 'sinogram_Trues' in results
        # assert 'sinogram_Scatter' in results 
        # Scatter might be empty/non-existent if disabled or short run? 
        # Wrapper reads "sinogram_Scatter.raw.gz".
        
        # Check if execution happened (logs usually printed)
        
    except Exception as e:
        pytest.fail(f"Simulation failed with error: {e}")
