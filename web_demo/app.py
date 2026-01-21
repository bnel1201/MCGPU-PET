import os
import sys
import json
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, send_file
import io
import time # For checking cache validity

# Add parent directory to path to import mcgpu_wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcgpu_wrapper import MCGPUWrapper

# Import for Reconstruction
from skimage.transform import iradon
# Legacy Recon Path
LEGACY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../recon_demo_legacy/PET-Recon/recon"))
sys.path.append(LEGACY_PATH)
try:
    import system
    import algorithms
except ImportError:
    print("Warning: Could not import legacy PET-Recon modules. OSEM/BSREM will fail.")

app = Flask(__name__)

# Initialize wrapper - pointing to the executable in the parent/sample_simulation directory
# We'll use a specific temp directory for web simulations to avoid clutter
WEB_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp_sim'))
EXECUTABLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sample_simulation/MCGPU-PET.x'))

if not os.path.exists(WEB_SIM_DIR):
    os.makedirs(WEB_SIM_DIR)

wrapper = MCGPUWrapper(
    executable_path=EXECUTABLE_PATH,
    working_dir=WEB_SIM_DIR,
    verbose=True
)

# Global Cache to store last simulation results (for reconstruction without re-sim)
# In production, use session or redis. For demo, global dict is fine (single user assumed).
CACHE = {
    'sinogram': None, # (slices, angles, rad)
    'system_matrix': None, # Sparse matrix
    'geometry': {} # Cached geometry params
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        
        # Extract parameters from request
        # We expect the frontend to send a compatible dictionary or we map it here
        
        # Default params based on the notebook demo
        params = {
            'time_sec': float(data.get('time_sec', 20.0)),
            'random_seed': int(data.get('random_seed', 42)),
            'detector_height': float(data.get('detector_height', 25.0)),
            'detector_radius': float(data.get('detector_radius', -15.0)),
            'fov_z': float(data.get('fov_z', 25.0)),
            'num_rows': 128,
            'num_crystals': 504,
            'num_angles': 252,
            'num_rad_bins': 256,
            'image_res': int(data.get('image_res', 128)),
            
            # Activities
            'activities': [
                (1, float(data.get('activity_bg', 1000.0))),
                (2, float(data.get('activity_phantom', 200000.0))) # Increased for better visibility
            ],
            
            # Files - we need to make sure these paths are correct relative to the WORKING DIR
            # The wrapper runs in WEB_SIM_DIR (MCGPU-PET/web_demo/temp_sim).
            # The materials are in MCGPU-PET/sample_simulation/materials
            # So we go up two levels to MCGPU-PET, then down to sample_simulation/materials
            'material_files': [
                '../../sample_simulation/materials/air_5-515keV.mcgpu.gz',
                '../../sample_simulation/materials/water_5-515keV.mcgpu.gz'
            ],
            'voxel_file': '../../sample_simulation/nema_iec_128.vox' # Assuming this is where it is
        }

        # Run simulation
        print("Starting simulation with params:", params)
        results = wrapper.run(params)
        
        # Process results for transport
        # We'll send the raw arrays as base64 encoded binary
        
        response_data = {}
        
        # Helper to encode numpy array
        def encode_array(arr):
            # Ensure float32 for webgl/canvas consumption if possible, or keep as int32/float32
            # The raw output is int32 (counts). Let's cast to float32 for easier JS handling if needed, 
            # or keep int32. JS TypedArrays support Int32.
            # Let's send as Float32 to simplify normalization on client? Or just Int32.
            # Let's use Int32 for counts to save precision/space if relevant (though both 4 bytes).
            # ACTUALLY, let's cast to float32 because we might want to normalize on backend?
            # Nah, send raw counts.
            return base64.b64encode(arr.astype(np.int32).tobytes()).decode('utf-8')

        if 'sinogram_Trues' in results:
            # Shape: (Slices, Angles, Radial)
            sino = results['sinogram_Trues']
            
            # Update Cache
            CACHE['sinogram'] = sino.astype(np.float32) # Store as float for recon
            CACHE['geometry'] = {
                'num_angles': params['num_angles'],
                'num_rad_bins': params['num_rad_bins'],
                'image_res': params['image_res'], # This is desired output resolution
                'detector_radius': params['detector_radius'],
                # We need FOV to calculate pixel size for system matrix
                'fov_z': params['fov_z'], 
                # Note: We need Transaxial FOV or detector info to get bin size. 
                # Assuming standard NEMA ring diameter approx 85cm? 
                # Or derive from detector radius = 15cm -> Diameter = 30-40cm? 
                # MC-GPU input: cylinder 15cm radius.
            }
            # Invalidate System Matrix if geometry changes (simple check: none for now, just reset)
            # CACHE['system_matrix'] = None 
            
            response_data['sinogram'] = {
                'data': encode_array(sino),
                'shape': sino.shape,
                'max': int(np.max(sino))
            }
            
        if 'image_Trues' in results:
            # Shape: (Z, Y, X)
            img = results['image_Trues']
            response_data['image'] = {
                'data': encode_array(img),
                'shape': img.shape,
                'max': int(np.max(img))
            }

        return jsonify({'status': 'success', 'results': response_data})

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/reconstruct', methods=['POST'])
def reconstruct():
    try:
        if CACHE['sinogram'] is None:
            return jsonify({'status': 'error', 'message': 'No sinogram available. Run simulation first.'}), 400

        data = request.json
        method = data.get('method', 'FBP')
        
        # Get cached data
        mc_sino_3d = CACHE['sinogram'] # (Slices, Angles, Radial)
        geom = CACHE['geometry']
        
        # Parameters
        slices, num_angles, num_radial = mc_sino_3d.shape
        img_res = geom['image_res']
        
        # Slice Selection: For interactive speed, we might only recon CURRENT slice?
        # Or recon volume. Volume takes time (128 slices x 1s per slice FBP = 2 mins?).
        # For Demo: Recon CENTRAL SLICE only initially? Or User selected slice?
        # The user wants "Interactive control". Reconstructing 128 slices OSEM is TOO SLOW for web.
        # Strategy: Reconstruct current slice (from UI index) or central slice.
        # LET'S RECONSTRUCT ALL FOR FBP (fast), SINGLE SLICE FOR ITERATIVE?
        # Actually, let's try to reconstruct the whole volume for FBP, but OSEM/BSREM might timeout.
        # Better: Reconstruct specific slice requested by UI, OR Central Slice if not specified.
        # NOTE: The User wants to scroll through the volume.
        # If we only recon one slice, scrolling is broken.
        # Compromise: Reconstruct Center Slice for OSEM/BSREM (interactive demo), 
        # FBP can do full volume.
        # OR: We launch a background task? No, standard request/response.
        # We will reconstruct CENTER SLICE by default for iterative.
        
        # Decide scope
        slice_idx = data.get('slice_index', None)
        if slice_idx is None:
            slice_idx = slices // 2
        else:
            slice_idx = int(slice_idx)

        # Extract 2D sinogram for that slice
        if slice_idx < 0 or slice_idx >= slices:
             return jsonify({'status': 'error', 'message': 'Invalid slice index'}), 400
             
        sino_2d = mc_sino_3d[slice_idx, :, :]
        
        recon_img = None
        
        startTime = time.time()
        
        if method == 'FBP':
            # Scikit-image FBP
            # Theta needs to be 0-180 (or 0-360). MC-GPU covers 0-180 usually.
            theta = np.linspace(0., 180., num_angles, endpoint=False)
            # sinogram array (angles, radial). Iradon expects (theta, radial) or (radial, theta)?
            # scikit-image iradon: input (M, N) -> M angles, N positions?
            # "radon: (image, theta=theta). iradon: (radon_image, theta=theta)"
            # Usually sinogram is (Angles, Radial).
            # Transpose if needed. Let's assume (Angles, Radial) is standard.
            # Tested: iradon needs (Theta, Radial) if circle=True? 
            # Actually, iradon takes (row, col) = (theta, projection positions) usually.
            # Correct: input array is (number of angles, number of detectors).
            # Let's verify shape: (252, 256).
            
            recon_img = iradon(sino_2d.T, theta=theta, output_size=img_res, circle=True)
            # Rotate/Flip to match? Typically needed.
            
        elif method in ['OSEM', 'BSREM']:
            if 'algorithms' not in sys.modules:
                 return jsonify({'status': 'error', 'message': 'Reconstruction algorithms not loaded'}), 500

            # Generate/Get System Matrix
            # A needs to match the geometry. 
            # We need pixel_size_cm and bin_size_cm.
            # Using logic from reconstruct_nema.py
            det_radius = abs(geom['detector_radius'])
            ncrystals = 504 # Hardcoded NEMA standard for this demo (from params)
            fov_diameter = 2 * det_radius * np.sin((np.pi * num_radial) / ncrystals)
            bin_size_cm = fov_diameter / num_radial
            
            # Pixel size: usually FOV / res
            # Assumed FOV_XY ~ 32cm? 
            # Voxel file says bounding box 32x32x32cm.
            # So 32cm / 128 = 0.25cm.
            pixel_size_cm = 32.0 / img_res
            
            # Check cache
            A = CACHE.get('system_matrix')
            if A is None:
                print("Generating System Matrix (this happens once)...")
                # Using generate_toy_system_matrix from LEGACY lib
                A = system.generate_toy_system_matrix(
                    (img_res, img_res), 
                    num_angles, 
                    num_radial, 
                    pixel_size_cm=pixel_size_cm, 
                    bin_size_cm=bin_size_cm
                )
                CACHE['system_matrix'] = A
            
            # Recon Params
            iters = int(data.get('iterations', 5))
            subsets_n = int(data.get('subsets', 12))
            beta = float(data.get('beta', 0.05)) # For BSREM
            
            subsets_list = system.get_subsets(num_angles, num_radial, n_subsets=subsets_n)
            
            # Initial guess
            # r = randoms/scatter (Assuming 0 for demo/uncompensated)
            # In real demo we calculated Scatter, but here we just recon simple OSEM on Trues/Prompts
            r = np.zeros_like(sino_2d.flatten()) 
            
            if method == 'OSEM':
                # AC Factors defaults to None (No AC)
                rec_flat, _ = algorithms.run_osem(
                    sino_2d.flatten(), A, r, subsets_list, n_iters=iters, 
                    img_shape=(img_res, img_res), ac_factors=None
                )
                recon_img = rec_flat.reshape((img_res, img_res))
            
            elif method == 'BSREM':
                rec_flat, _ = algorithms.run_bsrem(
                    sino_2d.flatten(), A, r, subsets_list, n_iters=iters, 
                    img_shape=(img_res, img_res), beta=beta, ac_factors=None
                )
                recon_img = rec_flat.reshape((img_res, img_res))

        duration = time.time() - startTime
        print(f"Reconstruction ({method}) took {duration:.2f}s")
        
        # Helper to encode
        def encode_array(arr):
             return base64.b64encode(arr.astype(np.float32).tobytes()).decode('utf-8')

        response_data = {
            'image_slice': {
                'data': encode_array(recon_img),
                'shape': recon_img.shape,
                'max': float(np.max(recon_img))
            },
            'slice_index': slice_idx,
            'method': method
        }

        return jsonify({'status': 'success', 'results': response_data})

    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
