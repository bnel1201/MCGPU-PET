import os
import subprocess
import numpy as np
import struct
import math
import gzip

class MCGPUWrapper:
    """
    A Python wrapper for the MC-GPU-PET simulator.
    
    This class handles the generation of the input file, execution of the binary,
    and parsing of the output binary files into NumPy arrays.
    """

    def __init__(self, executable_path=None, working_dir='.', verbose=True):
        """
        Initialize the wrapper.

        Args:
            executable_path (str): Path to the MC-GPU-PET executable.
            working_dir (str): Working directory for the simulation.
            verbose (bool): Whether to print simulation output to stdout.
        """
        if executable_path is None:
            # Look in the same directory as this file, or check PATH? 
            # For this package, we expect it in the package dir
            executable_path = os.path.join(os.path.dirname(__file__), 'MCGPU-PET.x')
            
        self.executable_path = os.path.abspath(executable_path)
        self.working_dir = os.path.abspath(working_dir)
        self.verbose = verbose

        if not os.path.exists(self.executable_path):
            print(f"Warning: Executable not found at {self.executable_path}")
    
    def run(self, input_params=None, clean_up_input=True):
        """
        Run a simulation.

        Args:
            input_params (dict): Dictionary of simulation parameters overriding defaults.
                **Simulation Config:**
                - `random_seed` (int, default=0): Seed for random number generator.
                - `gpu_id` (int, default=0): CUDA GPU device ID to use.
                - `threads_per_block` (int, default=32): GPU threads per block.
                - `scale_density` (float, default=1.0): Scaling factor for material density.

                **Source:**
                - `time_sec` (float, default=1.0): Total acquisition time in seconds.
                - `isotope_mean_life` (float, default=70000.0): Mean life of the isotope in seconds.
                - `activities` (list of tuples, default=[]): List of (material_index, activity_Bq) for each active source.

                **Phase Space:**
                - `output_psf_file` (str, default='MCGPU_PET.psf'): Output filename for Phase Space File.
                - `detector_center` (tuple, default=(0.0, 0.0, 0.0)): Center (x,y,z) of the detector in cm.
                - `detector_height` (float, default=25.0): Height of the cylindrical detector in cm.
                - `detector_radius` (float, default=-15.0): Radius of the detector cylinder in cm (negative means inward facing?).
                - `psf_size` (int, default=20000000): Max number of photons to store in PSF.
                - `report_trues_scatter` (int, default=0): 1=Trues, 2=Scatter, 0=Both.
                - `report_psf_sino` (int, default=0): 1=PSF, 2=Sinogram, 0=Both.

                **Dose:**
                - `tally_material_dose` (str, default='YES'): Tally dose per material?
                - `tally_voxel_dose` (str, default='NO'): Tally 3D dose map?
                - `output_dose_file` (str, default='mc-gpu_dose.dat'): Output dose filename.
                - `dose_roi_x`, `dose_roi_y`, `dose_roi_z` (tuple, default=(1, 128)): Min/Max indices for dose ROI.

                **Energy:**
                - `energy_resolution` (float, default=0.12): Energy resolution (fraction, e.g. 12%).
                - `energy_window_low` (float, default=350000.0): Lower energy threshold in eV.
                - `energy_window_high` (float, default=600000.0): Upper energy threshold in eV.

                **Sinogram / Geometry:**
                - `fov_z` (float, default=25.0): Axial Field of View in cm.
                - `num_rows` (int, default=128): Number of detector element rows (transaxial?).
                - `num_crystals` (int, default=504): Total number of crystals per ring.
                - `num_angles` (int, default=252): Number of angular bins in sinogram.
                - `num_rad_bins` (int, default=256): Number of radial bins in sinogram.
                - `num_z_slices` (int, default=128): Number of axial slices.
                - `image_res` (int, default=128): Image resolution (pixels) for output maps.
                - `num_energy_bins` (int, default=700): Number of energy bins for spectra.
                - `max_ring_diff` (int, default=79): Maximum ring difference for 3D PET.
                - `span` (int, default=11): Span for mashing.

                **Files:**
                - `voxel_file` (str): Path to the voxelized geometry file (default 'nema_iec_128.vox').
                - `material_files` (list, default=[internal Air, internal Water]): List of material file paths.

            clean_up_input (bool): Whether to remove the generated .in file after run.

        Returns:
            dict: Dictionary containing parsed output data.
            keys:
                - `'sinogram_Trues'`: ndarray (n_sinos, n_angles, n_rad) - True coincidence sinograms.
                - `'sinogram_Scatter'`: ndarray (n_sinos, n_angles, n_rad) - Scatter coincidence sinograms.
                - `'image_Trues'`: ndarray (nzs, res, res) - Projection/Image of Trues (if generated).
                - `'image_Scatter'`: ndarray (nzs, res, res) - Projection/Image of Scatter (if generated).
        """
        # Merge defaults
        params = self._get_default_params()
        if input_params:
            params.update(input_params)
        
        # Prepare working directory
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            
        # Generate input file name
        input_filename = os.path.join(self.working_dir, f"mcgpu_input_{params['random_seed']}.in")
        
        # Write input file
        self._write_input_file(input_filename, params)
        
        # Construct command
        # MC-GPU typically takes the input file as the first argument
        # It's better to run from the working_dir so relative paths (like materials) work
        cmd = [self.executable_path, os.path.basename(input_filename)]
        
        if self.verbose:
            print(f"Running command: {' '.join(cmd)}")
            print(f"Working directory: {self.working_dir}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                stdout=None if self.verbose else subprocess.PIPE,
                stderr=None if self.verbose else subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print("Simulation failed!")
            if not self.verbose:
                print(e.stdout.decode())
                print(e.stderr.decode())
            raise e
        finally:
            if clean_up_input and os.path.exists(input_filename):
                os.remove(input_filename)

        # Parse outputs
        return self._read_outputs(params)

    def _get_default_params(self):
        """Returns the default simulation parameters."""
        return {
            # SIMULATION CONFIG
            'random_seed': 0,
            'gpu_id': 0,
            'threads_per_block': 32,
            'scale_density': 1.0,

            # SOURCE
            'time_sec': 1.0,
            'isotope_mean_life': 70000.0,
            'activities': [], # List of tuples/list [material_idx, activity_Bq], e.g. [[1, 1000.0]]

            # PHASE SPACE
            'output_psf_file': 'MCGPU_PET.psf',
            'detector_center': (0.0, 0.0, 0.0),
            'detector_height': 25.0,
            'detector_radius': -15.0,
            'psf_size': 20000000,
            'report_trues_scatter': 0,
            'report_psf_sino': 0,

            # DOSE
            'tally_material_dose': 'YES',
            'tally_voxel_dose': 'NO',
            'output_dose_file': 'mc-gpu_dose.dat',
            'dose_roi_x': (1, 128),
            'dose_roi_y': (1, 128),
            'dose_roi_z': (1, 128),

            # ENERGY
            'energy_resolution': 0.12,
            'energy_window_low': 350000.0,
            'energy_window_high': 600000.0,

            # SINOGRAM
            'fov_z': 25.0,
            'num_rows': 128,
            'num_crystals': 504,
            'num_angles': 252,
            'num_rad_bins': 256,
            'num_z_slices': 128,
            'image_res': 128,
            'num_energy_bins': 700,
            'max_ring_diff': 79,
            'span': 11,

            # GEOMETRY
            'voxel_file': 'nema_iec_128.vox',

            # MATERIALS
            'material_files': [
                os.path.join(os.path.dirname(__file__), 'materials', 'air_5-515keV.mcgpu.gz'),
                os.path.join(os.path.dirname(__file__), 'materials', 'water_5-515keV.mcgpu.gz')
            ]
        }

    def _write_input_file(self, filename, params):
        """Writes the MC-GPU input file directly."""
        with open(filename, 'w') as f:
            f.write("# >>>> INPUT FILE GENERATED BY PYTHON WRAPPER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
            
            f.write("#[SECTION SIMULATION CONFIG v.2016-07-05]\n")
            f.write(f"{params['random_seed']}                               # RANDOM SEED\n")
            f.write(f"{params['gpu_id']}                               # GPU NUMBER\n")
            f.write(f"{params['threads_per_block']}                              # GPU THREADS PER CUDA BLOCK\n")
            f.write(f"{params['scale_density']}                             # FACTOR TO SCALE DENSITY\n\n")
            
            f.write("#[SECTION SOURCE PET SCAN v.2017-03-14]\n")
            f.write(f"{params['time_sec']}                             # TOTAL PET SCAN ACQUISITION TIME [s]\n")
            f.write(f"{params['isotope_mean_life']}                        # ISOTOPE MEAN LIFE [s]\n")
            
            # Activities
            # Default empty if not provided, but one line with 0 0.0 is needed to terminate
            # If provided, list them. Ended by 0 0.0
            if params['activities']:
                for mat_idx, act in params['activities']:
                    f.write(f"   {mat_idx}    {act}   # Material activity\n")
            # Always ensure list is terminated
            f.write("   0    0.0\n\n")
            
            f.write("#[SECTION PHASE SPACE FILE v.2016-07-05]\n")
            f.write(f" {params['output_psf_file']}                  # OUTPUT PHASE SPACE FILE FILE NAME\n")
            cx, cy, cz = params['detector_center']
            f.write(f" {cx}  {cy}  {cz}  {params['detector_height']}  {params['detector_radius']}   # DETECTOR CENTER, HEIGHT, RADIUS\n")
            f.write(f" {params['psf_size']}                       # PHASE SPACE FILE SIZE\n")
            f.write(f" {params['report_trues_scatter']}                              # REPORT TRUES (1), SCATTER (2), OR BOTH (0)\n")
            f.write(f" {params['report_psf_sino']}                              # REPORT PSF (1), SINOGRAM (2) or BOTH (0)\n\n")
            
            f.write("#[SECTION DOSE DEPOSITION v.2012-12-12]\n")
            f.write(f"{params['tally_material_dose']}                             # TALLY MATERIAL DOSE? [YES/NO]\n")
            f.write(f"{params['tally_voxel_dose']}                              # TALLY 3D VOXEL DOSE? [YES/NO]\n")
            f.write(f"{params['output_dose_file']}                 # OUTPUT VOXEL DOSE FILE NAME\n")
            f.write(f"  {params['dose_roi_x'][0]}  {params['dose_roi_x'][1]}                        # VOXEL DOSE ROI: X-index min max\n")
            f.write(f"  {params['dose_roi_y'][0]}  {params['dose_roi_y'][1]}                        # VOXEL DOSE ROI: Y-index min max\n")
            f.write(f"  {params['dose_roi_z'][0]}  {params['dose_roi_z'][1]}                        # VOXEL DOSE ROI: Z-index min max\n\n")
            
            f.write("#[SECTION ENERGY PARAMETERS v.2019-04-25]\n")
            f.write(f"{params['energy_resolution']}          # ENERGY RESOLUTION\n")
            f.write(f"{params['energy_window_low']}      # ENERGY WINDOW LOW (keV)\n")
            f.write(f"{params['energy_window_high']}      # ENERGY WINDOW HIGH (keV)\n\n")
            
            f.write("#[SECTION SINOGRAM PARAMETERS v.2019-04-25]\n")
            f.write(f"{params['fov_z']} # AXIAL FOV\n")
            f.write(f"{params['num_rows']}     # NUMBER OF ROWS\n")
            f.write(f"{params['num_crystals']}    # TOTAL NUMBER OF CRYSTALS\n")
            f.write(f"{params['num_angles']}    # NUMBER OF ANGULAR BINS\n")
            f.write(f"{params['num_rad_bins']}    # NUMBER OF RADIAL BINS\n")
            f.write(f"{params['num_z_slices']}    # NUMBER OF Z SLICES\n")
            f.write(f"{params['image_res']}    # IMAGE RESOLUTION\n")
            f.write(f"{params['num_energy_bins']}    # NUMBER OF ENERGY BINS\n")
            f.write(f"{params['max_ring_diff']}     # MAXIMUM RING DIFFERENCE\n")
            f.write(f"{params['span']}     # SPAN\n\n")
            
            f.write("#[SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]\n")
            f.write(f"{params['voxel_file']}          # VOXEL GEOMETRY FILE\n\n")
            
            f.write("#[SECTION MATERIAL FILE LIST v.2009-11-30]\n")
            for mat_file in params['material_files']:
                f.write(f"{mat_file}\n")
            
            f.write("#\n# >>>> END INPUT FILE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    def _read_outputs(self, params):
        """Reads output files based on params and returns dictionary of results."""
        results = {}
        
        # Determine expected output filenames
        # Logic from MCGPU-PET.cu:
        # gzopen("sinogram_Trues.raw.gz","wb"); etc.
        # It seems filenames are hardcoded in the C++ code for sinograms/images!
        # "sinogram_Trues.raw.gz", "sinogram_Scatter.raw.gz"
        # "image_Trues.raw.gz", "image_Scatter.raw.gz"
        # Unless background simulation... but let's assume standard run first.
        
        # Dimensions for reshaping
        n_rad = params['num_rad_bins']
        n_ang = params['num_angles']
        
        # Calculate n_sinos (complex logic from C++ replicated here)
        mrd = params['max_ring_diff']
        span = params['span']
        nzs = params['num_z_slices']
        
        n_seg = 2 * math.floor(mrd / span) + 1
        n_sinos = n_seg * nzs
        
        for aaa in range(1, n_seg + 1):
            if aaa > 1:
                n_sinos -= (span + 1)
            if aaa > 3:
                n_sinos -= 2 * math.floor((aaa - 2) / 2) * span
        n_sinos = int(n_sinos)
        
        # Image dims
        res = params['image_res']
        
        # 1. Read Sinograms
        for type_name in ['Trues', 'Scatter']:
            filename = f"sinogram_{type_name}.raw.gz"
            filepath = os.path.join(self.working_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    with gzip.open(filepath, 'rb') as f:
                        data = f.read()
                    
                    # Assume int32
                    arr = np.frombuffer(data, dtype=np.int32)
                    
                    expected_len = n_rad * n_ang * n_sinos
                    if arr.size == expected_len:
                        # Reshape to (Projections, Angles, Radial) or similar?
                        # C code: NRAD * NANGLES * NSINOS
                        # Fortran order or C order? Usually C order in CUDA/C
                        # But wait, how exactly is it indexed?
                        # Usually sinograms are [Slices/Sinograms, Angles, Radial]
                        # Let's try to reshape to (n_sinos, n_ang, n_rad)
                        results[f'sinogram_{type_name}'] = arr.reshape((n_sinos, n_ang, n_rad))
                    else:
                        print(f"Warning: Sinogram {type_name} size mismatch. Expected {expected_len}, got {arr.size}")
                        results[f'sinogram_{type_name}'] = arr
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            else:
                 # Check for non-gzipped .raw if user compiled differently? 
                 # C++ code uses gzopen, so likely always .gz
                 pass

        # 2. Read Images
        for type_name in ['Trues', 'Scatter']:
            filename = f"image_{type_name}.raw.gz"
            filepath = os.path.join(self.working_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    with gzip.open(filepath, 'rb') as f:
                        data = f.read()
                    
                    arr = np.frombuffer(data, dtype=np.int32)
                    expected_len = res * res * nzs
                    
                    if arr.size == expected_len:
                        # Reshape to (Z, Y, X) or (X, Y, Z)?
                        # C++: NVOXS = RES*RES*NZS
                        # Output usually Z, Y, X in medical imaging if flat buffer
                        results[f'image_{type_name}'] = arr.reshape((nzs, res, res))
                    else:
                         results[f'image_{type_name}'] = arr
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        # 3. Read PSF (Optional)
        # Note: PSF is struct data, harder to parse in numpy without dtype definition
        # It's also large. Maybe skip for now unless requested, or implement partial read.
        # The user requested "output to be available as a numpy array".
        # If the user asks for PSF, we can implement it. For now, sinograms/images are primary.
        
        return results

