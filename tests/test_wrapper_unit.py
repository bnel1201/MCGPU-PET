
import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
import numpy as np

# Adjust path to import package if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcgpu.wrapper import MCGPUWrapper

class TestMCGPUWrapperUnit(unittest.TestCase):
    def setUp(self):
        self.mock_executable = '/tmp/fake_mcgpu.x'
        self.wrapper = MCGPUWrapper(executable_path=self.mock_executable, verbose=False)

    def test_init_defaults(self):
        """Test initialization with default paths"""
        # When relying on package path, we can't easily assert exact path without mocking __file__ in wrapper
        # But we can check if it tries to resolve it.
        # For this unit test, we passed a specific executable path.
        self.assertEqual(self.wrapper.executable_path, self.mock_executable)
        self.assertEqual(self.wrapper.working_dir, os.path.abspath('.'))

    def test_get_default_params(self):
        """Test default parameters dictionary"""
        params = self.wrapper._get_default_params()
        self.assertIn('random_seed', params)
        self.assertIn('time_sec', params)
        self.assertIn('material_files', params)
        # Check defaults
        self.assertEqual(params['random_seed'], 0)
        self.assertEqual(params['threads_per_block'], 32)
        
    @patch('builtins.open', new_callable=mock_open)
    def test_write_input_file(self, mock_file):
        """Test writing the input file"""
        params = self.wrapper._get_default_params()
        
        # Modify some params for uniqueness
        params['random_seed'] = 999
        params['time_sec'] = 123.4
        
        filename = 'test.in'
        self.wrapper._write_input_file(filename, params)
        
        mock_file.assert_called_with(filename, 'w')
        handle = mock_file()
        
        # Check for specific content write calls
        handle.write.assert_any_call("999                               # RANDOM SEED\n")
        handle.write.assert_any_call("123.4                             # TOTAL PET SCAN ACQUISITION TIME [s]\n")

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('mcgpu.wrapper.os.remove')
    @patch('mcgpu.wrapper.MCGPUWrapper._read_outputs')
    @patch('mcgpu.wrapper.os.path.exists')
    def test_run_logic(self, mock_exists, mock_read_outputs, mock_remove, mock_file_open, mock_subprocess):
        """Test run logic calls subprocess and cleanup"""
        # mock_exists needs to return True for the input file cleanup check
        mock_exists.return_value = True
        
        mock_read_outputs.return_value = {'test': 'data'}
        
        result = self.wrapper.run(input_params={'random_seed': 42}, clean_up_input=True)
        
        # Verify subprocess call
        # Expected command: [executable, input_filename]
        expected_input = os.path.abspath(f"mcgpu_input_42.in")
        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], self.mock_executable)
        self.assertIn('mcgpu_input_42.in', cmd[1]) # Just the basename is passed usually if cwd is set
        
        # Verify cleanup
        # Since we mocked exists=True, it should try to remove
        mock_remove.assert_called_with(expected_input)
        
        # Verify result
        self.assertEqual(result, {'test': 'data'})

    @patch('mcgpu.wrapper.gzip.open', new_callable=mock_open, read_data=b'\x00\x00\x00\x00' * 100)
    @patch('mcgpu.wrapper.os.path.exists')
    def test_read_outputs(self, mock_exists, mock_gzip):
        """Test reading output sinograms and images"""
        mock_exists.return_value = True # Pretend files exist
        
        params = self.wrapper._get_default_params()
        # Set small dimensions for testing
        params['num_rad_bins'] = 2
        params['num_angles'] = 2
        params['num_z_slices'] = 1
        params['max_ring_diff'] = 0
        params['span'] = 1
        params['image_res'] = 2
        
        # Mock reading actual bytes?
        # We need byte length to match expected for "successful" reshape
        # Let's just check it attempts to read
        
        results = self.wrapper._read_outputs(params)
        
        self.assertIn('sinogram_Trues', results)
        self.assertIn('image_Trues', results)

if __name__ == '__main__':
    unittest.main()
