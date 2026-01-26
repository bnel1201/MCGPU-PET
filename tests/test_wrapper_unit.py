
import pytest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
import numpy as np

# Adjust path to import package if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcgpu.wrapper import MCGPUWrapper

@pytest.fixture
def wrapper():
    mock_executable = '/tmp/fake_mcgpu.x'
    return MCGPUWrapper(executable_path=mock_executable, verbose=False)

def test_init_defaults(wrapper):
    """Test initialization with default paths"""
    # When relying on package path, we can't easily assert exact path without mocking __file__ in wrapper
    # But we can check if it tries to resolve it.
    assert wrapper.executable_path == '/tmp/fake_mcgpu.x'
    assert wrapper.working_dir == os.path.abspath('.')

def test_get_default_params(wrapper):
    """Test default parameters dictionary"""
    params = wrapper._get_default_params()
    assert 'random_seed' in params
    assert 'time_sec' in params
    assert 'material_files' in params
    # Check defaults
    assert params['random_seed'] == 0
    assert params['threads_per_block'] == 32
    
@patch('builtins.open', new_callable=mock_open)
def test_write_input_file(mock_file, wrapper):
    """Test writing the input file"""
    params = wrapper._get_default_params()
    
    # Modify some params for uniqueness
    params['random_seed'] = 999
    params['time_sec'] = 123.4
    
    filename = 'test.in'
    wrapper._write_input_file(filename, params)
    
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
def test_run_logic(mock_exists, mock_read_outputs, mock_remove, mock_file_open, mock_subprocess, wrapper):
    """Test run logic calls subprocess and cleanup"""
    # mock_exists needs to return True for the input file cleanup check
    mock_exists.return_value = True
    
    mock_read_outputs.return_value = {'test': 'data'}
    
    result = wrapper.run(input_params={'random_seed': 42}, clean_up_input=True)
    
    # Verify subprocess call
    # Expected command: [executable, input_filename]
    expected_input = os.path.abspath(f"mcgpu_input_42.in")
    mock_subprocess.assert_called_once()
    args, kwargs = mock_subprocess.call_args
    cmd = args[0]
    assert cmd[0] == wrapper.executable_path
    assert 'mcgpu_input_42.in' in cmd[1] # Just the basename is passed usually if cwd is set
    
    # Verify cleanup
    # Since we mocked exists=True, it should try to remove
    mock_remove.assert_called_with(expected_input)
    
    # Verify result
    assert result == {'test': 'data'}

@patch('mcgpu.wrapper.gzip.open', new_callable=mock_open, read_data=b'\x00\x00\x00\x00' * 100)
@patch('mcgpu.wrapper.os.path.exists')
def test_read_outputs(mock_exists, mock_gzip, wrapper):
    """Test reading output sinograms and images"""
    mock_exists.return_value = True # Pretend files exist
    
    params = wrapper._get_default_params()
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
    
    results = wrapper._read_outputs(params)
    
    assert 'sinogram_Trues' in results
    assert 'image_Trues' in results
