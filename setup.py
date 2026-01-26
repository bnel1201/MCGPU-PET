import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

class BuildMCGPU(build_py):
    def run(self):
        # Run standard build first to copy python files
        super().run()
        
        # Define paths
        source_root = os.path.dirname(os.path.abspath(__file__))
        build_lib = self.build_lib
        target_dir = os.path.join(build_lib, 'mcgpu')
        executable_name = 'MCGPU-PET.x'
        target_path = os.path.join(target_dir, executable_name)
        
        # Ensure target directory exists (it should after super().run())
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        print(f"Compiling {executable_name} to {target_path}...")
        
        # CUDA Compilation Command
        # Based on Makefile:
        # nvcc -DUSING_CUDA -O3 -use_fast_math -m64 -I./ -I/usr/include -I./cuda-samples/Common/ -L./cuda-samples/Common/ -L/usr/lib/x86_64-linux-gnu -lcudart -lm -lz --ptxas-options=-v MCGPU-PET.cu -o MCGPU-PET.x
        
        cuda_sdk_path = os.path.join(source_root, 'cuda-samples', 'Common')
        
        cmd = [
            'nvcc',
            '-DUSING_CUDA', '-O3', '-use_fast_math', '-m64',
            f'-I{source_root}', 
            '-I/usr/include',
            f'-I{cuda_sdk_path}',
            f'-L{cuda_sdk_path}',
            # '-L/usr/lib/x86_64-linux-gnu', # Let nvcc handle system libs usually
            '-lcudart', '-lm', '-lz',
            '--ptxas-options=-v',
            os.path.join(source_root, 'MCGPU-PET.cu'),
            '-o', target_path
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print("ERROR: Compilation failed.")
            print("Please ensure you have the CUDA Toolkit installed (nvcc) and available in your PATH.")
            raise e
        except FileNotFoundError:
            print("ERROR: nvcc not found.")
            print("Please ensure you have the CUDA Toolkit installed and 'nvcc' is in your PATH.")
            raise

setup(
    cmdclass={
        'build_py': BuildMCGPU,
    },
)
