# ========================================================================================
#                                  MAKEFILE MC-GPU-PET
#
# 
#   ** Simple script to compile the code MC-GPU-PET
#
#      Using the default installation path for the CUDA toolkit and SDK (http://www.nvidia.com/cuda). 
#      The code can also be compiled for specific GPU architectures using the "-gencode=arch=compute_61,code=sm_61"
#      option, where in this case 61 refers to compute capability 6.1.
#      The zlib.h library is used to allow gzip-ed input files.
#
#      Default paths:
#         CUDA:  /usr/local/cuda
#         SDK:   /usr/local/cuda/samples
#
# 
#                      @file    Makefile
#                      @author  Andreu Badal [Andreu.Badal-Soler (at) fda.hhs.gov]
#                      @date    2022/02/02
#   
# ========================================================================================

SHELL = /bin/sh

# Suffixes:
.SUFFIXES: .cu .o

# Compilers and linker:
CC = nvcc

# Program's name:
PROG = MCGPU-PET.x

# Include and library paths:
# Include and library paths:
CUDA_PATH = /usr
CUDA_LIB_PATH = /usr/lib/x86_64-linux-gnu
CUDA_SDK_PATH = ./cuda-samples/Common/
CUDA_SDK_LIB_PATH = ./cuda-samples/Common/

#  NOTE: you can compile the code for a specific GPU compute capability. For example, for compute capabilities 5.0 and 6.1, use flags:
GPU_COMPUTE_CAPABILITY = -gencode=arch=compute_75,code=sm_75

# Compiler's flags:
CFLAGS = -DUSING_CUDA -O3 -use_fast_math -m64 -I./ -I$(CUDA_PATH)/include -I$(CUDA_SDK_PATH) -L$(CUDA_SDK_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lm -lz --ptxas-options=-v $(GPU_COMPUTE_CAPABILITY)


# Command to erase files:
RM = /bin/rm -vf

# .cu files path:
SRCS = MCGPU-PET.cu

# Building the application:
default: $(PROG)
$(PROG):
	$(CC) $(CFLAGS) $(SRCS) -o $(PROG)

# Rule for cleaning re-compilable files
# Rule for cleaning re-compilable files and simulation results
clean:
	$(RM) $(PROG)
	$(RM) sample_simulation/*.raw.gz
	$(RM) sample_simulation/*.out
	$(RM) sample_simulation/*.vox
	$(RM) -r sample_simulation/outputs/*
	$(RM) -r sample_simulation/scripts/results_recon
	$(RM) -r sample_simulation/scripts/results_recon_vol
	$(RM) -r sample_simulation/scripts/__pycache__
	@echo "Cleaned all simulation and reconstruction results."

# ========================================================================================
#                                     HELPERS
# ========================================================================================

# ========================================================================================
#                                     HELPERS
# ========================================================================================

# Intermediate target for phantom generation
phantom:
	@echo "Generating NEMA phantom..."
	cd sample_simulation/scripts && python3 generate_nema_phantom.py
	@echo "Moving phantom to simulation dir..."
	mv sample_simulation/scripts/nema_iec_128.vox sample_simulation/

run-nema: $(PROG) phantom
	@echo "Running NEMA simulation (Standard Window)..."
	cd sample_simulation && ./MCGPU-PET.x NEMA_IEC.in
	@echo "Backing up Standard results..."
	mkdir -p sample_simulation/output_std
	cp sample_simulation/*.raw.gz sample_simulation/output_std/

run-narrow: $(PROG) phantom
	@echo "Running NEMA Simulation (Narrow Window)..."
	cd sample_simulation && ../MCGPU-PET.x NEMA_IEC_narrow.in
	@echo "Backing up Narrow results..."
	mkdir -p sample_simulation/output_narrow
	cp sample_simulation/*.raw.gz sample_simulation/output_narrow/

visualize:
	@echo "Running Visualization..."
	cd sample_simulation/scripts && python3 visualize_results.py

recon-nema:
	@echo "Running NEMA Reconstruction Comparison..."
	cd sample_simulation/scripts && python3 reconstruct_nema.py

recon-volume:
	@echo "Running Volume Reconstruction (OSEM, 5mm slice thickness)..."
	cd sample_simulation/scripts && python3 reconstruct_volume.py

compare-windows:
	@echo "Comparing Energy Windows..."
	cd sample_simulation/scripts && python3 compare_energy_windows.py

# Full Pipeline
# Order: 
# 1. run-narrow (Leaves narrow files) -> Backup
# 2. run-nema (Overwrites with Standard files) -> Backup & Leave for Recon
# 3. Analyze & Reconstruct (uses Standard files currently in folder)
all: run-narrow run-nema visualize recon-nema recon-volume compare-windows
	@echo "================================================================================"
	@echo "   Pipeline Finalized Successfully."
	@echo "================================================================================"

help:
	@echo "================================================================================"
	@echo "   MC-GPU-PET Makefile Helper"
	@echo "================================================================================"
	@echo "   Available commands:"
	@echo "      make             : Compile the MCGPU-PET.x executable."
	@echo "      make clean       : Remove the executable and all simulation results."
	@echo "      make all         : Run FULL pipeline: Sim (Std+Narrow) -> Recon -> Analysis."
	@echo "      make phantom     : Generate NEMA IEC phantom voxel file."
	@echo "      make run-nema    : Run Standard NEMA simulation (350-600 keV)."
	@echo "      make run-narrow  : Run Narrow NEMA simulation (450-550 keV)."
	@echo "      make recon-nema  : Run 2D Reconstruction Comparison."
	@echo "      make recon-volume: Run 3D Volume Reconstruction."
	@echo "      make visualize   : Generate histograms and basic plots."
	@echo "      make compare-windows: Compare Standard vs Narrow stats."
	@echo "================================================================================"
