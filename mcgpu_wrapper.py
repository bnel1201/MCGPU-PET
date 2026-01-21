import warnings
# Import the class from the new package
try:
    from mcgpu import MCGPUWrapper
except ImportError:
    # If package is not installed/found, try relative import if possible or fail
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'mcgpu'))
    from wrapper import MCGPUWrapper

# Emit warning
warnings.warn(
    "Importing from 'mcgpu_wrapper' is deprecated. "
    "Please update your code to use `from mcgpu import MCGPUWrapper`. "
    "This module will be removed in future versions.",
    DeprecationWarning,
    stacklevel=2
)
