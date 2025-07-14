"""
FANN Rust Core - High-Performance Neural Network Library for Python

A comprehensive neural network library built in Rust with Python bindings via PyO3.
Provides fast, safe, and efficient neural network computations with support for
SIMD optimization, parallel processing, and edge deployment.

Key Features:
- Memory-safe neural network operations
- SIMD-optimized computations (AVX2, NEON)
- Parallel processing support
- Model quantization for edge deployment
- Comprehensive training algorithms
- Neural swarm integration capabilities

Example Usage:
    >>> import fann_rust_core as fann
    >>> 
    >>> # Initialize the library
    >>> fann.initialize()
    >>> 
    >>> # Create a neural network
    >>> network = fann.NeuralNetwork([784, 128, 64, 10], activation="relu")
    >>> 
    >>> # Train the network
    >>> import numpy as np
    >>> inputs = np.random.rand(1000, 784).astype(np.float32)
    >>> targets = np.random.rand(1000, 10).astype(np.float32)
    >>> results = network.train(inputs, targets, epochs=100, verbose=True)
    >>> 
    >>> # Make predictions
    >>> test_input = np.random.rand(1, 784).astype(np.float32)
    >>> prediction = network.predict(test_input)
    >>> print(f"Prediction: {prediction}")

Classes:
    NeuralNetwork: Main neural network class with training and inference capabilities
    TrainingData: Container for training datasets with utilities
    TrainingResults: Results and metrics from training sessions

Modules:
    utils: Utility functions for data processing and mathematics
    benchmarks: Performance benchmarking utilities
"""

from .fann_rust_core import (
    # Core classes
    NeuralNetwork,
    TrainingData, 
    TrainingResults,
    
    # Utility functions
    get_library_info,
    get_simd_features,
    initialize,
    
    # Submodules
    utils,
    benchmarks,
    
    # Version information
    __version__,
    __name__ as __lib_name__,
    __description__,
)

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import warnings

# Library metadata
__author__ = "Neural Swarm Team"
__email__ = "neural-swarm@example.com"
__license__ = "MIT OR Apache-2.0"
__url__ = "https://github.com/neural-swarm/fann-rust-core"

# Version information from Rust
__version__ = __version__
__lib_description__ = __description__

# Export main classes and functions
__all__ = [
    # Core classes
    "NeuralNetwork",
    "TrainingData",
    "TrainingResults",
    
    # Utility functions  
    "get_library_info",
    "get_simd_features",
    "initialize",
    "create_network",
    "load_network",
    
    # Submodules
    "utils",
    "benchmarks",
    
    # Helper functions
    "check_environment",
    "get_performance_info",
]

def create_network(
    layers: List[int],
    activation: str = "relu",
    output_activation: Optional[str] = None,
    **kwargs
) -> NeuralNetwork:
    """
    Create a new neural network with the specified architecture.
    
    Args:
        layers: List of layer sizes (including input and output layers)
        activation: Activation function for hidden layers ("relu", "sigmoid", "tanh", etc.)
        output_activation: Activation function for output layer (defaults to activation)
        **kwargs: Additional arguments passed to NeuralNetwork constructor
        
    Returns:
        NeuralNetwork: Configured neural network ready for training
        
    Example:
        >>> network = create_network([784, 128, 10], activation="relu", output_activation="softmax")
        >>> print(network.architecture())
    """
    try:
        return NeuralNetwork(layers, activation, output_activation, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create network: {e}")

def load_network(path: str) -> NeuralNetwork:
    """
    Load a pre-trained neural network from file.
    
    Args:
        path: Path to the saved network file
        
    Returns:
        NeuralNetwork: Loaded neural network
        
    Example:
        >>> network = load_network("my_model.json")
        >>> prediction = network.predict(test_data)
    """
    try:
        return NeuralNetwork.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load network from {path}: {e}")

def check_environment() -> Dict[str, Any]:
    """
    Check the current environment and library capabilities.
    
    Returns:
        Dictionary with environment information including:
        - Library version and features
        - Available SIMD instructions
        - Thread count
        - System capabilities
        
    Example:
        >>> env_info = check_environment()
        >>> print(f"SIMD support: {env_info['simd_features']}")
        >>> print(f"Thread count: {env_info['thread_count']}")
    """
    try:
        lib_info = get_library_info()
        simd_features = get_simd_features()
        
        return {
            "library_version": lib_info["version"],
            "library_name": lib_info["name"],
            "library_description": lib_info["description"],
            "enabled_features": lib_info["features"],
            "simd_features": simd_features,
            "thread_count": lib_info["thread_count"],
            "numpy_version": np.__version__,
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
        }
    except Exception as e:
        warnings.warn(f"Failed to get complete environment info: {e}")
        return {"error": str(e)}

def get_performance_info() -> Dict[str, Any]:
    """
    Get performance information about the current system.
    
    Returns:
        Dictionary with performance metrics including:
        - Available SIMD features
        - Benchmark results for key operations
        - Memory usage information
        
    Example:
        >>> perf_info = get_performance_info()
        >>> print(f"Matrix multiply performance: {perf_info['matrix_multiply_benchmark']}")
    """
    try:
        perf_info = {}
        
        # Get SIMD capabilities
        perf_info["simd_features"] = get_simd_features()
        
        # Run quick benchmarks
        try:
            matrix_bench = benchmarks.benchmark_matrix_multiply(size=128, iterations=10)
            perf_info["matrix_multiply_benchmark"] = matrix_bench
        except Exception as e:
            perf_info["matrix_multiply_benchmark"] = {"error": str(e)}
        
        try:
            activation_bench = benchmarks.benchmark_activations(size=1000, iterations=100)
            perf_info["activation_benchmark"] = activation_bench
        except Exception as e:
            perf_info["activation_benchmark"] = {"error": str(e)}
        
        return perf_info
    except Exception as e:
        return {"error": str(e)}

def _validate_numpy_array(arr: np.ndarray, name: str, expected_dtype: type = np.float32) -> np.ndarray:
    """
    Validate and convert NumPy array to expected format.
    
    Args:
        arr: Input array to validate
        name: Name of the array for error messages
        expected_dtype: Expected data type
        
    Returns:
        Validated and potentially converted array
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(arr)}")
    
    if arr.dtype != expected_dtype:
        warnings.warn(f"{name} will be converted from {arr.dtype} to {expected_dtype}")
        arr = arr.astype(expected_dtype)
    
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    
    return arr

# Monkey patch validation helpers to main classes
def _patch_neural_network():
    """Add Python-specific enhancements to NeuralNetwork class."""
    
    original_predict = NeuralNetwork.predict
    original_train = NeuralNetwork.train
    original_evaluate = NeuralNetwork.evaluate
    
    def enhanced_predict(self, input_data: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Enhanced predict method with input validation."""
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32)
        
        input_data = _validate_numpy_array(input_data, "input_data")
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        elif input_data.ndim != 2:
            raise ValueError(f"Input must be 1D or 2D array, got {input_data.ndim}D")
        
        if input_data.shape[1] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {input_data.shape[1]}")
        
        if input_data.shape[0] == 1:
            return original_predict(self, input_data[0])
        else:
            # Batch prediction
            results = []
            for i in range(input_data.shape[0]):
                result = original_predict(self, input_data[i])
                results.append(result)
            return np.array(results)
    
    def enhanced_train(self, inputs: Union[np.ndarray, List[List[float]]], 
                      targets: Union[np.ndarray, List[List[float]]], **kwargs) -> TrainingResults:
        """Enhanced train method with input validation."""
        if isinstance(inputs, list):
            inputs = np.array(inputs, dtype=np.float32)
        if isinstance(targets, list):
            targets = np.array(targets, dtype=np.float32)
        
        inputs = _validate_numpy_array(inputs, "inputs")
        targets = _validate_numpy_array(targets, "targets")
        
        if inputs.ndim != 2 or targets.ndim != 2:
            raise ValueError("Inputs and targets must be 2D arrays")
        
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"Number of input samples ({inputs.shape[0]}) must match number of target samples ({targets.shape[0]})")
        
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {inputs.shape[1]}")
        
        if targets.shape[1] != self.output_size:
            raise ValueError(f"Output size mismatch: expected {self.output_size}, got {targets.shape[1]}")
        
        return original_train(self, inputs, targets, **kwargs)
    
    def enhanced_evaluate(self, inputs: Union[np.ndarray, List[List[float]]], 
                         targets: Union[np.ndarray, List[List[float]]]) -> Dict[str, float]:
        """Enhanced evaluate method with input validation."""
        if isinstance(inputs, list):
            inputs = np.array(inputs, dtype=np.float32)
        if isinstance(targets, list):
            targets = np.array(targets, dtype=np.float32)
        
        inputs = _validate_numpy_array(inputs, "inputs")
        targets = _validate_numpy_array(targets, "targets")
        
        return original_evaluate(self, inputs, targets)
    
    # Apply patches
    NeuralNetwork.predict = enhanced_predict
    NeuralNetwork.train = enhanced_train
    NeuralNetwork.evaluate = enhanced_evaluate

# Apply patches when module is imported
try:
    _patch_neural_network()
except Exception as e:
    warnings.warn(f"Failed to apply Python enhancements: {e}")

# Initialize library on import (with error handling)
try:
    initialize()
    print(f"FANN Rust Core {__version__} initialized successfully")
    
    # Show capabilities if in interactive mode
    if hasattr(__import__('sys'), 'ps1'):  # Interactive mode
        env_info = check_environment()
        if 'simd_features' in env_info and env_info['simd_features']:
            print(f"SIMD support: {', '.join(env_info['simd_features'])}")
        else:
            print("SIMD support: None detected")
        print(f"Thread count: {env_info.get('thread_count', 'Unknown')}")
        
except Exception as e:
    warnings.warn(f"Failed to initialize FANN Rust Core: {e}")

# Cleanup
del np, warnings, List, Tuple, Optional, Union, Dict, Any