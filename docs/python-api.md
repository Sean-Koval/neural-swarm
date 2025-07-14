# Python API Documentation

## Overview

The Python bindings for FANN-Rust-Core provide a Pythonic interface to the high-performance Rust neural network library. Built with PyO3, these bindings offer near-native performance while maintaining ease of use.

## Installation

```bash
pip install fann-rust-core
```

## Quick Start

```python
import fann_rust_core as fann
import numpy as np

# Create a neural network
network = fann.NeuralNetwork([784, 128, 64, 10])

# Load training data
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randn(1000, 10).astype(np.float32)

# Train the network
training_error = network.train(X_train.tolist(), y_train.tolist())
print(f"Training error: {training_error}")

# Make predictions
X_test = np.random.randn(100, 784).astype(np.float32)
predictions = network.predict_batch(X_test.tolist())
```

## Core Classes

### NeuralNetwork

The main neural network class for creating and training networks.

```python
class NeuralNetwork:
    def __init__(self, layers: List[int], **kwargs):
        """
        Create a neural network.
        
        Args:
            layers: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            **kwargs: Optional configuration parameters
        """
```

#### Constructor Parameters

```python
# Basic network
network = fann.NeuralNetwork([784, 128, 10])

# Advanced configuration
network = fann.NeuralNetwork(
    layers=[784, 256, 128, 10],
    activation='relu',
    output_activation='softmax',
    learning_rate=0.001,
    optimizer='adam',
    use_simd=True,
    quantization='int8'
)
```

#### Methods

##### train()

```python
def train(self, inputs: List[List[float]], targets: List[List[float]], 
          epochs: int = 1000, **kwargs) -> float:
    """
    Train the neural network.
    
    Args:
        inputs: Training input data
        targets: Training target data
        epochs: Number of training epochs
        **kwargs: Training configuration
        
    Returns:
        Final training error
    """

# Example usage
error = network.train(
    inputs=X_train.tolist(),
    targets=y_train.tolist(),
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    early_stopping=True,
    patience=10
)
```

##### predict()

```python
def predict(self, input_data: List[float]) -> List[float]:
    """
    Make a single prediction.
    
    Args:
        input_data: Input vector
        
    Returns:
        Output vector
    """

# Single prediction
input_vector = [0.5] * 784
output = network.predict(input_vector)
print(f"Prediction: {output}")
```

##### predict_batch()

```python
def predict_batch(self, inputs: List[List[float]]) -> List[List[float]]:
    """
    Make batch predictions.
    
    Args:
        inputs: List of input vectors
        
    Returns:
        List of output vectors
    """

# Batch prediction
inputs = [[0.5] * 784 for _ in range(100)]
outputs = network.predict_batch(inputs)
```

##### save() / load()

```python
def save(self, filepath: str, format: str = 'binary') -> None:
    """Save the network to file."""

def load(filepath: str, format: str = 'binary') -> 'NeuralNetwork':
    """Load network from file."""

# Save and load
network.save('model.fann')
loaded_network = fann.NeuralNetwork.load('model.fann')
```

## Advanced Features

### NetworkBuilder

For complex network architectures:

```python
from fann_rust_core import NetworkBuilder, LayerConfig

builder = NetworkBuilder()

# Add layers with custom configuration
builder.add_layer(LayerConfig(
    size=784,
    activation='relu',
    dropout=0.2,
    batch_normalization=True
))

builder.add_layer(LayerConfig(
    size=128,
    activation='relu',
    dropout=0.5
))

builder.add_layer(LayerConfig(
    size=10,
    activation='softmax'
))

# Configure optimizer and training
builder.set_optimizer('adam', learning_rate=0.001)
builder.set_loss_function('cross_entropy')
builder.set_regularization(l1=0.0001, l2=0.001)

# Build the network
network = builder.build()
```

### Optimization Features

#### SIMD Acceleration

```python
# Enable SIMD optimizations
network = fann.NeuralNetwork(
    layers=[784, 128, 10],
    use_simd=True,
    simd_features={
        'avx2': True,
        'avx512': False,  # Auto-detected
        'neon': True      # For ARM processors
    }
)
```

#### Quantization

```python
from fann_rust_core import QuantizationEngine

# Train a full-precision model
network = fann.NeuralNetwork([784, 128, 10])
network.train(X_train, y_train)

# Quantize the model
quantizer = QuantizationEngine()
calibration_data = X_train[:100]  # Sample for calibration

quantized_network = quantizer.quantize(
    network=network,
    calibration_data=calibration_data,
    quantization_type='int8'
)

# Use quantized model for inference
output = quantized_network.predict(test_input)
```

#### Sparse Networks

```python
from fann_rust_core import SparseNetwork

# Convert dense network to sparse
sparse_network = SparseNetwork.from_dense(
    dense_network=network,
    sparsity_threshold=0.01
)

# Check compression
stats = sparse_network.efficiency_stats()
print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
print(f"Memory reduction: {stats['compression_ratio']:.1f}x")
```

## Edge Deployment

### Adaptive Networks

```python
from fann_rust_core import AdaptiveEngine, ComplexityLevel

# Create adaptive engine
engine = AdaptiveEngine()

# Register models of different complexity
engine.register_model(ComplexityLevel.ULTRA_LOW, ultra_low_model)
engine.register_model(ComplexityLevel.MEDIUM, medium_model)
engine.register_model(ComplexityLevel.HIGH, high_model)

# Adaptive inference
result = engine.adaptive_inference(
    input_data=test_input,
    resource_constraints={
        'memory_limit': 50_000_000,  # 50MB
        'time_budget': 0.1,          # 100ms
        'power_budget': 0.5          # 0.5 watts
    }
)
```

### Power Optimization

```python
from fann_rust_core import PowerOptimizer

optimizer = PowerOptimizer()

# Energy-efficient inference
result = optimizer.energy_efficient_inference(
    input_data=test_input,
    energy_budget=0.1  # 0.1 joules
)

print(f"Energy consumed: {result['energy_used']} J")
print(f"Inference time: {result['time_taken']} ms")
```

## NumPy Integration

### Direct NumPy Support

```python
import numpy as np
import fann_rust_core as fann

# Create network
network = fann.NeuralNetwork([784, 128, 10])

# NumPy arrays are automatically converted
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randn(1000, 10).astype(np.float32)

# Train with NumPy arrays
error = network.train_numpy(X_train, y_train)

# Predict with NumPy arrays
X_test = np.random.randn(100, 784).astype(np.float32)
predictions = network.predict_numpy(X_test)
```

### Memory-Efficient Operations

```python
# Zero-copy operations where possible
input_array = np.ascontiguousarray(X_test.astype(np.float32))
output_array = network.predict_numpy_inplace(input_array)
```

## Scikit-learn Compatibility

### Estimator Interface

```python
from fann_rust_core import FANNClassifier, FANNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Classification
classifier = FANNClassifier(
    layers=[784, 128, 10],
    epochs=1000,
    learning_rate=0.001
)

# Fit the model
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Regression
regressor = FANNRegressor(
    layers=[13, 64, 32, 1],
    epochs=500
)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', FANNClassifier(layers=[784, 128, 10]))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
```

## Callbacks and Monitoring

### Training Callbacks

```python
from fann_rust_core import TrainingCallback

class CustomCallback(TrainingCallback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {logs['loss']:.4f}")
    
    def on_training_end(self, logs):
        print(f"Training completed. Final loss: {logs['final_loss']:.4f}")

# Use callback
network = fann.NeuralNetwork([784, 128, 10])
callback = CustomCallback()

network.train(
    X_train, y_train,
    callbacks=[callback],
    epochs=1000
)
```

### Performance Monitoring

```python
from fann_rust_core import PerformanceMonitor

monitor = PerformanceMonitor()

# Start monitoring
monitor.start()

# Perform operations
result = network.predict(test_input)

# Get statistics
stats = monitor.get_stats()
print(f"Inference time: {stats['inference_time']:.2f} ms")
print(f"Memory usage: {stats['memory_usage']:.1f} MB")
print(f"Energy consumption: {stats['energy']:.3f} J")
```

## Error Handling

### Exception Types

```python
from fann_rust_core import (
    NetworkError,
    TrainingError,
    OptimizationError,
    SerializationError
)

try:
    network = fann.NeuralNetwork([784, 128, 10])
    network.train(X_train, y_train)
    
except NetworkError as e:
    print(f"Network error: {e}")
    
except TrainingError as e:
    print(f"Training failed: {e}")
    
except OptimizationError as e:
    print(f"Optimization error: {e}")
```

### Validation and Debugging

```python
# Input validation
try:
    network = fann.NeuralNetwork([784, 128, 10])
    invalid_input = [0.5] * 100  # Wrong size
    result = network.predict(invalid_input)
    
except ValueError as e:
    print(f"Input size error: {e}")

# Debug mode
network = fann.NeuralNetwork(
    layers=[784, 128, 10],
    debug=True  # Enable debug output
)
```

## Distributed Training

### Multi-GPU Support

```python
from fann_rust_core import DistributedTrainer

# Configure distributed training
trainer = DistributedTrainer(
    backend='nccl',
    devices=['cuda:0', 'cuda:1'],
    strategy='data_parallel'
)

# Train across multiple GPUs
trainer.train(
    network=network,
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    epochs=1000
)
```

### Swarm Integration

```python
from fann_rust_core import SwarmClient

# Connect to neural swarm
client = SwarmClient(
    blackboard_url='ws://localhost:8080/blackboard',
    agent_id='python-neural-agent'
)

# Register network capability
client.register_network('classifier', network)

# Process coordination requests
async def process_requests():
    async for request in client.listen_for_requests():
        if request.type == 'inference':
            result = network.predict(request.input_data)
            await client.send_response(request.id, result)

# Run event loop
import asyncio
asyncio.run(process_requests())
```

## Utilities and Helpers

### Data Loading

```python
from fann_rust_core.utils import DataLoader

# Load common datasets
loader = DataLoader()

# MNIST
(X_train, y_train), (X_test, y_test) = loader.load_mnist()

# Custom CSV
data = loader.load_csv('data.csv', target_column='label')

# Image data
images, labels = loader.load_images('images/', format='auto')
```

### Preprocessing

```python
from fann_rust_core.preprocessing import Normalizer, Encoder

# Normalization
normalizer = Normalizer(method='minmax')
X_normalized = normalizer.fit_transform(X_train)

# Encoding
encoder = Encoder(method='onehot')
y_encoded = encoder.fit_transform(y_train)
```

### Visualization

```python
from fann_rust_core.visualization import NetworkVisualizer

# Visualize network architecture
visualizer = NetworkVisualizer()
visualizer.plot_architecture(network, save_path='network.png')

# Plot training history
history = network.get_training_history()
visualizer.plot_training_curves(history)

# Visualize activations
activations = network.get_layer_activations(test_input)
visualizer.plot_activations(activations)
```

## Configuration

### Global Settings

```python
import fann_rust_core as fann

# Configure global settings
fann.config.set_num_threads(8)
fann.config.set_memory_limit('1GB')
fann.config.enable_simd(True)
fann.config.set_log_level('INFO')

# Check system capabilities
caps = fann.get_system_capabilities()
print(f"SIMD support: {caps['simd']}")
print(f"GPU support: {caps['cuda']}")
print(f"Available memory: {caps['memory']} MB")
```

### Environment Variables

```python
import os

# Performance tuning
os.environ['FANN_NUM_THREADS'] = '8'
os.environ['FANN_USE_SIMD'] = '1'
os.environ['FANN_MEMORY_POOL_SIZE'] = '512MB'

# Debugging
os.environ['FANN_DEBUG'] = '1'
os.environ['FANN_LOG_LEVEL'] = 'DEBUG'
```

## Examples

### Complete MNIST Example

```python
import fann_rust_core as fann
import numpy as np
from sklearn.metrics import accuracy_score

# Load data
(X_train, y_train), (X_test, y_test) = fann.utils.load_mnist()

# Preprocess
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# One-hot encode labels
y_train_onehot = fann.utils.to_categorical(y_train, 10)
y_test_onehot = fann.utils.to_categorical(y_test, 10)

# Create network
network = fann.NeuralNetwork(
    layers=[784, 256, 128, 10],
    activation='relu',
    output_activation='softmax',
    optimizer='adam',
    learning_rate=0.001
)

# Train
error = network.train(
    X_train.reshape(-1, 784).tolist(),
    y_train_onehot.tolist(),
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    early_stopping=True
)

# Evaluate
predictions = network.predict_batch(X_test.reshape(-1, 784).tolist())
y_pred = np.argmax(predictions, axis=1)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.4f}")

# Save model
network.save('mnist_model.fann')
```

### Real-time Inference Server

```python
from flask import Flask, request, jsonify
import fann_rust_core as fann
import numpy as np

app = Flask(__name__)

# Load pre-trained model
network = fann.NeuralNetwork.load('model.fann')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array(data['input'], dtype=np.float32)
        
        prediction = network.predict(input_data.tolist())
        
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

This Python API documentation provides comprehensive coverage of the Python bindings for FANN-Rust-Core, including basic usage, advanced features, and practical examples.