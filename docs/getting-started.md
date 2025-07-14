# Getting Started with FANN-Rust-Core

## Overview

FANN-Rust-Core is a high-performance neural network library written in Rust, providing both standalone functionality and seamless integration with neural swarm coordination systems. This guide will walk you through installation, basic usage, and advanced features.

## Installation

### Prerequisites

- **Rust**: 1.70.0 or later
- **Python**: 3.8+ (for Python bindings)
- **C Compiler**: GCC or Clang (for C API compatibility)

### System Requirements

| Platform | Architecture | SIMD Support |
|----------|-------------|--------------|
| Linux    | x86_64, ARM64 | AVX2, AVX-512, NEON |
| Windows  | x86_64 | AVX2, AVX-512 |
| macOS    | x86_64, ARM64 | AVX2, NEON |
| WebAssembly | wasm32 | Limited |

### Rust Installation

#### From Source

```bash
# Clone the repository
git clone https://github.com/neural-swarm/fann-rust-core
cd fann-rust-core

# Build in release mode
cargo build --release

# Run tests
cargo test

# Install locally
cargo install --path .
```

#### Using Cargo

Add to your `Cargo.toml`:

```toml
[dependencies]
fann-rust-core = "0.1.0"

# Optional features
fann-rust-core = { version = "0.1.0", features = ["simd", "quantization", "swarm"] }
```

### Python Installation

```bash
# Install from PyPI
pip install fann-rust-core

# Or build from source
pip install maturin
git clone https://github.com/neural-swarm/fann-rust-core
cd fann-rust-core/python
maturin develop --release
```

### Feature Flags

Enable optional features during compilation:

```toml
[dependencies]
fann-rust-core = { version = "0.1.0", features = [
    "simd",           # SIMD optimizations
    "quantization",   # Model quantization
    "parallel",       # Parallel training
    "swarm",          # Swarm integration
    "gpu",            # GPU acceleration (experimental)
    "python",         # Python bindings
    "c-api",          # C API compatibility
] }
```

## Quick Start

### Rust: Basic Neural Network

```rust
use fann_rust_core::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple neural network
    let mut network = NetworkBuilder::new()
        .layers(&[2, 4, 1])  // 2 inputs, 4 hidden, 1 output
        .activation(ActivationFunction::Sigmoid)
        .build()?;

    // Prepare training data (XOR problem)
    let training_data = TrainingData::new(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
        ],
    )?;

    // Train the network
    let results = network.train(&training_data, TrainingConfig::default())?;
    println!("Training completed with error: {:.6}", results.final_error);

    // Test the network
    let output = network.forward(&[1.0, 0.0])?;
    println!("1 XOR 0 = {:.3}", output[0]);

    Ok(())
}
```

### Python: Basic Usage

```python
import fann_rust_core as fann
import numpy as np

# Create network
network = fann.NeuralNetwork([2, 4, 1])

# Training data (XOR)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# Train
error = network.train(X, y, epochs=1000)
print(f"Training error: {error}")

# Predict
result = network.predict([1, 0])
print(f"1 XOR 0 = {result[0]:.3f}")
```

## Core Concepts

### Neural Network Architecture

FANN-Rust-Core supports various network architectures:

```rust
// Feedforward network
let feedforward = NetworkBuilder::new()
    .layers(&[784, 256, 128, 10])
    .build()?;

// Custom layer configuration
let custom = NetworkBuilder::new()
    .add_layer(LayerConfig::dense(784))
    .add_layer(LayerConfig::dense(256)
        .activation(ActivationFunction::ReLU)
        .dropout(0.3))
    .add_layer(LayerConfig::dense(10)
        .activation(ActivationFunction::Softmax))
    .build()?;
```

### Activation Functions

Supported activation functions with SIMD optimization:

```rust
// Built-in functions
ActivationFunction::ReLU
ActivationFunction::Sigmoid
ActivationFunction::Tanh
ActivationFunction::Softmax
ActivationFunction::LeakyReLU(alpha)
ActivationFunction::ELU(alpha)
ActivationFunction::Swish

// Custom activation function
#[derive(Clone)]
struct CustomActivation;

impl ActivationFunction for CustomActivation {
    fn compute(&self, x: f32) -> f32 {
        // Your custom function
        x.max(0.0) // ReLU example
    }
    
    fn derivative(&self, x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
```

### Training Configuration

Comprehensive training options:

```rust
let config = TrainingConfig {
    epochs: 1000,
    learning_rate: 0.001,
    batch_size: 32,
    validation_split: 0.2,
    
    // Early stopping
    early_stopping: Some(EarlyStoppingConfig {
        patience: 10,
        min_delta: 0.001,
    }),
    
    // Learning rate scheduling
    lr_schedule: Some(LRSchedule::StepDecay {
        step_size: 100,
        gamma: 0.9,
    }),
    
    // Regularization
    l1_regularization: 0.0001,
    l2_regularization: 0.001,
    
    // Progress monitoring
    progress_callback: Some(Box::new(|epoch, loss, accuracy| {
        if epoch % 10 == 0 {
            println!("Epoch {}: loss={:.4}, acc={:.4}", epoch, loss, accuracy);
        }
    })),
};
```

## Performance Optimization

### SIMD Acceleration

Enable SIMD for automatic vectorization:

```rust
let network = NetworkBuilder::new()
    .layers(&[784, 128, 10])
    .optimization(OptimizationConfig {
        use_simd: true,
        simd_features: SIMDFeatures::auto_detect(),
        ..Default::default()
    })
    .build()?;
```

### Parallel Training

Utilize multiple CPU cores:

```rust
let config = TrainingConfig {
    parallel_training: true,
    num_threads: num_cpus::get(),
    batch_size: 64,  // Larger batches for better parallelization
    ..Default::default()
};
```

### Memory Optimization

Configure memory usage:

```rust
let network = NetworkBuilder::new()
    .layers(&[784, 128, 10])
    .memory_config(MemoryConfig {
        use_memory_pool: true,
        pool_size: 1024 * 1024,  // 1MB
        alignment: 64,           // Cache line alignment
    })
    .build()?;
```

## Model Persistence

### Saving and Loading

```rust
// Save model
network.save_to_file("model.fann")?;
network.save_json("model.json")?;
network.save_safetensors("model.safetensors")?;

// Load model
let network1 = FeedforwardNetwork::load_from_file("model.fann")?;
let network2 = FeedforwardNetwork::load_json("model.json")?;
let network3 = FeedforwardNetwork::load_safetensors("model.safetensors")?;

// Export weights only
let weights = network.get_weights();
let mut new_network = NetworkBuilder::new()
    .layers(&[784, 128, 10])
    .build()?;
new_network.set_weights(weights)?;
```

### Format Compatibility

| Format | Use Case | Compatibility |
|--------|----------|---------------|
| `.fann` | FANN compatibility | Original FANN library |
| `.json` | Human-readable | Cross-language |
| `.safetensors` | Secure tensors | Modern ML frameworks |
| Binary | Fast loading | Rust applications |

## Advanced Features

### Model Quantization

Compress models for edge deployment:

```rust
use fann_rust_core::optimization::quantization::*;

// Train full-precision model
let mut network = train_model()?;

// Quantize to 8-bit integers
let quantizer = QuantizationEngine::new();
let calibration_data = load_calibration_samples()?;

let quantized = quantizer.quantize_model(
    &network,
    &calibration_data,
    QuantizationType::Int8
)?;

// 4-10x compression with minimal accuracy loss
let compression_ratio = quantized.compression_ratio();
println!("Compression: {:.1f}x", compression_ratio);
```

### Sparse Networks

Reduce memory usage with sparsity:

```rust
use fann_rust_core::optimization::sparse::*;

let sparse_network = SparseNeuralNetwork::from_dense_network(
    &dense_network,
    0.01  // Sparsity threshold
)?;

let efficiency = sparse_network.memory_efficiency();
println!("Memory saved: {:.1f}%", efficiency.sparsity_ratio * 100.0);
```

### Edge Deployment

Adaptive computation for resource-constrained environments:

```rust
use fann_rust_core::edge::*;

let adaptive_engine = AdaptiveNeuralEngine::new()?;

// Register models of different complexity levels
adaptive_engine.register_model(ComplexityLevel::Low, lightweight_model)?;
adaptive_engine.register_model(ComplexityLevel::High, full_model)?;

// Automatic model selection based on available resources
let result = adaptive_engine.adaptive_inference(
    &input_data,
    ResourceConstraints {
        memory_limit: 50_000_000,  // 50MB
        time_budget: Duration::from_millis(100),
        power_budget: Some(0.5),   // 0.5 watts
    }
).await?;
```

## Integration with Swarm Systems

### Blackboard Coordination

```rust
use fann_rust_core::swarm::*;

// Create swarm-aware neural engine
let mut swarm_engine = SwarmNeuralEngine::new(SwarmConfig {
    blackboard_url: "ws://localhost:8080/blackboard".to_string(),
    agent_id: "neural-worker-1".to_string(),
    coordination: CoordinationConfig {
        share_model_updates: true,
        distributed_training: true,
        sync_frequency: Duration::from_secs(30),
    },
}).await?;

// Register network capabilities
swarm_engine.register_network(
    "classifier".to_string(),
    Box::new(network)
).await?;

// Process coordination requests
swarm_engine.process_computation_requests().await?;
```

### Distributed Training

```rust
// Coordinate distributed training across multiple agents
let training_spec = DistributedTrainingSpec {
    model_name: "global_classifier".to_string(),
    training_data: local_data,
    sync_method: SynchronizationMethod::AllReduce,
    communication_backend: CommunicationBackend::TCP,
};

swarm_engine.coordinate_distributed_training(training_spec).await?;
```

## Debugging and Profiling

### Debug Mode

Enable detailed logging:

```rust
let network = NetworkBuilder::new()
    .layers(&[784, 128, 10])
    .debug_config(DebugConfig {
        log_level: LogLevel::Debug,
        track_gradients: true,
        save_activations: true,
        performance_profiling: true,
    })
    .build()?;
```

### Performance Profiling

```rust
use fann_rust_core::profiling::*;

let profiler = PerformanceProfiler::new();
profiler.start_profiling();

// Run operations
let result = network.forward(&input)?;

let profile = profiler.get_profile();
println!("Forward pass time: {:.2}ms", profile.forward_time_ms);
println!("Memory usage: {:.1f}MB", profile.peak_memory_mb);
```

### Benchmarking

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_inference(c: &mut Criterion) {
    let network = create_test_network();
    let input = vec![0.5; 784];
    
    c.bench_function("neural_inference", |b| {
        b.iter(|| network.forward(&input))
    });
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
```

## Error Handling

### Common Error Types

```rust
use fann_rust_core::error::*;

match network.train(&training_data, config) {
    Ok(results) => {
        println!("Training successful: {:.4}", results.final_error);
    }
    Err(NetworkError::InvalidArchitecture(msg)) => {
        eprintln!("Architecture error: {}", msg);
    }
    Err(NetworkError::TrainingError(msg)) => {
        eprintln!("Training failed: {}", msg);
    }
    Err(NetworkError::InputSizeMismatch { expected, actual }) => {
        eprintln!("Input size error: expected {}, got {}", expected, actual);
    }
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

### Validation and Safety

```rust
// Input validation
let validated_input = network.validate_input(&input_data)?;

// Safe weight updates
network.update_weights_safe(&weight_updates)?;

// Bounds checking
let output = network.forward_checked(&input)?;
```

## Next Steps

### Learning Path

1. **Start Simple**: Begin with basic feedforward networks
2. **Add Optimization**: Enable SIMD and parallel training
3. **Explore Advanced**: Try quantization and sparse networks
4. **Scale Up**: Implement swarm coordination
5. **Deploy**: Optimize for edge environments

### Resources

- [API Documentation](rust-api.md)
- [Python Bindings](python-api.md)
- [Examples Repository](../examples/)
- [Performance Benchmarks](benchmarks.md)
- [Swarm Integration Guide](swarm-integration.md)

### Community

- **GitHub**: [Issues and Discussions](https://github.com/neural-swarm/fann-rust-core)
- **Documentation**: [Online Docs](https://neural-swarm.github.io/fann-rust-core/)
- **Examples**: [Community Examples](https://github.com/neural-swarm/fann-examples)

## Troubleshooting

### Common Issues

**Build Failures**
```bash
# Update Rust toolchain
rustup update

# Install required system dependencies
sudo apt-get install build-essential cmake

# Clear cargo cache
cargo clean
```

**SIMD Not Working**
```rust
// Check CPU features
let features = fann_rust_core::simd::detect_cpu_features();
println!("Available SIMD: {:?}", features);

// Force enable specific features
let config = OptimizationConfig {
    use_simd: true,
    force_simd_features: Some(vec!["avx2"]),
    ..Default::default()
};
```

**Memory Issues**
```rust
// Reduce memory usage
let config = TrainingConfig {
    batch_size: 16,  // Smaller batches
    gradient_accumulation: 4,  // Accumulate gradients
    ..Default::default()
};
```

**Performance Problems**
```bash
# Profile the application
cargo flamegraph --bin your_app

# Enable optimizations
export RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

This getting started guide provides a comprehensive introduction to FANN-Rust-Core. Continue with the specific API documentation and examples to dive deeper into advanced features.