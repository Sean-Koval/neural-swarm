# Rust API Documentation

## Overview

The FANN-Rust-Core library provides a comprehensive API for neural network creation, training, and inference with advanced optimization features. The API is designed with multiple layers to accommodate different use cases.

## Core Traits

### NeuralNetwork Trait

The foundational trait for all neural network implementations.

```rust
pub trait NeuralNetwork: Send + Sync {
    type Input;
    type Output;
    type Error;
    
    fn forward(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;
    fn backward(&mut self, target: &Self::Output) -> Result<f32, Self::Error>;
    fn train_epoch(&mut self, data: &TrainingSet) -> Result<TrainingStats, Self::Error>;
    fn get_weights(&self) -> &[f32];
    fn set_weights(&mut self, weights: &[f32]) -> Result<(), Self::Error>;
}
```

#### Example Usage

```rust
use fann_rust_core::prelude::*;

let mut network = FeedforwardNetwork::new(&[784, 128, 10])?;

// Forward pass
let output = network.forward(&input_data)?;

// Training
let training_data = TrainingData::new(inputs, targets)?;
let stats = network.train_epoch(&training_data)?;

println!("Training error: {}", stats.avg_error);
```

### ActivationFunction Trait

Defines activation functions with SIMD optimization support.

```rust
pub trait ActivationFunction: Send + Sync + Clone {
    fn compute(&self, x: f32) -> f32;
    fn derivative(&self, x: f32) -> f32;
    
    // SIMD-optimized versions
    fn compute_simd(&self, input: &[f32], output: &mut [f32]);
    fn derivative_simd(&self, input: &[f32], output: &mut [f32]);
}
```

#### Built-in Activation Functions

```rust
// Available activation functions
let relu = ActivationFunction::ReLU;
let sigmoid = ActivationFunction::Sigmoid;
let tanh = ActivationFunction::Tanh;
let softmax = ActivationFunction::Softmax;
let leaky_relu = ActivationFunction::LeakyReLU(0.01);

// Custom activation function
#[derive(Clone)]
struct Swish;

impl ActivationFunction for Swish {
    fn compute(&self, x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
    
    fn derivative(&self, x: f32) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        sigmoid + x * sigmoid * (1.0 - sigmoid)
    }
    
    // SIMD implementations...
}
```

## Simple API

For quick prototyping and basic use cases.

### SimpleNetwork

```rust
use fann_rust_core::simple::*;

// Create a network
let mut network = create_network(&[784, 128, 64, 10])?;

// Train with data
let error = network.train(training_inputs, training_targets)?;

// Make predictions
let prediction = network.predict(&test_input)?;

// Save and load
network.save("model.fann")?;
let loaded_network = SimpleNetwork::load("model.fann")?;
```

### Configuration Options

```rust
// Training configuration
let config = TrainingConfig {
    epochs: 1000,
    learning_rate: 0.001,
    batch_size: 32,
    validation_split: 0.2,
    early_stopping: Some(EarlyStoppingConfig {
        patience: 10,
        min_delta: 0.001,
    }),
};

let results = network.train_with_config(&training_data, config)?;
```

## Advanced API

For custom architectures and fine-grained control.

### NetworkBuilder

```rust
use fann_rust_core::advanced::*;

let network = NetworkBuilder::new()
    .add_layer(LayerConfig::dense(784)
        .activation(ActivationFunction::ReLU)
        .dropout(0.2)
        .batch_norm())
    .add_layer(LayerConfig::dense(128)
        .activation(ActivationFunction::ReLU)
        .dropout(0.5))
    .add_layer(LayerConfig::dense(10)
        .activation(ActivationFunction::Softmax))
    .optimizer(AdamOptimizer::new(0.001))
    .loss_function(CrossEntropyLoss)
    .regularization(RegularizationConfig {
        l1_weight: 0.0001,
        l2_weight: 0.001,
    })
    .optimization(OptimizationConfig {
        use_simd: true,
        parallel_training: true,
        quantization: Some(QuantizationType::Int8),
    })
    .build()?;
```

### Custom Optimizers

```rust
#[derive(Debug, Clone)]
pub struct CustomOptimizer {
    learning_rate: f32,
    momentum: f32,
}

impl Optimizer for CustomOptimizer {
    fn update_weights(&mut self, weights: &mut [f32], gradients: &[f32]) {
        for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
            *weight -= self.learning_rate * gradient;
        }
    }
    
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }
}
```

## Performance Optimization

### SIMD Operations

```rust
use fann_rust_core::optimization::simd::*;

// Enable SIMD optimizations
let config = OptimizationConfig {
    use_simd: true,
    simd_features: SIMDFeatures {
        avx2: true,
        avx512: false,  // Auto-detected
        neon: true,     // For ARM
    },
    ..Default::default()
};

let network = NetworkBuilder::new()
    .layers(&[784, 128, 10])
    .optimization(config)
    .build()?;
```

### Quantization

```rust
use fann_rust_core::optimization::quantization::*;

// Quantize a trained model
let quantizer = QuantizationEngine::new();
let calibration_data = load_calibration_data()?;

// Calibrate quantization parameters
let params = quantizer.calibrate(&network, &calibration_data)?;

// Create quantized model
let quantized_network = quantizer.quantize(&network, &params)?;

// Quantized inference
let output = quantized_network.forward_quantized(&input)?;
```

### Sparse Networks

```rust
use fann_rust_core::optimization::sparse::*;

// Convert dense network to sparse
let sparse_network = SparseNeuralNetwork::from_dense_network(
    &dense_network, 
    0.01  // sparsity threshold
)?;

// Check memory savings
let efficiency = sparse_network.memory_efficiency();
println!("Sparsity ratio: {:.2}%", efficiency.sparsity_ratio * 100.0);
println!("Memory reduction: {}x", efficiency.compression_ratio);
```

## Edge Deployment

### Adaptive Networks

```rust
use fann_rust_core::edge::*;

let adaptive_engine = AdaptiveNeuralEngine::new()?;

// Register models of different complexity
adaptive_engine.register_model(ComplexityLevel::UltraLow, ultra_low_model)?;
adaptive_engine.register_model(ComplexityLevel::Medium, medium_model)?;
adaptive_engine.register_model(ComplexityLevel::High, high_model)?;

// Adaptive inference based on available resources
let result = adaptive_engine.adaptive_inference(&input).await?;
```

### Power Optimization

```rust
use fann_rust_core::edge::power::*;

let power_engine = PowerOptimizedEngine::new()?;

// Energy-constrained inference
let energy_budget = 0.5; // joules
let result = power_engine.energy_efficient_inference(&input, energy_budget).await?;
```

## Swarm Integration

### Blackboard Coordination

```rust
use fann_rust_core::swarm::*;

let mut swarm_engine = SwarmNeuralEngine::new(SwarmConfig {
    blackboard_url: "ws://localhost:8080/blackboard".to_string(),
    coordination: CoordinationConfig {
        agent_id: "neural-agent-1".to_string(),
        share_model_updates: true,
        enable_distributed_training: true,
        coordination_frequency: Duration::from_secs(30),
    },
    performance_monitoring: true,
    energy_optimization: true,
}).await?;

// Register network capabilities
swarm_engine.register_network("classifier".to_string(), Box::new(network)).await?;

// Process coordination requests
swarm_engine.process_computation_requests().await?;
```

### Distributed Training

```rust
let training_spec = DistributedTrainingSpec {
    model_name: "mnist_classifier".to_string(),
    training_data: distributed_data,
    synchronization_method: SynchronizationMethod::AllReduce,
    communication_backend: CommunicationBackend::NCCL,
};

swarm_engine.coordinate_distributed_training(training_spec).await?;
```

## Error Handling

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("Invalid network architecture: {0}")]
    InvalidArchitecture(String),
    
    #[error("Training error: {0}")]
    TrainingError(String),
    
    #[error("Input size mismatch: expected {expected}, got {actual}")]
    InputSizeMismatch { expected: usize, actual: usize },
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[derive(Debug, thiserror::Error)]
pub enum OptimizationError {
    #[error("SIMD not supported on this platform")]
    SIMDNotSupported,
    
    #[error("Quantization failed: {0}")]
    QuantizationFailed(String),
    
    #[error("Memory allocation failed")]
    MemoryAllocationFailed,
}
```

### Error Handling Patterns

```rust
use fann_rust_core::prelude::*;

// Result handling with context
let network = NetworkBuilder::new()
    .layers(&[784, 128, 10])
    .build()
    .map_err(|e| format!("Failed to create network: {}", e))?;

// Custom error handling
match network.train(&training_data, config) {
    Ok(results) => println!("Training completed: {:.4}", results.final_error),
    Err(NetworkError::InputSizeMismatch { expected, actual }) => {
        eprintln!("Input size error: expected {}, got {}", expected, actual);
    },
    Err(e) => eprintln!("Training failed: {}", e),
}
```

## Testing and Validation

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_creation() {
        let network = NetworkBuilder::new()
            .layers(&[2, 3, 1])
            .build()
            .unwrap();
        
        assert_eq!(network.input_size(), 2);
        assert_eq!(network.output_size(), 1);
    }
    
    #[test]
    fn test_forward_pass() {
        let network = create_test_network();
        let input = vec![0.5, -0.3];
        let output = network.forward(&input).unwrap();
        
        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }
}
```

### Benchmarking

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_inference(c: &mut Criterion) {
    let network = create_benchmark_network();
    let input = vec![0.5; 784];
    
    c.bench_function("neural_inference", |b| {
        b.iter(|| network.forward(black_box(&input)))
    });
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
```

## Memory Management

### Custom Allocators

```rust
use fann_rust_core::memory::*;

// Use custom memory pool for better performance
let pool = MemoryPool::new(1024 * 1024)?; // 1MB pool
let allocator = PoolAllocator::new(pool);

let network = NetworkBuilder::new()
    .layers(&[784, 128, 10])
    .allocator(allocator)
    .build()?;
```

### Memory Monitoring

```rust
use fann_rust_core::profiling::*;

let monitor = MemoryMonitor::new();
monitor.start_monitoring();

// Perform neural network operations
let result = network.train(&training_data, config)?;

let stats = monitor.get_statistics();
println!("Peak memory usage: {} MB", stats.peak_memory_mb);
println!("Total allocations: {}", stats.allocation_count);
```

## Threading and Concurrency

### Parallel Training

```rust
use fann_rust_core::parallel::*;

let parallel_config = ParallelConfig {
    num_threads: num_cpus::get(),
    batch_processing: true,
    thread_pool_size: 8,
};

let network = NetworkBuilder::new()
    .layers(&[784, 128, 10])
    .parallel_config(parallel_config)
    .build()?;
```

### Thread Safety

```rust
use std::sync::Arc;

// Networks are Send + Sync
let network = Arc::new(create_network(&[784, 128, 10])?);

// Share across threads safely
let handles: Vec<_> = (0..4).map(|_| {
    let net = network.clone();
    std::thread::spawn(move || {
        let input = generate_random_input();
        net.forward(&input)
    })
}).collect();

for handle in handles {
    let result = handle.join().unwrap()?;
    println!("Result: {:?}", result);
}
```

## Serialization and Persistence

### Model Serialization

```rust
use fann_rust_core::serialization::*;

// Binary format (FANN compatible)
network.save_binary("model.fann")?;
let loaded = FeedforwardNetwork::load_binary("model.fann")?;

// JSON format
network.save_json("model.json")?;
let loaded = FeedforwardNetwork::load_json("model.json")?;

// SafeTensors format
network.save_safetensors("model.safetensors")?;
let loaded = FeedforwardNetwork::load_safetensors("model.safetensors")?;
```

### Custom Serialization

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct NetworkConfig {
    layers: Vec<usize>,
    activation: String,
    learning_rate: f32,
}

impl From<NetworkConfig> for NetworkBuilder {
    fn from(config: NetworkConfig) -> Self {
        NetworkBuilder::new()
            .layers(&config.layers)
            .activation(ActivationFunction::from_string(&config.activation))
            .learning_rate(config.learning_rate)
    }
}
```

This API documentation provides comprehensive coverage of the FANN-Rust-Core library's capabilities, from basic usage to advanced optimization and swarm integration features.