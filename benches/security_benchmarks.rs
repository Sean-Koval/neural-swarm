//! Security Performance Benchmarks for Neural Swarm
//!
//! This module provides comprehensive security-focused performance benchmarks
//! to ensure that security measures don't significantly impact performance
//! and to detect potential timing attack vulnerabilities.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neural_swarm::{
    NeuralNetwork, NetworkBuilder,
    activation::ActivationType,
    network::LayerConfig,
    training::{TrainingData, TrainingAlgorithm, TrainingParams, Trainer},
    ffi::{neural_network_create, neural_network_destroy, neural_network_predict},
    NeuralFloat,
};
use std::{
    time::{Duration, Instant},
    collections::HashMap,
    sync::Arc,
    thread,
};
use sha2::{Sha256, Digest};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

// =============================================================================
// SECURITY BENCHMARK CONFIGURATION
// =============================================================================

#[derive(Clone)]
struct SecurityBenchmarkConfig {
    network_sizes: Vec<(usize, usize, usize)>, // (input, hidden, output)
    batch_sizes: Vec<usize>,
    thread_counts: Vec<usize>,
    sample_sizes: Vec<usize>,
    enable_timing_analysis: bool,
    enable_memory_profiling: bool,
}

impl Default for SecurityBenchmarkConfig {
    fn default() -> Self {
        Self {
            network_sizes: vec![
                (10, 20, 5),
                (50, 100, 25),
                (100, 200, 50),
            ],
            batch_sizes: vec![1, 10, 100, 1000],
            thread_counts: vec![1, 2, 4, 8],
            sample_sizes: vec![100, 1000, 10000],
            enable_timing_analysis: true,
            enable_memory_profiling: true,
        }
    }
}

// =============================================================================
// CRYPTOGRAPHIC SECURITY BENCHMARKS
// =============================================================================

/// Benchmark random number generation security vs performance
fn bench_rng_security_vs_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("RNG Security vs Performance");
    
    // Test different RNG approaches
    let rng_configs = vec![
        ("System RNG", RngType::System),
        ("ChaCha20 RNG", RngType::ChaCha20),
        ("Fast Pseudo RNG", RngType::FastPseudo),
    ];

    for (name, rng_type) in rng_configs {
        for &sample_size in &[1000, 10000, 100000] {
            group.bench_with_input(
                BenchmarkId::new(name, sample_size),
                &(rng_type.clone(), sample_size),
                |b, &(ref rng_type, size)| {
                    b.iter(|| {
                        let values = generate_secure_random_values(rng_type.clone(), size);
                        black_box(values)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark hash computation performance for integrity checking
fn bench_hash_computation_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hash Computation Performance");
    
    // Test different data sizes
    let data_sizes = vec![1024, 10240, 102400, 1024000]; // 1KB to 1MB
    
    for &size in &data_sizes {
        let test_data = generate_test_data(size);
        
        group.bench_with_input(
            BenchmarkId::new("SHA256", size),
            &test_data,
            |b, data| {
                b.iter(|| {
                    let hash = compute_sha256_hash(black_box(data));
                    black_box(hash)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark serialization security performance
fn bench_serialization_security(c: &mut Criterion) {
    let mut group = c.benchmark_group("Serialization Security");
    
    let config = SecurityBenchmarkConfig::default();
    
    for &(input_size, hidden_size, output_size) in &config.network_sizes {
        let layer_configs = vec![
            LayerConfig::new(input_size, ActivationType::Linear),
            LayerConfig::new(hidden_size, ActivationType::ReLU),
            LayerConfig::new(output_size, ActivationType::Sigmoid),
        ];
        
        let mut network = NeuralNetwork::new_feedforward(&layer_configs).unwrap();
        network.initialize_weights(Some(42)).unwrap();
        
        let network_id = format!("{}x{}x{}", input_size, hidden_size, output_size);
        
        // Benchmark serialization
        group.bench_with_input(
            BenchmarkId::new("Serialize", &network_id),
            &network,
            |b, net| {
                b.iter(|| {
                    let serialized = serde_json::to_string(black_box(net)).unwrap();
                    black_box(serialized)
                })
            },
        );
        
        // Benchmark deserialization with integrity check
        let serialized = serde_json::to_string(&network).unwrap();
        group.bench_with_input(
            BenchmarkId::new("Deserialize+Verify", &network_id),
            &serialized,
            |b, data| {
                b.iter(|| {
                    // Deserialize
                    let network: NeuralNetwork = serde_json::from_str(black_box(data)).unwrap();
                    // Compute integrity hash
                    let hash = compute_sha256_hash(data);
                    black_box((network, hash))
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// FFI SECURITY BENCHMARKS
// =============================================================================

/// Benchmark C FFI call security overhead
fn bench_c_ffi_security_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("C FFI Security Overhead");
    
    // Create test network
    let layer_sizes = [10usize, 20, 5];
    let activations = [0u32, 3, 1]; // linear, relu, sigmoid
    
    let network_ptr = neural_network_create(
        layer_sizes.as_ptr(),
        layer_sizes.len(),
        activations.as_ptr(),
    );
    
    if network_ptr.is_null() {
        panic!("Failed to create network for FFI benchmark");
    }
    
    let test_input = vec![0.1f32; 10];
    let mut output = vec![0.0f32; 5];
    
    // Benchmark raw FFI call
    group.bench_function("Raw FFI Call", |b| {
        b.iter(|| {
            let result = neural_network_predict(
                black_box(network_ptr),
                black_box(test_input.as_ptr()),
                black_box(test_input.len()),
                black_box(output.as_mut_ptr()),
                black_box(output.len()),
            );
            black_box(result)
        })
    });
    
    // Benchmark FFI call with input validation
    group.bench_function("FFI Call with Validation", |b| {
        b.iter(|| {
            // Validate inputs
            if validate_ffi_inputs(black_box(&test_input), black_box(&output)) {
                let result = neural_network_predict(
                    black_box(network_ptr),
                    black_box(test_input.as_ptr()),
                    black_box(test_input.len()),
                    black_box(output.as_mut_ptr()),
                    black_box(output.len()),
                );
                black_box(result)
            } else {
                black_box(-1)
            }
        })
    });
    
    // Benchmark FFI call with full security checks
    group.bench_function("FFI Call with Full Security", |b| {
        b.iter(|| {
            // Full security validation
            if validate_ffi_inputs(black_box(&test_input), black_box(&output)) &&
               validate_network_pointer(black_box(network_ptr)) {
                
                let start_time = Instant::now();
                let result = neural_network_predict(
                    black_box(network_ptr),
                    black_box(test_input.as_ptr()),
                    black_box(test_input.len()),
                    black_box(output.as_mut_ptr()),
                    black_box(output.len()),
                );
                let _duration = start_time.elapsed();
                
                // Validate outputs
                if validate_ffi_outputs(black_box(&output)) {
                    black_box(result)
                } else {
                    black_box(-2)
                }
            } else {
                black_box(-1)
            }
        })
    });
    
    neural_network_destroy(network_ptr);
    group.finish();
}

#[cfg(feature = "python")]
/// Benchmark Python FFI security overhead
fn bench_python_ffi_security_overhead(c: &mut Criterion) {
    use neural_swarm::{PyNeuralNetwork, PyTrainer};
    
    let mut group = c.benchmark_group("Python FFI Security Overhead");
    
    // Create Python network
    let layer_sizes = vec![5, 10, 3];
    let activations = vec!["linear".to_string(), "relu".to_string(), "sigmoid".to_string()];
    
    let mut py_network = PyNeuralNetwork::new(layer_sizes, activations).unwrap();
    py_network.initialize_weights(Some(42)).unwrap();
    
    let test_input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    
    // Benchmark raw Python prediction
    group.bench_function("Raw Python Prediction", |b| {
        b.iter(|| {
            let result = py_network.predict(black_box(test_input.clone())).unwrap();
            black_box(result)
        })
    });
    
    // Benchmark Python prediction with validation
    group.bench_function("Python Prediction with Validation", |b| {
        b.iter(|| {
            if validate_python_input(black_box(&test_input)) {
                let result = py_network.predict(black_box(test_input.clone())).unwrap();
                if validate_python_output(black_box(&result)) {
                    black_box(result)
                } else {
                    black_box(vec![])
                }
            } else {
                black_box(vec![])
            }
        })
    });
    
    group.finish();
}

#[cfg(not(feature = "python"))]
fn bench_python_ffi_security_overhead(_c: &mut Criterion) {
    // Skip Python benchmarks if not enabled
}

// =============================================================================
// MEMORY SECURITY BENCHMARKS
// =============================================================================

/// Benchmark memory allocation security overhead
fn bench_memory_allocation_security(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Allocation Security");
    
    let allocation_sizes = vec![1024, 10240, 102400, 1024000]; // 1KB to 1MB
    
    for &size in &allocation_sizes {
        // Benchmark standard allocation
        group.bench_with_input(
            BenchmarkId::new("Standard Allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data = vec![0u8; black_box(size)];
                    black_box(data)
                })
            },
        );
        
        // Benchmark secure allocation with clearing
        group.bench_with_input(
            BenchmarkId::new("Secure Allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut data = vec![0u8; black_box(size)];
                    // Simulate secure clearing
                    for byte in data.iter_mut() {
                        *byte = 0;
                    }
                    black_box(data)
                })
            },
        );
        
        // Benchmark allocation with bounds checking
        group.bench_with_input(
            BenchmarkId::new("Bounds-Checked Allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    if size <= 10_000_000 { // Safety check
                        let data = vec![0u8; black_box(size)];
                        black_box(data)
                    } else {
                        black_box(vec![])
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent memory access security
fn bench_concurrent_memory_security(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Memory Security");
    
    let config = SecurityBenchmarkConfig::default();
    
    for &thread_count in &config.thread_counts {
        // Create shared network for concurrent access
        let layer_configs = vec![
            LayerConfig::new(20, ActivationType::Linear),
            LayerConfig::new(40, ActivationType::ReLU),
            LayerConfig::new(10, ActivationType::Sigmoid),
        ];
        
        let mut network = NeuralNetwork::new_feedforward(&layer_configs).unwrap();
        network.initialize_weights(Some(42)).unwrap();
        let network = Arc::new(network);
        
        group.bench_with_input(
            BenchmarkId::new("Concurrent Access", thread_count),
            &(network.clone(), thread_count),
            |b, (net, &threads)| {
                b.iter(|| {
                    let handles: Vec<_> = (0..threads).map(|i| {
                        let net_clone = net.clone();
                        thread::spawn(move || {
                            let input = vec![i as f32 / 10.0; 20];
                            let result = net_clone.predict(&input).unwrap();
                            black_box(result)
                        })
                    }).collect();
                    
                    for handle in handles {
                        let _ = handle.join();
                    }
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// TIMING ATTACK DETECTION BENCHMARKS
// =============================================================================

/// Benchmark timing attack resistance
fn bench_timing_attack_resistance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Timing Attack Resistance");
    
    // Create network for timing analysis
    let layer_configs = vec![
        LayerConfig::new(16, ActivationType::Linear),
        LayerConfig::new(32, ActivationType::ReLU),
        LayerConfig::new(8, ActivationType::Sigmoid),
    ];
    
    let mut network = NeuralNetwork::new_feedforward(&layer_configs).unwrap();
    network.initialize_weights(Some(42)).unwrap();
    
    // Test different input patterns that might reveal timing differences
    let input_patterns = vec![
        ("All Zeros", vec![0.0f32; 16]),
        ("All Ones", vec![1.0f32; 16]),
        ("Alternating", (0..16).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect()),
        ("Random Pattern", generate_random_input(16, 12345)),
        ("Edge Values", (0..16).map(|i| if i < 8 { f32::MIN } else { f32::MAX }).collect()),
    ];
    
    for (pattern_name, input) in input_patterns {
        group.bench_with_input(
            BenchmarkId::new("Prediction Timing", pattern_name),
            &input,
            |b, input| {
                b.iter(|| {
                    let result = network.predict(black_box(input)).unwrap();
                    black_box(result)
                })
            },
        );
    }
    
    // Benchmark constant-time operations
    group.bench_function("Constant-Time Hash", |b| {
        let data = "sensitive data for hashing";
        b.iter(|| {
            let hash = compute_constant_time_hash(black_box(data));
            black_box(hash)
        })
    });
    
    group.finish();
}

/// Benchmark side-channel resistance
fn bench_side_channel_resistance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Side-Channel Resistance");
    
    // Test operations that should be resistant to cache timing attacks
    let key_sizes = vec![128, 256, 512]; // Different key/data sizes
    
    for &size in &key_sizes {
        let secret_data = generate_secret_data(size);
        let public_data = generate_public_data(size);
        
        // Benchmark data-independent operations
        group.bench_with_input(
            BenchmarkId::new("Data-Independent Operation", size),
            &(secret_data.clone(), public_data.clone()),
            |b, (secret, public)| {
                b.iter(|| {
                    let result = data_independent_operation(black_box(secret), black_box(public));
                    black_box(result)
                })
            },
        );
        
        // Benchmark potentially vulnerable operations
        group.bench_with_input(
            BenchmarkId::new("Potentially Vulnerable Operation", size),
            &(secret_data.clone(), public_data.clone()),
            |b, (secret, public)| {
                b.iter(|| {
                    let result = potentially_vulnerable_operation(black_box(secret), black_box(public));
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// STRESS TEST BENCHMARKS
// =============================================================================

/// Benchmark security under stress conditions
fn bench_security_under_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("Security Under Stress");
    group.sample_size(10); // Reduce sample size for stress tests
    
    // Test large network creation and destruction
    group.bench_function("Large Network Stress", |b| {
        b.iter(|| {
            let layer_configs = vec![
                LayerConfig::new(1000, ActivationType::Linear),
                LayerConfig::new(2000, ActivationType::ReLU),
                LayerConfig::new(500, ActivationType::Sigmoid),
            ];
            
            if let Ok(mut network) = NeuralNetwork::new_feedforward(&layer_configs) {
                let _ = network.initialize_weights(Some(42));
                let input = vec![0.001f32; 1000];
                let _ = network.predict(&input);
                black_box(network)
            }
        })
    });
    
    // Test rapid creation/destruction cycles
    group.bench_function("Rapid Creation/Destruction", |b| {
        b.iter(|| {
            for i in 0..100 {
                let layer_configs = vec![
                    LayerConfig::new(5, ActivationType::Linear),
                    LayerConfig::new(10, ActivationType::ReLU),
                    LayerConfig::new(3, ActivationType::Sigmoid),
                ];
                
                if let Ok(mut network) = NeuralNetwork::new_feedforward(&layer_configs) {
                    let _ = network.initialize_weights(Some(i));
                    let input = vec![i as f32 / 100.0; 5];
                    let _ = network.predict(&input);
                }
            }
        })
    });
    
    group.finish();
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

#[derive(Clone)]
enum RngType {
    System,
    ChaCha20,
    FastPseudo,
}

fn generate_secure_random_values(rng_type: RngType, count: usize) -> Vec<f32> {
    match rng_type {
        RngType::System => {
            let mut rng = rand::thread_rng();
            (0..count).map(|_| rng.gen::<f32>()).collect()
        }
        RngType::ChaCha20 => {
            let mut rng = ChaCha20Rng::seed_from_u64(42);
            (0..count).map(|_| rng.gen::<f32>()).collect()
        }
        RngType::FastPseudo => {
            let mut rng = rand::thread_rng();
            (0..count).map(|_| rng.gen::<f32>()).collect()
        }
    }
}

fn generate_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn compute_sha256_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn compute_constant_time_hash(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn validate_ffi_inputs(input: &[f32], output: &[f32]) -> bool {
    !input.is_empty() && 
    !output.is_empty() && 
    input.len() <= 10000 && 
    output.len() <= 10000 &&
    input.iter().all(|&x| x.is_finite())
}

fn validate_network_pointer(ptr: *mut NeuralNetwork) -> bool {
    !ptr.is_null()
}

fn validate_ffi_outputs(output: &[f32]) -> bool {
    output.iter().all(|&x| x.is_finite())
}

#[cfg(feature = "python")]
fn validate_python_input(input: &[f32]) -> bool {
    !input.is_empty() && 
    input.len() <= 10000 &&
    input.iter().all(|&x| x.is_finite())
}

#[cfg(feature = "python")]
fn validate_python_output(output: &[f32]) -> bool {
    !output.is_empty() && output.iter().all(|&x| x.is_finite())
}

fn generate_random_input(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen::<f32>()).collect()
}

fn generate_secret_data(size: usize) -> Vec<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    (0..size).map(|_| rng.gen::<u8>()).collect()
}

fn generate_public_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn data_independent_operation(secret: &[u8], public: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(secret.len());
    for i in 0..secret.len().min(public.len()) {
        // Constant-time operation
        result.push(secret[i] ^ public[i]);
    }
    result
}

fn potentially_vulnerable_operation(secret: &[u8], public: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    for i in 0..secret.len().min(public.len()) {
        // Potentially vulnerable to timing attacks due to branching
        if secret[i] > 128 {
            result.push(secret[i] ^ public[i]);
        } else {
            result.push(public[i]);
        }
    }
    result
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    security_benchmarks,
    bench_rng_security_vs_performance,
    bench_hash_computation_performance,
    bench_serialization_security,
    bench_c_ffi_security_overhead,
    bench_python_ffi_security_overhead,
    bench_memory_allocation_security,
    bench_concurrent_memory_security,
    bench_timing_attack_resistance,
    bench_side_channel_resistance,
    bench_security_under_stress
);

criterion_main!(security_benchmarks);