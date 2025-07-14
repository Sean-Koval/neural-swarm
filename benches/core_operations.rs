use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use fann_rust_core::{
    matrix::{MatrixMultiplier, SIMDMatrixMultiplier, ScalarMatrixMultiplier},
    activations::{ActivationFunction, SIMDActivations, ScalarActivations},
    memory::{AlignedVec, MemoryPool},
};
use std::time::Duration;

/// Benchmark matrix multiplication operations with different implementations
fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    // Test different matrix sizes
    let sizes = vec![
        (64, 64, 64),     // Small matrices
        (128, 128, 128),  // Medium matrices
        (256, 256, 256),  // Large matrices
        (512, 512, 512),  // Very large matrices
        (784, 128, 64),   // MNIST-like dimensions
        (1024, 512, 256), // Deep network dimensions
    ];
    
    for (m, n, k) in sizes {
        let operations = 2 * m * n * k; // Multiply-add operations
        group.throughput(Throughput::Elements(operations as u64));
        
        // Generate test data
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        
        // Benchmark scalar implementation
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |bench, &(m, n, k)| {
                let scalar_multiplier = ScalarMatrixMultiplier::new();
                bench.iter_batched(
                    || {
                        let a = vec![1.0f32; m * k];
                        let b = vec![1.0f32; k * n];
                        let c = vec![0.0f32; m * n];
                        (a, b, c)
                    },
                    |(a, b, mut c)| {
                        scalar_multiplier.multiply(
                            black_box(&a),
                            black_box(&b),
                            black_box(&mut c),
                            m, n, k
                        )
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        
        // Benchmark SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |bench, &(m, n, k)| {
                let simd_multiplier = SIMDMatrixMultiplier::new();
                bench.iter_batched(
                    || {
                        let a = AlignedVec::<f32>::new_zeroed(m * k, 32);
                        let b = AlignedVec::<f32>::new_zeroed(k * n, 32);
                        let c = AlignedVec::<f32>::new_zeroed(m * n, 32);
                        (a, b, c)
                    },
                    |(a, b, mut c)| {
                        simd_multiplier.multiply(
                            black_box(&a),
                            black_box(&b),
                            black_box(&mut c),
                            m, n, k
                        )
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        
        // Benchmark blocked SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("simd_blocked", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |bench, &(m, n, k)| {
                let simd_multiplier = SIMDMatrixMultiplier::new();
                bench.iter_batched(
                    || {
                        let a = AlignedVec::<f32>::new_zeroed(m * k, 32);
                        let b = AlignedVec::<f32>::new_zeroed(k * n, 32);
                        let c = AlignedVec::<f32>::new_zeroed(m * n, 32);
                        (a, b, c)
                    },
                    |(a, b, mut c)| {
                        simd_multiplier.multiply_blocked(
                            black_box(&a),
                            black_box(&b),
                            black_box(&mut c),
                            m, n, k,
                            64 // Block size
                        )
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark activation functions with different implementations
fn benchmark_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");
    
    let sizes = vec![1024, 4096, 16384, 65536, 262144];
    let functions = vec!["relu", "sigmoid", "tanh", "gelu", "swish"];
    
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        let input = vec![0.5f32; size];
        
        for function in &functions {
            // Benchmark scalar implementation
            group.bench_with_input(
                BenchmarkId::new(format!("{}_scalar", function), size),
                &size,
                |bench, &size| {
                    let activations = ScalarActivations::new();
                    bench.iter_batched(
                        || {
                            let input = vec![0.5f32; size];
                            let output = vec![0.0f32; size];
                            (input, output)
                        },
                        |(input, mut output)| {
                            match function.as_str() {
                                "relu" => activations.relu(black_box(&input), black_box(&mut output)),
                                "sigmoid" => activations.sigmoid(black_box(&input), black_box(&mut output)),
                                "tanh" => activations.tanh(black_box(&input), black_box(&mut output)),
                                "gelu" => activations.gelu(black_box(&input), black_box(&mut output)),
                                "swish" => activations.swish(black_box(&input), black_box(&mut output)),
                                _ => panic!("Invalid SIMD configuration"),
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Benchmark SIMD implementation
            group.bench_with_input(
                BenchmarkId::new(format!("{}_simd", function), size),
                &size,
                |bench, &size| {
                    let activations = SIMDActivations::new();
                    bench.iter_batched(
                        || {
                            let input = AlignedVec::<f32>::new_zeroed(size, 32);
                            let output = AlignedVec::<f32>::new_zeroed(size, 32);
                            (input, output)
                        },
                        |(input, mut output)| {
                            match function.as_str() {
                                "relu" => activations.relu(black_box(&input), black_box(&mut output)),
                                "sigmoid" => activations.sigmoid(black_box(&input), black_box(&mut output)),
                                "tanh" => activations.tanh(black_box(&input), black_box(&mut output)),
                                "gelu" => activations.gelu(black_box(&input), black_box(&mut output)),
                                "swish" => activations.swish(black_box(&input), black_box(&mut output)),
                                _ => panic!("Invalid SIMD configuration"),
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark memory allocation patterns
fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    let sizes = vec![64, 256, 1024, 4096, 16384];
    
    for &size in &sizes {
        group.throughput(Throughput::Bytes((size * 4) as u64)); // 4 bytes per f32
        
        // Benchmark standard allocation
        group.bench_with_input(
            BenchmarkId::new("std_alloc", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let _vec: Vec<f32> = black_box(vec![0.0; size]);
                });
            },
        );
        
        // Benchmark aligned allocation
        group.bench_with_input(
            BenchmarkId::new("aligned_alloc", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let _vec = black_box(AlignedVec::<f32>::new_zeroed(size, 32));
                });
            },
        );
        
        // Benchmark memory pool allocation
        group.bench_with_input(
            BenchmarkId::new("pool_alloc", size),
            &size,
            |bench, &size| {
                let mut pool = MemoryPool::new(size * 4, 32);
                bench.iter(|| {
                    if let Some(ptr) = pool.allocate() {
                        black_box(ptr);
                        pool.deallocate(ptr);
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache efficiency with different access patterns
fn benchmark_cache_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_patterns");
    
    let size = 1024 * 1024; // 1M elements
    let data = vec![1.0f32; size];
    
    // Sequential access pattern
    group.bench_function("sequential_access", |bench| {
        bench.iter(|| {
            let mut sum = 0.0f32;
            for &value in black_box(&data) {
                sum += value;
            }
            black_box(sum);
        });
    });
    
    // Random access pattern
    group.bench_function("random_access", |bench| {
        let indices: Vec<usize> = (0..size).map(|i| (i * 1103515245 + 12345) % size).collect();
        bench.iter(|| {
            let mut sum = 0.0f32;
            for &idx in black_box(&indices[..1000]) { // Sample 1000 random accesses
                sum += data[idx];
            }
            black_box(sum);
        });
    });
    
    // Strided access pattern
    group.bench_function("strided_access", |bench| {
        bench.iter(|| {
            let mut sum = 0.0f32;
            let stride = 64; // Cache line size
            for i in (0..size).step_by(stride) {
                sum += data[i];
            }
            black_box(sum);
        });
    });
    
    group.finish();
}

/// Configure Criterion for comprehensive benchmarking
fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(10))  // Longer measurement for stable results
        .warm_up_time(Duration::from_secs(3))       // Adequate warmup
        .sample_size(100)                           // Good statistical sample
        .confidence_level(0.95)                     // 95% confidence intervals
        .significance_level(0.05)                   // 5% significance level
        .noise_threshold(0.02)                      // 2% noise threshold
        .with_plots()                               // Generate performance plots
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = 
        benchmark_matrix_multiplication,
        benchmark_activation_functions,
        benchmark_memory_operations,
        benchmark_cache_patterns
}

criterion_main!(benches);