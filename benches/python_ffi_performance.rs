// Python FFI Performance Benchmarks
// Comprehensive analysis of cross-language call overhead and optimization

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Benchmark Python FFI call overhead
fn benchmark_ffi_call_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_call_overhead");
    
    let call_counts = vec![1, 10, 100, 1000, 10000];
    let data_sizes = vec![64, 256, 1024, 4096, 16384]; // bytes
    
    for &call_count in &call_counts {
        for &data_size in &data_sizes {
            group.throughput(Throughput::Elements(call_count as u64));
            
            // Benchmark simple function calls (no data transfer)
            group.bench_with_input(
                BenchmarkId::new("simple_calls", format!("{}calls", call_count)),
                &call_count,
                |b, &call_count| {
                    b.iter_batched(
                        || MockPythonFFI::new(),
                        |mut ffi| {
                            for i in 0..call_count {
                                black_box(ffi.simple_call(i));
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Benchmark data transfer overhead
            group.bench_with_input(
                BenchmarkId::new("data_transfer", format!("{}calls_{}bytes", call_count, data_size)),
                &(call_count, data_size),
                |b, &(call_count, data_size)| {
                    b.iter_batched(
                        || {
                            let ffi = MockPythonFFI::new();
                            let data = vec![0u8; data_size];
                            (ffi, data)
                        },
                        |(mut ffi, data)| {
                            for i in 0..call_count {
                                black_box(ffi.data_transfer_call(i, &data));
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Benchmark array operations
            group.bench_with_input(
                BenchmarkId::new("array_operations", format!("{}calls_{}elements", call_count, data_size / 4)),
                &(call_count, data_size / 4),
                |b, &(call_count, element_count)| {
                    b.iter_batched(
                        || {
                            let ffi = MockPythonFFI::new();
                            let array = vec![1.0f32; element_count];
                            (ffi, array)
                        },
                        |(mut ffi, array)| {
                            for i in 0..call_count {
                                black_box(ffi.array_operation(i, &array));
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

/// Benchmark memory allocation efficiency in FFI
fn benchmark_ffi_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_memory_allocation");
    
    let allocation_sizes = vec![64, 256, 1024, 4096, 16384, 65536];
    let allocation_counts = vec![1, 10, 100, 1000];
    
    for &size in &allocation_sizes {
        for &count in &allocation_counts {
            group.throughput(Throughput::Bytes((size * count) as u64));
            
            // Python-side allocation
            group.bench_with_input(
                BenchmarkId::new("python_allocation", format!("{}x{}bytes", count, size)),
                &(count, size),
                |b, &(count, size)| {
                    b.iter_batched(
                        || MockPythonFFI::new(),
                        |mut ffi| {
                            for i in 0..count {
                                let result = ffi.allocate_python_buffer(size);
                                black_box(result);
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Rust-side allocation passed to Python
            group.bench_with_input(
                BenchmarkId::new("rust_allocation", format!("{}x{}bytes", count, size)),
                &(count, size),
                |b, &(count, size)| {
                    b.iter_batched(
                        || MockPythonFFI::new(),
                        |mut ffi| {
                            for i in 0..count {
                                let buffer = vec![0u8; size];
                                let result = ffi.use_rust_buffer(&buffer);
                                black_box(result);
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Zero-copy operations
            group.bench_with_input(
                BenchmarkId::new("zero_copy", format!("{}x{}bytes", count, size)),
                &(count, size),
                |b, &(count, size)| {
                    b.iter_batched(
                        || {
                            let ffi = MockPythonFFI::new();
                            let buffer = vec![0u8; size];
                            (ffi, buffer)
                        },
                        |(mut ffi, buffer)| {
                            for i in 0..count {
                                let result = ffi.zero_copy_operation(&buffer);
                                black_box(result);
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

/// Benchmark async operation performance
fn benchmark_async_ffi_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("async_ffi_operations");
    
    let concurrent_ops = vec![1, 4, 8, 16, 32];
    let operation_duration = vec![1, 10, 100]; // milliseconds
    
    for &concurrency in &concurrent_ops {
        for &duration in &operation_duration {
            group.throughput(Throughput::Elements(concurrency as u64));
            
            // Async operations without blocking
            group.bench_with_input(
                BenchmarkId::new("async_non_blocking", format!("{}concurrent_{}ms", concurrency, duration)),
                &(concurrency, duration),
                |b, &(concurrency, duration)| {
                    b.to_async(&rt).iter(|| async {
                        let ffi = MockPythonFFI::new();
                        let futures: Vec<_> = (0..concurrency)
                            .map(|i| ffi.async_operation(i, duration))
                            .collect();
                        
                        let results = futures::future::join_all(futures).await;
                        black_box(results)
                    })
                },
            );
            
            // Sequential async operations
            group.bench_with_input(
                BenchmarkId::new("async_sequential", format!("{}ops_{}ms", concurrency, duration)),
                &(concurrency, duration),
                |b, &(concurrency, duration)| {
                    b.to_async(&rt).iter(|| async {
                        let ffi = MockPythonFFI::new();
                        let mut results = Vec::new();
                        
                        for i in 0..concurrency {
                            let result = ffi.async_operation(i, duration).await;
                            results.push(result);
                        }
                        
                        black_box(results)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark GIL impact on concurrent operations
fn benchmark_gil_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("gil_impact");
    
    let thread_counts = vec![1, 2, 4, 8, 16];
    let operation_types = vec!["cpu_intensive", "io_bound", "mixed"];
    
    for &thread_count in &thread_counts {
        for op_type in &operation_types {
            group.throughput(Throughput::Elements(thread_count as u64));
            
            group.bench_with_input(
                BenchmarkId::new("gil_contention", format!("{}threads_{}", thread_count, op_type)),
                &(thread_count, op_type),
                |b, &(thread_count, op_type)| {
                    b.iter_batched(
                        || MockPythonFFI::new(),
                        |ffi| {
                            let handles: Vec<_> = (0..thread_count)
                                .map(|i| {
                                    let ffi_clone = ffi.clone();
                                    std::thread::spawn(move || {
                                        match op_type {
                                            "cpu_intensive" => ffi_clone.cpu_intensive_operation(i),
                                            "io_bound" => ffi_clone.io_bound_operation(i),
                                            "mixed" => ffi_clone.mixed_operation(i),
                                            _ => 0,
                                        }
                                    })
                                })
                                .collect();
                            
                            let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                            black_box(results)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark neural network operations through FFI
fn benchmark_neural_network_ffi(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_network_ffi");
    
    let network_sizes = vec![
        (784, 128, 10),    // MNIST-like
        (1000, 500, 100), // Medium network
        (2048, 1024, 256), // Large network
    ];
    
    let batch_sizes = vec![1, 8, 32, 128];
    
    for &(input_size, hidden_size, output_size) in &network_sizes {
        for &batch_size in &batch_sizes {
            let network_name = format!("{}x{}x{}", input_size, hidden_size, output_size);
            
            group.throughput(Throughput::Elements(batch_size as u64));
            
            // Forward pass through FFI
            group.bench_with_input(
                BenchmarkId::new("forward_pass_ffi", format!("{}_batch{}", network_name, batch_size)),
                &(input_size, hidden_size, output_size, batch_size),
                |b, &(input_size, hidden_size, output_size, batch_size)| {
                    b.iter_batched(
                        || {
                            let ffi = MockPythonFFI::new();
                            let network = ffi.create_network(input_size, hidden_size, output_size);
                            let batch_data = vec![vec![1.0f32; input_size]; batch_size];
                            (ffi, network, batch_data)
                        },
                        |(mut ffi, network, batch_data)| {
                            let result = ffi.forward_pass_batch(&network, &batch_data);
                            black_box(result)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Training step through FFI
            group.bench_with_input(
                BenchmarkId::new("training_step_ffi", format!("{}_batch{}", network_name, batch_size)),
                &(input_size, hidden_size, output_size, batch_size),
                |b, &(input_size, hidden_size, output_size, batch_size)| {
                    b.iter_batched(
                        || {
                            let ffi = MockPythonFFI::new();
                            let network = ffi.create_network(input_size, hidden_size, output_size);
                            let batch_data = vec![vec![1.0f32; input_size]; batch_size];
                            let targets = vec![vec![1.0f32; output_size]; batch_size];
                            (ffi, network, batch_data, targets)
                        },
                        |(mut ffi, network, batch_data, targets)| {
                            let result = ffi.training_step(&network, &batch_data, &targets);
                            black_box(result)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Direct Rust vs FFI comparison
            group.bench_with_input(
                BenchmarkId::new("direct_rust", format!("{}_batch{}", network_name, batch_size)),
                &(input_size, hidden_size, output_size, batch_size),
                |b, &(input_size, hidden_size, output_size, batch_size)| {
                    b.iter_batched(
                        || {
                            let network = MockRustNetwork::new(input_size, hidden_size, output_size);
                            let batch_data = vec![vec![1.0f32; input_size]; batch_size];
                            (network, batch_data)
                        },
                        |(network, batch_data)| {
                            let result = network.forward_pass_batch(&batch_data);
                            black_box(result)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    
    group.finish();
}

// Mock implementations for benchmarking

#[derive(Clone)]
struct MockPythonFFI {
    call_count: usize,
}

impl MockPythonFFI {
    fn new() -> Self {
        Self { call_count: 0 }
    }
    
    fn simple_call(&mut self, value: usize) -> usize {
        self.call_count += 1;
        
        // Simulate FFI call overhead
        std::thread::sleep(Duration::from_nanos(100));
        value * 2
    }
    
    fn data_transfer_call(&mut self, value: usize, data: &[u8]) -> usize {
        self.call_count += 1;
        
        // Simulate data transfer overhead
        let overhead = data.len() / 1000; // 1µs per KB
        std::thread::sleep(Duration::from_nanos(100 + overhead as u64));
        
        value + data.len()
    }
    
    fn array_operation(&mut self, value: usize, array: &[f32]) -> f32 {
        self.call_count += 1;
        
        // Simulate array processing overhead
        let overhead = array.len() / 100; // 10ns per element
        std::thread::sleep(Duration::from_nanos(100 + overhead as u64));
        
        array.iter().sum::<f32>() + value as f32
    }
    
    fn allocate_python_buffer(&mut self, size: usize) -> Vec<u8> {
        // Simulate Python memory allocation overhead
        std::thread::sleep(Duration::from_nanos(500 + (size / 10) as u64));
        vec![0u8; size]
    }
    
    fn use_rust_buffer(&mut self, buffer: &[u8]) -> usize {
        // Simulate using Rust-allocated buffer in Python
        std::thread::sleep(Duration::from_nanos(200));
        buffer.len()
    }
    
    fn zero_copy_operation(&mut self, buffer: &[u8]) -> usize {
        // Simulate zero-copy operation
        std::thread::sleep(Duration::from_nanos(50));
        buffer.len()
    }
    
    async fn async_operation(&self, id: usize, duration_ms: usize) -> usize {
        // Simulate async operation
        tokio::time::sleep(Duration::from_millis(duration_ms as u64)).await;
        id * 2
    }
    
    fn cpu_intensive_operation(&self, id: usize) -> usize {
        // Simulate CPU-intensive operation (affected by GIL)
        let mut result = 0;
        for i in 0..10000 {
            result += i * id;
        }
        result
    }
    
    fn io_bound_operation(&self, id: usize) -> usize {
        // Simulate I/O-bound operation (can release GIL)
        std::thread::sleep(Duration::from_millis(1));
        id * 2
    }
    
    fn mixed_operation(&self, id: usize) -> usize {
        // Simulate mixed CPU/I/O operation
        let mut result = 0;
        for i in 0..1000 {
            result += i * id;
        }
        std::thread::sleep(Duration::from_micros(100));
        result
    }
    
    fn create_network(&self, input_size: usize, hidden_size: usize, output_size: usize) -> MockNetworkHandle {
        // Simulate network creation overhead
        std::thread::sleep(Duration::from_millis(1));
        MockNetworkHandle {
            input_size,
            hidden_size,
            output_size,
        }
    }
    
    fn forward_pass_batch(&mut self, network: &MockNetworkHandle, batch_data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simulate forward pass through FFI
        let overhead = batch_data.len() * network.input_size / 1000; // 1µs per 1K elements
        std::thread::sleep(Duration::from_micros(overhead as u64));
        
        batch_data.iter().map(|input| {
            vec![0.5f32; network.output_size]
        }).collect()
    }
    
    fn training_step(&mut self, network: &MockNetworkHandle, batch_data: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        // Simulate training step through FFI
        let overhead = batch_data.len() * network.input_size / 500; // 2µs per 1K elements
        std::thread::sleep(Duration::from_micros(overhead as u64));
        
        0.1 // Mock loss value
    }
}

struct MockNetworkHandle {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

struct MockRustNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl MockRustNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
        }
    }
    
    fn forward_pass_batch(&self, batch_data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simulate direct Rust implementation (no FFI overhead)
        let overhead = batch_data.len() * self.input_size / 2000; // 0.5µs per 1K elements
        std::thread::sleep(Duration::from_micros(overhead as u64));
        
        batch_data.iter().map(|input| {
            vec![0.5f32; self.output_size]
        }).collect()
    }
}

criterion_group! {
    name = python_ffi_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100)
        .with_plots();
    targets = 
        benchmark_ffi_call_overhead,
        benchmark_ffi_memory_allocation,
        benchmark_async_ffi_operations,
        benchmark_gil_impact,
        benchmark_neural_network_ffi
}

criterion_main!(python_ffi_benches);