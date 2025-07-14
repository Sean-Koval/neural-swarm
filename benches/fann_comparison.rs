use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use fann_rust_core::{Network, NetworkBuilder, TrainingData, TrainingAlgorithm};
use std::time::Duration;
use std::process::Command;
use std::ffi::CString;

/// External FANN library bindings for comparison benchmarks
#[link(name = "fann")]
extern "C" {
    fn fann_create_standard(num_layers: u32, ...) -> *mut std::ffi::c_void;
    fn fann_destroy(ann: *mut std::ffi::c_void);
    fn fann_run(ann: *mut std::ffi::c_void, input: *const f32) -> *mut f32;
    fn fann_train_on_data(ann: *mut std::ffi::c_void, data: *const std::ffi::c_void, max_epochs: u32, epochs_between_reports: u32, desired_error: f32);
    fn fann_read_train_from_file(filename: *const i8) -> *mut std::ffi::c_void;
    fn fann_destroy_train(train_data: *mut std::ffi::c_void);
    fn fann_get_MSE(ann: *mut std::ffi::c_void) -> f32;
}

/// FANN-Rust vs Original FANN performance comparison
struct FANNComparison {
    rust_network: Network,
    fann_network: *mut std::ffi::c_void,
    training_data: TrainingData,
}

impl FANNComparison {
    fn new(layers: &[usize]) -> Self {
        // Create Rust network
        let mut builder = NetworkBuilder::new();
        for i in 0..layers.len() - 1 {
            builder = builder.add_layer(layers[i], layers[i + 1]);
        }
        let rust_network = builder.build().expect("Failed to create Rust network");
        
        // Create original FANN network
        let fann_network = unsafe {
            fann_create_standard(
                layers.len() as u32,
                layers[0] as u32,
                layers[1] as u32,
                layers.get(2).copied().unwrap_or(layers[1]) as u32,
            )
        };
        
        // Generate training data
        let training_data = Self::generate_training_data(layers[0], layers[layers.len() - 1], 1000);
        
        Self {
            rust_network,
            fann_network,
            training_data,
        }
    }
    
    fn generate_training_data(input_size: usize, output_size: usize, num_samples: usize) -> TrainingData {
        let mut data = TrainingData::new();
        
        for _ in 0..num_samples {
            let input: Vec<f32> = (0..input_size).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
            let target: Vec<f32> = (0..output_size).map(|_| rand::random::<f32>()).collect();
            
            data.add_sample(input, target);
        }
        
        data
    }
}

impl Drop for FANNComparison {
    fn drop(&mut self) {
        unsafe {
            fann_destroy(self.fann_network);
        }
    }
}

/// Benchmark network creation performance
fn benchmark_network_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_creation");
    
    let architectures = vec![
        vec![784, 100, 10],           // MNIST classifier
        vec![784, 300, 100, 10],      // Deep MNIST classifier
        vec![4, 8, 8, 1],             // XOR problem
        vec![10, 50, 50, 50, 1],      // Deep regression
        vec![100, 200, 200, 100, 50], // Large network
    ];
    
    for architecture in &architectures {
        let arch_name = format!("{:?}", architecture);
        
        // Benchmark Rust implementation
        group.bench_with_input(
            BenchmarkId::new("rust", &arch_name),
            architecture,
            |bench, arch| {
                bench.iter(|| {
                    let mut builder = NetworkBuilder::new();
                    for i in 0..arch.len() - 1 {
                        builder = builder.add_layer(arch[i], arch[i + 1]);
                    }
                    black_box(builder.build().expect("Failed to create network"));
                });
            },
        );
        
        // Benchmark original FANN implementation
        group.bench_with_input(
            BenchmarkId::new("fann", &arch_name),
            architecture,
            |bench, arch| {
                bench.iter(|| {
                    let network = unsafe {
                        match arch.len() {
                            3 => fann_create_standard(3, arch[0] as u32, arch[1] as u32, arch[2] as u32),
                            4 => fann_create_standard(4, arch[0] as u32, arch[1] as u32, arch[2] as u32, arch[3] as u32),
                            5 => fann_create_standard(5, arch[0] as u32, arch[1] as u32, arch[2] as u32, arch[3] as u32, arch[4] as u32),
                            _ => panic!("Unsupported architecture length"),
                        }
                    };
                    black_box(network);
                    unsafe { fann_destroy(network); }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark forward pass performance
fn benchmark_forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass");
    
    let architectures = vec![
        vec![784, 100, 10],           // MNIST classifier
        vec![784, 300, 100, 10],      // Deep MNIST classifier
        vec![4, 8, 8, 1],             // XOR problem
        vec![32, 64, 64, 32, 1],      // Medium network
    ];
    
    for architecture in &architectures {
        let arch_name = format!("{:?}", architecture);
        let comparison = FANNComparison::new(architecture);
        let input: Vec<f32> = (0..architecture[0]).map(|_| rand::random::<f32>()).collect();
        
        group.throughput(Throughput::Elements(1));
        
        // Benchmark Rust implementation
        group.bench_with_input(
            BenchmarkId::new("rust", &arch_name),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let output = comparison.rust_network.forward(black_box(input));
                    black_box(output);
                });
            },
        );
        
        // Benchmark original FANN implementation
        group.bench_with_input(
            BenchmarkId::new("fann", &arch_name),
            &input,
            |bench, input| {
                bench.iter(|| {
                    let output = unsafe {
                        fann_run(comparison.fann_network, black_box(input.as_ptr()))
                    };
                    black_box(output);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark training performance
fn benchmark_training_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_performance");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for training
    
    let architectures = vec![
        vec![4, 8, 1],              // XOR problem
        vec![10, 20, 10, 1],        // Small regression
        vec![784, 100, 10],         // MNIST classifier (reduced for speed)
    ];
    
    for architecture in &architectures {
        let arch_name = format!("{:?}", architecture);
        
        // Benchmark Rust implementation
        group.bench_with_input(
            BenchmarkId::new("rust", &arch_name),
            architecture,
            |bench, arch| {
                bench.iter_batched(
                    || {
                        let mut builder = NetworkBuilder::new();
                        for i in 0..arch.len() - 1 {
                            builder = builder.add_layer(arch[i], arch[i + 1]);
                        }
                        let network = builder.build().expect("Failed to create network");
                        let training_data = FANNComparison::generate_training_data(arch[0], arch[arch.len() - 1], 100);
                        (network, training_data)
                    },
                    |(mut network, training_data)| {
                        network.train(
                            black_box(&training_data),
                            10, // epochs
                            0.1, // learning rate
                            TrainingAlgorithm::Backpropagation
                        ).expect("Training failed");
                        black_box(network);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch processing performance
fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let batch_sizes = vec![1, 8, 32, 64, 128];
    let architecture = vec![784, 128, 10];
    
    for &batch_size in &batch_sizes {
        let comparison = FANNComparison::new(&architecture);
        let batch_input: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| (0..architecture[0]).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        // Benchmark Rust batch processing
        group.bench_with_input(
            BenchmarkId::new("rust_batch", batch_size),
            &batch_input,
            |bench, batch| {
                bench.iter(|| {
                    let outputs: Vec<_> = batch.iter()
                        .map(|input| comparison.rust_network.forward(black_box(input)))
                        .collect();
                    black_box(outputs);
                });
            },
        );
        
        // Benchmark original FANN individual processing
        group.bench_with_input(
            BenchmarkId::new("fann_individual", batch_size),
            &batch_input,
            |bench, batch| {
                bench.iter(|| {
                    let outputs: Vec<_> = batch.iter()
                        .map(|input| unsafe {
                            fann_run(comparison.fann_network, black_box(input.as_ptr()))
                        })
                        .collect();
                    black_box(outputs);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage patterns
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    let architectures = vec![
        vec![100, 50, 10],
        vec![784, 300, 100, 10],
        vec![1000, 500, 250, 100, 10],
    ];
    
    for architecture in &architectures {
        let arch_name = format!("{:?}", architecture);
        
        // Measure Rust implementation memory usage
        group.bench_with_input(
            BenchmarkId::new("rust_memory", &arch_name),
            architecture,
            |bench, arch| {
                bench.iter_batched(
                    || {
                        let mut builder = NetworkBuilder::new();
                        for i in 0..arch.len() - 1 {
                            builder = builder.add_layer(arch[i], arch[i + 1]);
                        }
                        builder.build().expect("Failed to create network")
                    },
                    |network| {
                        let memory_usage = network.memory_footprint();
                        black_box(memory_usage);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark accuracy comparison
fn benchmark_accuracy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_comparison");
    group.measurement_time(Duration::from_secs(60)); // Extended time for training
    
    let architecture = vec![4, 10, 1]; // XOR-like problem
    
    group.bench_function("xor_learning_rust", |bench| {
        bench.iter_batched(
            || {
                let mut builder = NetworkBuilder::new();
                builder = builder.add_layer(4, 10).add_layer(10, 1);
                let network = builder.build().expect("Failed to create network");
                
                // XOR training data
                let mut training_data = TrainingData::new();
                training_data.add_sample(vec![0.0, 0.0, 0.0, 0.0], vec![0.0]);
                training_data.add_sample(vec![0.0, 1.0, 1.0, 0.0], vec![1.0]);
                training_data.add_sample(vec![1.0, 0.0, 0.0, 1.0], vec![1.0]);
                training_data.add_sample(vec![1.0, 1.0, 1.0, 1.0], vec![0.0]);
                
                (network, training_data)
            },
            |(mut network, training_data)| {
                network.train(
                    black_box(&training_data),
                    1000, // epochs
                    0.1,  // learning rate
                    TrainingAlgorithm::Backpropagation
                ).expect("Training failed");
                
                let final_error = network.get_mse();
                black_box(final_error);
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Performance regression detection
fn benchmark_regression_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_detection");
    
    // Load baseline performance data if available
    let baseline_path = "benches/baseline_performance.json";
    
    group.bench_function("performance_regression_check", |bench| {
        let architecture = vec![784, 100, 10];
        let comparison = FANNComparison::new(&architecture);
        let test_input: Vec<f32> = (0..784).map(|_| rand::random::<f32>()).collect();
        
        bench.iter(|| {
            let start = std::time::Instant::now();
            let _output = comparison.rust_network.forward(black_box(&test_input));
            let duration = start.elapsed();
            
            // Check if performance has regressed
            let baseline_duration = Duration::from_micros(100); // Expected baseline
            let regression_threshold = 1.1; // 10% regression threshold
            
            if duration > baseline_duration.mul_f64(regression_threshold) {
                eprintln!("WARNING: Performance regression detected!");
                eprintln!("Current: {:?}, Baseline: {:?}", duration, baseline_duration);
            }
            
            black_box(duration);
        });
    });
    
    group.finish();
}

criterion_group! {
    name = fann_comparison_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(5))
        .sample_size(50)
        .with_plots();
    targets = 
        benchmark_network_creation,
        benchmark_forward_pass,
        benchmark_training_performance,
        benchmark_batch_processing,
        benchmark_memory_usage,
        benchmark_accuracy_comparison,
        benchmark_regression_detection
}

criterion_main!(fann_comparison_benches);