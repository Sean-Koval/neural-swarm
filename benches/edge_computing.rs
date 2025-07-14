use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use fann_rust_core::{
    edge::{EdgeOptimizedNetwork, EdgeDeployment, ResourceConstraint, DeviceProfile},
    quantization::{QuantizationType, MixedPrecisionNetwork},
    compression::{NetworkCompressor, CompressionLevel},
    power::{PowerManager, ThermalController},
};
use std::time::Duration;

/// Benchmark edge computing performance characteristics
fn benchmark_edge_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_inference");
    
    let device_profiles = vec![
        DeviceProfile::IoTSensor {
            memory_limit: 64 * 1024,      // 64KB
            power_budget: 0.1,            // 100mW
            cpu_freq: 48_000_000,         // 48MHz
        },
        DeviceProfile::MicroController {
            memory_limit: 256 * 1024,     // 256KB
            power_budget: 0.5,            // 500mW
            cpu_freq: 168_000_000,        // 168MHz
        },
        DeviceProfile::EdgeGateway {
            memory_limit: 16 * 1024 * 1024, // 16MB
            power_budget: 5.0,             // 5W
            cpu_freq: 1_000_000_000,       // 1GHz
        },
        DeviceProfile::MobileDevice {
            memory_limit: 512 * 1024 * 1024, // 512MB
            power_budget: 2.0,              // 2W
            cpu_freq: 2_400_000_000,        // 2.4GHz
        },
    ];
    
    let network_sizes = vec![
        (10, 5, 1),      // Tiny network
        (28, 16, 10),    // Small classification
        (100, 50, 10),   // Medium network
        (784, 128, 10),  // MNIST-like (for mobile only)
    ];
    
    for device_profile in &device_profiles {
        let device_name = device_profile.name();
        
        for &(input_size, hidden_size, output_size) in &network_sizes {
            // Skip large networks on resource-constrained devices
            if device_profile.memory_limit() < 1024 * 1024 && input_size > 100 {
                continue;
            }
            
            let network_name = format!("{}x{}x{}", input_size, hidden_size, output_size);
            
            group.throughput(Throughput::Elements(1));
            
            // Benchmark optimized edge network
            group.bench_with_input(
                BenchmarkId::new(format!("{}_optimized", device_name), &network_name),
                &(device_profile, input_size, hidden_size, output_size),
                |bench, &(device_profile, input_size, hidden_size, output_size)| {
                    let edge_network = EdgeOptimizedNetwork::new(
                        &[input_size, hidden_size, output_size],
                        device_profile.clone(),
                    ).expect("Failed to create edge network");
                    
                    let input = vec![0.5f32; input_size];
                    
                    bench.iter(|| {
                        let output = edge_network.inference(black_box(&input));
                        black_box(output);
                    });
                },
            );
            
            // Benchmark quantized network
            group.bench_with_input(
                BenchmarkId::new(format!("{}_quantized", device_name), &network_name),
                &(device_profile, input_size, hidden_size, output_size),
                |bench, &(device_profile, input_size, hidden_size, output_size)| {
                    let mut edge_network = EdgeOptimizedNetwork::new(
                        &[input_size, hidden_size, output_size],
                        device_profile.clone(),
                    ).expect("Failed to create edge network");
                    
                    let quantization_type = match device_profile.memory_limit() {
                        limit if limit < 128 * 1024 => QuantizationType::Binary,
                        limit if limit < 1024 * 1024 => QuantizationType::INT4,
                        _ => QuantizationType::INT8,
                    };
                    
                    edge_network.apply_quantization(quantization_type)
                        .expect("Failed to apply quantization");
                    
                    let input = vec![0.5f32; input_size];
                    
                    bench.iter(|| {
                        let output = edge_network.inference(black_box(&input));
                        black_box(output);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark memory usage under resource constraints
fn benchmark_memory_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_constraints");
    
    let memory_limits = vec![
        32 * 1024,     // 32KB - severe constraint
        128 * 1024,    // 128KB - moderate constraint
        1024 * 1024,   // 1MB - light constraint
        16 * 1024 * 1024, // 16MB - no significant constraint
    ];
    
    for &memory_limit in &memory_limits {
        group.bench_with_input(
            BenchmarkId::new("memory_usage", memory_limit),
            &memory_limit,
            |bench, &memory_limit| {
                bench.iter_batched(
                    || {
                        let constraint = ResourceConstraint {
                            memory_limit,
                            power_budget: 1.0,
                            compute_budget: 1.0,
                            latency_target: Duration::from_millis(100),
                        };
                        
                        EdgeDeployment::new(constraint)
                    },
                    |mut deployment| {
                        // Create network that fits within memory constraint
                        let network_size = deployment.optimal_network_size(&[100, 50, 10]);
                        let edge_network = EdgeOptimizedNetwork::new(
                            &network_size,
                            DeviceProfile::Custom {
                                memory_limit,
                                power_budget: 1.0,
                                compute_capability: 1.0,
                            },
                        ).expect("Failed to create network");
                        
                        // Measure actual memory usage
                        let memory_usage = edge_network.memory_footprint();
                        assert!(memory_usage <= memory_limit, 
                            "Network exceeds memory limit: {} > {}", memory_usage, memory_limit);
                        
                        // Perform inference to test memory allocation
                        let input = vec![0.5f32; network_size[0]];
                        let output = edge_network.inference(black_box(&input));
                        
                        black_box((memory_usage, output));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark power consumption characteristics
fn benchmark_power_consumption(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_consumption");
    
    let power_budgets = vec![0.1, 0.5, 1.0, 2.0, 5.0]; // Watts
    let network_configs = vec![
        (10, 5, 1),
        (50, 25, 5),
        (100, 50, 10),
    ];
    
    for &power_budget in &power_budgets {
        for &(input_size, hidden_size, output_size) in &network_configs {
            let config_name = format!("{:.1}W_{}x{}x{}", power_budget, input_size, hidden_size, output_size);
            
            group.bench_with_input(
                BenchmarkId::new("power_managed_inference", &config_name),
                &(power_budget, input_size, hidden_size, output_size),
                |bench, &(power_budget, input_size, hidden_size, output_size)| {
                    let device_profile = DeviceProfile::Custom {
                        memory_limit: 1024 * 1024,
                        power_budget,
                        compute_capability: power_budget, // Assume compute scales with power
                    };
                    
                    let mut edge_network = EdgeOptimizedNetwork::new(
                        &[input_size, hidden_size, output_size],
                        device_profile,
                    ).expect("Failed to create edge network");
                    
                    let mut power_manager = PowerManager::new(power_budget);
                    edge_network.set_power_manager(power_manager);
                    
                    let input = vec![0.5f32; input_size];
                    
                    bench.iter(|| {
                        let start_energy = edge_network.current_energy_usage();
                        let output = edge_network.inference(black_box(&input));
                        let energy_consumed = edge_network.current_energy_usage() - start_energy;
                        
                        black_box((output, energy_consumed));
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark thermal throttling behavior
fn benchmark_thermal_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermal_performance");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for thermal effects
    
    let thermal_limits = vec![60.0, 70.0, 80.0, 90.0]; // Celsius
    
    for &thermal_limit in &thermal_limits {
        group.bench_with_input(
            BenchmarkId::new("thermal_throttling", thermal_limit as u32),
            &thermal_limit,
            |bench, &thermal_limit| {
                bench.iter_batched(
                    || {
                        let device_profile = DeviceProfile::MobileDevice {
                            memory_limit: 512 * 1024 * 1024,
                            power_budget: 3.0,
                            cpu_freq: 2_400_000_000,
                        };
                        
                        let mut edge_network = EdgeOptimizedNetwork::new(
                            &[784, 128, 10],
                            device_profile,
                        ).expect("Failed to create edge network");
                        
                        let mut thermal_controller = ThermalController::new(thermal_limit);
                        edge_network.set_thermal_controller(thermal_controller);
                        
                        (edge_network, vec![0.5f32; 784])
                    },
                    |(mut edge_network, input)| {
                        // Simulate sustained inference load
                        let mut total_inferences = 0;
                        let mut throttling_events = 0;
                        let start_time = std::time::Instant::now();
                        
                        while start_time.elapsed() < Duration::from_millis(100) {
                            let temperature_before = edge_network.current_temperature();
                            let output = edge_network.inference(black_box(&input));
                            let temperature_after = edge_network.current_temperature();
                            
                            total_inferences += 1;
                            
                            if temperature_after > thermal_limit {
                                throttling_events += 1;
                            }
                            
                            black_box(output);
                        }
                        
                        black_box((total_inferences, throttling_events));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark network compression techniques
fn benchmark_compression_techniques(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_techniques");
    
    let compression_levels = vec![
        CompressionLevel::None,
        CompressionLevel::Light,   // 25% reduction
        CompressionLevel::Medium,  // 50% reduction
        CompressionLevel::Heavy,   // 75% reduction
        CompressionLevel::Extreme, // 90% reduction
    ];
    
    let base_network = &[784, 128, 64, 10];
    
    for compression_level in &compression_levels {
        let level_name = format!("{:?}", compression_level);
        
        // Benchmark compression process
        group.bench_with_input(
            BenchmarkId::new("compression_time", &level_name),
            compression_level,
            |bench, compression_level| {
                bench.iter_batched(
                    || {
                        EdgeOptimizedNetwork::new(
                            base_network,
                            DeviceProfile::EdgeGateway {
                                memory_limit: 16 * 1024 * 1024,
                                power_budget: 5.0,
                                cpu_freq: 1_000_000_000,
                            },
                        ).expect("Failed to create network")
                    },
                    |network| {
                        let mut compressor = NetworkCompressor::new();
                        let compressed_network = compressor.compress(
                            black_box(network),
                            *compression_level
                        ).expect("Compression failed");
                        black_box(compressed_network);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        
        // Benchmark inference on compressed network
        group.bench_with_input(
            BenchmarkId::new("compressed_inference", &level_name),
            compression_level,
            |bench, compression_level| {
                let mut network = EdgeOptimizedNetwork::new(
                    base_network,
                    DeviceProfile::EdgeGateway {
                        memory_limit: 16 * 1024 * 1024,
                        power_budget: 5.0,
                        cpu_freq: 1_000_000_000,
                    },
                ).expect("Failed to create network");
                
                let mut compressor = NetworkCompressor::new();
                network = compressor.compress(network, *compression_level)
                    .expect("Compression failed");
                
                let input = vec![0.5f32; 784];
                
                bench.iter(|| {
                    let output = network.inference(black_box(&input));
                    black_box(output);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch processing on edge devices
fn benchmark_edge_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_batch_processing");
    
    let batch_sizes = vec![1, 4, 8, 16, 32];
    let device_profiles = vec![
        DeviceProfile::MicroController {
            memory_limit: 256 * 1024,
            power_budget: 0.5,
            cpu_freq: 168_000_000,
        },
        DeviceProfile::EdgeGateway {
            memory_limit: 16 * 1024 * 1024,
            power_budget: 5.0,
            cpu_freq: 1_000_000_000,
        },
    ];
    
    for device_profile in &device_profiles {
        let device_name = device_profile.name();
        
        for &batch_size in &batch_sizes {
            // Skip large batches on constrained devices
            if device_profile.memory_limit() < 1024 * 1024 && batch_size > 8 {
                continue;
            }
            
            group.throughput(Throughput::Elements(batch_size as u64));
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}_batch", device_name), batch_size),
                &(device_profile, batch_size),
                |bench, &(device_profile, batch_size)| {
                    // Adjust network size based on device constraints
                    let network_size = if device_profile.memory_limit() < 1024 * 1024 {
                        vec![28, 16, 10]  // Small network
                    } else {
                        vec![100, 50, 10] // Medium network
                    };
                    
                    let edge_network = EdgeOptimizedNetwork::new(
                        &network_size,
                        device_profile.clone(),
                    ).expect("Failed to create edge network");
                    
                    let batch_input: Vec<Vec<f32>> = (0..batch_size)
                        .map(|_| vec![0.5f32; network_size[0]])
                        .collect();
                    
                    bench.iter(|| {
                        let outputs: Vec<_> = batch_input.iter()
                            .map(|input| edge_network.inference(black_box(input)))
                            .collect();
                        black_box(outputs);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark real-time performance constraints
fn benchmark_realtime_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("realtime_constraints");
    
    let latency_targets = vec![
        Duration::from_millis(1),   // 1ms - hard real-time
        Duration::from_millis(10),  // 10ms - soft real-time
        Duration::from_millis(100), // 100ms - interactive
        Duration::from_millis(1000), // 1s - batch processing
    ];
    
    for &latency_target in &latency_targets {
        let target_name = format!("{}ms", latency_target.as_millis());
        
        group.bench_with_input(
            BenchmarkId::new("realtime_inference", &target_name),
            &latency_target,
            |bench, &latency_target| {
                let constraint = ResourceConstraint {
                    memory_limit: 1024 * 1024,
                    power_budget: 2.0,
                    compute_budget: 1.0,
                    latency_target,
                };
                
                let deployment = EdgeDeployment::new(constraint);
                let optimal_size = deployment.optimal_network_size(&[100, 50, 10]);
                
                let mut edge_network = EdgeOptimizedNetwork::new(
                    &optimal_size,
                    DeviceProfile::MobileDevice {
                        memory_limit: 1024 * 1024,
                        power_budget: 2.0,
                        cpu_freq: 2_000_000_000,
                    },
                ).expect("Failed to create edge network");
                
                // Apply optimizations to meet latency target
                edge_network.optimize_for_latency(latency_target)
                    .expect("Failed to optimize for latency");
                
                let input = vec![0.5f32; optimal_size[0]];
                
                bench.iter(|| {
                    let start = std::time::Instant::now();
                    let output = edge_network.inference(black_box(&input));
                    let actual_latency = start.elapsed();
                    
                    // Verify latency constraint is met
                    let constraint_met = actual_latency <= latency_target;
                    
                    black_box((output, actual_latency, constraint_met));
                });
            },
        );
    }
    
    group.finish();
}

criterion_group! {
    name = edge_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(5))
        .sample_size(50)
        .with_plots();
    targets = 
        benchmark_edge_inference,
        benchmark_memory_constraints,
        benchmark_power_consumption,
        benchmark_thermal_performance,
        benchmark_compression_techniques,
        benchmark_edge_batch_processing,
        benchmark_realtime_constraints
}

criterion_main!(edge_benches);