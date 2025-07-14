// Neural Network Performance Benchmarks
// Comprehensive benchmarking suite for FANN-compatible neural networks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neural_swarm::*;
use std::time::Duration;

// Test data for benchmarking
const XOR_DATA: &[(Vec<f32>, Vec<f32>)] = &[
    (vec![0.0, 0.0], vec![0.0]),
    (vec![0.0, 1.0], vec![1.0]),
    (vec![1.0, 0.0], vec![1.0]),
    (vec![1.0, 1.0], vec![0.0]),
];

/// Benchmark XOR network forward pass performance
fn bench_xor_forward_pass(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let network = rt.block_on(async { MockXorNetwork::new() });
    
    c.bench_function("xor_forward_pass", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(network.forward(black_box(&[0.5, 0.5])).await.unwrap())
        })
    });
}

/// Benchmark XOR network training performance
fn bench_xor_training(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("xor_training");
    
    for iterations in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            iterations,
            |b, &iterations| {
                b.to_async(&rt).iter(|| async {
                    let mut network = MockXorNetwork::new();
                    black_box(network.train_batch(black_box(XOR_DATA), iterations).await.unwrap())
                });
            },
        );
    }
    group.finish();
}

/// Benchmark cascade network training performance
fn bench_cascade_training(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("cascade_training");
    
    for max_neurons in [5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("max_neurons", max_neurons),
            max_neurons,
            |b, &max_neurons| {
                b.to_async(&rt).iter(|| async {
                    let mut network = MockCascadeNetwork::new(2, 1);
                    black_box(network.cascade_train(
                        black_box(XOR_DATA), 
                        max_neurons, 
                        1, 
                        0.01
                    ).await.unwrap())
                });
            },
        );
    }
    group.finish();
}

/// Benchmark network serialization performance
fn bench_network_serialization(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let network = rt.block_on(async { MockXorNetwork::new() });
    let temp_dir = tempfile::TempDir::new().unwrap();
    let network_path = temp_dir.path().join("benchmark_network.fann");
    
    c.bench_function("network_save", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(network.save_network(black_box(&network_path)).await.unwrap())
        })
    });
    
    // Ensure file exists for load benchmark
    rt.block_on(async { network.save_network(&network_path).await.unwrap() });
    
    c.bench_function("network_load", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(MockXorNetwork::load_network(black_box(&network_path)).await.unwrap())
        })
    });
}

/// Benchmark data parsing performance
fn bench_data_parsing(c: &mut Criterion) {
    let xor_data_string = "4 2 1\n0 0\n0\n0 1\n1\n1 0\n1\n1 1\n0\n";
    
    c.bench_function("fann_data_parse", |b| {
        b.iter(|| {
            black_box(parse_fann_data(black_box(xor_data_string)).unwrap())
        })
    });
    
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    c.bench_function("fann_data_generate", |b| {
        b.iter(|| {
            black_box(generate_fann_data(black_box(&training_data)))
        })
    });
}

/// Benchmark multiple concurrent networks
fn bench_concurrent_networks(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_networks");
    
    for num_networks in [1, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("networks", num_networks),
            num_networks,
            |b, &num_networks| {
                b.to_async(&rt).iter(|| async {
                    let networks: Vec<_> = (0..num_networks)
                        .map(|_| MockXorNetwork::new())
                        .collect();
                    
                    let futures: Vec<_> = networks.iter().map(|network| async {
                        network.forward(&[0.5, 0.5]).await.unwrap()
                    }).collect();
                    
                    black_box(futures::future::join_all(futures).await)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark network scaling with different architectures
fn bench_network_scaling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("network_scaling");
    
    let architectures = vec![
        ("small", vec![3]),
        ("medium", vec![8]),
        ("large", vec![16]),
        ("deep", vec![8, 4]),
        ("very_deep", vec![16, 8, 4]),
    ];
    
    for (name, hidden_layers) in architectures {
        group.bench_with_input(
            BenchmarkId::new("architecture", name),
            &hidden_layers,
            |b, hidden_layers| {
                b.to_async(&rt).iter(|| async {
                    let network = MockXorNetwork::with_architecture(2, 1, hidden_layers);
                    black_box(network.forward(black_box(&[0.5, 0.5])).await.unwrap())
                });
            },
        );
    }
    group.finish();
}

// Mock implementations for benchmarking
use async_trait::async_trait;
use std::collections::HashMap;

struct MockXorNetwork {
    weights: HashMap<String, f32>,
    architecture: Vec<u32>,
}

impl MockXorNetwork {
    fn new() -> Self {
        Self {
            weights: HashMap::new(),
            architecture: vec![2, 4, 1],
        }
    }
    
    fn with_architecture(input_size: u32, output_size: u32, hidden_layers: &[u32]) -> Self {
        let mut arch = vec![input_size];
        arch.extend_from_slice(hidden_layers);
        arch.push(output_size);
        
        Self {
            weights: HashMap::new(),
            architecture: arch,
        }
    }
    
    async fn load_network(_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new())
    }
}

#[async_trait]
impl NeuralNetwork for MockXorNetwork {
    async fn forward(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Simulate computation based on architecture complexity
        let complexity = self.architecture.iter().sum::<u32>() as f32;
        let computation_cycles = (complexity * 10.0) as usize;
        
        // Simulate computational work
        let mut result = 0.0;
        for i in 0..computation_cycles {
            result += (i as f32 * input[0] * input[1]).sin();
        }
        
        // Return XOR-like result
        let xor_result = if (input[0] > 0.5) != (input[1] > 0.5) { 1.0 } else { 0.0 };
        Ok(vec![xor_result + (result * 0.001).sin() * 0.1])
    }
    
    async fn train_batch(&mut self, training_data: &[(Vec<f32>, Vec<f32>)], iterations: u32) -> Result<(), Box<dyn std::error::Error>> {
        // Simulate training computation
        let computation_load = training_data.len() * iterations as usize;
        let mut accumulator = 0.0;
        
        for i in 0..computation_load {
            accumulator += (i as f32).sin();
        }
        
        // Store some "learning"
        self.weights.insert("learned".to_string(), accumulator * 0.0001);
        Ok(())
    }
    
    async fn save_network(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let data = format!("weights: {:?}\narchitecture: {:?}", self.weights, self.architecture);
        tokio::fs::write(path, data).await?;
        Ok(())
    }
    
    async fn get_mse(&self, test_data: &[(Vec<f32>, Vec<f32>)]) -> Result<f32, Box<dyn std::error::Error>> {
        let mut total_error = 0.0;
        for (input, expected) in test_data {
            let output = self.forward(input).await?;
            total_error += (output[0] - expected[0]).powi(2);
        }
        Ok(total_error / test_data.len() as f32)
    }
}

struct MockCascadeNetwork {
    input_size: u32,
    output_size: u32,
    neurons: u32,
}

impl MockCascadeNetwork {
    fn new(input_size: u32, output_size: u32) -> Self {
        Self {
            input_size,
            output_size,
            neurons: output_size,
        }
    }
    
    async fn cascade_train(
        &mut self,
        training_data: &[(Vec<f32>, Vec<f32>)],
        max_neurons: u32,
        _neurons_between_reports: u32,
        desired_error: f32
    ) -> Result<CascadeTrainingResult, Box<dyn std::error::Error>> {
        let initial_neurons = self.neurons;
        let mut current_error = 1.0;
        let mut iterations = 0;
        
        // Simulate cascade training with computational work
        while self.neurons < initial_neurons + max_neurons && current_error > desired_error {
            iterations += 100;
            
            // Simulate candidate neuron evaluation
            let candidates = 5;
            let mut best_improvement = 0.0;
            
            for candidate in 0..candidates {
                // Simulate training candidate
                let candidate_work = training_data.len() * 50; // Simulate work
                let mut work_result = 0.0;
                for i in 0..candidate_work {
                    work_result += ((i + candidate) as f32).sin();
                }
                
                let improvement = work_result.abs() * 0.001;
                if improvement > best_improvement {
                    best_improvement = improvement;
                }
            }
            
            // Add neuron if improvement is significant
            if best_improvement > current_error * 0.1 {
                self.neurons += 1;
                current_error *= 0.8;
            } else {
                current_error *= 0.95;
            }
            
            if current_error <= desired_error {
                break;
            }
        }
        
        Ok(CascadeTrainingResult {
            total_iterations: iterations,
            final_mse: current_error,
            final_bit_fails: ((current_error * training_data.len() as f32) * 2.0) as u32,
        })
    }
}

#[derive(Debug)]
struct CascadeTrainingResult {
    total_iterations: u32,
    final_mse: f32,
    final_bit_fails: u32,
}

// Mock trait implementations
#[async_trait]
pub trait NeuralNetwork: Send + Sync {
    async fn forward(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>>;
    async fn train_batch(&mut self, training_data: &[(Vec<f32>, Vec<f32>)], iterations: u32) -> Result<(), Box<dyn std::error::Error>>;
    async fn save_network(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>>;
    async fn get_mse(&self, test_data: &[(Vec<f32>, Vec<f32>)]) -> Result<f32, Box<dyn std::error::Error>>;
}

// Data parsing functions for benchmarking
fn parse_fann_data(data: &str) -> Result<Vec<(Vec<f32>, Vec<f32>)>, Box<dyn std::error::Error>> {
    let lines: Vec<&str> = data.trim().split('\n').collect();
    if lines.is_empty() {
        return Err("Empty data".into());
    }
    
    let header: Vec<u32> = lines[0].split_whitespace()
        .map(|s| s.parse())
        .collect::<Result<Vec<_>, _>>()?;
    
    let num_patterns = header[0] as usize;
    let num_inputs = header[1] as usize;
    let num_outputs = header[2] as usize;
    
    let mut training_data = Vec::new();
    let mut line_idx = 1;
    
    for _ in 0..num_patterns {
        let inputs: Vec<f32> = lines[line_idx].split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<Vec<_>, _>>()?;
        
        let outputs: Vec<f32> = lines[line_idx + 1].split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<Vec<_>, _>>()?;
        
        training_data.push((inputs, outputs));
        line_idx += 2;
    }
    
    Ok(training_data)
}

fn generate_fann_data(data: &[(Vec<f32>, Vec<f32>)]) -> String {
    if data.is_empty() {
        return "0 0 0\n".to_string();
    }

    let num_patterns = data.len();
    let num_inputs = data[0].0.len();
    let num_outputs = data[0].1.len();

    let mut content = format!("{} {} {}\n", num_patterns, num_inputs, num_outputs);

    for (inputs, outputs) in data {
        content.push_str(&inputs.iter()
            .map(|x| format!("{}", x))
            .collect::<Vec<_>>()
            .join(" "));
        content.push('\n');

        content.push_str(&outputs.iter()
            .map(|x| format!("{}", x))
            .collect::<Vec<_>>()
            .join(" "));
        content.push('\n');
    }

    content
}

criterion_group!(
    benches,
    bench_xor_forward_pass,
    bench_xor_training,
    bench_cascade_training,
    bench_network_serialization,
    bench_data_parsing,
    bench_concurrent_networks,
    bench_network_scaling
);

criterion_main!(benches);