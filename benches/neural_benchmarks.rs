use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neural_swarm::{
    activation::ActivationType,
    inference::{ExecutionMode, InferenceConfig, InferenceEngine},
    network::{LayerConfig, NeuralNetwork},
    training::{TrainingAlgorithm, TrainingData, TrainingParams, Trainer},
};
use std::sync::Arc;

fn create_benchmark_network(input_size: usize, hidden_size: usize, output_size: usize) -> NeuralNetwork {
    let configs = vec![
        LayerConfig::new(input_size, ActivationType::Linear),
        LayerConfig::new(hidden_size, ActivationType::ReLU),
        LayerConfig::new(hidden_size / 2, ActivationType::ReLU),
        LayerConfig::new(output_size, ActivationType::Sigmoid),
    ];
    
    let mut network = NeuralNetwork::new_feedforward(&configs).unwrap();
    network.initialize_weights(Some(42)).unwrap();
    network
}

fn create_benchmark_data(num_samples: usize, input_size: usize, output_size: usize) -> TrainingData {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = TrainingData::new();
    
    for _ in 0..num_samples {
        let input: Vec<f32> = (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let target: Vec<f32> = (0..output_size).map(|_| rng.gen_range(0.0..1.0)).collect();
        data.add_sample(input, target);
    }
    
    data
}

fn bench_forward_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_propagation");
    
    for &size in &[10, 50, 100, 500, 1000] {
        let network = create_benchmark_network(size, size * 2, size / 2);
        let input: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_sample", size),
            &size,
            |b, _| {
                let mut net = network.clone();
                b.iter(|| {
                    let result = net.predict(black_box(&input));
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_training_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_algorithms");
    
    let network = create_benchmark_network(10, 20, 5);
    let training_data = create_benchmark_data(100, 10, 5);
    
    let algorithms = [
        TrainingAlgorithm::Backpropagation,
        TrainingAlgorithm::BackpropagationMomentum,
        TrainingAlgorithm::Rprop,
        TrainingAlgorithm::Quickprop,
    ];
    
    for algorithm in &algorithms {
        group.bench_function(
            BenchmarkId::new("single_epoch", format!("{:?}", algorithm)),
            |b| {
                b.iter(|| {
                    let mut net = network.clone();
                    let mut params = TrainingParams::default();
                    params.max_epochs = 1; // Single epoch for benchmarking
                    
                    let mut trainer = Trainer::new(*algorithm, params);
                    let result = trainer.train(&mut net, &training_data, None);
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inference");
    
    let network = Arc::new(create_benchmark_network(50, 100, 10));
    
    for &batch_size in &[1, 8, 16, 32, 64, 128] {
        let inputs: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| (0..50).map(|j| (i * j) as f32 / 100.0).collect())
            .collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        // Sequential execution
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &batch_size,
            |b, _| {
                let config = InferenceConfig {
                    mode: ExecutionMode::Sequential,
                    batch_size,
                    ..Default::default()
                };
                let mut engine = InferenceEngine::new(network.clone(), config);
                
                b.iter(|| {
                    let result = engine.predict_batch(black_box(&inputs));
                    black_box(result)
                });
            },
        );
        
        // Parallel execution
        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            &batch_size,
            |b, _| {
                let config = InferenceConfig {
                    mode: ExecutionMode::Parallel,
                    batch_size,
                    ..Default::default()
                };
                let mut engine = InferenceEngine::new(network.clone(), config);
                
                b.iter(|| {
                    let result = engine.predict_batch(black_box(&inputs));
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    use neural_swarm::activation::*;
    
    let mut group = c.benchmark_group("activation_functions");
    
    let input_data: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 100.0).collect();
    let mut output_data = vec![0.0f32; 1000];
    
    let activations = [
        ("linear", Box::new(Linear) as Box<dyn ActivationFunction>),
        ("sigmoid", Box::new(Sigmoid) as Box<dyn ActivationFunction>),
        ("tanh", Box::new(Tanh) as Box<dyn ActivationFunction>),
        ("relu", Box::new(ReLU) as Box<dyn ActivationFunction>),
        ("leaky_relu", Box::new(LeakyReLU::default()) as Box<dyn ActivationFunction>),
        ("elu", Box::new(ELU::default()) as Box<dyn ActivationFunction>),
        ("swish", Box::new(Swish) as Box<dyn ActivationFunction>),
        ("gelu", Box::new(GELU) as Box<dyn ActivationFunction>),
    ];
    
    for (name, activation) in activations {
        group.throughput(Throughput::Elements(1000));
        group.bench_function(
            BenchmarkId::new("vectorized", name),
            |b| {
                b.iter(|| {
                    activation.activate_slice(black_box(&input_data), black_box(&mut output_data));
                });
            },
        );
        
        group.bench_function(
            BenchmarkId::new("scalar", name),
            |b| {
                b.iter(|| {
                    for &x in &input_data {
                        black_box(activation.activate(black_box(x)));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_network_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_sizes");
    
    let network_configs = [
        (10, 20, 5, "small"),
        (50, 100, 25, "medium"),
        (100, 200, 50, "large"),
        (500, 1000, 100, "xlarge"),
    ];
    
    for &(input_size, hidden_size, output_size, name) in &network_configs {
        let network = create_benchmark_network(input_size, hidden_size, output_size);
        let input: Vec<f32> = (0..input_size).map(|i| i as f32 / input_size as f32).collect();
        
        group.throughput(Throughput::Elements(1));
        group.bench_function(
            BenchmarkId::new("prediction", name),
            |b| {
                let mut net = network.clone();
                b.iter(|| {
                    let result = net.predict(black_box(&input));
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    let network = Arc::new(create_benchmark_network(100, 200, 50));
    
    for &num_samples in &[10, 100, 1000, 10000] {
        let inputs: Vec<Vec<f32>> = (0..num_samples)
            .map(|i| (0..100).map(|j| (i * j) as f32 / 1000.0).collect())
            .collect();
        
        group.throughput(Throughput::Elements(num_samples as u64));
        
        // With memory optimization
        group.bench_with_input(
            BenchmarkId::new("optimized", num_samples),
            &num_samples,
            |b, _| {
                let config = InferenceConfig {
                    mode: ExecutionMode::Sequential,
                    batch_size: 32,
                    memory_optimize: true,
                    ..Default::default()
                };
                let mut engine = InferenceEngine::new(network.clone(), config);
                
                b.iter(|| {
                    let result = engine.predict_batch(black_box(&inputs));
                    black_box(result)
                });
            },
        );
        
        // Without memory optimization
        group.bench_with_input(
            BenchmarkId::new("unoptimized", num_samples),
            &num_samples,
            |b, _| {
                let config = InferenceConfig {
                    mode: ExecutionMode::Sequential,
                    batch_size: 32,
                    memory_optimize: false,
                    ..Default::default()
                };
                let mut engine = InferenceEngine::new(network.clone(), config);
                
                b.iter(|| {
                    let result = engine.predict_batch(black_box(&inputs));
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_forward_propagation,
    bench_training_algorithms,
    bench_batch_inference,
    bench_activation_functions,
    bench_network_sizes,
    bench_memory_usage
);
criterion_main!(benches);