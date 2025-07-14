//! XOR neural network training example
//!
//! This example demonstrates training a neural network to learn the XOR function,
//! which is a classic non-linearly separable problem.

use neural_swarm::{
    activation::ActivationType,
    network::{LayerConfig, NeuralNetwork},
    training::{TrainingAlgorithm, TrainingData, TrainingParams, Trainer, DefaultCallback},
    inference::{ExecutionMode, InferenceConfig, InferenceEngine},
    utils::{DataNormalizer, NormalizationMethod, metrics},
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Neural Swarm XOR Example");
    println!("========================");

    // Create XOR training data
    let mut training_data = TrainingData::new();
    training_data.add_sample(vec![0.0, 0.0], vec![0.0]);
    training_data.add_sample(vec![0.0, 1.0], vec![1.0]);
    training_data.add_sample(vec![1.0, 0.0], vec![1.0]);
    training_data.add_sample(vec![1.0, 1.0], vec![0.0]);

    println!("Training data:");
    for sample in &training_data.samples {
        println!("  {:?} -> {:?}", sample.input, sample.target);
    }
    println!();

    // Create neural network architecture
    let layer_configs = vec![
        LayerConfig::new(2, ActivationType::Linear),      // Input layer
        LayerConfig::new(4, ActivationType::Sigmoid),     // Hidden layer
        LayerConfig::new(1, ActivationType::Sigmoid),     // Output layer
    ];

    let mut network = NeuralNetwork::new_feedforward(&layer_configs)?;
    network.initialize_weights(Some(42))?;

    println!("Network architecture:");
    println!("  Input size: {}", network.input_size());
    println!("  Output size: {}", network.output_size());
    println!("  Total weights: {}", network.num_weights());
    println!("  Total biases: {}", network.num_biases());
    println!();

    // Test different training algorithms
    let algorithms = vec![
        ("Backpropagation", TrainingAlgorithm::Backpropagation),
        ("Momentum", TrainingAlgorithm::BackpropagationMomentum),
        ("RPROP", TrainingAlgorithm::Rprop),
        ("Quickprop", TrainingAlgorithm::Quickprop),
    ];

    for (name, algorithm) in algorithms {
        println!("Training with {}...", name);
        
        let mut network_copy = network.clone();
        let mut params = TrainingParams::default();
        params.max_epochs = 1000;
        params.learning_rate = if algorithm == TrainingAlgorithm::Rprop { 0.1 } else { 0.7 };
        params.momentum = 0.5;
        params.desired_error = 0.01;
        params.report_interval = 200;

        let mut trainer = Trainer::new(algorithm, params);
        let mut callback = DefaultCallback::new(200);
        
        let start_time = std::time::Instant::now();
        let progress = trainer.train(&mut network_copy, &training_data, Some(&mut callback))?;
        let training_time = start_time.elapsed();

        println!("  Final error: {:.6}", progress.error);
        println!("  Best error: {:.6}", progress.best_error);
        println!("  Epochs: {}", progress.epoch);
        println!("  Training time: {:.2}s", training_time.as_secs_f32());

        // Test the trained network
        println!("  Test results:");
        let mut total_error = 0.0;
        for sample in &training_data.samples {
            let output = network_copy.predict(&sample.input)?;
            let error = (output[0] - sample.target[0]).abs();
            total_error += error;
            
            println!("    {:?} -> {:.3} (target: {:.1}, error: {:.3})", 
                    sample.input, output[0], sample.target[0], error);
        }
        println!("  Average absolute error: {:.3}", total_error / training_data.len() as f32);
        println!();
    }

    // Demonstrate inference engine usage
    println!("Inference Engine Demo");
    println!("====================");

    // Use the RPROP trained network for inference demo
    let mut network_copy = network.clone();
    let mut params = TrainingParams::default();
    params.max_epochs = 1000;
    params.learning_rate = 0.1;
    params.desired_error = 0.01;

    let mut trainer = Trainer::new(TrainingAlgorithm::Rprop, params);
    trainer.train(&mut network_copy, &training_data, None)?;

    let network_arc = Arc::new(network_copy);

    // Test different execution modes
    let execution_modes = vec![
        ("Sequential", ExecutionMode::Sequential),
        ("Parallel", ExecutionMode::Parallel),
    ];

    let test_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
        vec![0.2, 0.8],
    ];

    for (mode_name, execution_mode) in execution_modes {
        println!("Testing {} execution mode:", mode_name);
        
        let config = InferenceConfig {
            mode: execution_mode,
            batch_size: 4,
            ..Default::default()
        };
        
        let mut engine = InferenceEngine::new(network_arc.clone(), config);
        
        let start_time = std::time::Instant::now();
        let result = engine.predict_batch(&test_inputs)?;
        let inference_time = start_time.elapsed();
        
        println!("  Inference time: {:.2} μs", result.inference_time_us);
        println!("  Throughput: {:.0} samples/sec", result.throughput);
        println!("  Memory usage: {} bytes", result.memory_usage);
        println!("  Results:");
        
        for (input, output) in test_inputs.iter().zip(result.outputs.iter()) {
            println!("    {:?} -> {:.3}", input, output[0]);
        }
        println!();
    }

    // Demonstrate data normalization
    println!("Data Normalization Demo");
    println!("======================");

    let raw_data = vec![
        vec![10.0, 100.0],
        vec![20.0, 200.0],
        vec![30.0, 300.0],
        vec![40.0, 400.0],
    ];

    let normalization_methods = vec![
        ("MinMax", NormalizationMethod::MinMax),
        ("Z-Score", NormalizationMethod::ZScore),
        ("Robust", NormalizationMethod::Robust),
    ];

    for (method_name, method) in normalization_methods {
        println!("Using {} normalization:", method_name);
        
        let mut normalizer = DataNormalizer::new(method);
        let normalized_data = normalizer.fit_transform(&raw_data)?;
        
        println!("  Original -> Normalized");
        for (original, normalized) in raw_data.iter().zip(normalized_data.iter()) {
            println!("    {:?} -> {:?}", original, normalized);
        }
        
        // Test inverse transformation
        let restored_data = normalizer.inverse_transform(&normalized_data)?;
        println!("  Inverse transformation accuracy:");
        for (original, restored) in raw_data.iter().zip(restored_data.iter()) {
            let error = metrics::mae(original, restored)?;
            println!("    MAE: {:.6}", error);
        }
        println!();
    }

    // Performance comparison
    println!("Performance Comparison");
    println!("=====================");

    let large_test_data: Vec<Vec<f32>> = (0..1000)
        .map(|i| vec![(i % 2) as f32, ((i + 1) % 2) as f32])
        .collect();

    let config = InferenceConfig {
        mode: ExecutionMode::Sequential,
        batch_size: 32,
        ..Default::default()
    };
    
    let mut engine = InferenceEngine::new(network_arc.clone(), config);
    
    // Warmup
    engine.predict_batch(&large_test_data[0..10])?;
    
    // Benchmark
    let num_iterations = 10;
    let mut total_time = std::time::Duration::ZERO;
    
    for _ in 0..num_iterations {
        let start = std::time::Instant::now();
        engine.predict_batch(&large_test_data)?;
        total_time += start.elapsed();
    }
    
    let avg_time = total_time / num_iterations;
    let throughput = large_test_data.len() as f64 / avg_time.as_secs_f64();
    
    println!("Batch inference performance:");
    println!("  Samples: {}", large_test_data.len());
    println!("  Average time: {:.2} ms", avg_time.as_millis());
    println!("  Throughput: {:.0} samples/sec", throughput);

    println!("\nXOR example completed successfully!");
    Ok(())
}