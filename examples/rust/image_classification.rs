/*!
 * Image Classification Example
 * 
 * This example demonstrates using FANN-Rust-Core for MNIST digit classification
 * with advanced features like SIMD optimization, quantization, and performance monitoring.
 */

use fann_rust_core::prelude::*;
use fann_rust_core::optimization::{simd::*, quantization::*};
use fann_rust_core::utils::{data_loader::*, metrics::*};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    println!("FANN-Rust-Core: Image Classification Example");
    println!("============================================");

    // Load MNIST dataset
    println!("📁 Loading MNIST dataset...");
    let (train_images, train_labels, test_images, test_labels) = load_mnist_data()?;
    
    println!("✓ Dataset loaded:");
    println!("   Training samples: {}", train_images.len());
    println!("   Test samples: {}", test_images.len());
    println!("   Image size: {}x{}", 28, 28);

    // Create optimized neural network
    println!("\n🏗️  Creating optimized neural network...");
    let mut network = create_optimized_network()?;

    // Train the network
    println!("\n🚀 Training the network...");
    let training_results = train_network(&mut network, &train_images, &train_labels)?;

    // Evaluate on test set
    println!("\n📊 Evaluating on test set...");
    let test_accuracy = evaluate_network(&network, &test_images, &test_labels)?;

    // Demonstrate quantization
    println!("\n⚡ Demonstrating model quantization...");
    let quantized_network = quantize_model(&network, &train_images[..1000])?;

    // Compare performance
    println!("\n🏁 Performance comparison...");
    compare_performance(&network, &quantized_network, &test_images[..100])?;

    // Save models
    network.save_to_file("mnist_classifier.fann")?;
    quantized_network.save_to_file("mnist_classifier_quantized.fann")?;
    println!("✓ Models saved to disk");

    Ok(())
}

/// Create an optimized neural network for MNIST classification
fn create_optimized_network() -> Result<FeedforwardNetwork, NetworkError> {
    let network = NetworkBuilder::new()
        .add_layer(LayerConfig::dense(784)  // 28x28 input
            .activation(ActivationFunction::ReLU))
        .add_layer(LayerConfig::dense(256)
            .activation(ActivationFunction::ReLU)
            .dropout(0.2)
            .batch_norm())
        .add_layer(LayerConfig::dense(128)
            .activation(ActivationFunction::ReLU)
            .dropout(0.5))
        .add_layer(LayerConfig::dense(10)   // 10 digit classes
            .activation(ActivationFunction::Softmax))
        .optimizer(AdamOptimizer::new(0.001))
        .loss_function(CrossEntropyLoss)
        .regularization(RegularizationConfig {
            l1_weight: 0.0,
            l2_weight: 0.0001,
        })
        .optimization(OptimizationConfig {
            use_simd: true,
            parallel_training: true,
            quantization: None, // We'll quantize manually later
            memory_optimization: true,
        })
        .build()?;

    println!("✓ Created network with SIMD and parallel optimization");
    println!("   Architecture: 784 -> 256 -> 128 -> 10");
    println!("   Optimizer: Adam (lr=0.001)");
    println!("   Regularization: L2 (0.0001)");

    Ok(network)
}

/// Train the neural network
fn train_network(
    network: &mut FeedforwardNetwork,
    images: &[Vec<f32>],
    labels: &[Vec<f32>]
) -> Result<TrainingResults, Box<dyn Error>> {
    let training_data = TrainingData::new(images.to_vec(), labels.to_vec())?;

    let config = TrainingConfig {
        epochs: 50,
        learning_rate: 0.001,
        batch_size: 64,
        validation_split: 0.1,
        early_stopping: Some(EarlyStoppingConfig {
            patience: 5,
            min_delta: 0.001,
        }),
        progress_callback: Some(Box::new(|epoch, loss, accuracy| {
            if epoch % 5 == 0 {
                println!("   Epoch {}: loss={:.4}, accuracy={:.4}", epoch, loss, accuracy);
            }
        })),
        ..Default::default()
    };

    let start_time = Instant::now();
    let results = network.train(&training_data, config)?;
    let training_time = start_time.elapsed();

    println!("✅ Training completed!");
    println!("   Final loss: {:.6}", results.final_error);
    println!("   Best validation accuracy: {:.4}", results.best_validation_accuracy.unwrap_or(0.0));
    println!("   Training time: {:.2}s", training_time.as_secs_f32());
    println!("   Epochs trained: {}/{}", results.epochs_trained, 50);

    Ok(results)
}

/// Evaluate network performance on test set
fn evaluate_network(
    network: &FeedforwardNetwork,
    test_images: &[Vec<f32>],
    test_labels: &[Vec<f32>]
) -> Result<f32, Box<dyn Error>> {
    let mut correct_predictions = 0;
    let mut total_predictions = 0;
    let mut confusion_matrix = vec![vec![0; 10]; 10];

    println!("   Evaluating {} test samples...", test_images.len());

    for (image, label) in test_images.iter().zip(test_labels.iter()) {
        let prediction = network.forward(image)?;
        
        let predicted_class = argmax(&prediction);
        let actual_class = argmax(label);
        
        confusion_matrix[actual_class][predicted_class] += 1;
        
        if predicted_class == actual_class {
            correct_predictions += 1;
        }
        total_predictions += 1;
    }

    let accuracy = correct_predictions as f32 / total_predictions as f32;
    
    println!("✅ Test accuracy: {:.4} ({}/{} correct)", 
             accuracy, correct_predictions, total_predictions);

    // Print per-class accuracy
    println!("\n   Per-class accuracy:");
    for class in 0..10 {
        let class_total: i32 = confusion_matrix[class].iter().sum();
        let class_correct = confusion_matrix[class][class];
        let class_accuracy = if class_total > 0 {
            class_correct as f32 / class_total as f32
        } else {
            0.0
        };
        println!("     Class {}: {:.3}", class, class_accuracy);
    }

    Ok(accuracy)
}

/// Quantize the trained model
fn quantize_model(
    network: &FeedforwardNetwork,
    calibration_data: &[Vec<f32>]
) -> Result<QuantizedNetwork, Box<dyn Error>> {
    let mut quantizer = QuantizationEngine::new();
    
    // Calibrate quantization parameters
    println!("   Calibrating quantization with {} samples...", calibration_data.len());
    let params = quantizer.calibrate_quantization(network, calibration_data)?;
    
    // Quantize the model
    let quantized_network = quantizer.quantize_model(network, &params)?;
    
    let efficiency = quantized_network.memory_efficiency();
    println!("✓ Model quantized successfully");
    println!("   Memory reduction: {:.1f}x", efficiency.compression_ratio);
    println!("   Model size: {:.1f}KB -> {:.1f}KB", 
             efficiency.estimated_dense_memory as f32 / 1024.0,
             efficiency.sparse_memory_bytes as f32 / 1024.0);

    Ok(quantized_network)
}

/// Compare performance between original and quantized models
fn compare_performance(
    original: &FeedforwardNetwork,
    quantized: &QuantizedNetwork,
    test_samples: &[Vec<f32>]
) -> Result<(), Box<dyn Error>> {
    println!("   Testing on {} samples...", test_samples.len());

    // Benchmark original model
    let start_time = Instant::now();
    for sample in test_samples {
        let _ = original.forward(sample)?;
    }
    let original_time = start_time.elapsed();

    // Benchmark quantized model
    let start_time = Instant::now();
    for sample in test_samples {
        let _ = quantized.forward_quantized(sample);
    }
    let quantized_time = start_time.elapsed();

    // Compare accuracy on sample
    let mut accuracy_diff = 0.0;
    for sample in test_samples.iter().take(10) {
        let original_output = original.forward(sample)?;
        let quantized_output = quantized.forward_quantized(sample);
        
        let diff: f32 = original_output.iter()
            .zip(quantized_output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / original_output.len() as f32;
        
        accuracy_diff += diff;
    }
    accuracy_diff /= 10.0;

    println!("📈 Performance comparison:");
    println!("   Original model:");
    println!("     Inference time: {:.2}ms", original_time.as_millis() as f32 / test_samples.len() as f32);
    println!("   Quantized model:");
    println!("     Inference time: {:.2}ms", quantized_time.as_millis() as f32 / test_samples.len() as f32);
    println!("     Speedup: {:.1f}x", original_time.as_secs_f32() / quantized_time.as_secs_f32());
    println!("     Average output difference: {:.6}", accuracy_diff);

    Ok(())
}

/// Load MNIST dataset (mock implementation)
fn load_mnist_data() -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>), Box<dyn Error>> {
    // In a real implementation, this would load actual MNIST data
    // For this example, we'll generate mock data
    
    let train_size = 1000;
    let test_size = 200;
    
    let mut train_images = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_images = Vec::new();
    let mut test_labels = Vec::new();
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Generate training data
    for _ in 0..train_size {
        let image: Vec<f32> = (0..784).map(|_| rng.gen::<f32>()).collect();
        let class = rng.gen_range(0..10);
        let mut label = vec![0.0; 10];
        label[class] = 1.0;
        
        train_images.push(image);
        train_labels.push(label);
    }
    
    // Generate test data
    for _ in 0..test_size {
        let image: Vec<f32> = (0..784).map(|_| rng.gen::<f32>()).collect();
        let class = rng.gen_range(0..10);
        let mut label = vec![0.0; 10];
        label[class] = 1.0;
        
        test_images.push(image);
        test_labels.push(label);
    }
    
    Ok((train_images, train_labels, test_images, test_labels))
}

/// Find the index of the maximum value in a vector
fn argmax(values: &[f32]) -> usize {
    values.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let values = vec![0.1, 0.9, 0.3, 0.2];
        assert_eq!(argmax(&values), 1);
        
        let values = vec![0.8, 0.1, 0.1];
        assert_eq!(argmax(&values), 0);
    }

    #[test]
    fn test_network_creation() {
        let network = create_optimized_network().unwrap();
        assert_eq!(network.input_size(), 784);
        assert_eq!(network.output_size(), 10);
    }

    #[test]
    fn test_mock_data_generation() {
        let (train_images, train_labels, test_images, test_labels) = load_mnist_data().unwrap();
        
        assert_eq!(train_images.len(), 1000);
        assert_eq!(train_labels.len(), 1000);
        assert_eq!(test_images.len(), 200);
        assert_eq!(test_labels.len(), 200);
        
        // Check data dimensions
        assert_eq!(train_images[0].len(), 784);
        assert_eq!(train_labels[0].len(), 10);
    }
}