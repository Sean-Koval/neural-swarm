/*!
 * Basic Neural Network Example
 * 
 * This example demonstrates the fundamental usage of FANN-Rust-Core for creating,
 * training, and using a neural network for classification tasks.
 */

use fann_rust_core::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("FANN-Rust-Core: Basic Neural Network Example");
    println!("===========================================");

    // Create a simple feedforward neural network
    // Architecture: 2 inputs -> 3 hidden -> 1 output
    let mut network = NetworkBuilder::new()
        .layers(&[2, 3, 1])
        .activation(ActivationFunction::Sigmoid)
        .output_activation(ActivationFunction::Sigmoid)
        .build()?;

    println!("✓ Created neural network with architecture: [2, 3, 1]");

    // Generate simple XOR training data
    let training_data = generate_xor_data();
    println!("✓ Generated XOR training data ({} samples)", training_data.len());

    // Configure training parameters
    let training_config = TrainingConfig {
        epochs: 5000,
        learning_rate: 0.1,
        batch_size: 4,
        validation_split: 0.0, // Use all data for training in this example
        early_stopping: Some(EarlyStoppingConfig {
            patience: 500,
            min_delta: 0.001,
        }),
        ..Default::default()
    };

    println!("✓ Configured training parameters");

    // Train the network
    println!("\n🚀 Starting training...");
    let training_results = network.train(&training_data, training_config)?;

    println!("✅ Training completed!");
    println!("   Final error: {:.6}", training_results.final_error);
    println!("   Epochs trained: {}", training_results.epochs_trained);
    println!("   Training time: {:.2}s", training_results.training_time.as_secs_f32());

    // Test the trained network
    println!("\n🧪 Testing the trained network:");
    test_network(&network)?;

    // Save the trained network
    network.save_to_file("xor_network.fann")?;
    println!("✓ Network saved to 'xor_network.fann'");

    // Demonstrate loading a saved network
    let loaded_network = FeedforwardNetwork::load_from_file("xor_network.fann")?;
    println!("✓ Network loaded from file");

    // Verify the loaded network works the same
    println!("\n🔄 Verifying loaded network:");
    test_network(&loaded_network)?;

    Ok(())
}

/// Generate XOR training data
fn generate_xor_data() -> TrainingData {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![
        vec![0.0],  // 0 XOR 0 = 0
        vec![1.0],  // 0 XOR 1 = 1
        vec![1.0],  // 1 XOR 0 = 1
        vec![0.0],  // 1 XOR 1 = 0
    ];

    TrainingData::new(inputs, targets).expect("Failed to create training data")
}

/// Test the network with XOR test cases
fn test_network(network: &FeedforwardNetwork) -> Result<(), Box<dyn Error>> {
    let test_cases = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    println!("   Input  | Expected | Actual   | Error");
    println!("   -------|----------|----------|-------");

    let mut total_error = 0.0;

    for (input, expected) in &test_cases {
        let output = network.forward(&input.to_vec())?;
        let actual = output[0];
        let error = (expected - actual).abs();
        total_error += error;

        println!("   {:?} |   {:.1}    |  {:.4}  | {:.4}", 
                 input, expected, actual, error);
    }

    let avg_error = total_error / test_cases.len() as f32;
    println!("   -------|----------|----------|-------");
    println!("   Average error: {:.6}", avg_error);

    if avg_error < 0.1 {
        println!("   ✅ Network learned XOR function successfully!");
    } else {
        println!("   ⚠️  Network may need more training (error > 0.1)");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_data_generation() {
        let data = generate_xor_data();
        assert_eq!(data.len(), 4);
        
        let samples = data.samples();
        assert_eq!(samples[0].input, vec![0.0, 0.0]);
        assert_eq!(samples[0].target, vec![0.0]);
        assert_eq!(samples[1].input, vec![0.0, 1.0]);
        assert_eq!(samples[1].target, vec![1.0]);
    }

    #[test]
    fn test_network_creation() {
        let network = NetworkBuilder::new()
            .layers(&[2, 3, 1])
            .build()
            .unwrap();

        assert_eq!(network.input_size(), 2);
        assert_eq!(network.output_size(), 1);
        assert_eq!(network.layer_count(), 3);
    }

    #[test]
    fn test_forward_pass() {
        let network = NetworkBuilder::new()
            .layers(&[2, 3, 1])
            .build()
            .unwrap();

        let input = vec![0.5, -0.3];
        let output = network.forward(&input).unwrap();

        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0); // Sigmoid output range
    }
}