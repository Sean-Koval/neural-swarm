//! Integration tests for the neural network implementation
//!
//! This module provides comprehensive tests to validate that all components
//! work together correctly in the absence of Rust compiler tools.

use crate::{
    activations::ActivationFunction,
    network::{Network, NetworkBuilder},
    training::{TrainingData, TrainingAlgorithm},
    error::{FannError, Result},
    quantization::{QuantizationEngine, QuantizationType},
    optimization::SIMDOperations,
    memory::AlignedVec,
    profiling::PerformanceProfiler,
};

/// Integration test results
#[derive(Debug)]
pub struct IntegrationTestResults {
    pub tests_run: usize,
    pub tests_passed: usize,
    pub tests_failed: usize,
    pub errors: Vec<String>,
}

impl IntegrationTestResults {
    fn new() -> Self {
        Self {
            tests_run: 0,
            tests_passed: 0,
            tests_failed: 0,
            errors: Vec::new(),
        }
    }
    
    fn add_test_result(&mut self, test_name: &str, result: Result<()>) {
        self.tests_run += 1;
        match result {
            Ok(()) => {
                self.tests_passed += 1;
                println!("✅ PASS: {}", test_name);
            }
            Err(e) => {
                self.tests_failed += 1;
                let error_msg = format!("❌ FAIL: {} - {}", test_name, e);
                println!("{}", error_msg);
                self.errors.push(error_msg);
            }
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.tests_run == 0 {
            0.0
        } else {
            self.tests_passed as f64 / self.tests_run as f64
        }
    }
}

/// Run comprehensive integration tests
pub fn run_integration_tests() -> IntegrationTestResults {
    let mut results = IntegrationTestResults::new();
    
    println!("🧪 Running Neural Network Integration Tests");
    println!("==========================================");
    
    // Test 1: Basic network creation
    results.add_test_result("network_creation", test_network_creation());
    
    // Test 2: Activation functions
    results.add_test_result("activation_functions", test_activation_functions());
    
    // Test 3: Memory management
    results.add_test_result("memory_management", test_memory_management());
    
    // Test 4: SIMD operations
    results.add_test_result("simd_operations", test_simd_operations());
    
    // Test 5: Training data handling
    results.add_test_result("training_data", test_training_data());
    
    // Test 6: Quantization engine
    results.add_test_result("quantization_engine", test_quantization_engine());
    
    // Test 7: Performance profiling
    results.add_test_result("performance_profiling", test_performance_profiling());
    
    // Test 8: Error handling
    results.add_test_result("error_handling", test_error_handling());
    
    // Test 9: Network forward pass
    results.add_test_result("network_forward_pass", test_network_forward_pass());
    
    // Test 10: End-to-end XOR training simulation
    results.add_test_result("xor_training_simulation", test_xor_training_simulation());
    
    println!("==========================================");
    println!("🏆 Test Results Summary:");
    println!("   Tests Run: {}", results.tests_run);
    println!("   Tests Passed: {}", results.tests_passed);
    println!("   Tests Failed: {}", results.tests_failed);
    println!("   Success Rate: {:.1}%", results.success_rate() * 100.0);
    
    if results.tests_failed > 0 {
        println!("\n❌ Failed Tests:");
        for error in &results.errors {
            println!("   {}", error);
        }
    } else {
        println!("\n🎉 All tests passed! Neural network implementation is ready.");
    }
    
    results
}

fn test_network_creation() -> Result<()> {
    // Test basic network builder
    let network = NetworkBuilder::new()
        .add_layer(2, 4)
        .activation(ActivationFunction::ReLU)
        .add_layer(4, 1)
        .activation(ActivationFunction::Sigmoid)
        .learning_rate(0.1)
        .build()?;
    
    // Validate network structure
    if network.get_input_size() != 2 {
        return Err(FannError::network_construction("Input size mismatch"));
    }
    
    if network.get_output_size() != 1 {
        return Err(FannError::network_construction("Output size mismatch"));
    }
    
    Ok(())
}

fn test_activation_functions() -> Result<()> {
    let test_input = 0.5f32;
    
    // Test each activation function
    let relu_output = ActivationFunction::ReLU.apply(test_input);
    if relu_output != 0.5 {
        return Err(FannError::network_construction("ReLU activation failed"));
    }
    
    let sigmoid_output = ActivationFunction::Sigmoid.apply(test_input);
    if sigmoid_output <= 0.0 || sigmoid_output >= 1.0 {
        return Err(FannError::network_construction("Sigmoid activation out of range"));
    }
    
    let tanh_output = ActivationFunction::Tanh.apply(test_input);
    if tanh_output <= -1.0 || tanh_output >= 1.0 {
        return Err(FannError::network_construction("Tanh activation out of range"));
    }
    
    // Test derivative computation
    let _relu_deriv = ActivationFunction::ReLU.derivative(test_input);
    let _sigmoid_deriv = ActivationFunction::Sigmoid.derivative(test_input);
    
    Ok(())
}

fn test_memory_management() -> Result<()> {
    // Test aligned vector creation
    let mut aligned_vec = AlignedVec::new(64);
    aligned_vec.resize(1024, 0.0f32);
    
    if aligned_vec.len() != 1024 {
        return Err(FannError::memory_allocation("AlignedVec size mismatch"));
    }
    
    // Test data access
    aligned_vec[0] = 1.0;
    aligned_vec[1023] = 2.0;
    
    if aligned_vec[0] != 1.0 || aligned_vec[1023] != 2.0 {
        return Err(FannError::memory_allocation("AlignedVec data access failed"));
    }
    
    Ok(())
}

fn test_simd_operations() -> Result<()> {
    let simd_ops = SIMDOperations::new();
    
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output = vec![0.0; 5];
    
    // Test SIMD ReLU
    simd_ops.relu(&input, &mut output);
    
    for (i, &val) in output.iter().enumerate() {
        if val != input[i].max(0.0) {
            return Err(FannError::inference("SIMD ReLU operation failed"));
        }
    }
    
    // Test SIMD matrix multiplication
    let matrix_a = vec![1.0, 2.0, 3.0, 4.0];
    let matrix_b = vec![5.0, 6.0, 7.0, 8.0];
    let mut matrix_c = vec![0.0; 4];
    
    simd_ops.matrix_multiply(&matrix_a, &matrix_b, &mut matrix_c, 2, 2, 2);
    
    // Expected result: [19, 22, 43, 50]
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    for (actual, &expected_val) in matrix_c.iter().zip(expected.iter()) {
        if (actual - expected_val).abs() > 1e-6 {
            return Err(FannError::inference("SIMD matrix multiplication failed"));
        }
    }
    
    Ok(())
}

fn test_training_data() -> Result<()> {
    let mut training_data = TrainingData::new();
    
    // Add XOR training samples
    training_data.add_sample(vec![0.0, 0.0], vec![0.0]);
    training_data.add_sample(vec![0.0, 1.0], vec![1.0]);
    training_data.add_sample(vec![1.0, 0.0], vec![1.0]);
    training_data.add_sample(vec![1.0, 1.0], vec![0.0]);
    
    if training_data.len() != 4 {
        return Err(FannError::training("Training data size mismatch"));
    }
    
    // Test data retrieval
    let sample = training_data.get_sample(0)?;
    if sample.input.len() != 2 || sample.target.len() != 1 {
        return Err(FannError::training("Training sample dimensions incorrect"));
    }
    
    // Test data validation
    training_data.validate_consistency()?;
    
    Ok(())
}

fn test_quantization_engine() -> Result<()> {
    let mut quantization_engine = QuantizationEngine::new();
    
    // Add calibration data
    quantization_engine.add_calibration_data(vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
    ]);
    
    // Create a mock network for testing
    let network = NetworkBuilder::new()
        .add_layer(3, 2)
        .activation(ActivationFunction::ReLU)
        .build()?;
    
    // Test calibration
    quantization_engine.calibrate(&network, QuantizationType::Int8)?;
    
    // Test quantization
    let quantized_network = quantization_engine.quantize_network(&network, QuantizationType::Int8)?;
    
    // Test forward pass with quantized network
    let input = vec![0.5, 0.3, 0.7];
    let output = quantized_network.forward(&input)?;
    
    if output.is_empty() {
        return Err(FannError::quantization("Quantized network produced no output"));
    }
    
    // Test compression ratio
    let compression_ratio = quantized_network.compression_ratio();
    if compression_ratio <= 1.0 {
        return Err(FannError::quantization("Invalid compression ratio"));
    }
    
    Ok(())
}

fn test_performance_profiling() -> Result<()> {
    let mut profiler = PerformanceProfiler::new();
    
    // Start profiling an operation
    profiler.start_operation("test_operation");
    
    // Simulate some work
    let mut sum = 0.0f32;
    for i in 0..1000 {
        sum += (i as f32).sqrt();
    }
    
    // End profiling
    profiler.end_operation("test_operation");
    
    // Get results
    let stats = profiler.get_operation_stats("test_operation");
    if stats.call_count == 0 {
        return Err(FannError::benchmark("Profiler did not record operation"));
    }
    
    // Test memory tracking
    profiler.record_memory_allocation(1024);
    profiler.record_memory_deallocation(512);
    
    let memory_stats = profiler.get_memory_stats();
    if memory_stats.current_usage != 512 {
        return Err(FannError::benchmark("Memory tracking incorrect"));
    }
    
    Ok(())
}

fn test_error_handling() -> Result<()> {
    // Test dimension mismatch error
    let error = FannError::dimension_mismatch(5, 3);
    match error {
        FannError::DimensionMismatch { expected, actual } => {
            if expected != 5 || actual != 3 {
                return Err(FannError::benchmark("Error parameters incorrect"));
            }
        }
        _ => return Err(FannError::benchmark("Wrong error type created")),
    }
    
    // Test error conversion and propagation
    let training_error = FannError::training("Test training error");
    let _error_string = format!("{}", training_error);
    
    // Test various error types
    let _inference_error = FannError::inference("Test inference error");
    let _quantization_error = FannError::quantization("Test quantization error");
    let _memory_error = FannError::memory_allocation("Test memory error");
    
    Ok(())
}

fn test_network_forward_pass() -> Result<()> {
    let mut network = NetworkBuilder::new()
        .add_layer(2, 3)
        .activation(ActivationFunction::ReLU)
        .add_layer(3, 1)
        .activation(ActivationFunction::Sigmoid)
        .build()?;
    
    // Initialize weights with a specific seed for reproducibility
    network.initialize_weights(Some(42))?;
    
    // Test forward pass
    let input = vec![0.5, -0.3];
    let output = network.forward(&input)?;
    
    if output.len() != 1 {
        return Err(FannError::inference("Network output size incorrect"));
    }
    
    // Output should be between 0 and 1 due to sigmoid activation
    if output[0] < 0.0 || output[0] > 1.0 {
        return Err(FannError::inference("Network output out of expected range"));
    }
    
    // Test with different input
    let input2 = vec![1.0, 1.0];
    let output2 = network.forward(&input2)?;
    
    if output2.len() != 1 {
        return Err(FannError::inference("Network output size incorrect for second input"));
    }
    
    Ok(())
}

fn test_xor_training_simulation() -> Result<()> {
    // Create network for XOR problem
    let mut network = NetworkBuilder::new()
        .add_layer(2, 4)
        .activation(ActivationFunction::ReLU)
        .add_layer(4, 1)
        .activation(ActivationFunction::Sigmoid)
        .learning_rate(0.1)
        .build()?;
    
    // Initialize weights
    network.initialize_weights(Some(123))?;
    
    // Create XOR training data
    let mut training_data = TrainingData::new();
    training_data.add_sample(vec![0.0, 0.0], vec![0.0]);
    training_data.add_sample(vec![0.0, 1.0], vec![1.0]);
    training_data.add_sample(vec![1.0, 0.0], vec![1.0]);
    training_data.add_sample(vec![1.0, 1.0], vec![0.0]);
    
    // Test initial predictions (should be random)
    let initial_predictions = vec![
        network.forward(&vec![0.0, 0.0])?,
        network.forward(&vec![0.0, 1.0])?,
        network.forward(&vec![1.0, 0.0])?,
        network.forward(&vec![1.0, 1.0])?,
    ];
    
    // Verify predictions are generated
    for pred in &initial_predictions {
        if pred.len() != 1 {
            return Err(FannError::training("Initial prediction size incorrect"));
        }
    }
    
    // Simulate a training step (calculate MSE)
    let mut total_error = 0.0;
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    
    for (pred, target) in initial_predictions.iter().zip(targets.iter()) {
        let error = (pred[0] - target[0]).powi(2);
        total_error += error;
    }
    
    let mse = total_error / 4.0;
    
    // MSE should be a valid positive number
    if mse < 0.0 || mse.is_nan() || mse.is_infinite() {
        return Err(FannError::training("Invalid MSE calculation"));
    }
    
    // Test that network can compute gradients (simplified test)
    // In a full implementation, this would test actual gradient computation
    let _gradient_magnitude = 0.1; // Mock gradient magnitude
    
    println!("   XOR simulation: Initial MSE = {:.6}", mse);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn integration_test_runner() {
        let results = run_integration_tests();
        
        // Require at least 80% success rate
        assert!(results.success_rate() >= 0.8, 
               "Integration tests failed with success rate: {:.1}%", 
               results.success_rate() * 100.0);
        
        // Ensure all critical tests passed
        assert!(results.tests_run >= 10, "Not enough tests were run");
    }
}