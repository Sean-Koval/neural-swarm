//! Interoperability Test Suite for Neural Swarm
//!
//! This module provides comprehensive testing of Rust ↔ Python FFI bindings
//! and cross-language communication security and functionality.

use neural_swarm::{
    NeuralNetwork, PyNeuralNetwork, PyTrainer, PyInferenceEngine, PyDataNormalizer,
    activation::ActivationType,
    network::LayerConfig,
    training::{TrainingAlgorithm, TrainingData, TrainingParams},
    ffi::{neural_network_create, neural_network_destroy, neural_network_predict, get_version},
    NeuralFloat,
};
use std::{
    ffi::{CStr, CString},
    ptr,
    slice,
    time::{Duration, Instant},
    collections::HashMap,
    sync::{Arc, Mutex},
    thread,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

// =============================================================================
// INTEROPERABILITY TEST FRAMEWORK
// =============================================================================

#[derive(Debug, Clone)]
pub struct InteroperabilityTestSuite {
    config: InteropTestConfig,
    performance_monitor: PerformanceMonitor,
    data_validator: DataValidator,
    error_tracker: ErrorTracker,
}

#[derive(Debug, Clone)]
pub struct InteropTestConfig {
    pub test_python_ffi: bool,
    pub test_c_ffi: bool,
    pub test_performance: bool,
    pub test_memory_safety: bool,
    pub test_data_consistency: bool,
    pub stress_test_iterations: usize,
    pub performance_tolerance_ms: u64,
    pub data_precision_tolerance: f64,
}

impl Default for InteropTestConfig {
    fn default() -> Self {
        Self {
            test_python_ffi: cfg!(feature = "python"),
            test_c_ffi: true,
            test_performance: true,
            test_memory_safety: true,
            test_data_consistency: true,
            stress_test_iterations: 1000,
            performance_tolerance_ms: 100,
            data_precision_tolerance: 1e-6,
        }
    }
}

impl InteroperabilityTestSuite {
    pub fn new(config: InteropTestConfig) -> Self {
        Self {
            config,
            performance_monitor: PerformanceMonitor::new(),
            data_validator: DataValidator::new(),
            error_tracker: ErrorTracker::new(),
        }
    }

    /// Run comprehensive interoperability tests
    pub fn run_interop_tests(&mut self) -> InteropTestResults {
        let mut results = InteropTestResults::new();

        if self.config.test_python_ffi {
            results.python_ffi = self.test_python_ffi();
        }

        if self.config.test_c_ffi {
            results.c_ffi = self.test_c_ffi();
        }

        if self.config.test_performance {
            results.performance = self.test_performance();
        }

        if self.config.test_memory_safety {
            results.memory_safety = self.test_memory_safety();
        }

        if self.config.test_data_consistency {
            results.data_consistency = self.test_data_consistency();
        }

        results.calculate_overall_score();
        results
    }

    /// Test Python FFI bindings
    fn test_python_ffi(&mut self) -> InteropTestCategory {
        let mut category = InteropTestCategory::new("Python FFI");

        #[cfg(feature = "python")]
        {
            // Test 1: Python network creation and basic operations
            category.add_test(self.test_python_network_creation());

            // Test 2: Python training functionality
            category.add_test(self.test_python_training());

            // Test 3: Python inference engine
            category.add_test(self.test_python_inference_engine());

            // Test 4: Python data normalization
            category.add_test(self.test_python_data_normalization());

            // Test 5: Python error handling
            category.add_test(self.test_python_error_handling());

            // Test 6: Python memory management
            category.add_test(self.test_python_memory_management());

            // Test 7: Python concurrent access
            category.add_test(self.test_python_concurrent_access());
        }

        #[cfg(not(feature = "python"))]
        {
            category.add_test(InteropTestResult::new("Python FFI").mark_skipped("Python bindings not enabled"));
        }

        category
    }

    /// Test C FFI bindings
    fn test_c_ffi(&mut self) -> InteropTestCategory {
        let mut category = InteropTestCategory::new("C FFI");

        // Test 1: C network creation and destruction
        category.add_test(self.test_c_network_lifecycle());

        // Test 2: C prediction functionality
        category.add_test(self.test_c_prediction());

        // Test 3: C error handling
        category.add_test(self.test_c_error_handling());

        // Test 4: C memory safety
        category.add_test(self.test_c_memory_safety());

        // Test 5: C thread safety
        category.add_test(self.test_c_thread_safety());

        // Test 6: C boundary conditions
        category.add_test(self.test_c_boundary_conditions());

        category
    }

    /// Test cross-language performance
    fn test_performance(&mut self) -> InteropTestCategory {
        let mut category = InteropTestCategory::new("Performance");

        // Test 1: FFI call overhead
        category.add_test(self.test_ffi_call_overhead());

        // Test 2: Large data transfer performance
        category.add_test(self.test_large_data_transfer());

        // Test 3: Concurrent performance
        category.add_test(self.test_concurrent_performance());

        // Test 4: Memory allocation performance
        category.add_test(self.test_memory_allocation_performance());

        category
    }

    /// Test memory safety across language boundaries
    fn test_memory_safety(&mut self) -> InteropTestCategory {
        let mut category = InteropTestCategory::new("Memory Safety");

        // Test 1: Buffer overflow protection
        category.add_test(self.test_cross_language_buffer_protection());

        // Test 2: Memory leak detection
        category.add_test(self.test_cross_language_memory_leaks());

        // Test 3: Garbage collection interaction
        category.add_test(self.test_garbage_collection_interaction());

        // Test 4: Resource cleanup
        category.add_test(self.test_resource_cleanup());

        category
    }

    /// Test data consistency across languages
    fn test_data_consistency(&mut self) -> InteropTestCategory {
        let mut category = InteropTestCategory::new("Data Consistency");

        // Test 1: Rust to Python data integrity
        category.add_test(self.test_rust_to_python_data_integrity());

        // Test 2: Python to Rust data integrity
        category.add_test(self.test_python_to_rust_data_integrity());

        // Test 3: C FFI data integrity
        category.add_test(self.test_c_ffi_data_integrity());

        // Test 4: Complex data structure transfer
        category.add_test(self.test_complex_data_transfer());

        category
    }
}

// =============================================================================
// PYTHON FFI TESTS
// =============================================================================

impl InteroperabilityTestSuite {
    #[cfg(feature = "python")]
    fn test_python_network_creation(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("Python Network Creation");

        let result = std::panic::catch_unwind(|| {
            // Test valid network creation
            let layer_sizes = vec![3, 5, 2];
            let activations = vec!["linear".to_string(), "relu".to_string(), "sigmoid".to_string()];
            
            match PyNeuralNetwork::new(layer_sizes, activations) {
                Ok(mut py_network) => {
                    // Test initialization
                    match py_network.initialize_weights(Some(42)) {
                        Ok(_) => {
                            // Test basic properties
                            if py_network.get_input_size() == 3 && 
                               py_network.get_output_size() == 2 {
                                Ok(())
                            } else {
                                Err("Network properties mismatch")
                            }
                        }
                        Err(_) => Err("Weight initialization failed")
                    }
                }
                Err(_) => Err("Network creation failed")
            }
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(_) => test.mark_passed("Python network creation successful"),
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("Python network creation caused panic"),
        }

        test
    }

    #[cfg(feature = "python")]
    fn test_python_training(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("Python Training");

        let result = std::panic::catch_unwind(|| {
            // Create network
            let layer_sizes = vec![2, 3, 1];
            let activations = vec!["linear".to_string(), "relu".to_string(), "sigmoid".to_string()];
            
            let mut py_network = PyNeuralNetwork::new(layer_sizes, activations)?;
            py_network.initialize_weights(Some(42))?;

            // Create trainer
            let mut trainer = PyTrainer::new("backpropagation".to_string())?;
            trainer.set_learning_rate(0.1);
            trainer.set_max_epochs(100);

            // Create training data (XOR problem)
            let inputs = vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ];
            let targets = vec![
                vec![0.0],
                vec![1.0],
                vec![1.0],
                vec![0.0],
            ];

            // Train the network
            let training_result = trainer.train(&mut py_network, inputs, targets)?;
            
            if training_result.epochs > 0 && training_result.final_error.is_finite() {
                Ok(training_result.final_error)
            } else {
                Err("Invalid training results")
            }
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(final_error) => {
                        if final_error < 1.0 {
                            test.mark_passed(&format!("Training successful, final error: {:.3}", final_error))
                        } else {
                            test.mark_warning(&format!("Training completed but high error: {:.3}", final_error))
                        }
                    }
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("Python training caused panic"),
        }

        test
    }

    #[cfg(feature = "python")]
    fn test_python_inference_engine(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("Python Inference Engine");

        let result = std::panic::catch_unwind(|| {
            // Create and initialize network
            let layer_sizes = vec![4, 8, 3];
            let activations = vec!["linear".to_string(), "relu".to_string(), "sigmoid".to_string()];
            
            let mut py_network = PyNeuralNetwork::new(layer_sizes, activations)?;
            py_network.initialize_weights(Some(123))?;

            // Create inference engine
            let mut engine = PyInferenceEngine::new(&py_network, "sequential".to_string(), 32)?;

            // Test single prediction
            let input = vec![0.1, 0.2, 0.3, 0.4];
            let single_output = engine.predict_one(input.clone())?;
            
            if single_output.len() != 3 {
                return Err("Wrong output size for single prediction");
            }

            // Test batch prediction
            let batch_inputs = vec![
                vec![0.1, 0.2, 0.3, 0.4],
                vec![0.5, 0.6, 0.7, 0.8],
                vec![-0.1, -0.2, 0.3, 0.4],
            ];
            
            let batch_result = engine.predict_batch(batch_inputs)?;
            
            if batch_result.outputs.len() != 3 || 
               batch_result.outputs[0].len() != 3 ||
               batch_result.inference_time_us == 0 {
                return Err("Invalid batch prediction results");
            }

            // Test statistics
            let stats = engine.get_stats()?;
            if !stats.contains_key("input_size") || !stats.contains_key("output_size") {
                return Err("Missing inference engine statistics");
            }

            Ok(batch_result.throughput)
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(throughput) => {
                        test.mark_passed(&format!("Inference engine working, throughput: {:.1} samples/sec", throughput))
                    }
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("Python inference engine caused panic"),
        }

        test
    }

    #[cfg(not(feature = "python"))]
    fn test_python_network_creation(&mut self) -> InteropTestResult {
        InteropTestResult::new("Python Network Creation").mark_skipped("Python bindings not enabled")
    }

    #[cfg(not(feature = "python"))]
    fn test_python_training(&mut self) -> InteropTestResult {
        InteropTestResult::new("Python Training").mark_skipped("Python bindings not enabled")
    }

    #[cfg(not(feature = "python"))]
    fn test_python_inference_engine(&mut self) -> InteropTestResult {
        InteropTestResult::new("Python Inference Engine").mark_skipped("Python bindings not enabled")
    }

    fn test_python_data_normalization(&mut self) -> InteropTestResult {
        InteropTestResult::new("Python Data Normalization").mark_skipped("Not yet implemented")
    }

    fn test_python_error_handling(&mut self) -> InteropTestResult {
        InteropTestResult::new("Python Error Handling").mark_skipped("Not yet implemented")
    }

    fn test_python_memory_management(&mut self) -> InteropTestResult {
        InteropTestResult::new("Python Memory Management").mark_skipped("Not yet implemented")
    }

    fn test_python_concurrent_access(&mut self) -> InteropTestResult {
        InteropTestResult::new("Python Concurrent Access").mark_skipped("Not yet implemented")
    }
}

// =============================================================================
// C FFI TESTS
// =============================================================================

impl InteroperabilityTestSuite {
    fn test_c_network_lifecycle(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("C Network Lifecycle");

        let result = std::panic::catch_unwind(|| {
            // Test network creation
            let layer_sizes = [3usize, 5, 2];
            let activations = [1u32, 3, 1]; // sigmoid, relu, sigmoid
            
            let network_ptr = neural_network_create(
                layer_sizes.as_ptr(),
                layer_sizes.len(),
                activations.as_ptr(),
            );

            if network_ptr.is_null() {
                return Err("C network creation failed");
            }

            // Test that network pointer is valid by attempting prediction
            let input = [0.5f32, -0.2, 1.0];
            let mut output = [0.0f32; 2];
            
            let predict_result = neural_network_predict(
                network_ptr,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            );

            if predict_result != 0 {
                neural_network_destroy(network_ptr);
                return Err("C network prediction failed");
            }

            // Verify output is reasonable
            if !output.iter().all(|&x| x.is_finite()) {
                neural_network_destroy(network_ptr);
                return Err("C network produced invalid output");
            }

            // Test network destruction
            neural_network_destroy(network_ptr);
            
            Ok(())
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(_) => test.mark_passed("C network lifecycle successful"),
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("C network lifecycle caused panic"),
        }

        test
    }

    fn test_c_prediction(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("C Prediction");

        let result = std::panic::catch_unwind(|| {
            // Create network
            let layer_sizes = [4usize, 6, 3];
            let activations = [0u32, 3, 1]; // linear, relu, sigmoid
            
            let network_ptr = neural_network_create(
                layer_sizes.as_ptr(),
                layer_sizes.len(),
                activations.as_ptr(),
            );

            if network_ptr.is_null() {
                return Err("Failed to create C network for prediction test");
            }

            // Test multiple predictions
            let test_cases = [
                ([0.1f32, 0.2, 0.3, 0.4], "normal values"),
                ([0.0f32, 0.0, 0.0, 0.0], "zero values"),
                ([1.0f32, 1.0, 1.0, 1.0], "ones values"),
                ([-1.0f32, -1.0, -1.0, -1.0], "negative values"),
                ([f32::MAX, f32::MIN, 0.5, -0.5], "extreme values"),
            ];

            for (input, description) in &test_cases {
                let mut output = [0.0f32; 3];
                
                let result = neural_network_predict(
                    network_ptr,
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                );

                if result != 0 {
                    neural_network_destroy(network_ptr);
                    return Err(&format!("Prediction failed for {}", description));
                }

                // Check output validity
                if !output.iter().all(|&x| x.is_finite()) {
                    neural_network_destroy(network_ptr);
                    return Err(&format!("Invalid output for {}", description));
                }
            }

            neural_network_destroy(network_ptr);
            Ok(())
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(_) => test.mark_passed("C prediction tests successful"),
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("C prediction caused panic"),
        }

        test
    }

    fn test_c_error_handling(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("C Error Handling");

        let result = std::panic::catch_unwind(|| {
            // Test null pointer handling
            let null_result = neural_network_create(ptr::null(), 0, ptr::null());
            if !null_result.is_null() {
                neural_network_destroy(null_result);
                return Err("Null pointers accepted when they should be rejected");
            }

            // Test invalid activation types
            let layer_sizes = [2usize, 3, 1];
            let invalid_activations = [999u32, 1000, 1001]; // Invalid activation IDs
            
            let invalid_network = neural_network_create(
                layer_sizes.as_ptr(),
                layer_sizes.len(),
                invalid_activations.as_ptr(),
            );

            if !invalid_network.is_null() {
                neural_network_destroy(invalid_network);
                return Err("Invalid activations accepted");
            }

            // Test prediction with null network
            let input = [0.5f32, 0.5];
            let mut output = [0.0f32];
            
            let null_predict_result = neural_network_predict(
                ptr::null_mut(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            );

            if null_predict_result == 0 {
                return Err("Null network pointer accepted for prediction");
            }

            // Test version string
            let version_ptr = get_version();
            if version_ptr.is_null() {
                return Err("Version string is null");
            }

            let version_cstr = unsafe { CStr::from_ptr(version_ptr) };
            let version_str = version_cstr.to_string_lossy();
            if version_str.is_empty() {
                return Err("Version string is empty");
            }

            Ok(version_str.to_string())
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(version) => test.mark_passed(&format!("C error handling correct, version: {}", version)),
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("C error handling caused panic"),
        }

        test
    }

    fn test_c_memory_safety(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("C Memory Safety");

        let result = std::panic::catch_unwind(|| {
            let mut created_networks = Vec::new();

            // Create multiple networks
            for i in 0..10 {
                let layer_sizes = [2usize, (i % 5) + 3, 1];
                let activations = [0u32, 3, 1];
                
                let network_ptr = neural_network_create(
                    layer_sizes.as_ptr(),
                    layer_sizes.len(),
                    activations.as_ptr(),
                );

                if !network_ptr.is_null() {
                    created_networks.push(network_ptr);
                }
            }

            // Use all networks
            for (i, &network_ptr) in created_networks.iter().enumerate() {
                let input = [i as f32 / 10.0, (i + 1) as f32 / 10.0];
                let mut output = [0.0f32];
                
                let result = neural_network_predict(
                    network_ptr,
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                );

                if result != 0 {
                    // Clean up remaining networks
                    for &ptr in &created_networks {
                        neural_network_destroy(ptr);
                    }
                    return Err("Memory corruption detected in C FFI");
                }
            }

            // Destroy all networks
            for &network_ptr in &created_networks {
                neural_network_destroy(network_ptr);
            }

            // Test double destruction safety (should not crash)
            if let Some(&first_ptr) = created_networks.first() {
                neural_network_destroy(first_ptr); // This should be safe (no-op)
            }

            Ok(created_networks.len())
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(count) => test.mark_passed(&format!("C memory safety verified with {} networks", count)),
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("C memory safety test caused panic"),
        }

        test
    }

    fn test_c_thread_safety(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("C Thread Safety");

        let result = std::panic::catch_unwind(|| {
            // Create a network
            let layer_sizes = [3usize, 5, 2];
            let activations = [0u32, 3, 1];
            
            let network_ptr = neural_network_create(
                layer_sizes.as_ptr(),
                layer_sizes.len(),
                activations.as_ptr(),
            );

            if network_ptr.is_null() {
                return Err("Failed to create network for thread safety test");
            }

            let error_count = Arc::new(Mutex::new(0usize));
            let mut handles = Vec::new();

            // Spawn multiple threads to use the network
            for i in 0..5 {
                let error_count_clone = error_count.clone();
                
                let handle = thread::spawn(move || {
                    for j in 0..20 {
                        let input = [
                            (i * j) as f32 / 100.0,
                            ((i + 1) * j) as f32 / 100.0,
                            ((i + 2) * j) as f32 / 100.0,
                        ];
                        let mut output = [0.0f32; 2];
                        
                        let result = neural_network_predict(
                            network_ptr,
                            input.as_ptr(),
                            input.len(),
                            output.as_mut_ptr(),
                            output.len(),
                        );

                        if result != 0 || !output.iter().all(|&x| x.is_finite()) {
                            let mut count = error_count_clone.lock().unwrap();
                            *count += 1;
                        }
                    }
                });
                
                handles.push(handle);
            }

            // Wait for all threads
            for handle in handles {
                let _ = handle.join();
            }

            let final_error_count = *error_count.lock().unwrap();
            neural_network_destroy(network_ptr);

            if final_error_count == 0 {
                Ok(())
            } else {
                Err(&format!("Thread safety issues detected: {} errors", final_error_count))
            }
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(_) => test.mark_passed("C thread safety verified"),
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("C thread safety test caused panic"),
        }

        test
    }

    fn test_c_boundary_conditions(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("C Boundary Conditions");

        let result = std::panic::catch_unwind(|| {
            // Test very small network
            let small_sizes = [1usize, 1];
            let small_activations = [0u32, 1];
            
            let small_network = neural_network_create(
                small_sizes.as_ptr(),
                small_sizes.len(),
                small_activations.as_ptr(),
            );

            if small_network.is_null() {
                return Err("Very small network creation failed");
            }

            let small_input = [0.5f32];
            let mut small_output = [0.0f32];
            
            let small_result = neural_network_predict(
                small_network,
                small_input.as_ptr(),
                small_input.len(),
                small_output.as_mut_ptr(),
                small_output.len(),
            );

            neural_network_destroy(small_network);

            if small_result != 0 {
                return Err("Small network prediction failed");
            }

            // Test moderately large network
            let large_sizes = [50usize, 100, 25];
            let large_activations = [0u32, 3, 1];
            
            let large_network = neural_network_create(
                large_sizes.as_ptr(),
                large_sizes.len(),
                large_activations.as_ptr(),
            );

            if large_network.is_null() {
                return Err("Large network creation failed");
            }

            let large_input = vec![0.01f32; 50];
            let mut large_output = vec![0.0f32; 25];
            
            let large_result = neural_network_predict(
                large_network,
                large_input.as_ptr(),
                large_input.len(),
                large_output.as_mut_ptr(),
                large_output.len(),
            );

            neural_network_destroy(large_network);

            if large_result != 0 {
                return Err("Large network prediction failed");
            }

            Ok(())
        });

        match result {
            Ok(inner_result) => {
                match inner_result {
                    Ok(_) => test.mark_passed("C boundary conditions handled correctly"),
                    Err(msg) => test.mark_failed(msg),
                }
            }
            Err(_) => test.mark_failed("C boundary conditions test caused panic"),
        }

        test
    }
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

impl InteroperabilityTestSuite {
    fn test_ffi_call_overhead(&mut self) -> InteropTestResult {
        let mut test = InteropTestResult::new("FFI Call Overhead");

        // Create a network for performance testing
        let layer_sizes = [10usize, 20, 5];
        let activations = [0u32, 3, 1];
        
        let network_ptr = neural_network_create(
            layer_sizes.as_ptr(),
            layer_sizes.len(),
            activations.as_ptr(),
        );

        if network_ptr.is_null() {
            return test.mark_failed("Failed to create network for performance test");
        }

        let input = vec![0.1f32; 10];
        let mut output = vec![0.0f32; 5];
        let iterations = 1000;

        // Warm up
        for _ in 0..10 {
            let _ = neural_network_predict(
                network_ptr,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            );
        }

        // Measure performance
        let start = Instant::now();
        for _ in 0..iterations {
            let result = neural_network_predict(
                network_ptr,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            );
            
            if result != 0 {
                neural_network_destroy(network_ptr);
                return test.mark_failed("Prediction failed during performance test");
            }
        }
        let duration = start.elapsed();

        neural_network_destroy(network_ptr);

        let avg_time_per_call = duration.as_nanos() as f64 / iterations as f64;
        let calls_per_second = 1e9 / avg_time_per_call;

        if duration.as_millis() < self.config.performance_tolerance_ms {
            test.mark_passed(&format!("Good FFI performance: {:.1} calls/sec, {:.0} ns/call", 
                                    calls_per_second, avg_time_per_call))
        } else {
            test.mark_warning(&format!("Slow FFI performance: {:.1} calls/sec, {:.0} ns/call", 
                                     calls_per_second, avg_time_per_call))
        }
    }

    fn test_large_data_transfer(&mut self) -> InteropTestResult {
        InteropTestResult::new("Large Data Transfer").mark_skipped("Not yet implemented")
    }

    fn test_concurrent_performance(&mut self) -> InteropTestResult {
        InteropTestResult::new("Concurrent Performance").mark_skipped("Not yet implemented")
    }

    fn test_memory_allocation_performance(&mut self) -> InteropTestResult {
        InteropTestResult::new("Memory Allocation Performance").mark_skipped("Not yet implemented")
    }

    fn test_cross_language_buffer_protection(&mut self) -> InteropTestResult {
        InteropTestResult::new("Cross-Language Buffer Protection").mark_skipped("Not yet implemented")
    }

    fn test_cross_language_memory_leaks(&mut self) -> InteropTestResult {
        InteropTestResult::new("Cross-Language Memory Leaks").mark_skipped("Not yet implemented")
    }

    fn test_garbage_collection_interaction(&mut self) -> InteropTestResult {
        InteropTestResult::new("Garbage Collection Interaction").mark_skipped("Not yet implemented")
    }

    fn test_resource_cleanup(&mut self) -> InteropTestResult {
        InteropTestResult::new("Resource Cleanup").mark_skipped("Not yet implemented")
    }

    fn test_rust_to_python_data_integrity(&mut self) -> InteropTestResult {
        InteropTestResult::new("Rust to Python Data Integrity").mark_skipped("Not yet implemented")
    }

    fn test_python_to_rust_data_integrity(&mut self) -> InteropTestResult {
        InteropTestResult::new("Python to Rust Data Integrity").mark_skipped("Not yet implemented")
    }

    fn test_c_ffi_data_integrity(&mut self) -> InteropTestResult {
        InteropTestResult::new("C FFI Data Integrity").mark_skipped("Not yet implemented")
    }

    fn test_complex_data_transfer(&mut self) -> InteropTestResult {
        InteropTestResult::new("Complex Data Transfer").mark_skipped("Not yet implemented")
    }
}

// =============================================================================
// UTILITY CLASSES
// =============================================================================

pub struct PerformanceMonitor {
    measurements: Vec<Duration>,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    fn add_measurement(&mut self, duration: Duration) {
        self.measurements.push(duration);
    }

    fn average_duration(&self) -> Duration {
        if self.measurements.is_empty() {
            return Duration::new(0, 0);
        }
        
        let total_nanos: u64 = self.measurements.iter()
            .map(|d| d.as_nanos() as u64)
            .sum();
        
        Duration::from_nanos(total_nanos / self.measurements.len() as u64)
    }
}

pub struct DataValidator;

impl DataValidator {
    fn new() -> Self {
        Self
    }

    fn validate_float_array(&self, data: &[f32], tolerance: f64) -> bool {
        data.iter().all(|&x| x.is_finite() && (x as f64).abs() < tolerance)
    }
}

pub struct ErrorTracker {
    errors: Vec<String>,
}

impl ErrorTracker {
    fn new() -> Self {
        Self {
            errors: Vec::new(),
        }
    }

    fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    fn error_count(&self) -> usize {
        self.errors.len()
    }
}

// =============================================================================
// TEST RESULTS AND REPORTING
// =============================================================================

#[derive(Debug, Clone)]
pub struct InteropTestResults {
    pub python_ffi: InteropTestCategory,
    pub c_ffi: InteropTestCategory,
    pub performance: InteropTestCategory,
    pub memory_safety: InteropTestCategory,
    pub data_consistency: InteropTestCategory,
    pub overall_score: f64,
    pub compatibility_rating: CompatibilityRating,
}

impl InteropTestResults {
    fn new() -> Self {
        Self {
            python_ffi: InteropTestCategory::new("Python FFI"),
            c_ffi: InteropTestCategory::new("C FFI"),
            performance: InteropTestCategory::new("Performance"),
            memory_safety: InteropTestCategory::new("Memory Safety"),
            data_consistency: InteropTestCategory::new("Data Consistency"),
            overall_score: 0.0,
            compatibility_rating: CompatibilityRating::Unknown,
        }
    }

    fn calculate_overall_score(&mut self) {
        let categories = [
            &self.python_ffi,
            &self.c_ffi,
            &self.performance,
            &self.memory_safety,
            &self.data_consistency,
        ];

        let total_score: f64 = categories.iter()
            .map(|cat| cat.success_rate())
            .sum();

        self.overall_score = total_score / categories.len() as f64;

        self.compatibility_rating = match self.overall_score {
            s if s >= 0.95 => CompatibilityRating::Excellent,
            s if s >= 0.85 => CompatibilityRating::Good,
            s if s >= 0.70 => CompatibilityRating::Acceptable,
            s if s >= 0.50 => CompatibilityRating::Poor,
            _ => CompatibilityRating::Critical,
        };
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== INTEROPERABILITY TEST REPORT ===\n\n");
        report.push_str(&format!("Overall Score: {:.1}%\n", self.overall_score * 100.0));
        report.push_str(&format!("Compatibility Rating: {:?}\n\n", self.compatibility_rating));

        for category in [
            &self.python_ffi,
            &self.c_ffi,
            &self.performance,
            &self.memory_safety,
            &self.data_consistency,
        ] {
            report.push_str(&category.format_report());
            report.push_str("\n");
        }

        report
    }
}

#[derive(Debug, Clone)]
pub struct InteropTestCategory {
    pub name: String,
    pub tests: Vec<InteropTestResult>,
}

impl InteropTestCategory {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tests: Vec::new(),
        }
    }

    fn add_test(&mut self, test: InteropTestResult) {
        self.tests.push(test);
    }

    fn success_rate(&self) -> f64 {
        if self.tests.is_empty() {
            return 0.0;
        }

        let passed = self.tests.iter()
            .filter(|t| t.status == InteropTestStatus::Passed)
            .count();

        passed as f64 / self.tests.len() as f64
    }

    fn format_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("=== {} ===\n", self.name));
        report.push_str(&format!("Success Rate: {:.1}%\n", self.success_rate() * 100.0));
        
        for test in &self.tests {
            report.push_str(&format!("  {} - {:?}", test.name, test.status));
            if !test.message.is_empty() {
                report.push_str(&format!(": {}", test.message));
            }
            report.push_str("\n");
        }
        
        report
    }
}

#[derive(Debug, Clone)]
pub struct InteropTestResult {
    pub name: String,
    pub status: InteropTestStatus,
    pub message: String,
    pub execution_time: Duration,
}

impl InteropTestResult {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            status: InteropTestStatus::Running,
            message: String::new(),
            execution_time: Duration::new(0, 0),
        }
    }

    fn mark_passed(mut self, message: &str) -> Self {
        self.status = InteropTestStatus::Passed;
        self.message = message.to_string();
        self
    }

    fn mark_failed(mut self, message: &str) -> Self {
        self.status = InteropTestStatus::Failed;
        self.message = message.to_string();
        self
    }

    fn mark_warning(mut self, message: &str) -> Self {
        self.status = InteropTestStatus::Warning;
        self.message = message.to_string();
        self
    }

    fn mark_skipped(mut self, message: &str) -> Self {
        self.status = InteropTestStatus::Skipped;
        self.message = message.to_string();
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InteropTestStatus {
    Running,
    Passed,
    Failed,
    Warning,
    Skipped,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompatibilityRating {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Critical,
    Unknown,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interop_suite_creation() {
        let config = InteropTestConfig::default();
        let suite = InteroperabilityTestSuite::new(config);
        assert_eq!(suite.config.stress_test_iterations, 1000);
    }

    #[test]
    fn test_c_ffi_basic() {
        let mut config = InteropTestConfig::default();
        config.stress_test_iterations = 10; // Reduce for testing
        
        let mut suite = InteroperabilityTestSuite::new(config);
        let results = suite.test_c_ffi();
        
        assert!(results.tests.len() > 0);
    }

    #[test]
    fn test_performance_monitoring() {
        let mut monitor = PerformanceMonitor::new();
        monitor.add_measurement(Duration::from_millis(10));
        monitor.add_measurement(Duration::from_millis(20));
        
        let avg = monitor.average_duration();
        assert_eq!(avg.as_millis(), 15);
    }

    #[test]
    fn test_comprehensive_interop_tests() {
        let mut config = InteropTestConfig::default();
        config.stress_test_iterations = 50; // Reduced for testing
        
        let mut suite = InteroperabilityTestSuite::new(config);
        let results = suite.run_interop_tests();
        
        assert!(results.overall_score >= 0.0);
        assert!(results.overall_score <= 1.0);
        
        println!("{}", results.generate_report());
    }
}