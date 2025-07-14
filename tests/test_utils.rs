// Test Utilities for Neural Swarm FANN Tests
// Common functionality and helpers for comprehensive neural network testing

use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};

/// FANN-compatible training data format parser
#[derive(Debug, Clone)]
pub struct FannDataParser;

impl FannDataParser {
    /// Parse FANN training data format into internal representation
    /// Format: first line "num_patterns num_inputs num_outputs"
    /// Then alternating lines of inputs and outputs
    pub fn parse_training_file(content: &str) -> Result<Vec<(Vec<f32>, Vec<f32>)>, FannParseError> {
        let lines: Vec<&str> = content.trim().split('\n').collect();
        if lines.is_empty() {
            return Err(FannParseError::EmptyFile);
        }

        // Parse header
        let header: Vec<u32> = lines[0]
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| FannParseError::InvalidHeader)?;

        if header.len() != 3 {
            return Err(FannParseError::InvalidHeaderFormat);
        }

        let num_patterns = header[0] as usize;
        let num_inputs = header[1] as usize;
        let num_outputs = header[2] as usize;

        let mut training_data = Vec::new();
        let mut line_idx = 1;

        for pattern_idx in 0..num_patterns {
            if line_idx + 1 >= lines.len() {
                return Err(FannParseError::InsufficientData {
                    pattern: pattern_idx,
                    expected_lines: num_patterns * 2,
                    actual_lines: lines.len() - 1,
                });
            }

            // Parse inputs
            let inputs: Vec<f32> = lines[line_idx]
                .split_whitespace()
                .map(|s| s.parse())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| FannParseError::InvalidInputData { pattern: pattern_idx })?;

            // Parse outputs
            let outputs: Vec<f32> = lines[line_idx + 1]
                .split_whitespace()
                .map(|s| s.parse())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| FannParseError::InvalidOutputData { pattern: pattern_idx })?;

            if inputs.len() != num_inputs {
                return Err(FannParseError::WrongInputSize {
                    pattern: pattern_idx,
                    expected: num_inputs,
                    actual: inputs.len(),
                });
            }

            if outputs.len() != num_outputs {
                return Err(FannParseError::WrongOutputSize {
                    pattern: pattern_idx,
                    expected: num_outputs,
                    actual: outputs.len(),
                });
            }

            training_data.push((inputs, outputs));
            line_idx += 2;
        }

        Ok(training_data)
    }

    /// Generate FANN format data file content
    pub fn generate_training_file(data: &[(Vec<f32>, Vec<f32>)]) -> String {
        if data.is_empty() {
            return "0 0 0\n".to_string();
        }

        let num_patterns = data.len();
        let num_inputs = data[0].0.len();
        let num_outputs = data[0].1.len();

        let mut content = format!("{} {} {}\n", num_patterns, num_inputs, num_outputs);

        for (inputs, outputs) in data {
            // Input line
            content.push_str(&inputs.iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<_>>()
                .join(" "));
            content.push('\n');

            // Output line
            content.push_str(&outputs.iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<_>>()
                .join(" "));
            content.push('\n');
        }

        content
    }

    /// Create canonical XOR training data in FANN format
    pub fn xor_data() -> String {
        let data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        Self::generate_training_file(&data)
    }

    /// Create scaled training data (normalize between -1 and 1)
    pub fn scale_data(data: &[(Vec<f32>, Vec<f32>)], min_val: f32, max_val: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
        data.iter().map(|(inputs, outputs)| {
            let scaled_inputs = inputs.iter()
                .map(|&x| min_val + (max_val - min_val) * x)
                .collect();
            let scaled_outputs = outputs.iter()
                .map(|&x| min_val + (max_val - min_val) * x)
                .collect();
            (scaled_inputs, scaled_outputs)
        }).collect()
    }
}

/// Errors that can occur during FANN data parsing
#[derive(Debug, thiserror::Error)]
pub enum FannParseError {
    #[error("Empty data file")]
    EmptyFile,
    
    #[error("Invalid header format")]
    InvalidHeader,
    
    #[error("Header must have exactly 3 numbers: num_patterns num_inputs num_outputs")]
    InvalidHeaderFormat,
    
    #[error("Insufficient data: pattern {pattern}, expected {expected_lines} lines, got {actual_lines}")]
    InsufficientData {
        pattern: usize,
        expected_lines: usize,
        actual_lines: usize,
    },
    
    #[error("Invalid input data at pattern {pattern}")]
    InvalidInputData { pattern: usize },
    
    #[error("Invalid output data at pattern {pattern}")]
    InvalidOutputData { pattern: usize },
    
    #[error("Wrong input size at pattern {pattern}: expected {expected}, got {actual}")]
    WrongInputSize {
        pattern: usize,
        expected: usize,
        actual: usize,
    },
    
    #[error("Wrong output size at pattern {pattern}: expected {expected}, got {actual}")]
    WrongOutputSize {
        pattern: usize,
        expected: usize,
        actual: usize,
    },
}

/// Network validation utilities
#[derive(Debug, Clone)]
pub struct NetworkValidator;

impl NetworkValidator {
    /// Validate network architecture parameters
    pub fn validate_architecture(
        input_size: u32,
        output_size: u32,
        hidden_layers: &[u32]
    ) -> Result<(), ValidationError> {
        if input_size == 0 {
            return Err(ValidationError::InvalidInputSize(input_size));
        }
        
        if output_size == 0 {
            return Err(ValidationError::InvalidOutputSize(output_size));
        }
        
        for (i, &layer_size) in hidden_layers.iter().enumerate() {
            if layer_size == 0 {
                return Err(ValidationError::InvalidHiddenLayerSize { layer: i, size: layer_size });
            }
        }
        
        Ok(())
    }

    /// Validate training parameters
    pub fn validate_training_params(
        learning_rate: f32,
        desired_error: f32,
        max_iterations: u32
    ) -> Result<(), ValidationError> {
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(ValidationError::InvalidLearningRate(learning_rate));
        }
        
        if desired_error < 0.0 {
            return Err(ValidationError::InvalidDesiredError(desired_error));
        }
        
        if max_iterations == 0 {
            return Err(ValidationError::InvalidMaxIterations(max_iterations));
        }
        
        Ok(())
    }

    /// Calculate classification accuracy with tolerance
    pub fn calculate_accuracy(
        predictions: &[(Vec<f32>, Vec<f32>)], // (predicted, expected)
        tolerance: f32
    ) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }

        let mut correct = 0;
        for (predicted, expected) in predictions {
            let mut case_correct = true;
            for (pred, exp) in predicted.iter().zip(expected.iter()) {
                if (pred - exp).abs() > tolerance {
                    case_correct = false;
                    break;
                }
            }
            if case_correct {
                correct += 1;
            }
        }

        correct as f32 / predictions.len() as f32
    }

    /// Calculate Mean Squared Error
    pub fn calculate_mse(predictions: &[(Vec<f32>, Vec<f32>)]) -> f32 {
        if predictions.is_empty() {
            return f32::INFINITY;
        }

        let mut total_error = 0.0;
        let mut total_elements = 0;

        for (predicted, expected) in predictions {
            for (pred, exp) in predicted.iter().zip(expected.iter()) {
                total_error += (pred - exp).powi(2);
                total_elements += 1;
            }
        }

        total_error / total_elements as f32
    }

    /// Calculate bit fail count (predictions outside tolerance)
    pub fn calculate_bit_fails(
        predictions: &[(Vec<f32>, Vec<f32>)],
        bit_fail_limit: f32
    ) -> u32 {
        let mut bit_fails = 0;
        for (predicted, expected) in predictions {
            for (pred, exp) in predicted.iter().zip(expected.iter()) {
                if (pred - exp).abs() > bit_fail_limit {
                    bit_fails += 1;
                }
            }
        }
        bit_fails
    }
}

/// Validation errors
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Invalid input size: {0} (must be > 0)")]
    InvalidInputSize(u32),
    
    #[error("Invalid output size: {0} (must be > 0)")]
    InvalidOutputSize(u32),
    
    #[error("Invalid hidden layer {layer} size: {size} (must be > 0)")]
    InvalidHiddenLayerSize { layer: usize, size: u32 },
    
    #[error("Invalid learning rate: {0} (must be in (0, 1])")]
    InvalidLearningRate(f32),
    
    #[error("Invalid desired error: {0} (must be >= 0)")]
    InvalidDesiredError(f32),
    
    #[error("Invalid max iterations: {0} (must be > 0)")]
    InvalidMaxIterations(u32),
}

/// Test data generators for common neural network problems
#[derive(Debug, Clone)]
pub struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate XOR problem data
    pub fn xor() -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ]
    }

    /// Generate AND problem data
    pub fn and() -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![0.0]),
            (vec![1.0, 0.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ]
    }

    /// Generate OR problem data
    pub fn or() -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ]
    }

    /// Generate n-bit parity problem data
    pub fn parity(bits: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut data = Vec::new();
        
        for i in 0..(1 << bits) {
            let mut input = Vec::new();
            let mut ones_count = 0;
            
            for bit in 0..bits {
                let value = if (i >> bit) & 1 == 1 { 1.0 } else { 0.0 };
                input.push(value);
                if value > 0.5 {
                    ones_count += 1;
                }
            }
            
            let output = vec![if ones_count % 2 == 1 { 1.0 } else { 0.0 }];
            data.push((input, output));
        }
        
        data
    }

    /// Generate sine wave approximation data
    pub fn sine_wave(samples: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut data = Vec::new();
        
        for i in 0..samples {
            let x = (i as f32 / samples as f32) * 2.0 * std::f32::consts::PI;
            let y = x.sin();
            data.push((vec![x], vec![y]));
        }
        
        data
    }

    /// Generate simple linear regression data
    pub fn linear_regression(samples: usize, slope: f32, intercept: f32, noise: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut data = Vec::new();
        
        for i in 0..samples {
            let x = (i as f32 / samples as f32) * 10.0 - 5.0; // Range [-5, 5]
            let noise_val = (i as f32 * 0.1).sin() * noise; // Deterministic "noise"
            let y = slope * x + intercept + noise_val;
            data.push((vec![x], vec![y]));
        }
        
        data
    }

    /// Generate circle classification data
    pub fn circle_classification(samples_per_class: usize, radius: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut data = Vec::new();
        
        // Inner circle (class 0)
        for i in 0..samples_per_class {
            let angle = (i as f32 / samples_per_class as f32) * 2.0 * std::f32::consts::PI;
            let r = radius * 0.5 * (1.0 + 0.1 * (i as f32).sin()); // Slight variation
            let x = r * angle.cos();
            let y = r * angle.sin();
            data.push((vec![x, y], vec![0.0]));
        }
        
        // Outer circle (class 1)
        for i in 0..samples_per_class {
            let angle = (i as f32 / samples_per_class as f32) * 2.0 * std::f32::consts::PI;
            let r = radius * (1.2 + 0.1 * (i as f32).cos()); // Slight variation
            let x = r * angle.cos();
            let y = r * angle.sin();
            data.push((vec![x, y], vec![1.0]));
        }
        
        data
    }
}

/// Performance measurement utilities
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    measurements: HashMap<String, Vec<u64>>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            measurements: HashMap::new(),
        }
    }

    /// Time a function execution
    pub async fn time_async<F, Fut, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = std::time::Instant::now();
        let result = f().await;
        let duration = start.elapsed();
        
        self.measurements
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration.as_micros() as u64);
        
        result
    }

    /// Get statistics for a measurement
    pub fn get_stats(&self, name: &str) -> Option<PerformanceStats> {
        self.measurements.get(name).map(|times| {
            let mut sorted_times = times.clone();
            sorted_times.sort();
            
            let count = sorted_times.len();
            let sum: u64 = sorted_times.iter().sum();
            let mean = sum as f64 / count as f64;
            
            let min = *sorted_times.first().unwrap_or(&0);
            let max = *sorted_times.last().unwrap_or(&0);
            let median = if count % 2 == 0 {
                (sorted_times[count / 2 - 1] + sorted_times[count / 2]) as f64 / 2.0
            } else {
                sorted_times[count / 2] as f64
            };

            PerformanceStats {
                name: name.to_string(),
                count,
                mean_us: mean,
                median_us: median,
                min_us: min,
                max_us: max,
                total_us: sum,
            }
        })
    }

    /// Get all statistics
    pub fn get_all_stats(&self) -> Vec<PerformanceStats> {
        self.measurements
            .keys()
            .filter_map(|name| self.get_stats(name))
            .collect()
    }

    /// Print performance report
    pub fn print_report(&self) {
        println!("\n📊 Performance Report");
        println!("=====================");
        
        for stats in self.get_all_stats() {
            println!("{}", stats);
        }
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub name: String,
    pub count: usize,
    pub mean_us: f64,
    pub median_us: f64,
    pub min_us: u64,
    pub max_us: u64,
    pub total_us: u64,
}

impl std::fmt::Display for PerformanceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} calls, mean: {:.1}μs, median: {:.1}μs, range: [{}, {}]μs",
            self.name, self.count, self.mean_us, self.median_us, self.min_us, self.max_us
        )
    }
}

/// File system utilities for tests
#[derive(Debug, Clone)]
pub struct TestFileManager;

impl TestFileManager {
    /// Create a temporary FANN training data file
    pub fn create_temp_training_file(data: &[(Vec<f32>, Vec<f32>)]) -> Result<tempfile::NamedTempFile, std::io::Error> {
        let content = FannDataParser::generate_training_file(data);
        let mut temp_file = tempfile::NamedTempFile::new()?;
        std::io::Write::write_all(&mut temp_file, content.as_bytes())?;
        Ok(temp_file)
    }

    /// Load training data from file
    pub fn load_training_data<P: AsRef<Path>>(path: P) -> Result<Vec<(Vec<f32>, Vec<f32>)>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(FannDataParser::parse_training_file(&content)?)
    }

    /// Save training data to file
    pub fn save_training_data<P: AsRef<Path>>(
        path: P,
        data: &[(Vec<f32>, Vec<f32>)]
    ) -> Result<(), std::io::Error> {
        let content = FannDataParser::generate_training_file(data);
        std::fs::write(path, content)
    }
}

/// Common test assertions for neural networks
pub struct NeuralTestAssertions;

impl NeuralTestAssertions {
    /// Assert that a network can learn XOR with reasonable accuracy
    pub fn assert_xor_learning(accuracy: f32, min_expected: f32) {
        assert!(
            accuracy >= min_expected,
            "XOR learning accuracy {:.1}% should be at least {:.1}%",
            accuracy * 100.0,
            min_expected * 100.0
        );
    }

    /// Assert that training improves network performance
    pub fn assert_training_improvement(initial_error: f32, final_error: f32) {
        assert!(
            final_error <= initial_error,
            "Training should improve network: initial error {:.6} -> final error {:.6}",
            initial_error,
            final_error
        );
    }

    /// Assert network output is in valid range
    pub fn assert_output_range(output: &[f32], min: f32, max: f32) {
        for (i, &value) in output.iter().enumerate() {
            assert!(
                value >= min && value <= max,
                "Output {} value {:.6} should be in range [{}, {}]",
                i, value, min, max
            );
        }
    }

    /// Assert training converges within iteration limit
    pub fn assert_convergence(final_iteration: u32, max_iterations: u32, converged: bool) {
        if !converged {
            assert!(
                final_iteration < max_iterations,
                "Training should converge within {} iterations, ran for {}",
                max_iterations,
                final_iteration
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fann_data_parser() {
        let xor_data = FannDataParser::xor_data();
        let parsed = FannDataParser::parse_training_file(&xor_data).unwrap();
        
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0], (vec![0.0, 0.0], vec![0.0]));
        assert_eq!(parsed[1], (vec![0.0, 1.0], vec![1.0]));
        assert_eq!(parsed[2], (vec![1.0, 0.0], vec![1.0]));
        assert_eq!(parsed[3], (vec![1.0, 1.0], vec![0.0]));
    }

    #[test]
    fn test_network_validator() {
        // Valid architecture
        assert!(NetworkValidator::validate_architecture(2, 1, &[4]).is_ok());
        
        // Invalid input size
        assert!(NetworkValidator::validate_architecture(0, 1, &[4]).is_err());
        
        // Valid training params
        assert!(NetworkValidator::validate_training_params(0.7, 0.001, 1000).is_ok());
        
        // Invalid learning rate
        assert!(NetworkValidator::validate_training_params(1.5, 0.001, 1000).is_err());
    }

    #[test]
    fn test_accuracy_calculation() {
        let predictions = vec![
            (vec![0.1], vec![0.0]), // Close enough
            (vec![0.9], vec![1.0]), // Close enough
            (vec![0.5], vec![0.0]), // Too far
            (vec![0.6], vec![1.0]), // Too far
        ];
        
        let accuracy = NetworkValidator::calculate_accuracy(&predictions, 0.2);
        assert_eq!(accuracy, 0.5); // 2 out of 4 correct
    }

    #[test]
    fn test_test_data_generators() {
        let xor_data = TestDataGenerator::xor();
        assert_eq!(xor_data.len(), 4);
        
        let parity_data = TestDataGenerator::parity(3);
        assert_eq!(parity_data.len(), 8); // 2^3
        
        let sine_data = TestDataGenerator::sine_wave(10);
        assert_eq!(sine_data.len(), 10);
    }

    #[tokio::test]
    async fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();
        
        let result = profiler.time_async("test_operation", || async {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            42
        }).await;
        
        assert_eq!(result, 42);
        
        let stats = profiler.get_stats("test_operation").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.mean_us > 1000); // At least 1ms
    }
}