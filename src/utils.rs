//! Utility functions and helpers
//!
//! This module provides various utility functions used throughout the library.

use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Timing utilities for performance measurement
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    pub fn log_elapsed(&self) {
        log::info!("{}: {:.3}ms", self.name, self.elapsed_ms());
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("{}: {:.3}ms", self.name, self.elapsed_ms());
        }
    }
}

/// Memory usage tracking
pub struct MemoryTracker {
    initial_usage: usize,
    name: String,
}

impl MemoryTracker {
    pub fn new(name: &str) -> Self {
        Self {
            initial_usage: get_memory_usage(),
            name: name.to_string(),
        }
    }
    
    pub fn current_usage(&self) -> usize {
        get_memory_usage() - self.initial_usage
    }
    
    pub fn log_usage(&self) {
        let usage = self.current_usage();
        log::info!("{}: {:.2} MB", self.name, usage as f64 / (1024.0 * 1024.0));
    }
}

#[cfg(target_os = "linux")]
fn get_memory_usage() -> usize {
    use std::fs;
    
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<usize>() {
                        return kb * 1024; // Convert KB to bytes
                    }
                }
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn get_memory_usage() -> usize {
    // Fallback implementation for other platforms
    0
}

/// Random number utilities
pub struct RandomUtils;

impl RandomUtils {
    /// Generate random weights using Xavier/Glorot initialization
    pub fn xavier_weights(input_size: usize, output_size: usize) -> Vec<f32> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};
        
        let std_dev = (2.0 / (input_size + output_size) as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        let mut rng = rand::thread_rng();
        
        (0..input_size * output_size)
            .map(|_| normal.sample(&mut rng))
            .collect()
    }
    
    /// Generate random weights using He initialization (for ReLU networks)
    pub fn he_weights(input_size: usize, output_size: usize) -> Vec<f32> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};
        
        let std_dev = (2.0 / input_size as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        let mut rng = rand::thread_rng();
        
        (0..input_size * output_size)
            .map(|_| normal.sample(&mut rng))
            .collect()
    }
    
    /// Generate uniform random weights
    pub fn uniform_weights(input_size: usize, output_size: usize, range: f32) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        (0..input_size * output_size)
            .map(|_| rng.gen_range(-range..range))
            .collect()
    }
    
    /// Shuffle a vector in place
    pub fn shuffle<T>(vec: &mut [T]) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        vec.shuffle(&mut rng);
    }
}

/// Mathematical utilities
pub struct MathUtils;

impl MathUtils {
    /// Compute softmax with numerical stability
    pub fn softmax(input: &[f32]) -> Vec<f32> {
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        
        exp_values.iter().map(|&x| x / sum).collect()
    }
    
    /// Compute mean of a slice
    pub fn mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            0.0
        } else {
            data.iter().sum::<f32>() / data.len() as f32
        }
    }
    
    /// Compute variance of a slice
    pub fn variance(data: &[f32]) -> f32 {
        if data.len() <= 1 {
            0.0
        } else {
            let mean = Self::mean(data);
            let variance = data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / (data.len() - 1) as f32;
            variance
        }
    }
    
    /// Compute standard deviation
    pub fn std_dev(data: &[f32]) -> f32 {
        Self::variance(data).sqrt()
    }
    
    /// Normalize data to zero mean and unit variance
    pub fn normalize(data: &mut [f32]) {
        let mean = Self::mean(data);
        let std = Self::std_dev(data);
        
        if std > 1e-8 {
            for value in data.iter_mut() {
                *value = (*value - mean) / std;
            }
        }
    }
    
    /// Min-max normalization to [0, 1]
    pub fn min_max_normalize(data: &mut [f32]) {
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        
        if range > 1e-8 {
            for value in data.iter_mut() {
                *value = (*value - min_val) / range;
            }
        }
    }
    
    /// Compute L2 norm
    pub fn l2_norm(data: &[f32]) -> f32 {
        data.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
    
    /// Clamp values to a range
    pub fn clamp(data: &mut [f32], min_val: f32, max_val: f32) {
        for value in data.iter_mut() {
            *value = value.clamp(min_val, max_val);
        }
    }
}

/// Data format conversion utilities
pub struct DataUtils;

impl DataUtils {
    /// Convert flat array to 2D matrix representation
    pub fn flat_to_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
        assert_eq!(data.len(), rows * cols);
        
        data.chunks(cols)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    /// Convert 2D matrix to flat array
    pub fn matrix_to_flat(matrix: &[Vec<f32>]) -> Vec<f32> {
        matrix.iter().flat_map(|row| row.iter().cloned()).collect()
    }
    
    /// Pad vector to specified length with zeros
    pub fn pad_zeros(data: &mut Vec<f32>, target_length: usize) {
        if data.len() < target_length {
            data.resize(target_length, 0.0);
        }
    }
    
    /// Truncate vector to specified length
    pub fn truncate(data: &mut Vec<f32>, max_length: usize) {
        if data.len() > max_length {
            data.truncate(max_length);
        }
    }
    
    /// One-hot encode a class index
    pub fn one_hot_encode(class_index: usize, num_classes: usize) -> Vec<f32> {
        let mut result = vec![0.0; num_classes];
        if class_index < num_classes {
            result[class_index] = 1.0;
        }
        result
    }
    
    /// Convert one-hot encoded vector to class index
    pub fn one_hot_decode(one_hot: &[f32]) -> usize {
        one_hot.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Configuration utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub network: NetworkConfig,
    pub training: TrainingConfig,
    pub optimization: OptimizationConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub default_activation: String,
    pub weight_initialization: String,
    pub bias_initialization: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub default_epochs: usize,
    pub default_batch_size: usize,
    pub default_learning_rate: f32,
    pub early_stopping_patience: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub thread_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub enable_performance_logging: bool,
    pub enable_memory_logging: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            network: NetworkConfig {
                default_activation: "relu".to_string(),
                weight_initialization: "xavier".to_string(),
                bias_initialization: 0.0,
            },
            training: TrainingConfig {
                default_epochs: 100,
                default_batch_size: 32,
                default_learning_rate: 0.001,
                early_stopping_patience: 10,
            },
            optimization: OptimizationConfig {
                enable_simd: true,
                enable_parallel: true,
                thread_count: None,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                enable_performance_logging: true,
                enable_memory_logging: false,
            },
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: &str) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> crate::error::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Logging utilities
pub struct LoggingUtils;

impl LoggingUtils {
    /// Initialize logging with configuration
    pub fn init(config: &LoggingConfig) {
        let level = match config.level.to_lowercase().as_str() {
            "error" => log::LevelFilter::Error,
            "warn" => log::LevelFilter::Warn,
            "info" => log::LevelFilter::Info,
            "debug" => log::LevelFilter::Debug,
            "trace" => log::LevelFilter::Trace,
            _ => log::LevelFilter::Info,
        };
        
        env_logger::Builder::from_default_env()
            .filter_level(level)
            .init();
    }
    
    /// Log performance metrics
    pub fn log_performance(operation: &str, duration: Duration, throughput: Option<f64>) {
        if let Some(ops_per_sec) = throughput {
            log::info!("Performance [{}]: {:.3}ms ({:.2} ops/sec)", 
                      operation, duration.as_millis(), ops_per_sec);
        } else {
            log::info!("Performance [{}]: {:.3}ms", operation, duration.as_millis());
        }
    }
    
    /// Log memory usage
    pub fn log_memory(operation: &str, bytes: usize) {
        let mb = bytes as f64 / (1024.0 * 1024.0);
        log::info!("Memory [{}]: {:.2} MB", operation, mb);
    }
}

/// File I/O utilities
pub struct FileUtils;

impl FileUtils {
    /// Ensure directory exists
    pub fn ensure_dir(path: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(path)
    }
    
    /// Check if file exists
    pub fn file_exists(path: &str) -> bool {
        std::path::Path::new(path).exists()
    }
    
    /// Get file size in bytes
    pub fn file_size(path: &str) -> std::io::Result<u64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    /// Create backup of file
    pub fn backup_file(path: &str) -> std::io::Result<String> {
        let backup_path = format!("{}.backup", path);
        std::fs::copy(path, &backup_path)?;
        Ok(backup_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_math_utils() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_relative_eq!(MathUtils::mean(&data), 3.0, epsilon = 1e-6);
        assert_relative_eq!(MathUtils::l2_norm(&data), (1.0 + 4.0 + 9.0 + 16.0 + 25.0_f32).sqrt(), epsilon = 1e-6);
    }
    
    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let output = MathUtils::softmax(&input);
        
        // Check that outputs sum to 1
        let sum: f32 = output.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        
        // Check that all outputs are positive
        for &val in &output {
            assert!(val > 0.0);
        }
    }
    
    #[test]
    fn test_one_hot_encoding() {
        let encoded = DataUtils::one_hot_encode(2, 5);
        assert_eq!(encoded, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        
        let decoded = DataUtils::one_hot_decode(&encoded);
        assert_eq!(decoded, 2);
    }
    
    #[test]
    fn test_data_conversion() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = DataUtils::flat_to_matrix(&flat, 2, 3);
        
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(matrix[1], vec![4.0, 5.0, 6.0]);
        
        let flat_again = DataUtils::matrix_to_flat(&matrix);
        assert_eq!(flat_again, flat);
    }
    
    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
    }
}