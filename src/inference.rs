//! High-performance inference engine for neural networks
//!
//! This module provides optimized inference capabilities including batch processing,
//! SIMD optimizations, and various execution modes for maximum performance.

use crate::{network::NeuralNetwork, NeuralFloat, NeuralError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "simd")]
use wide::*;

/// Inference execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Single-threaded execution
    Sequential,
    /// Multi-threaded execution using Rayon
    Parallel,
    /// SIMD-optimized execution
    Simd,
    /// Combined SIMD + parallel execution
    SimdParallel,
}

/// Batch inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Execution mode
    pub mode: ExecutionMode,
    /// Batch size for processing
    pub batch_size: usize,
    /// Number of threads for parallel execution
    pub num_threads: Option<usize>,
    /// Enable memory optimization
    pub memory_optimize: bool,
    /// Enable input validation
    pub validate_inputs: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::SimdParallel,
            batch_size: 32,
            num_threads: None, // Use all available cores
            memory_optimize: true,
            validate_inputs: true,
        }
    }
}

/// Inference result with timing and metadata
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Output predictions
    pub outputs: Vec<Vec<NeuralFloat>>,
    /// Inference time in microseconds
    pub inference_time_us: u64,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

/// High-performance inference engine
pub struct InferenceEngine {
    /// Network reference
    network: Arc<NeuralNetwork>,
    /// Inference configuration
    config: InferenceConfig,
    /// Pre-allocated batch buffers
    batch_inputs: Option<Array2<NeuralFloat>>,
    batch_outputs: Option<Array2<NeuralFloat>>,
    /// Layer intermediate values for batch processing
    layer_activations: Vec<Array2<NeuralFloat>>,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(network: Arc<NeuralNetwork>, config: InferenceConfig) -> Self {
        let mut engine = Self {
            network,
            config,
            batch_inputs: None,
            batch_outputs: None,
            layer_activations: Vec::new(),
        };
        
        if config.memory_optimize {
            engine.preallocate_buffers();
        }
        
        engine
    }

    /// Create with default configuration
    pub fn new_default(network: Arc<NeuralNetwork>) -> Self {
        Self::new(network, InferenceConfig::default())
    }

    /// Preallocate memory buffers for batch processing
    fn preallocate_buffers(&mut self) {
        let batch_size = self.config.batch_size;
        let input_size = self.network.input_size();
        let output_size = self.network.output_size();

        // Preallocate batch buffers
        self.batch_inputs = Some(Array2::zeros((batch_size, input_size)));
        self.batch_outputs = Some(Array2::zeros((batch_size, output_size)));

        // Preallocate layer activation buffers
        self.layer_activations.clear();
        for layer in &self.network.layers {
            self.layer_activations.push(Array2::zeros((batch_size, layer.config.size)));
        }
    }

    /// Single sample inference
    pub fn predict_one(&self, input: &[NeuralFloat]) -> Result<Vec<NeuralFloat>> {
        if self.config.validate_inputs {
            self.validate_input(input)?;
        }

        match self.config.mode {
            ExecutionMode::Simd => self.predict_one_simd(input),
            _ => self.predict_one_standard(input),
        }
    }

    /// Standard single sample prediction
    fn predict_one_standard(&self, input: &[NeuralFloat]) -> Result<Vec<NeuralFloat>> {
        // Clone network for thread safety (not optimal, but works)
        let mut network_copy = (*self.network).clone();
        network_copy.set_input(input)?;
        network_copy.forward_propagate()?;
        Ok(network_copy.get_output().to_vec())
    }

    /// SIMD-optimized single sample prediction
    #[cfg(feature = "simd")]
    fn predict_one_simd(&self, input: &[NeuralFloat]) -> Result<Vec<NeuralFloat>> {
        // For single samples, SIMD provides limited benefit
        // This is more of a demonstration - batch processing is where SIMD shines
        self.predict_one_standard(input)
    }

    #[cfg(not(feature = "simd"))]
    fn predict_one_simd(&self, input: &[NeuralFloat]) -> Result<Vec<NeuralFloat>> {
        self.predict_one_standard(input)
    }

    /// Batch inference for multiple samples
    pub fn predict_batch(&mut self, inputs: &[Vec<NeuralFloat>]) -> Result<InferenceResult> {
        if inputs.is_empty() {
            return Err(NeuralError::inference("Input batch is empty"));
        }

        if self.config.validate_inputs {
            for input in inputs {
                self.validate_input(input)?;
            }
        }

        let start_time = std::time::Instant::now();

        let outputs = match self.config.mode {
            ExecutionMode::Sequential => self.predict_batch_sequential(inputs)?,
            ExecutionMode::Parallel => self.predict_batch_parallel(inputs)?,
            ExecutionMode::Simd => self.predict_batch_simd(inputs)?,
            ExecutionMode::SimdParallel => self.predict_batch_simd_parallel(inputs)?,
        };

        let elapsed = start_time.elapsed();
        let inference_time_us = elapsed.as_micros() as u64;
        let throughput = inputs.len() as f64 / elapsed.as_secs_f64();

        // Estimate memory usage
        let memory_usage = self.estimate_memory_usage(inputs.len());

        Ok(InferenceResult {
            outputs,
            inference_time_us,
            throughput,
            memory_usage,
        })
    }

    /// Sequential batch processing
    fn predict_batch_sequential(&self, inputs: &[Vec<NeuralFloat>]) -> Result<Vec<Vec<NeuralFloat>>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            outputs.push(self.predict_one_standard(input)?);
        }
        
        Ok(outputs)
    }

    /// Parallel batch processing using Rayon
    fn predict_batch_parallel(&self, inputs: &[Vec<NeuralFloat>]) -> Result<Vec<Vec<NeuralFloat>>> {
        // Configure Rayon thread pool if specified
        if let Some(num_threads) = self.config.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .map_err(|e| NeuralError::inference(format!("Failed to set thread count: {}", e)))?;
        }

        let results: Result<Vec<Vec<NeuralFloat>>> = inputs
            .par_iter()
            .map(|input| self.predict_one_standard(input))
            .collect();

        results
    }

    /// SIMD-optimized batch processing
    #[cfg(feature = "simd")]
    fn predict_batch_simd(&mut self, inputs: &[Vec<NeuralFloat>]) -> Result<Vec<Vec<NeuralFloat>>> {
        // Process in chunks that align with SIMD width
        const SIMD_WIDTH: usize = 8; // f32x8
        
        if inputs.len() < SIMD_WIDTH {
            return self.predict_batch_sequential(inputs);
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        
        // Process SIMD-aligned chunks
        for chunk in inputs.chunks(SIMD_WIDTH) {
            if chunk.len() == SIMD_WIDTH {
                outputs.extend(self.process_simd_chunk(chunk)?);
            } else {
                // Handle remainder with sequential processing
                for input in chunk {
                    outputs.push(self.predict_one_standard(input)?);
                }
            }
        }
        
        Ok(outputs)
    }

    #[cfg(not(feature = "simd"))]
    fn predict_batch_simd(&mut self, inputs: &[Vec<NeuralFloat>]) -> Result<Vec<Vec<NeuralFloat>>> {
        self.predict_batch_sequential(inputs)
    }

    /// Combined SIMD + parallel processing
    fn predict_batch_simd_parallel(&mut self, inputs: &[Vec<NeuralFloat>]) -> Result<Vec<Vec<NeuralFloat>>> {
        // For simplicity, fall back to parallel processing
        // In a real implementation, you'd want to combine both optimizations
        self.predict_batch_parallel(inputs)
    }

    /// Process a SIMD-aligned chunk of inputs
    #[cfg(feature = "simd")]
    fn process_simd_chunk(&self, chunk: &[Vec<NeuralFloat>]) -> Result<Vec<Vec<NeuralFloat>>> {
        // This is a simplified SIMD implementation
        // In practice, you'd want to vectorize the entire forward pass
        
        let mut results = Vec::with_capacity(chunk.len());
        
        // For demonstration, process each sample individually
        // Real SIMD implementation would process multiple samples simultaneously
        for input in chunk {
            results.push(self.predict_one_standard(input)?);
        }
        
        Ok(results)
    }

    /// Streaming inference for continuous data
    pub fn predict_stream<I>(&mut self, inputs: I) -> impl Iterator<Item = Result<Vec<NeuralFloat>>> + '_
    where
        I: Iterator<Item = Vec<NeuralFloat>>,
    {
        inputs.map(move |input| self.predict_one(&input))
    }

    /// Batch streaming with configurable batch size
    pub fn predict_batch_stream<I>(&mut self, inputs: I) -> impl Iterator<Item = Result<InferenceResult>> + '_
    where
        I: Iterator<Item = Vec<NeuralFloat>>,
    {
        BatchIterator::new(inputs, self.config.batch_size).map(move |batch| {
            self.predict_batch(&batch)
        })
    }

    /// Get inference statistics
    pub fn get_stats(&self) -> InferenceStats {
        InferenceStats {
            network_input_size: self.network.input_size(),
            network_output_size: self.network.output_size(),
            num_layers: self.network.layers.len(),
            num_weights: self.network.num_weights(),
            num_biases: self.network.num_biases(),
            execution_mode: self.config.mode,
            batch_size: self.config.batch_size,
            memory_optimized: self.config.memory_optimize,
        }
    }

    /// Validate input dimensions
    fn validate_input(&self, input: &[NeuralFloat]) -> Result<()> {
        if input.len() != self.network.input_size() {
            return Err(NeuralError::invalid_dimensions(
                self.network.input_size(),
                input.len(),
            ));
        }
        
        // Check for NaN or infinite values
        for &value in input {
            if !value.is_finite() {
                return Err(NeuralError::inference("Input contains NaN or infinite values"));
            }
        }
        
        Ok(())
    }

    /// Estimate memory usage for batch processing
    fn estimate_memory_usage(&self, batch_size: usize) -> usize {
        let mut memory = 0;
        
        // Input and output buffers
        memory += batch_size * self.network.input_size() * std::mem::size_of::<NeuralFloat>();
        memory += batch_size * self.network.output_size() * std::mem::size_of::<NeuralFloat>();
        
        // Layer activations
        for layer in &self.network.layers {
            memory += batch_size * layer.config.size * std::mem::size_of::<NeuralFloat>();
        }
        
        // Network weights (shared, but count once)
        memory += self.network.num_weights() * std::mem::size_of::<NeuralFloat>();
        memory += self.network.num_biases() * std::mem::size_of::<NeuralFloat>();
        
        memory
    }

    /// Update configuration
    pub fn set_config(&mut self, config: InferenceConfig) {
        let need_realloc = config.batch_size != self.config.batch_size 
                          || config.memory_optimize != self.config.memory_optimize;
        
        self.config = config;
        
        if need_realloc && self.config.memory_optimize {
            self.preallocate_buffers();
        }
    }

    /// Get current configuration
    pub fn get_config(&self) -> &InferenceConfig {
        &self.config
    }
}

/// Batch iterator for streaming inference
struct BatchIterator<I> {
    inner: I,
    batch_size: usize,
}

impl<I> BatchIterator<I>
where
    I: Iterator<Item = Vec<NeuralFloat>>,
{
    fn new(inner: I, batch_size: usize) -> Self {
        Self { inner, batch_size }
    }
}

impl<I> Iterator for BatchIterator<I>
where
    I: Iterator<Item = Vec<NeuralFloat>>,
{
    type Item = Vec<Vec<NeuralFloat>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        
        for _ in 0..self.batch_size {
            match self.inner.next() {
                Some(item) => batch.push(item),
                None => break,
            }
        }
        
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

/// Inference engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    pub network_input_size: usize,
    pub network_output_size: usize,
    pub num_layers: usize,
    pub num_weights: usize,
    pub num_biases: usize,
    pub execution_mode: ExecutionMode,
    pub batch_size: usize,
    pub memory_optimized: bool,
}

/// Utility functions for inference optimization
pub mod utils {
    use super::*;

    /// Benchmark inference performance
    pub fn benchmark_inference(
        engine: &mut InferenceEngine,
        test_inputs: &[Vec<NeuralFloat>],
        num_iterations: usize,
    ) -> Result<BenchmarkResult> {
        let mut total_time = std::time::Duration::ZERO;
        let mut total_samples = 0;

        for _ in 0..num_iterations {
            let start = std::time::Instant::now();
            let _result = engine.predict_batch(test_inputs)?;
            total_time += start.elapsed();
            total_samples += test_inputs.len();
        }

        let avg_time_per_batch = total_time / num_iterations as u32;
        let throughput = total_samples as f64 / total_time.as_secs_f64();
        let latency_per_sample = total_time / total_samples as u32;

        Ok(BenchmarkResult {
            avg_time_per_batch,
            throughput,
            latency_per_sample,
            total_samples,
            num_iterations,
        })
    }

    /// Optimize batch size for given hardware
    pub fn optimize_batch_size(
        engine: &mut InferenceEngine,
        test_inputs: &[Vec<NeuralFloat>],
        max_batch_size: usize,
    ) -> Result<usize> {
        let mut best_batch_size = 1;
        let mut best_throughput = 0.0;

        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            if batch_size > max_batch_size || batch_size > test_inputs.len() {
                break;
            }

            let mut config = engine.get_config().clone();
            config.batch_size = batch_size;
            engine.set_config(config);

            let benchmark = benchmark_inference(engine, test_inputs, 5)?;
            
            if benchmark.throughput > best_throughput {
                best_throughput = benchmark.throughput;
                best_batch_size = batch_size;
            }
        }

        Ok(best_batch_size)
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub avg_time_per_batch: std::time::Duration,
    pub throughput: f64, // samples per second
    pub latency_per_sample: std::time::Duration,
    pub total_samples: usize,
    pub num_iterations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{activation::ActivationType, network::LayerConfig};

    fn create_test_network() -> Arc<NeuralNetwork> {
        let configs = vec![
            LayerConfig::new(4, ActivationType::Linear),
            LayerConfig::new(5, ActivationType::ReLU),
            LayerConfig::new(3, ActivationType::Sigmoid),
        ];
        
        let mut network = NeuralNetwork::new_feedforward(&configs).unwrap();
        network.initialize_weights(Some(42)).unwrap();
        
        Arc::new(network)
    }

    #[test]
    fn test_inference_engine_creation() {
        let network = create_test_network();
        let engine = InferenceEngine::new_default(network);
        
        let stats = engine.get_stats();
        assert_eq!(stats.network_input_size, 4);
        assert_eq!(stats.network_output_size, 3);
        assert_eq!(stats.num_layers, 3);
    }

    #[test]
    fn test_single_prediction() {
        let network = create_test_network();
        let engine = InferenceEngine::new_default(network);
        
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = engine.predict_one(&input);
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_batch_prediction() {
        let network = create_test_network();
        let mut engine = InferenceEngine::new_default(network);
        
        let inputs = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        
        let result = engine.predict_batch(&inputs);
        assert!(result.is_ok());
        
        let inference_result = result.unwrap();
        assert_eq!(inference_result.outputs.len(), 3);
        assert!(inference_result.throughput > 0.0);
        assert!(inference_result.inference_time_us > 0);
    }

    #[test]
    fn test_execution_modes() {
        let network = create_test_network();
        let inputs = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        for mode in [ExecutionMode::Sequential, ExecutionMode::Parallel] {
            let config = InferenceConfig {
                mode,
                batch_size: 2,
                ..Default::default()
            };
            
            let mut engine = InferenceEngine::new(network.clone(), config);
            let result = engine.predict_batch(&inputs);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_input_validation() {
        let network = create_test_network();
        let engine = InferenceEngine::new_default(network);
        
        // Wrong input size
        let wrong_input = vec![1.0, 2.0]; // Should be 4 elements
        let result = engine.predict_one(&wrong_input);
        assert!(result.is_err());
    }
}