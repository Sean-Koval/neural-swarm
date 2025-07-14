//! Activation functions for neural networks
//!
//! This module provides various activation functions with both scalar and
//! SIMD-optimized implementations for high performance.

use serde::{Deserialize, Serialize};
use std::f32::consts::E;

/// Supported activation functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    Linear,
    ReLU,
    LeakyReLU(f32), // slope parameter
    Sigmoid,
    Tanh,
    Swish,
    GELU,
    ELU(f32), // alpha parameter
    Softmax,
    Softplus,
    Mish,
}

impl ActivationFunction {
    /// Compute activation function for a single value
    pub fn compute(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Linear => x,
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU(slope) => {
                if x > 0.0 { x } else { slope * x }
            },
            ActivationFunction::Sigmoid => {
                1.0 / (1.0 + (-x).exp())
            },
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Swish => x * sigmoid(x),
            ActivationFunction::GELU => {
                0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            },
            ActivationFunction::ELU(alpha) => {
                if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
            },
            ActivationFunction::Softmax => {
                // Note: Softmax should be computed over entire vector, not single value
                // This is just for completeness, use compute_vector for proper softmax
                x.exp()
            },
            ActivationFunction::Softplus => (1.0 + x.exp()).ln(),
            ActivationFunction::Mish => x * (1.0 + x.exp()).ln().tanh(),
        }
    }
    
    /// Compute derivative of activation function for a single value
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Linear => 1.0,
            ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::LeakyReLU(slope) => {
                if x > 0.0 { 1.0 } else { *slope }
            },
            ActivationFunction::Sigmoid => {
                let s = sigmoid(x);
                s * (1.0 - s)
            },
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            },
            ActivationFunction::Swish => {
                let s = sigmoid(x);
                s + x * s * (1.0 - s)
            },
            ActivationFunction::GELU => {
                // Approximate derivative
                let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
                let tanh_input = sqrt_2_pi * (x + 0.044715 * x.powi(3));
                let tanh_val = tanh_input.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;
                
                0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x.powi(2))
            },
            ActivationFunction::ELU(alpha) => {
                if x > 0.0 { 1.0 } else { alpha * x.exp() }
            },
            ActivationFunction::Softmax => {
                // Softmax derivative is complex and depends on the entire vector
                // This is a placeholder - use derivative_vector for proper implementation
                let s = x.exp();
                s * (1.0 - s)
            },
            ActivationFunction::Softplus => sigmoid(x),
            ActivationFunction::Mish => {
                let exp_x = x.exp();
                let softplus = (1.0 + exp_x).ln();
                let tanh_softplus = softplus.tanh();
                let delta = 4.0 * (x + 1.0) + 4.0 * exp_x * (1.0 + x) + exp_x * exp_x * (1.0 + x);
                let denominator = 2.0 * exp_x + exp_x * exp_x + 2.0;
                
                tanh_softplus + x * (1.0 - tanh_softplus * tanh_softplus) * delta / denominator
            },
        }
    }
    
    /// Compute activation function for a vector of values
    pub fn compute_vector(&self, input: &[f32], output: &mut [f32]) {
        match self {
            ActivationFunction::Softmax => {
                softmax_vector(input, output);
            },
            _ => {
                for (i, &x) in input.iter().enumerate() {
                    output[i] = self.compute(x);
                }
            }
        }
    }
    
    /// Compute derivative for a vector of values
    pub fn derivative_vector(&self, input: &[f32], output: &mut [f32]) {
        match self {
            ActivationFunction::Softmax => {
                // Softmax derivative is more complex - this is a simplified version
                // For proper implementation, need the Jacobian matrix
                softmax_derivative_vector(input, output);
            },
            _ => {
                for (i, &x) in input.iter().enumerate() {
                    output[i] = self.derivative(x);
                }
            }
        }
    }
    
    /// SIMD-optimized computation (when available)
    #[cfg(feature = "simd")]
    pub fn compute_simd(&self, input: &[f32], output: &mut [f32]) {
        use crate::optimization::simd::*;
        
        match self {
            ActivationFunction::ReLU => relu_simd(input, output),
            ActivationFunction::Sigmoid => sigmoid_simd(input, output),
            ActivationFunction::Tanh => tanh_simd(input, output),
            _ => self.compute_vector(input, output), // Fallback to scalar
        }
    }
    
    /// Get the name of the activation function
    pub fn name(&self) -> &'static str {
        match self {
            ActivationFunction::Linear => "linear",
            ActivationFunction::ReLU => "relu",
            ActivationFunction::LeakyReLU(_) => "leaky_relu",
            ActivationFunction::Sigmoid => "sigmoid",
            ActivationFunction::Tanh => "tanh",
            ActivationFunction::Swish => "swish",
            ActivationFunction::GELU => "gelu",
            ActivationFunction::ELU(_) => "elu",
            ActivationFunction::Softmax => "softmax",
            ActivationFunction::Softplus => "softplus",
            ActivationFunction::Mish => "mish",
        }
    }
    
    /// Check if the activation function has parameters
    pub fn has_parameters(&self) -> bool {
        matches!(self, ActivationFunction::LeakyReLU(_) | ActivationFunction::ELU(_))
    }
    
    /// Get parameters (if any)
    pub fn parameters(&self) -> Vec<f32> {
        match self {
            ActivationFunction::LeakyReLU(slope) => vec![*slope],
            ActivationFunction::ELU(alpha) => vec![*alpha],
            _ => vec![],
        }
    }
}

impl Default for ActivationFunction {
    fn default() -> Self {
        ActivationFunction::ReLU
    }
}

impl std::fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationFunction::LeakyReLU(slope) => write!(f, "leaky_relu(slope={})", slope),
            ActivationFunction::ELU(alpha) => write!(f, "elu(alpha={})", alpha),
            _ => write!(f, "{}", self.name()),
        }
    }
}

/// Helper function for sigmoid computation
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax implementation for vector
fn softmax_vector(input: &[f32], output: &mut [f32]) {
    // Find maximum for numerical stability
    let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for (i, &x) in input.iter().enumerate() {
        let exp_val = (x - max_val).exp();
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for val in output.iter_mut() {
        *val /= sum;
    }
}

/// Simplified softmax derivative (diagonal elements only)
fn softmax_derivative_vector(input: &[f32], output: &mut [f32]) {
    let mut softmax_output = vec![0.0; input.len()];
    softmax_vector(input, &mut softmax_output);
    
    for (i, &s) in softmax_output.iter().enumerate() {
        output[i] = s * (1.0 - s);
    }
}

/// Activation function factory for creating functions from strings
pub struct ActivationFactory;

impl ActivationFactory {
    pub fn create(name: &str, parameters: &[f32]) -> Option<ActivationFunction> {
        match name.to_lowercase().as_str() {
            "linear" => Some(ActivationFunction::Linear),
            "relu" => Some(ActivationFunction::ReLU),
            "leaky_relu" | "leakyrelu" => {
                let slope = parameters.get(0).copied().unwrap_or(0.01);
                Some(ActivationFunction::LeakyReLU(slope))
            },
            "sigmoid" => Some(ActivationFunction::Sigmoid),
            "tanh" => Some(ActivationFunction::Tanh),
            "swish" => Some(ActivationFunction::Swish),
            "gelu" => Some(ActivationFunction::GELU),
            "elu" => {
                let alpha = parameters.get(0).copied().unwrap_or(1.0);
                Some(ActivationFunction::ELU(alpha))
            },
            "softmax" => Some(ActivationFunction::Softmax),
            "softplus" => Some(ActivationFunction::Softplus),
            "mish" => Some(ActivationFunction::Mish),
            _ => None,
        }
    }
    
    pub fn list_available() -> Vec<&'static str> {
        vec![
            "linear", "relu", "leaky_relu", "sigmoid", "tanh",
            "swish", "gelu", "elu", "softmax", "softplus", "mish"
        ]
    }
}

/// Activation function benchmarking utilities
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    pub struct ActivationBenchmark {
        pub function: ActivationFunction,
        pub input_size: usize,
        pub iterations: usize,
    }
    
    impl ActivationBenchmark {
        pub fn new(function: ActivationFunction, input_size: usize, iterations: usize) -> Self {
            Self { function, input_size, iterations }
        }
        
        pub fn run_scalar_benchmark(&self) -> f64 {
            let input: Vec<f32> = (0..self.input_size).map(|i| (i as f32 - self.input_size as f32 / 2.0) * 0.1).collect();
            let mut output = vec![0.0; self.input_size];
            
            let start = Instant::now();
            for _ in 0..self.iterations {
                self.function.compute_vector(&input, &mut output);
            }
            let elapsed = start.elapsed();
            
            elapsed.as_secs_f64()
        }
        
        #[cfg(feature = "simd")]
        pub fn run_simd_benchmark(&self) -> f64 {
            let input: Vec<f32> = (0..self.input_size).map(|i| (i as f32 - self.input_size as f32 / 2.0) * 0.1).collect();
            let mut output = vec![0.0; self.input_size];
            
            let start = Instant::now();
            for _ in 0..self.iterations {
                self.function.compute_simd(&input, &mut output);
            }
            let elapsed = start.elapsed();
            
            elapsed.as_secs_f64()
        }
        
        pub fn compare_implementations(&self) -> BenchmarkResults {
            let scalar_time = self.run_scalar_benchmark();
            
            #[cfg(feature = "simd")]
            let simd_time = self.run_simd_benchmark();
            #[cfg(not(feature = "simd"))]
            let simd_time = scalar_time;
            
            BenchmarkResults {
                function: self.function.clone(),
                input_size: self.input_size,
                iterations: self.iterations,
                scalar_time,
                simd_time,
                speedup: scalar_time / simd_time,
            }
        }
    }
    
    #[derive(Debug)]
    pub struct BenchmarkResults {
        pub function: ActivationFunction,
        pub input_size: usize,
        pub iterations: usize,
        pub scalar_time: f64,
        pub simd_time: f64,
        pub speedup: f64,
    }
    
    impl std::fmt::Display for BenchmarkResults {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, 
                "Activation: {} | Size: {} | Iterations: {} | Scalar: {:.4}s | SIMD: {:.4}s | Speedup: {:.2}x",
                self.function, self.input_size, self.iterations, 
                self.scalar_time, self.simd_time, self.speedup
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_activation_functions() {
        let functions = vec![
            ActivationFunction::Linear,
            ActivationFunction::ReLU,
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
        ];
        
        let test_inputs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        
        for function in functions {
            for &input in &test_inputs {
                let output = function.compute(input);
                let derivative = function.derivative(input);
                
                // Check that outputs are finite
                assert!(output.is_finite(), "Output not finite for {} with input {}", function, input);
                assert!(derivative.is_finite(), "Derivative not finite for {} with input {}", function, input);
            }
        }
    }
    
    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        
        let softmax = ActivationFunction::Softmax;
        softmax.compute_vector(&input, &mut output);
        
        // Check that outputs sum to 1
        let sum: f32 = output.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        
        // Check that all outputs are positive
        for &val in &output {
            assert!(val > 0.0);
        }
    }
    
    #[test]
    fn test_activation_factory() {
        assert!(matches!(ActivationFactory::create("relu", &[]), Some(ActivationFunction::ReLU)));
        assert!(matches!(ActivationFactory::create("sigmoid", &[]), Some(ActivationFunction::Sigmoid)));
        assert!(matches!(ActivationFactory::create("leaky_relu", &[0.1]), Some(ActivationFunction::LeakyReLU(0.1))));
        assert!(ActivationFactory::create("unknown", &[]).is_none());
    }
}