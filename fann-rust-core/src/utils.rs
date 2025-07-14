//! Utility types and functions for the FANN Rust Core library.
//!
//! This module provides common data structures and helper functions
//! used throughout the neural network implementation.

use crate::error::{FannError, Result};
use nalgebra::{DMatrix, DVector};
use num_traits::Float;

/// Floating point type used throughout the library
pub type FannFloat = f32;

/// Vector type for neural network computations
pub type Vector = DVector<FannFloat>;

/// Matrix type for neural network computations  
pub type Matrix = DMatrix<FannFloat>;

/// Configuration for a neural network layer
#[derive(Debug, Clone, PartialEq)]
pub struct LayerConfig {
    /// Number of neurons in the layer
    pub num_neurons: usize,
    /// Activation function for the layer
    pub activation_function: ActivationFunction,
    /// Dropout rate (0.0 = no dropout, 1.0 = all neurons dropped)
    pub dropout_rate: f32,
    /// Whether to use bias neurons
    pub use_bias: bool,
}

impl LayerConfig {
    /// Create a new layer configuration
    pub fn new(num_neurons: usize, activation_function: ActivationFunction) -> Self {
        Self {
            num_neurons,
            activation_function,
            dropout_rate: 0.0,
            use_bias: true,
        }
    }

    /// Set the dropout rate
    pub fn with_dropout(mut self, rate: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(FannError::invalid_config(
                "Dropout rate must be between 0.0 and 1.0"
            ));
        }
        self.dropout_rate = rate;
        Ok(self)
    }

    /// Disable bias neurons
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }
}

/// Configuration for connections between layers
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectionConfig {
    /// Weight initialization method
    pub weight_init: WeightInitialization,
    /// Learning rate for this connection
    pub learning_rate: f32,
    /// Whether connections are sparse
    pub sparse: bool,
    /// Sparsity level (0.0 = dense, 1.0 = no connections)
    pub sparsity: f32,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            weight_init: WeightInitialization::Xavier,
            learning_rate: 0.01,
            sparse: false,
            sparsity: 0.0,
        }
    }
}

/// Available activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    /// Linear activation: f(x) = x
    Linear,
    /// Sigmoid activation: f(x) = 1 / (1 + e^(-x))
    Sigmoid,
    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,
    /// ReLU activation: f(x) = max(0, x)
    Relu,
    /// Leaky ReLU: f(x) = max(αx, x) where α = 0.01
    LeakyRelu,
    /// ELU activation: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
    Elu,
    /// Swish activation: f(x) = x * sigmoid(x)
    Swish,
    /// GELU activation: f(x) = x * Φ(x)
    Gelu,
    /// Softmax activation (for output layers)
    Softmax,
    /// Custom activation function (index into custom function array)
    Custom(usize),
}

impl ActivationFunction {
    /// Apply the activation function to a single value
    pub fn apply(&self, x: FannFloat) -> FannFloat {
        match self {
            Self::Linear => x,
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::Relu => x.max(0.0),
            Self::LeakyRelu => {
                if x > 0.0 { x } else { 0.01 * x }
            },
            Self::Elu => {
                if x > 0.0 { x } else { (x.exp() - 1.0) }
            },
            Self::Swish => x * (1.0 / (1.0 + (-x).exp())),
            Self::Gelu => {
                0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
            },
            Self::Softmax => x, // Softmax is handled differently (vector operation)
            Self::Custom(_) => x, // Custom functions handled externally
        }
    }

    /// Apply the derivative of the activation function
    pub fn derivative(&self, x: FannFloat) -> FannFloat {
        match self {
            Self::Linear => 1.0,
            Self::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            },
            Self::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            },
            Self::Relu => if x > 0.0 { 1.0 } else { 0.0 },
            Self::LeakyRelu => if x > 0.0 { 1.0 } else { 0.01 },
            Self::Elu => {
                if x > 0.0 { 1.0 } else { x.exp() }
            },
            Self::Swish => {
                let s = 1.0 / (1.0 + (-x).exp());
                s + x * s * (1.0 - s)
            },
            Self::Gelu => {
                let cdf = 0.5 * (1.0 + (x * 0.7978845608).tanh());
                let pdf = 0.3989422804 * (-0.5 * x * x).exp();
                cdf + x * pdf
            },
            Self::Softmax => 1.0, // Softmax derivative is handled differently
            Self::Custom(_) => 1.0, // Custom derivatives handled externally
        }
    }

    /// Check if this activation function requires special handling
    pub fn is_vector_function(&self) -> bool {
        matches!(self, Self::Softmax)
    }
}

/// Weight initialization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightInitialization {
    /// Zero initialization
    Zero,
    /// Random uniform initialization between -1 and 1
    Random,
    /// Xavier/Glorot initialization
    Xavier,
    /// He initialization (good for ReLU)
    He,
    /// LeCun initialization
    LeCun,
    /// Custom initialization with specified range
    Uniform { min: i32, max: i32 }, // Using i32 for serialization
    /// Normal distribution initialization
    Normal { mean: i32, std: i32 }, // Using i32 for serialization (scaled by 1000)
}

impl WeightInitialization {
    /// Generate initial weights for a connection matrix
    pub fn initialize_weights(&self, rows: usize, cols: usize) -> Matrix {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        match self {
            Self::Zero => Matrix::zeros(rows, cols),
            Self::Random => {
                Matrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0))
            },
            Self::Xavier => {
                let limit = (6.0 / (rows + cols) as f32).sqrt();
                Matrix::from_fn(rows, cols, |_, _| rng.gen_range(-limit..limit))
            },
            Self::He => {
                let std = (2.0 / rows as f32).sqrt();
                Matrix::from_fn(rows, cols, |_, _| rng.gen::<f32>() * std - std/2.0)
            },
            Self::LeCun => {
                let std = (1.0 / rows as f32).sqrt();
                Matrix::from_fn(rows, cols, |_, _| rng.gen::<f32>() * std - std/2.0)
            },
            Self::Uniform { min, max } => {
                let min_f = *min as f32 / 1000.0;
                let max_f = *max as f32 / 1000.0;
                Matrix::from_fn(rows, cols, |_, _| rng.gen_range(min_f..max_f))
            },
            Self::Normal { mean, std } => {
                let mean_f = *mean as f32 / 1000.0;
                let std_f = *std as f32 / 1000.0;
                Matrix::from_fn(rows, cols, |_, _| {
                    // Box-Muller transform for normal distribution
                    let u1: f32 = rng.gen();
                    let u2: f32 = rng.gen();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    mean_f + std_f * z
                })
            },
        }
    }
}

/// Network topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkTopology {
    /// Standard feedforward network
    Feedforward,
    /// Cascade correlation network
    Cascade,
    /// Shortcut connections allowed
    Shortcut,
    /// Custom topology
    Custom,
}

/// Training algorithms available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingAlgorithm {
    /// Standard backpropagation
    Backpropagation,
    /// Resilient backpropagation (RPROP)
    Rprop,
    /// Quickprop algorithm
    Quickprop,
    /// Scaled conjugate gradient
    ScaledConjugateGradient,
    /// Adam optimizer
    Adam,
    /// AdaGrad optimizer
    AdaGrad,
    /// RMSprop optimizer
    RmsProp,
}

/// Math utilities
pub mod math {
    use super::*;

    /// Apply softmax to a vector
    pub fn softmax(input: &Vector) -> Vector {
        let max_val = input.max();
        let exp_vals: Vector = input.map(|x| (x - max_val).exp());
        let sum = exp_vals.sum();
        exp_vals.map(|x| x / sum)
    }

    /// Calculate mean squared error
    pub fn mse(predicted: &Vector, actual: &Vector) -> Result<FannFloat> {
        if predicted.len() != actual.len() {
            return Err(FannError::invalid_config(
                "Predicted and actual vectors must have the same length"
            ));
        }

        let diff = predicted - actual;
        Ok(diff.dot(&diff) / predicted.len() as FannFloat)
    }

    /// Calculate cross-entropy loss
    pub fn cross_entropy(predicted: &Vector, actual: &Vector) -> Result<FannFloat> {
        if predicted.len() != actual.len() {
            return Err(FannError::invalid_config(
                "Predicted and actual vectors must have the same length"
            ));
        }

        let epsilon = 1e-15; // Prevent log(0)
        let mut loss = 0.0;
        
        for i in 0..predicted.len() {
            let p = predicted[i].max(epsilon).min(1.0 - epsilon);
            loss -= actual[i] * p.ln() + (1.0 - actual[i]) * (1.0 - p).ln();
        }
        
        Ok(loss / predicted.len() as FannFloat)
    }

    /// Calculate accuracy for classification
    pub fn accuracy(predicted: &Vector, actual: &Vector) -> Result<FannFloat> {
        if predicted.len() != actual.len() {
            return Err(FannError::invalid_config(
                "Predicted and actual vectors must have the same length"
            ));
        }

        let correct = predicted.iter()
            .zip(actual.iter())
            .map(|(p, a)| if (p - a).abs() < 0.5 { 1.0 } else { 0.0 })
            .sum::<FannFloat>();
            
        Ok(correct / predicted.len() as FannFloat)
    }
}

/// Memory management utilities
pub mod memory {
    use super::*;

    /// Memory pool for efficient allocation
    pub struct MemoryPool {
        vectors: Vec<Vector>,
        matrices: Vec<Matrix>,
    }

    impl MemoryPool {
        /// Create a new memory pool
        pub fn new() -> Self {
            Self {
                vectors: Vec::new(),
                matrices: Vec::new(),
            }
        }

        /// Get a vector from the pool or create a new one
        pub fn get_vector(&mut self, size: usize) -> Vector {
            if let Some(mut vec) = self.vectors.pop() {
                if vec.len() == size {
                    vec.fill(0.0);
                    return vec;
                }
            }
            Vector::zeros(size)
        }

        /// Return a vector to the pool
        pub fn return_vector(&mut self, vec: Vector) {
            if self.vectors.len() < 100 { // Limit pool size
                self.vectors.push(vec);
            }
        }

        /// Get a matrix from the pool or create a new one
        pub fn get_matrix(&mut self, rows: usize, cols: usize) -> Matrix {
            if let Some(mut mat) = self.matrices.pop() {
                if mat.nrows() == rows && mat.ncols() == cols {
                    mat.fill(0.0);
                    return mat;
                }
            }
            Matrix::zeros(rows, cols)
        }

        /// Return a matrix to the pool
        pub fn return_matrix(&mut self, mat: Matrix) {
            if self.matrices.len() < 50 { // Limit pool size
                self.matrices.push(mat);
            }
        }
    }

    impl Default for MemoryPool {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_activation_functions() {
        let x = 0.5;
        
        // Test sigmoid
        let sigmoid_val = ActivationFunction::Sigmoid.apply(x);
        assert!(sigmoid_val > 0.0 && sigmoid_val < 1.0);
        
        // Test ReLU
        assert_eq!(ActivationFunction::Relu.apply(-1.0), 0.0);
        assert_eq!(ActivationFunction::Relu.apply(1.0), 1.0);
        
        // Test linear
        assert_eq!(ActivationFunction::Linear.apply(x), x);
    }

    #[test]
    fn test_weight_initialization() {
        let init = WeightInitialization::Xavier;
        let weights = init.initialize_weights(3, 4);
        
        assert_eq!(weights.nrows(), 3);
        assert_eq!(weights.ncols(), 4);
        
        // Check that weights are in reasonable range for Xavier
        let limit = (6.0 / (3 + 4) as f32).sqrt();
        for val in weights.iter() {
            assert!(*val >= -limit && *val <= limit);
        }
    }

    #[test]
    fn test_math_utilities() {
        let predicted = Vector::from_vec(vec![0.8, 0.2, 0.1]);
        let actual = Vector::from_vec(vec![1.0, 0.0, 0.0]);
        
        let mse = math::mse(&predicted, &actual).unwrap();
        assert!(mse > 0.0);
        
        let accuracy = math::accuracy(&predicted, &actual).unwrap();
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_softmax() {
        let input = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let output = math::softmax(&input);
        
        // Check that outputs sum to 1
        assert_relative_eq!(output.sum(), 1.0, epsilon = 1e-6);
        
        // Check that all outputs are positive
        for val in output.iter() {
            assert!(*val > 0.0);
        }
    }
}