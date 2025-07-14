//! Neural network layer implementation.
//!
//! This module defines the layer structure and functionality for neural networks.

use crate::error::{FannError, Result};
use crate::utils::{FannFloat, Vector, ActivationFunction};
use nalgebra::DVector;

/// Type of neural network layer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Input layer
    Input,
    /// Hidden layer
    Hidden,
    /// Output layer
    Output,
    /// Bias layer (for networks with explicit bias layers)
    Bias,
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct Layer {
    /// Type of layer
    layer_type: LayerType,
    /// Number of neurons in the layer
    num_neurons: usize,
    /// Activation function for the layer
    activation_function: ActivationFunction,
    /// Whether this layer uses bias
    use_bias: bool,
    /// Neuron activations (output values)
    activations: Vector,
    /// Raw values before activation function (for derivative calculations)
    raw_values: Vector,
    /// Error values for backpropagation
    errors: Vector,
    /// Dropout mask (1.0 = keep, 0.0 = drop)
    dropout_mask: Option<Vector>,
    /// Dropout rate
    dropout_rate: FannFloat,
}

impl Layer {
    /// Create a new layer
    pub fn new(
        layer_type: LayerType,
        num_neurons: usize,
        activation_function: ActivationFunction,
        use_bias: bool,
    ) -> Result<Self> {
        if num_neurons == 0 {
            return Err(FannError::invalid_config(
                "Layer must have at least one neuron"
            ));
        }

        let size = if use_bias && layer_type != LayerType::Output {
            num_neurons + 1 // Add bias neuron
        } else {
            num_neurons
        };

        let mut layer = Self {
            layer_type,
            num_neurons,
            activation_function,
            use_bias: use_bias && layer_type != LayerType::Output,
            activations: Vector::zeros(size),
            raw_values: Vector::zeros(size),
            errors: Vector::zeros(size),
            dropout_mask: None,
            dropout_rate: 0.0,
        };

        // Set bias neuron to 1.0 if used
        if layer.use_bias {
            layer.activations[size - 1] = 1.0;
            layer.raw_values[size - 1] = 1.0;
        }

        Ok(layer)
    }

    /// Get the layer type
    pub fn layer_type(&self) -> LayerType {
        self.layer_type
    }

    /// Get the number of neurons (excluding bias)
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Get the total size (including bias if present)
    pub fn size(&self) -> usize {
        self.activations.len()
    }

    /// Check if layer uses bias
    pub fn uses_bias(&self) -> bool {
        self.use_bias
    }

    /// Get the activation function
    pub fn activation_function(&self) -> ActivationFunction {
        self.activation_function
    }

    /// Set the activation function
    pub fn set_activation_function(&mut self, activation: ActivationFunction) {
        self.activation_function = activation;
    }

    /// Get layer activations
    pub fn activations(&self) -> &Vector {
        &self.activations
    }

    /// Get mutable layer activations
    pub fn activations_mut(&mut self) -> &mut Vector {
        &mut self.activations
    }

    /// Get raw values (before activation)
    pub fn raw_values(&self) -> &Vector {
        &self.raw_values
    }

    /// Get layer errors
    pub fn errors(&self) -> &Vector {
        &self.errors
    }

    /// Get mutable layer errors
    pub fn errors_mut(&mut self) -> &mut Vector {
        &mut self.errors
    }

    /// Set input values for input layers
    pub fn set_inputs(&mut self, inputs: &[FannFloat]) -> Result<()> {
        if self.layer_type != LayerType::Input {
            return Err(FannError::invalid_config(
                "Can only set inputs on input layers"
            ));
        }

        if inputs.len() != self.num_neurons {
            return Err(FannError::InvalidInputDimensions {
                expected: self.num_neurons,
                actual: inputs.len(),
            });
        }

        // Copy inputs to activations
        for (i, &input) in inputs.iter().enumerate() {
            self.activations[i] = input;
            self.raw_values[i] = input;
        }

        Ok(())
    }

    /// Apply activation function to raw values
    pub fn apply_activation(&mut self) -> Result<()> {
        let end_idx = if self.use_bias {
            self.num_neurons // Don't apply activation to bias neuron
        } else {
            self.activations.len()
        };

        match self.activation_function {
            ActivationFunction::Softmax => {
                // Special handling for softmax (vector operation)
                let raw_slice = &self.raw_values.as_slice()[..end_idx];
                let max_val = raw_slice.iter().fold(FannFloat::NEG_INFINITY, |a, &b| a.max(b));
                
                let mut sum = 0.0;
                for i in 0..end_idx {
                    let exp_val = (self.raw_values[i] - max_val).exp();
                    self.activations[i] = exp_val;
                    sum += exp_val;
                }
                
                if sum > 0.0 {
                    for i in 0..end_idx {
                        self.activations[i] /= sum;
                    }
                }
            },
            _ => {
                // Regular activation functions
                for i in 0..end_idx {
                    self.activations[i] = self.activation_function.apply(self.raw_values[i]);
                }
            }
        }

        Ok(())
    }

    /// Calculate activation derivatives for backpropagation
    pub fn calculate_derivatives(&self) -> Result<Vector> {
        let end_idx = if self.use_bias {
            self.num_neurons
        } else {
            self.activations.len()
        };

        let mut derivatives = Vector::zeros(self.activations.len());

        match self.activation_function {
            ActivationFunction::Softmax => {
                // Softmax derivative: y_i * (δ_ij - y_j)
                for i in 0..end_idx {
                    for j in 0..end_idx {
                        let delta_ij = if i == j { 1.0 } else { 0.0 };
                        derivatives[i] += self.activations[i] * (delta_ij - self.activations[j]);
                    }
                }
            },
            _ => {
                // Regular activation function derivatives
                for i in 0..end_idx {
                    derivatives[i] = self.activation_function.derivative(self.raw_values[i]);
                }
            }
        }

        Ok(derivatives)
    }

    /// Set dropout rate
    pub fn set_dropout_rate(&mut self, rate: FannFloat) -> Result<()> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(FannError::invalid_config(
                "Dropout rate must be between 0.0 and 1.0"
            ));
        }
        self.dropout_rate = rate;
        Ok(())
    }

    /// Apply dropout during training
    pub fn apply_dropout(&mut self, training: bool) -> Result<()> {
        if !training || self.dropout_rate == 0.0 {
            self.dropout_mask = None;
            return Ok();
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let end_idx = if self.use_bias {
            self.num_neurons // Don't apply dropout to bias neuron
        } else {
            self.activations.len()
        };

        let mut mask = Vector::ones(self.activations.len());
        let scale_factor = 1.0 / (1.0 - self.dropout_rate);

        for i in 0..end_idx {
            if rng.gen::<FannFloat>() < self.dropout_rate {
                mask[i] = 0.0;
                self.activations[i] = 0.0;
            } else {
                mask[i] = scale_factor;
                self.activations[i] *= scale_factor;
            }
        }

        self.dropout_mask = Some(mask);
        Ok(())
    }

    /// Remove dropout (for inference)
    pub fn remove_dropout(&mut self) {
        if let Some(ref mask) = self.dropout_mask {
            for i in 0..self.activations.len() {
                if mask[i] == 0.0 {
                    // Restore activation based on raw value
                    self.activations[i] = self.activation_function.apply(self.raw_values[i]);
                } else if mask[i] != 1.0 {
                    // Remove scaling
                    self.activations[i] /= mask[i];
                }
            }
        }
        self.dropout_mask = None;
    }

    /// Reset layer state
    pub fn reset(&mut self) {
        self.activations.fill(0.0);
        self.raw_values.fill(0.0);
        self.errors.fill(0.0);
        
        // Reset bias neuron if present
        if self.use_bias {
            let bias_idx = self.activations.len() - 1;
            self.activations[bias_idx] = 1.0;
            self.raw_values[bias_idx] = 1.0;
        }
        
        self.dropout_mask = None;
    }

    /// Get layer statistics
    pub fn stats(&self) -> LayerStats {
        let end_idx = if self.use_bias {
            self.num_neurons
        } else {
            self.activations.len()
        };

        let activations_slice = &self.activations.as_slice()[..end_idx];
        let min_activation = activations_slice.iter().fold(FannFloat::INFINITY, |a, &b| a.min(b));
        let max_activation = activations_slice.iter().fold(FannFloat::NEG_INFINITY, |a, &b| a.max(b));
        let mean_activation = activations_slice.iter().sum::<FannFloat>() / activations_slice.len() as FannFloat;

        let errors_slice = &self.errors.as_slice()[..end_idx];
        let mean_error = errors_slice.iter().sum::<FannFloat>() / errors_slice.len() as FannFloat;
        let max_error = errors_slice.iter().fold(FannFloat::NEG_INFINITY, |a, &b| a.max(b.abs()));

        LayerStats {
            layer_type: self.layer_type,
            num_neurons: self.num_neurons,
            size: self.size(),
            activation_function: self.activation_function,
            uses_bias: self.use_bias,
            dropout_rate: self.dropout_rate,
            min_activation,
            max_activation,
            mean_activation,
            mean_error,
            max_error,
        }
    }
}

/// Layer statistics
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Layer type
    pub layer_type: LayerType,
    /// Number of neurons (excluding bias)
    pub num_neurons: usize,
    /// Total size (including bias)
    pub size: usize,
    /// Activation function
    pub activation_function: ActivationFunction,
    /// Whether layer uses bias
    pub uses_bias: bool,
    /// Dropout rate
    pub dropout_rate: FannFloat,
    /// Minimum activation value
    pub min_activation: FannFloat,
    /// Maximum activation value
    pub max_activation: FannFloat,
    /// Mean activation value
    pub mean_activation: FannFloat,
    /// Mean error value
    pub mean_error: FannFloat,
    /// Maximum error value
    pub max_error: FannFloat,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(
            LayerType::Hidden,
            3,
            ActivationFunction::Sigmoid,
            true,
        ).unwrap();

        assert_eq!(layer.layer_type(), LayerType::Hidden);
        assert_eq!(layer.num_neurons(), 3);
        assert_eq!(layer.size(), 4); // 3 neurons + 1 bias
        assert!(layer.uses_bias());
    }

    #[test]
    fn test_input_layer() {
        let mut layer = Layer::new(
            LayerType::Input,
            2,
            ActivationFunction::Linear,
            false,
        ).unwrap();

        let inputs = vec![0.5, -0.3];
        layer.set_inputs(&inputs).unwrap();

        assert_eq!(layer.activations()[0], 0.5);
        assert_eq!(layer.activations()[1], -0.3);
    }

    #[test]
    fn test_activation_functions() {
        let mut layer = Layer::new(
            LayerType::Hidden,
            2,
            ActivationFunction::Sigmoid,
            false,
        ).unwrap();

        // Set raw values
        layer.raw_values[0] = 0.0;
        layer.raw_values[1] = 1.0;

        layer.apply_activation().unwrap();

        assert_relative_eq!(layer.activations()[0], 0.5, epsilon = 1e-6);
        assert!(layer.activations()[1] > 0.7 && layer.activations()[1] < 0.8);
    }

    #[test]
    fn test_softmax_activation() {
        let mut layer = Layer::new(
            LayerType::Output,
            3,
            ActivationFunction::Softmax,
            false,
        ).unwrap();

        // Set raw values
        layer.raw_values[0] = 1.0;
        layer.raw_values[1] = 2.0;
        layer.raw_values[2] = 3.0;

        layer.apply_activation().unwrap();

        // Check that outputs sum to 1
        let sum: FannFloat = layer.activations().iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that all outputs are positive
        for &activation in layer.activations().iter() {
            assert!(activation > 0.0);
        }
    }

    #[test]
    fn test_dropout() {
        let mut layer = Layer::new(
            LayerType::Hidden,
            5,
            ActivationFunction::Relu,
            false,
        ).unwrap();

        // Set all activations to 1.0
        for i in 0..layer.activations.len() {
            layer.activations[i] = 1.0;
        }

        layer.set_dropout_rate(0.5).unwrap();
        layer.apply_dropout(true).unwrap();

        // Some neurons should be zeroed out
        let zero_count = layer.activations().iter()
            .filter(|&&x| x == 0.0)
            .count();
        
        assert!(zero_count > 0);
        assert!(zero_count < layer.size());
    }

    #[test]
    fn test_layer_reset() {
        let mut layer = Layer::new(
            LayerType::Hidden,
            3,
            ActivationFunction::Sigmoid,
            true,
        ).unwrap();

        // Modify activations and errors
        layer.activations[0] = 0.5;
        layer.errors[0] = 0.1;

        layer.reset();

        // All should be zero except bias
        assert_eq!(layer.activations()[0], 0.0);
        assert_eq!(layer.errors()[0], 0.0);
        assert_eq!(layer.activations()[3], 1.0); // Bias neuron
    }
}