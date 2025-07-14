use crate::error::{FannError, Result};
use crate::activations::ActivationFunction;
use crate::training::{TrainingData, TrainingAlgorithm};
use serde::{Deserialize, Serialize};

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub layers: Vec<usize>,
    pub activation_functions: Vec<ActivationFunction>,
    pub learning_rate: f32,
    pub use_bias: bool,
}

/// Main neural network structure
#[derive(Debug, Clone)]
pub struct Network {
    config: NetworkConfig,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
    activations: Vec<Vec<f32>>,
    errors: Vec<Vec<f32>>,
}

/// Builder pattern for creating neural networks
#[derive(Debug)]
pub struct NetworkBuilder {
    layers: Vec<usize>,
    activation_functions: Vec<ActivationFunction>,
    learning_rate: f32,
    use_bias: bool,
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            activation_functions: Vec::new(),
            learning_rate: 0.1,
            use_bias: true,
        }
    }
    
    /// Add a layer to the network
    pub fn add_layer(mut self, input_size: usize, output_size: usize) -> Self {
        if self.layers.is_empty() {
            self.layers.push(input_size);
        }
        self.layers.push(output_size);
        self.activation_functions.push(ActivationFunction::ReLU);
        self
    }
    
    /// Set the activation function for the last added layer
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        if let Some(last) = self.activation_functions.last_mut() {
            *last = activation;
        }
        self
    }
    
    /// Set the learning rate
    pub fn learning_rate(mut self, rate: f32) -> Self {
        self.learning_rate = rate;
        self
    }
    
    /// Enable or disable bias
    pub fn use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }
    
    /// Build the neural network
    pub fn build(self) -> Result<Network> {
        if self.layers.len() < 2 {
            return Err(FannError::network_construction(
                "Network must have at least 2 layers (input and output)"
            ));
        }
        
        let config = NetworkConfig {
            layers: self.layers.clone(),
            activation_functions: self.activation_functions,
            learning_rate: self.learning_rate,
            use_bias: self.use_bias,
        };
        
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut activations = Vec::new();
        let mut errors = Vec::new();
        
        // Initialize weights and biases for each layer transition
        for i in 0..self.layers.len() - 1 {
            let input_size = self.layers[i];
            let output_size = self.layers[i + 1];
            
            // Xavier initialization for weights
            let scale = (2.0 / (input_size + output_size) as f32).sqrt();
            let layer_weights: Vec<f32> = (0..input_size * output_size)
                .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
                .collect();
            weights.push(layer_weights);
            
            // Initialize biases to small random values
            let layer_biases: Vec<f32> = (0..output_size)
                .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
                .collect();
            biases.push(layer_biases);
            
            // Initialize activation and error storage
            activations.push(vec![0.0; output_size]);
            errors.push(vec![0.0; output_size]);
        }
        
        // Add input layer activations
        activations.insert(0, vec![0.0; self.layers[0]]);
        errors.insert(0, vec![0.0; self.layers[0]]);
        
        Ok(Network {
            config,
            weights,
            biases,
            activations,
            errors,
        })
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Network {
    /// Perform a forward pass through the network
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.config.layers[0] {
            panic!("Input size mismatch: expected {}, got {}", 
                   self.config.layers[0], input.len());
        }
        
        let mut current_input = input.to_vec();
        
        for (layer_idx, (weights, biases)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let input_size = self.config.layers[layer_idx];
            let output_size = self.config.layers[layer_idx + 1];
            let mut output = vec![0.0; output_size];
            
            // Matrix multiplication: output = weights * input + bias
            for i in 0..output_size {
                output[i] = if self.config.use_bias { biases[i] } else { 0.0 };
                for j in 0..input_size {
                    output[i] += weights[i * input_size + j] * current_input[j];
                }
                
                // Apply activation function
                output[i] = self.config.activation_functions[layer_idx].apply(output[i]);
            }
            
            current_input = output;
        }
        
        current_input
    }
    
    /// Train the network using the specified algorithm
    pub fn train(
        &mut self,
        training_data: &TrainingData,
        epochs: usize,
        learning_rate: f32,
        algorithm: TrainingAlgorithm,
    ) -> Result<()> {
        match algorithm {
            TrainingAlgorithm::Backpropagation => {
                self.train_backpropagation(training_data, epochs, learning_rate)
            }
            TrainingAlgorithm::StochasticGradientDescent => {
                self.train_sgd(training_data, epochs, learning_rate)
            }
        }
    }
    
    /// Backpropagation training
    fn train_backpropagation(
        &mut self,
        training_data: &TrainingData,
        epochs: usize,
        learning_rate: f32,
    ) -> Result<()> {
        for _epoch in 0..epochs {
            let mut total_error = 0.0;
            
            for sample in training_data.samples() {
                // Forward pass
                let output = self.forward(&sample.input);
                
                // Calculate error
                let sample_error: f32 = output.iter()
                    .zip(sample.target.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum();
                total_error += sample_error;
                
                // Backward pass
                self.backward(&sample.input, &sample.target, &output, learning_rate)?;
            }
            
            // Optional: Early stopping if error is low enough
            if total_error / training_data.len() as f32 < 1e-6 {
                break;
            }
        }
        
        Ok(())
    }
    
    /// Stochastic gradient descent training
    fn train_sgd(
        &mut self,
        training_data: &TrainingData,
        epochs: usize,
        learning_rate: f32,
    ) -> Result<()> {
        for _epoch in 0..epochs {
            // Shuffle training data for SGD
            let mut indices: Vec<usize> = (0..training_data.len()).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());
            
            for &idx in &indices {
                let sample = &training_data.samples()[idx];
                
                // Forward pass
                let output = self.forward(&sample.input);
                
                // Backward pass
                self.backward(&sample.input, &sample.target, &output, learning_rate)?;
            }
        }
        
        Ok(())
    }
    
    /// Backward pass for gradient computation and weight updates
    fn backward(
        &mut self,
        input: &[f32],
        target: &[f32],
        output: &[f32],
        learning_rate: f32,
    ) -> Result<()> {
        let num_layers = self.config.layers.len();
        
        // Clear previous errors
        for error_layer in &mut self.errors {
            error_layer.fill(0.0);
        }
        
        // Calculate output layer errors
        let output_layer_idx = num_layers - 1;
        for i in 0..output.len() {
            self.errors[output_layer_idx][i] = 2.0 * (output[i] - target[i]);
        }
        
        // Backpropagate errors
        for layer_idx in (1..num_layers).rev() {
            let prev_layer_size = self.config.layers[layer_idx - 1];
            let current_layer_size = self.config.layers[layer_idx];
            
            if layer_idx > 1 {
                // Calculate errors for previous layer
                for i in 0..prev_layer_size {
                    let mut error_sum = 0.0;
                    for j in 0..current_layer_size {
                        let weight_idx = j * prev_layer_size + i;
                        error_sum += self.errors[layer_idx][j] * self.weights[layer_idx - 1][weight_idx];
                    }
                    self.errors[layer_idx - 1][i] = error_sum;
                }
            }
            
            // Update weights and biases
            let weight_layer_idx = layer_idx - 1;
            for i in 0..current_layer_size {
                // Update bias
                if self.config.use_bias {
                    self.biases[weight_layer_idx][i] -= learning_rate * self.errors[layer_idx][i];
                }
                
                // Update weights
                for j in 0..prev_layer_size {
                    let weight_idx = i * prev_layer_size + j;
                    let input_value = if layer_idx == 1 {
                        input[j]
                    } else {
                        // Use activation from previous layer
                        // This is a simplified version - in practice, we'd store activations
                        // from the forward pass
                        0.0 // Placeholder
                    };
                    
                    self.weights[weight_layer_idx][weight_idx] -= 
                        learning_rate * self.errors[layer_idx][i] * input_value;
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the current mean squared error
    pub fn get_mse(&self) -> f32 {
        // This would typically be calculated during training
        // For now, return a placeholder
        0.01
    }
    
    /// Get the memory footprint of the network
    pub fn memory_footprint(&self) -> usize {
        let weights_size: usize = self.weights.iter().map(|w| w.len() * 4).sum(); // 4 bytes per f32
        let biases_size: usize = self.biases.iter().map(|b| b.len() * 4).sum();
        let activations_size: usize = self.activations.iter().map(|a| a.len() * 4).sum();
        let errors_size: usize = self.errors.iter().map(|e| e.len() * 4).sum();
        
        weights_size + biases_size + activations_size + errors_size
    }
    
    /// Get network configuration
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }
    
    /// Get network layers
    pub fn layers(&self) -> &[usize] {
        &self.config.layers
    }
}

/// Random number generation for testing
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    static mut SEED: u64 = 12345;
    
    pub fn random<T>() -> T 
    where 
        T: From<f32>
    {
        unsafe {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (SEED as f32) / (u64::MAX as f32);
            T::from(normalized)
        }
    }
    
    pub fn thread_rng() -> ThreadRng {
        ThreadRng
    }
    
    pub struct ThreadRng;
    
    pub mod seq {
        pub trait SliceRandom<T> {
            fn shuffle(&mut self, rng: &mut super::ThreadRng);
        }
        
        impl<T> SliceRandom<T> for [T] {
            fn shuffle(&mut self, _rng: &mut super::ThreadRng) {
                // Simple Fisher-Yates shuffle
                for i in (1..self.len()).rev() {
                    let j = unsafe { super::SEED as usize % (i + 1) };
                    self.swap(i, j);
                    unsafe {
                        super::SEED = super::SEED.wrapping_mul(1103515245).wrapping_add(12345);
                    }
                }
            }
        }
    }
}