//! Neural network core implementation.
//!
//! This module contains the main neural network structures and functionality,
//! including network creation, configuration, and basic operations.

use crate::error::{FannError, Result};
use crate::utils::{
    FannFloat, Vector, Matrix, LayerConfig, ConnectionConfig, 
    ActivationFunction, NetworkTopology, WeightInitialization,
    memory::MemoryPool
};
use nalgebra::DVector;
use std::collections::HashMap;

pub mod layer;
pub mod connection;
pub mod topology;

pub use layer::{Layer, LayerType};
pub use connection::{Connection, ConnectionMatrix};
pub use topology::{TopologyBuilder, NetworkStructure};

/// Main neural network structure
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    /// Network configuration
    pub config: NetworkConfig,
    /// Network layers
    pub layers: Vec<Layer>,
    /// Connections between layers
    pub connections: Vec<Connection>,
    /// Network topology information
    pub topology: NetworkStructure,
    /// Memory pool for efficient computations
    memory_pool: MemoryPool,
    /// Custom activation functions
    custom_activations: HashMap<usize, Box<dyn Fn(FannFloat) -> FannFloat + Send + Sync>>,
    /// Training state
    training_state: Option<TrainingState>,
}

/// Configuration for a neural network
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkConfig {
    /// Network topology type
    pub topology_type: NetworkTopology,
    /// Input layer size
    pub input_size: usize,
    /// Output layer size
    pub output_size: usize,
    /// Hidden layer configurations
    pub hidden_layers: Vec<LayerConfig>,
    /// Default connection configuration
    pub connection_config: ConnectionConfig,
    /// Whether to use bias neurons
    pub use_bias: bool,
    /// Random seed for reproducible results
    pub random_seed: Option<u64>,
    /// Learning rate
    pub learning_rate: FannFloat,
    /// Momentum for training
    pub momentum: FannFloat,
    /// L1 regularization coefficient
    pub l1_regularization: FannFloat,
    /// L2 regularization coefficient
    pub l2_regularization: FannFloat,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            topology_type: NetworkTopology::Feedforward,
            input_size: 1,
            output_size: 1,
            hidden_layers: vec![LayerConfig::new(2, ActivationFunction::Sigmoid)],
            connection_config: ConnectionConfig::default(),
            use_bias: true,
            random_seed: None,
            learning_rate: 0.01,
            momentum: 0.0,
            l1_regularization: 0.0,
            l2_regularization: 0.0,
        }
    }
}

/// Training state for the network
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Current error
    pub current_error: FannFloat,
    /// Error history
    pub error_history: Vec<FannFloat>,
    /// Learning rate schedule
    pub learning_rate_schedule: Option<LearningRateSchedule>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant,
    /// Exponential decay
    ExponentialDecay { decay_rate: FannFloat, decay_steps: usize },
    /// Step decay
    StepDecay { drop_rate: FannFloat, epochs_drop: usize },
    /// Cosine annealing
    CosineAnnealing { min_lr: FannFloat, max_lr: FannFloat, period: usize },
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Patience (epochs to wait without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: FannFloat,
    /// Whether to restore best weights
    pub restore_best_weights: bool,
}

/// Builder pattern for creating neural networks
pub struct NetworkBuilder {
    config: NetworkConfig,
    custom_layers: Vec<LayerConfig>,
    custom_connections: Vec<(usize, usize, ConnectionConfig)>,
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            config: NetworkConfig::default(),
            custom_layers: Vec::new(),
            custom_connections: Vec::new(),
        }
    }

    /// Set the network topology from layer sizes
    pub fn topology(mut self, layer_sizes: &[usize]) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(FannError::invalid_config(
                "Network must have at least input and output layers"
            ));
        }

        self.config.input_size = layer_sizes[0];
        self.config.output_size = *layer_sizes.last().unwrap();
        
        // Create hidden layers
        self.config.hidden_layers.clear();
        for &size in &layer_sizes[1..layer_sizes.len()-1] {
            self.config.hidden_layers.push(
                LayerConfig::new(size, ActivationFunction::Sigmoid)
            );
        }

        Ok(self)
    }

    /// Set the network topology type
    pub fn topology_type(mut self, topology: NetworkTopology) -> Self {
        self.config.topology_type = topology;
        self
    }

    /// Set the default activation function for hidden layers
    pub fn activation_function(mut self, activation: ActivationFunction) -> Self {
        for layer in &mut self.config.hidden_layers {
            layer.activation_function = activation;
        }
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: FannFloat) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set momentum
    pub fn momentum(mut self, momentum: FannFloat) -> Self {
        self.config.momentum = momentum;
        self
    }

    /// Enable L1 regularization
    pub fn l1_regularization(mut self, lambda: FannFloat) -> Self {
        self.config.l1_regularization = lambda;
        self
    }

    /// Enable L2 regularization  
    pub fn l2_regularization(mut self, lambda: FannFloat) -> Self {
        self.config.l2_regularization = lambda;
        self
    }

    /// Set weight initialization method
    pub fn weight_initialization(mut self, init: WeightInitialization) -> Self {
        self.config.connection_config.weight_init = init;
        self
    }

    /// Set random seed for reproducible results
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Add a custom layer configuration
    pub fn add_layer(mut self, layer_config: LayerConfig) -> Self {
        self.custom_layers.push(layer_config);
        self
    }

    /// Add a custom connection between layers
    pub fn add_connection(mut self, from_layer: usize, to_layer: usize, config: ConnectionConfig) -> Self {
        self.custom_connections.push((from_layer, to_layer, config));
        self
    }

    /// Build the neural network
    pub fn build(self) -> Result<NeuralNetwork> {
        // Validate configuration
        if self.config.input_size == 0 {
            return Err(FannError::invalid_config("Input size must be greater than 0"));
        }
        if self.config.output_size == 0 {
            return Err(FannError::invalid_config("Output size must be greater than 0"));
        }

        // Set random seed if specified
        if let Some(seed) = self.config.random_seed {
            use rand::SeedableRng;
            let _ = rand::rngs::StdRng::seed_from_u64(seed);
        }

        // Create network structure
        let topology = TopologyBuilder::new()
            .topology_type(self.config.topology_type)
            .input_size(self.config.input_size)
            .output_size(self.config.output_size)
            .hidden_layers(&self.config.hidden_layers)
            .custom_connections(&self.custom_connections)
            .build()?;

        // Create layers
        let mut layers = Vec::new();
        
        // Input layer
        layers.push(Layer::new(
            LayerType::Input,
            self.config.input_size,
            ActivationFunction::Linear,
            self.config.use_bias,
        )?);

        // Hidden layers
        for layer_config in &self.config.hidden_layers {
            layers.push(Layer::new(
                LayerType::Hidden,
                layer_config.num_neurons,
                layer_config.activation_function,
                layer_config.use_bias,
            )?);
        }

        // Output layer (use last hidden layer activation if no hidden layers)
        let output_activation = self.config.hidden_layers
            .last()
            .map(|l| l.activation_function)
            .unwrap_or(ActivationFunction::Linear);
        
        layers.push(Layer::new(
            LayerType::Output,
            self.config.output_size,
            output_activation,
            false, // Output layer typically doesn't use bias
        )?);

        // Create connections
        let mut connections = Vec::new();
        for (from_idx, to_idx) in topology.get_connections() {
            let from_size = layers[from_idx].size();
            let to_size = layers[to_idx].size();
            
            let connection = Connection::new(
                from_idx,
                to_idx,
                from_size,
                to_size,
                self.config.connection_config.clone(),
            )?;
            
            connections.push(connection);
        }

        Ok(NeuralNetwork {
            config: self.config,
            layers,
            connections,
            topology,
            memory_pool: MemoryPool::new(),
            custom_activations: HashMap::new(),
            training_state: None,
        })
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralNetwork {
    /// Create a new network builder
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    /// Get the network configuration
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get a layer by index
    pub fn layer(&self, index: usize) -> Result<&Layer> {
        self.layers.get(index)
            .ok_or_else(|| FannError::invalid_config(
                format!("Layer index {} out of bounds", index)
            ))
    }

    /// Get a mutable layer by index
    pub fn layer_mut(&mut self, index: usize) -> Result<&mut Layer> {
        self.layers.get_mut(index)
            .ok_or_else(|| FannError::invalid_config(
                format!("Layer index {} out of bounds", index)
            ))
    }

    /// Get the number of connections
    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }

    /// Get a connection by index
    pub fn connection(&self, index: usize) -> Result<&Connection> {
        self.connections.get(index)
            .ok_or_else(|| FannError::invalid_config(
                format!("Connection index {} out of bounds", index)
            ))
    }

    /// Get a mutable connection by index
    pub fn connection_mut(&mut self, index: usize) -> Result<&mut Connection> {
        self.connections.get_mut(index)
            .ok_or_else(|| FannError::invalid_config(
                format!("Connection index {} out of bounds", index)
            ))
    }

    /// Get the total number of weights in the network
    pub fn num_weights(&self) -> usize {
        self.connections.iter()
            .map(|c| c.num_weights())
            .sum()
    }

    /// Get the total number of neurons in the network
    pub fn num_neurons(&self) -> usize {
        self.layers.iter()
            .map(|l| l.num_neurons())
            .sum()
    }

    /// Reset the network weights
    pub fn reset_weights(&mut self) -> Result<()> {
        for connection in &mut self.connections {
            connection.reset_weights()?;
        }
        Ok(())
    }

    /// Get network statistics
    pub fn stats(&self) -> NetworkStats {
        NetworkStats {
            num_layers: self.num_layers(),
            num_neurons: self.num_neurons(),
            num_weights: self.num_weights(),
            num_connections: self.num_connections(),
            topology_type: self.config.topology_type,
        }
    }

    /// Add a custom activation function
    pub fn add_custom_activation<F>(&mut self, id: usize, func: F)
    where
        F: Fn(FannFloat) -> FannFloat + Send + Sync + 'static,
    {
        self.custom_activations.insert(id, Box::new(func));
    }

    /// Apply custom activation function
    pub fn apply_custom_activation(&self, id: usize, x: FannFloat) -> Result<FannFloat> {
        self.custom_activations
            .get(&id)
            .map(|f| f(x))
            .ok_or_else(|| FannError::invalid_config(
                format!("Custom activation function {} not found", id)
            ))
    }

    /// Get current training state
    pub fn training_state(&self) -> Option<&TrainingState> {
        self.training_state.as_ref()
    }

    /// Initialize training state
    pub fn init_training(&mut self) {
        self.training_state = Some(TrainingState {
            epoch: 0,
            current_error: FannFloat::INFINITY,
            error_history: Vec::new(),
            learning_rate_schedule: None,
            early_stopping: None,
        });
    }

    /// Update training state
    pub fn update_training_state(&mut self, error: FannFloat) {
        if let Some(ref mut state) = self.training_state {
            state.epoch += 1;
            state.current_error = error;
            state.error_history.push(error);
        }
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    /// Number of layers
    pub num_layers: usize,
    /// Number of neurons
    pub num_neurons: usize,
    /// Number of weights
    pub num_weights: usize,
    /// Number of connections
    pub num_connections: usize,
    /// Network topology type
    pub topology_type: NetworkTopology,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_builder() {
        let network = NeuralNetwork::builder()
            .topology(&[2, 3, 1])
            .unwrap()
            .activation_function(ActivationFunction::Sigmoid)
            .learning_rate(0.1)
            .build()
            .unwrap();

        assert_eq!(network.num_layers(), 3);
        assert_eq!(network.config.input_size, 2);
        assert_eq!(network.config.output_size, 1);
        assert_eq!(network.config.learning_rate, 0.1);
    }

    #[test]
    fn test_invalid_topology() {
        let result = NeuralNetwork::builder()
            .topology(&[1])
            .unwrap()
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_network_stats() {
        let network = NeuralNetwork::builder()
            .topology(&[2, 3, 1])
            .unwrap()
            .build()
            .unwrap();

        let stats = network.stats();
        assert_eq!(stats.num_layers, 3);
        assert_eq!(stats.num_neurons, 6); // 2 + 3 + 1
    }
}