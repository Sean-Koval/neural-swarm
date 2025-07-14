//! # FANN Rust Core
//!
//! A high-performance Fast Artificial Neural Network (FANN) implementation in Rust.
//! This library provides a complete neural network framework with:
//!
//! - Multiple network topologies (feedforward, cascade, etc.)
//! - Various training algorithms (backpropagation, RPROP, etc.)
//! - Fast inference engine with SIMD optimizations
//! - Serialization support for saving/loading networks
//! - FFI interfaces for language bindings
//! - GPU acceleration support (CUDA/OpenCL)
//!
//! ## Quick Start
//!
//! ```rust
//! use fann_rust_core::prelude::*;
//!
//! // Create a new neural network
//! let mut network = NeuralNetwork::builder()
//!     .topology(&[2, 3, 1])
//!     .activation_function(ActivationFunction::Sigmoid)
//!     .build()?;
//!
//! // Train the network
//! let training_data = TrainingData::from_arrays(&inputs, &targets)?;
//! network.train(&training_data, TrainingParams::default())?;
//!
//! // Run inference
//! let output = network.run(&[0.5, 0.8])?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![cfg_attr(feature = "no-std", no_std)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]

// Re-export core types for convenience
pub use crate::network::{NeuralNetwork, NetworkBuilder, NetworkConfig};
pub use crate::training::{TrainingData, TrainingParams, Trainer};
pub use crate::inference::InferenceEngine;
pub use crate::error::{FannError, Result};

// Core modules
pub mod network;
pub mod training;
pub mod inference;
pub mod serialization;
pub mod error;
pub mod utils;

// Optional FFI module
#[cfg(feature = "ffi")]
pub mod ffi;

/// Prelude module containing commonly used types and traits
pub mod prelude {
    pub use crate::network::{NeuralNetwork, NetworkBuilder, NetworkConfig};
    pub use crate::training::{
        TrainingData, TrainingParams, Trainer,
        ActivationFunction, TrainingAlgorithm
    };
    pub use crate::inference::{InferenceEngine, InferenceConfig};
    pub use crate::error::{FannError, Result};
    pub use crate::utils::{LayerConfig, ConnectionConfig};
}

// Library version information
/// The version of the fann-rust-core library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The major version number
pub const VERSION_MAJOR: u32 = 0;

/// The minor version number  
pub const VERSION_MINOR: u32 = 1;

/// The patch version number
pub const VERSION_PATCH: u32 = 0;