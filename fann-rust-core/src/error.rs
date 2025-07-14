//! Error handling for the FANN Rust Core library.
//!
//! This module provides comprehensive error types that cover all possible
//! failure modes in neural network operations.

use thiserror::Error;

/// Result type alias for FANN operations
pub type Result<T> = std::result::Result<T, FannError>;

/// Comprehensive error type for FANN operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum FannError {
    /// Invalid network configuration
    #[error("Invalid network configuration: {message}")]
    InvalidConfiguration { message: String },

    /// Training-related errors
    #[error("Training error: {message}")]
    TrainingError { message: String },

    /// Inference-related errors
    #[error("Inference error: {message}")]
    InferenceError { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    /// File I/O errors
    #[error("File I/O error: {message}")]
    IoError { message: String },

    /// Memory allocation errors
    #[error("Memory allocation error: {message}")]
    MemoryError { message: String },

    /// GPU/CUDA errors
    #[error("GPU error: {message}")]
    GpuError { message: String },

    /// Invalid input dimensions
    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    InvalidInputDimensions { expected: usize, actual: usize },

    /// Invalid output dimensions
    #[error("Invalid output dimensions: expected {expected}, got {actual}")]
    InvalidOutputDimensions { expected: usize, actual: usize },

    /// Invalid layer configuration
    #[error("Invalid layer configuration: {message}")]
    InvalidLayer { message: String },

    /// Invalid activation function
    #[error("Invalid activation function: {function}")]
    InvalidActivationFunction { function: String },

    /// Invalid training algorithm
    #[error("Invalid training algorithm: {algorithm}")]
    InvalidTrainingAlgorithm { algorithm: String },

    /// Convergence failure
    #[error("Training failed to converge after {epochs} epochs")]
    ConvergenceFailure { epochs: usize },

    /// Division by zero or numerical instability
    #[error("Numerical error: {message}")]
    NumericalError { message: String },

    /// FFI-specific errors
    #[cfg(feature = "ffi")]
    #[error("FFI error: {message}")]
    FfiError { message: String },

    /// WASM-specific errors
    #[cfg(feature = "wasm")]
    #[error("WASM error: {message}")]
    WasmError { message: String },
}

impl FannError {
    /// Create a new invalid configuration error
    pub fn invalid_config<S: Into<String>>(message: S) -> Self {
        Self::InvalidConfiguration {
            message: message.into(),
        }
    }

    /// Create a new training error
    pub fn training_error<S: Into<String>>(message: S) -> Self {
        Self::TrainingError {
            message: message.into(),
        }
    }

    /// Create a new inference error
    pub fn inference_error<S: Into<String>>(message: S) -> Self {
        Self::InferenceError {
            message: message.into(),
        }
    }

    /// Create a new serialization error
    pub fn serialization_error<S: Into<String>>(message: S) -> Self {
        Self::SerializationError {
            message: message.into(),
        }
    }

    /// Create a new I/O error
    pub fn io_error<S: Into<String>>(message: S) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }

    /// Create a new memory error
    pub fn memory_error<S: Into<String>>(message: S) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    /// Create a new GPU error
    pub fn gpu_error<S: Into<String>>(message: S) -> Self {
        Self::GpuError {
            message: message.into(),
        }
    }

    /// Create a new numerical error
    pub fn numerical_error<S: Into<String>>(message: S) -> Self {
        Self::NumericalError {
            message: message.into(),
        }
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::TrainingError { .. } => true,
            Self::ConvergenceFailure { .. } => true,
            Self::NumericalError { .. } => true,
            _ => false,
        }
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::InvalidConfiguration { .. } 
            | Self::InvalidInputDimensions { .. }
            | Self::InvalidOutputDimensions { .. }
            | Self::InvalidLayer { .. }
            | Self::InvalidActivationFunction { .. }
            | Self::InvalidTrainingAlgorithm { .. } => ErrorCategory::Configuration,
            
            Self::TrainingError { .. }
            | Self::ConvergenceFailure { .. } => ErrorCategory::Training,
            
            Self::InferenceError { .. } => ErrorCategory::Inference,
            
            Self::SerializationError { .. }
            | Self::IoError { .. } => ErrorCategory::Io,
            
            Self::MemoryError { .. } => ErrorCategory::Memory,
            
            Self::GpuError { .. } => ErrorCategory::Gpu,
            
            Self::NumericalError { .. } => ErrorCategory::Numerical,
            
            #[cfg(feature = "ffi")]
            Self::FfiError { .. } => ErrorCategory::Ffi,
            
            #[cfg(feature = "wasm")]
            Self::WasmError { .. } => ErrorCategory::Wasm,
        }
    }
}

/// Error categories for better error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Configuration-related errors
    Configuration,
    /// Training-related errors
    Training,
    /// Inference-related errors
    Inference,
    /// I/O and serialization errors
    Io,
    /// Memory allocation errors
    Memory,
    /// GPU acceleration errors
    Gpu,
    /// Numerical computation errors
    Numerical,
    /// FFI-related errors
    #[cfg(feature = "ffi")]
    Ffi,
    /// WASM-related errors
    #[cfg(feature = "wasm")]
    Wasm,
}

impl From<std::io::Error> for FannError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            message: err.to_string(),
        }
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for FannError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError {
            message: err.to_string(),
        }
    }
}

/// A specialized Result type for operations that might fail with recoverable errors
pub type RecoverableResult<T> = std::result::Result<T, (FannError, Option<T>)>;

/// Helper trait for converting errors to recoverable results
pub trait IntoRecoverable<T> {
    fn into_recoverable(self) -> RecoverableResult<T>;
    fn into_recoverable_with_partial(self, partial: T) -> RecoverableResult<T>;
}

impl<T> IntoRecoverable<T> for Result<T> {
    fn into_recoverable(self) -> RecoverableResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(err) => Err((err, None)),
        }
    }

    fn into_recoverable_with_partial(self, partial: T) -> RecoverableResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(err) if err.is_recoverable() => Err((err, Some(partial))),
            Err(err) => Err((err, None)),
        }
    }
}