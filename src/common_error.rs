//! Consolidated error types for the Neural Swarm ecosystem
//!
//! This module provides unified error handling across all components
//! to reduce duplication and improve consistency.

use thiserror::Error;

/// Unified result type for all Neural Swarm operations
pub type NeuralResult<T> = std::result::Result<T, NeuralSwarmError>;

/// Comprehensive error type covering all Neural Swarm components
#[derive(Error, Debug)]
pub enum NeuralSwarmError {
    // Network and Neural Errors
    #[error("Neural network error: {message}")]
    Neural { message: String },
    
    #[error("Network construction error: {message}")]
    NetworkConstruction { message: String },
    
    #[error("Training error: {message}")]
    Training { message: String },
    
    #[error("Inference error: {message}")]
    Inference { message: String },
    
    #[error("Convergence failure after {epochs} epochs")]
    ConvergenceFailure { epochs: usize },
    
    // Dimension and Configuration Errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },
    
    #[error("Invalid layer configuration: {message}")]
    InvalidLayer { message: String },
    
    #[error("Invalid activation function: {function}")]
    InvalidActivationFunction { function: String },
    
    // Memory and Performance Errors
    #[error("Memory allocation error: {message}")]
    MemoryAllocation { message: String },
    
    #[error("SIMD operation not supported on this architecture")]
    UnsupportedSIMD,
    
    #[error("Quantization error: {message}")]
    Quantization { message: String },
    
    #[error("Edge deployment error: {message}")]
    EdgeDeployment { message: String },
    
    // Communication Errors
    #[error("Cryptographic error: {message}")]
    Crypto { message: String },
    
    #[error("Channel error: {message}")]
    Channel { message: String },
    
    #[error("Protocol error: {message}")]
    Protocol { message: String },
    
    #[error("Authentication failed: {message}")]
    Authentication { message: String },
    
    #[error("Message validation failed: {message}")]
    Validation { message: String },
    
    // Task Decomposition Errors
    #[error("Task parsing failed: {message}")]
    TaskParsing { message: String },
    
    #[error("Context analysis failed: {message}")]
    ContextAnalysis { message: String },
    
    #[error("Graph construction failed: {message}")]
    GraphConstruction { message: String },
    
    #[error("Task decomposition failed: {message}")]
    TaskDecomposition { message: String },
    
    // Coordination and Swarm Errors
    #[error("Swarm coordination failed: {message}")]
    SwarmCoordination { message: String },
    
    #[error("Consensus failure: {message}")]
    Consensus { message: String },
    
    #[error("Fault tolerance failure: {message}")]
    FaultTolerance { message: String },
    
    // I/O and System Errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Operation timed out")]
    Timeout,
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Numerical error: {message}")]
    Numerical { message: String },
    
    // Generic and External Errors
    #[error("External dependency error: {source} - {message}")]
    External { source: String, message: String },
    
    #[error("Configuration error: {message}")]
    Configuration { message: String },
}

impl NeuralSwarmError {
    /// Check if this error is recoverable/retryable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Training { .. } | 
            Self::ConvergenceFailure { .. } |
            Self::Numerical { .. } |
            Self::Channel { .. } |
            Self::SwarmCoordination { .. } |
            Self::Io(_) |
            Self::Timeout |
            Self::ResourceExhausted { .. } => true,
            _ => false,
        }
    }
    
    /// Get the error category for metrics and logging
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Neural { .. } |
            Self::NetworkConstruction { .. } |
            Self::Training { .. } |
            Self::Inference { .. } |
            Self::ConvergenceFailure { .. } => ErrorCategory::Neural,
            
            Self::DimensionMismatch { .. } |
            Self::InvalidConfiguration { .. } |
            Self::InvalidLayer { .. } |
            Self::InvalidActivationFunction { .. } |
            Self::Configuration { .. } => ErrorCategory::Configuration,
            
            Self::MemoryAllocation { .. } |
            Self::UnsupportedSIMD |
            Self::Quantization { .. } |
            Self::EdgeDeployment { .. } => ErrorCategory::Performance,
            
            Self::Crypto { .. } |
            Self::Channel { .. } |
            Self::Protocol { .. } |
            Self::Authentication { .. } |
            Self::Validation { .. } => ErrorCategory::Communication,
            
            Self::TaskParsing { .. } |
            Self::ContextAnalysis { .. } |
            Self::GraphConstruction { .. } |
            Self::TaskDecomposition { .. } => ErrorCategory::TaskProcessing,
            
            Self::SwarmCoordination { .. } |
            Self::Consensus { .. } |
            Self::FaultTolerance { .. } => ErrorCategory::Coordination,
            
            Self::Io(_) |
            Self::Serialization(_) |
            Self::Timeout |
            Self::ResourceExhausted { .. } |
            Self::Numerical { .. } => ErrorCategory::System,
            
            Self::External { .. } => ErrorCategory::External,
        }
    }
    
    /// Create convenience constructors for common error types
    pub fn neural<S: Into<String>>(message: S) -> Self {
        Self::Neural { message: message.into() }
    }
    
    pub fn training<S: Into<String>>(message: S) -> Self {
        Self::Training { message: message.into() }
    }
    
    pub fn inference<S: Into<String>>(message: S) -> Self {
        Self::Inference { message: message.into() }
    }
    
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }
    
    pub fn memory_allocation<S: Into<String>>(message: S) -> Self {
        Self::MemoryAllocation { message: message.into() }
    }
    
    pub fn crypto<S: Into<String>>(message: S) -> Self {
        Self::Crypto { message: message.into() }
    }
    
    pub fn swarm_coordination<S: Into<String>>(message: S) -> Self {
        Self::SwarmCoordination { message: message.into() }
    }
    
    pub fn task_parsing<S: Into<String>>(message: S) -> Self {
        Self::TaskParsing { message: message.into() }
    }
}

/// Error categories for better organization and metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Neural network and AI-related errors
    Neural,
    /// Configuration and validation errors
    Configuration,
    /// Performance and optimization errors
    Performance,
    /// Communication and networking errors
    Communication,
    /// Task processing and decomposition errors
    TaskProcessing,
    /// Swarm coordination and consensus errors
    Coordination,
    /// System and I/O errors
    System,
    /// External dependency errors
    External,
}

// Compatibility type aliases for migration
pub type FannError = NeuralSwarmError;
pub type FannResult<T> = NeuralResult<T>;
pub type NeuralCommError = NeuralSwarmError;
pub type SplinterError = NeuralSwarmError;
pub type SplinterResult<T> = NeuralResult<T>;

// External error conversions
impl From<bincode::Error> for NeuralSwarmError {
    fn from(err: bincode::Error) -> Self {
        Self::Serialization(serde_json::Error::custom(err.to_string()))
    }
}

#[cfg(feature = "pyo3")]
impl From<pyo3::PyErr> for NeuralSwarmError {
    fn from(err: pyo3::PyErr) -> Self {
        Self::External {
            source: "python".to_string(),
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categorization() {
        let neural_err = NeuralSwarmError::neural("test");
        assert_eq!(neural_err.category(), ErrorCategory::Neural);
        
        let crypto_err = NeuralSwarmError::crypto("test");
        assert_eq!(crypto_err.category(), ErrorCategory::Communication);
    }
    
    #[test]
    fn test_error_recoverability() {
        let training_err = NeuralSwarmError::training("test");
        assert!(training_err.is_recoverable());
        
        let config_err = NeuralSwarmError::InvalidConfiguration { message: "test".to_string() };
        assert!(!config_err.is_recoverable());
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let err = NeuralSwarmError::dimension_mismatch(10, 5);
        assert_eq!(err.to_string(), "Dimension mismatch: expected 10, got 5");
    }
}