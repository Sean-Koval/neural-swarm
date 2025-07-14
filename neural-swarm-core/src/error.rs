//! Error handling for the splinter task decomposition engine

use thiserror::Error;

/// Result type for splinter operations
pub type Result<T> = std::result::Result<T, SplinterError>;

/// Error types for the splinter engine
#[derive(Error, Debug)]
pub enum SplinterError {
    /// Task parsing errors
    #[error("Task parsing failed: {message}")]
    ParseError { message: String },

    /// Context analysis errors
    #[error("Context analysis failed: {message}")]
    AnalysisError { message: String },

    /// Neural network errors
    #[error("Neural network error: {message}")]
    NeuralError { message: String },

    /// Graph construction errors
    #[error("Graph construction failed: {message}")]
    GraphError { message: String },

    /// Swarm coordination errors
    #[error("Swarm coordination failed: {message}")]
    SwarmError { message: String },

    /// Task decomposition errors
    #[error("Task decomposition failed: {message}")]
    DecompositionError { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// I/O errors
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Network communication errors
    #[error("Network error: {0}")]
    NetworkError(#[from] neural_comm::error::CommError),

    /// Distributed memory errors
    #[error("Memory error: {0}")]
    MemoryError(#[from] neuroplex::error::NeuroError),

    /// Neural network training errors
    #[error("Training error: {0}")]
    TrainingError(#[from] candle_core::Error),

    /// Python FFI errors
    #[cfg(feature = "python-ffi")]
    #[error("Python FFI error: {0}")]
    PythonError(#[from] pyo3::PyErr),

    /// Generic errors
    #[error("Generic error: {0}")]
    GenericError(#[from] anyhow::Error),
}

impl SplinterError {
    /// Create a new parse error
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::ParseError { message: message.into() }
    }

    /// Create a new analysis error
    pub fn analysis_error(message: impl Into<String>) -> Self {
        Self::AnalysisError { message: message.into() }
    }

    /// Create a new neural error
    pub fn neural_error(message: impl Into<String>) -> Self {
        Self::NeuralError { message: message.into() }
    }

    /// Create a new graph error
    pub fn graph_error(message: impl Into<String>) -> Self {
        Self::GraphError { message: message.into() }
    }

    /// Create a new swarm error
    pub fn swarm_error(message: impl Into<String>) -> Self {
        Self::SwarmError { message: message.into() }
    }

    /// Create a new decomposition error
    pub fn decomposition_error(message: impl Into<String>) -> Self {
        Self::DecompositionError { message: message.into() }
    }

    /// Create a new configuration error
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigError { message: message.into() }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            SplinterError::NetworkError(_) => true,
            SplinterError::MemoryError(_) => true,
            SplinterError::IoError(_) => true,
            SplinterError::SwarmError { .. } => true,
            _ => false,
        }
    }

    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            SplinterError::ParseError { .. } => "parse",
            SplinterError::AnalysisError { .. } => "analysis",
            SplinterError::NeuralError { .. } => "neural",
            SplinterError::GraphError { .. } => "graph",
            SplinterError::SwarmError { .. } => "swarm",
            SplinterError::DecompositionError { .. } => "decomposition",
            SplinterError::ConfigError { .. } => "config",
            SplinterError::IoError(_) => "io",
            SplinterError::SerializationError(_) => "serialization",
            SplinterError::NetworkError(_) => "network",
            SplinterError::MemoryError(_) => "memory",
            SplinterError::TrainingError(_) => "training",
            #[cfg(feature = "python-ffi")]
            SplinterError::PythonError(_) => "python",
            SplinterError::GenericError(_) => "generic",
        }
    }
}

// Implement conversion from common error types
impl From<std::num::ParseIntError> for SplinterError {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::parse_error(format!("Integer parsing failed: {}", err))
    }
}

impl From<std::num::ParseFloatError> for SplinterError {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::parse_error(format!("Float parsing failed: {}", err))
    }
}

impl From<regex::Error> for SplinterError {
    fn from(err: regex::Error) -> Self {
        Self::parse_error(format!("Regex error: {}", err))
    }
}

impl From<uuid::Error> for SplinterError {
    fn from(err: uuid::Error) -> Self {
        Self::parse_error(format!("UUID parsing failed: {}", err))
    }
}

impl From<chrono::ParseError> for SplinterError {
    fn from(err: chrono::ParseError) -> Self {
        Self::parse_error(format!("Date parsing failed: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = SplinterError::parse_error("test message");
        assert_eq!(err.category(), "parse");
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_retryable_errors() {
        let network_err = SplinterError::NetworkError(
            neural_comm::error::CommError::ConnectionFailed("test".to_string())
        );
        assert!(network_err.is_retryable());
        assert_eq!(network_err.category(), "network");
    }

    #[test]
    fn test_error_conversion() {
        let parse_err: SplinterError = "123.45.67".parse::<i32>().unwrap_err().into();
        assert_eq!(parse_err.category(), "parse");
    }
}