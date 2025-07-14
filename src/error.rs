use thiserror::Error;

/// Comprehensive error types for FANN-Rust operations
#[derive(Error, Debug)]
pub enum FannError {
    #[error("Network construction error: {message}")]
    NetworkConstruction { message: String },
    
    #[error("Training error: {message}")]
    Training { message: String },
    
    #[error("Inference error: {message}")]
    Inference { message: String },
    
    #[error("Input dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Memory allocation error: {message}")]
    MemoryAllocation { message: String },
    
    #[error("SIMD operation not supported on this architecture")]
    UnsupportedSIMD,
    
    #[error("Quantization error: {message}")]
    Quantization { message: String },
    
    #[error("Quantization calibration failed: {0}")]
    CalibrationFailed(String),
    
    #[error("Unsupported quantization type: {0}")]
    UnsupportedType(String),
    
    #[error("Quantization validation failed")]
    ValidationFailed,
    
    #[error("Precision loss: {loss_percentage}%")]
    PrecisionLoss { loss_percentage: f32 },
    
    #[error("Invalid quantization parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Edge deployment error: {message}")]
    EdgeDeployment { message: String },
    
    #[error("Benchmark error: {message}")]
    Benchmark { message: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type alias for FANN operations
pub type Result<T> = std::result::Result<T, FannError>;

/// Neural error alias for compatibility
pub type NeuralError = FannError;

impl FannError {
    /// Create a network construction error
    pub fn network_construction<S: Into<String>>(message: S) -> Self {
        Self::NetworkConstruction {
            message: message.into(),
        }
    }
    
    /// Create a training error
    pub fn training<S: Into<String>>(message: S) -> Self {
        Self::Training {
            message: message.into(),
        }
    }
    
    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }
    
    /// Create a memory allocation error
    pub fn memory_allocation<S: Into<String>>(message: S) -> Self {
        Self::MemoryAllocation {
            message: message.into(),
        }
    }
    
    /// Create a quantization error
    pub fn quantization<S: Into<String>>(message: S) -> Self {
        Self::Quantization {
            message: message.into(),
        }
    }
    
    /// Create an inference error
    pub fn inference<S: Into<String>>(message: S) -> Self {
        Self::Inference {
            message: message.into(),
        }
    }
    
    /// Create an invalid dimensions error
    pub fn invalid_dimensions(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }
    
    /// Create an edge deployment error
    pub fn edge_deployment<S: Into<String>>(message: S) -> Self {
        Self::EdgeDeployment {
            message: message.into(),
        }
    }
    
    /// Create a benchmark error
    pub fn benchmark<S: Into<String>>(message: S) -> Self {
        Self::Benchmark {
            message: message.into(),
        }
    }
}