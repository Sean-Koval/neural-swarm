//! Error types for neural communication

use thiserror::Error;

/// Result type for neural communication operations
pub type Result<T> = std::result::Result<T, NeuralCommError>;

/// Main error type for neural communication
#[derive(Error, Debug)]
pub enum NeuralCommError {
    /// Cryptographic operation failed
    #[error("Cryptographic error: {0}")]
    Crypto(#[from] CryptoError),

    /// Channel operation failed
    #[error("Channel error: {0}")]
    Channel(#[from] ChannelError),

    /// Protocol error
    #[error("Protocol error: {0}")]
    Protocol(#[from] ProtocolError),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Message validation failed
    #[error("Message validation failed: {0}")]
    Validation(String),

    /// Timeout occurred
    #[error("Operation timed out")]
    Timeout,

    /// Resource exhausted
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}

/// Cryptographic errors
#[derive(Error, Debug)]
pub enum CryptoError {
    /// Key generation failed
    #[error("Key generation failed: {0}")]
    KeyGeneration(String),

    /// Encryption failed
    #[error("Encryption failed: {0}")]
    Encryption(String),

    /// Decryption failed
    #[error("Decryption failed: {0}")]
    Decryption(String),

    /// Signature creation failed
    #[error("Signature creation failed: {0}")]
    Signing(String),

    /// Signature verification failed
    #[error("Signature verification failed: {0}")]
    Verification(String),

    /// Key exchange failed
    #[error("Key exchange failed: {0}")]
    KeyExchange(String),

    /// Invalid key format
    #[error("Invalid key format: {0}")]
    InvalidKey(String),

    /// Invalid nonce
    #[error("Invalid nonce: {0}")]
    InvalidNonce(String),

    /// Random number generation failed
    #[error("Random number generation failed: {0}")]
    RandomGeneration(String),
}

/// Channel errors
#[derive(Error, Debug)]
pub enum ChannelError {
    /// Channel not connected
    #[error("Channel not connected")]
    NotConnected,

    /// Channel already connected
    #[error("Channel already connected")]
    AlreadyConnected,

    /// Handshake failed
    #[error("Handshake failed: {0}")]
    HandshakeFailed(String),

    /// Message send failed
    #[error("Message send failed: {0}")]
    SendFailed(String),

    /// Message receive failed
    #[error("Message receive failed: {0}")]
    ReceiveFailed(String),

    /// Channel closed
    #[error("Channel closed")]
    Closed,

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// Peer disconnected
    #[error("Peer disconnected")]
    PeerDisconnected,

    /// Buffer overflow
    #[error("Buffer overflow")]
    BufferOverflow,
}

/// Protocol errors
#[derive(Error, Debug)]
pub enum ProtocolError {
    /// Invalid message format
    #[error("Invalid message format: {0}")]
    InvalidFormat(String),

    /// Unknown message type
    #[error("Unknown message type: {0}")]
    UnknownMessageType(u8),

    /// Message too large
    #[error("Message too large: {size} bytes (max: {max})")]
    MessageTooLarge { size: usize, max: usize },

    /// Invalid timestamp
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    /// Replay attack detected
    #[error("Replay attack detected")]
    ReplayAttack,

    /// Sequence number out of order
    #[error("Sequence number out of order: expected {expected}, got {actual}")]
    OutOfOrderSequence { expected: u64, actual: u64 },

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Protocol version mismatch
    #[error("Protocol version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    /// Compression error
    #[error("Compression error: {0}")]
    Compression(String),
}

impl From<chacha20poly1305::Error> for CryptoError {
    fn from(err: chacha20poly1305::Error) -> Self {
        CryptoError::Encryption(format!("ChaCha20Poly1305 error: {:?}", err))
    }
}

impl From<aes_gcm::Error> for CryptoError {
    fn from(err: aes_gcm::Error) -> Self {
        CryptoError::Encryption(format!("AES-GCM error: {:?}", err))
    }
}

impl From<ed25519_dalek::SignatureError> for CryptoError {
    fn from(err: ed25519_dalek::SignatureError) -> Self {
        CryptoError::Verification(format!("Ed25519 error: {}", err))
    }
}

impl From<p256::ecdsa::Error> for CryptoError {
    fn from(err: p256::ecdsa::Error) -> Self {
        CryptoError::Verification(format!("ECDSA error: {:?}", err))
    }
}

impl From<getrandom::Error> for CryptoError {
    fn from(err: getrandom::Error) -> Self {
        CryptoError::RandomGeneration(format!("Random generation error: {}", err))
    }
}

// Implement Send + Sync for error types
unsafe impl Send for NeuralCommError {}
unsafe impl Sync for NeuralCommError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let crypto_err = CryptoError::KeyGeneration("test".to_string());
        let main_err = NeuralCommError::Crypto(crypto_err);
        assert!(main_err.to_string().contains("Cryptographic error"));
    }

    #[test]
    fn error_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let main_err = NeuralCommError::Io(io_err);
        assert!(main_err.to_string().contains("I/O error"));
    }

    #[test]
    fn protocol_error_variants() {
        let err = ProtocolError::MessageTooLarge { size: 1000, max: 500 };
        assert!(err.to_string().contains("Message too large"));
        
        let err = ProtocolError::VersionMismatch { expected: 1, actual: 2 };
        assert!(err.to_string().contains("Protocol version mismatch"));
    }
}