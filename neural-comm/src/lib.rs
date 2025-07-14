//! # Neural Comm
//!
//! A high-performance cryptographic communication library for neural agent swarms.
//! This library provides secure, authenticated communication channels with modern
//! cryptographic primitives designed for agent-to-agent coordination.
//!
//! ## Security Features
//!
//! - **Strong Encryption**: ChaCha20-Poly1305, AES-GCM
//! - **Digital Signatures**: Ed25519, ECDSA (P-256)
//! - **Key Exchange**: X25519 Elliptic Curve Diffie-Hellman
//! - **Key Derivation**: HKDF with SHA-256/SHA-3
//! - **Password Hashing**: Argon2id
//! - **Forward Secrecy**: Ephemeral key exchange
//! - **Replay Protection**: Nonce-based anti-replay
//! - **Side-Channel Resistance**: Constant-time operations
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use neural_comm::{
//!     crypto::{CipherSuite, KeyPair},
//!     channels::{SecureChannel, ChannelConfig},
//!     protocols::{Message, MessageType},
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Generate keypair for agent
//!     let keypair = KeyPair::generate()?;
//!     
//!     // Create secure channel configuration
//!     let config = ChannelConfig::new()
//!         .cipher_suite(CipherSuite::ChaCha20Poly1305)
//!         .enable_forward_secrecy(true)
//!         .message_timeout(30); // seconds
//!     
//!     // Establish secure channel
//!     let mut channel = SecureChannel::new(config, keypair).await?;
//!     
//!     // Send encrypted message
//!     let message = Message::new(
//!         MessageType::Coordination,
//!         b"Hello, secure neural swarm!".to_vec()
//!     );
//!     
//!     channel.send(message).await?;
//!     
//!     Ok(())
//! }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod crypto;
pub mod channels;
pub mod protocols;
pub mod error;
pub mod memory;

#[cfg(feature = "python-bindings")]
pub mod ffi;

// Re-export core types
pub use crypto::{CipherSuite, KeyPair, Signature, SecretKey, PublicKey};
pub use channels::{SecureChannel, ChannelConfig, ChannelError};
pub use protocols::{Message, MessageType, ProtocolError};
pub use error::{NeuralCommError, Result};

/// Neural communication precision type
pub type CommFloat = f64;

/// Agent identifier type
pub type AgentId = [u8; 32];

/// Message identifier type
pub type MessageId = [u8; 16];

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Security parameters
pub mod security {
    /// Default key size in bytes
    pub const DEFAULT_KEY_SIZE: usize = 32;
    
    /// Default nonce size in bytes
    pub const DEFAULT_NONCE_SIZE: usize = 12;
    
    /// Default tag size in bytes
    pub const DEFAULT_TAG_SIZE: usize = 16;
    
    /// Maximum message size in bytes (16MB)
    pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;
    
    /// Default session key rotation interval (minutes)
    pub const DEFAULT_KEY_ROTATION_INTERVAL: u64 = 60;
    
    /// Maximum clock skew tolerance (seconds)
    pub const MAX_CLOCK_SKEW: u64 = 30;
}

/// Features available in this build
pub mod features {
    /// Check if async support is enabled
    pub const ASYNC: bool = cfg!(feature = "async");
    
    /// Check if Python bindings are enabled
    pub const PYTHON: bool = cfg!(feature = "python-bindings");
    
    /// Check if std support is enabled
    pub const STD: bool = cfg!(feature = "std");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_check() {
        assert!(!VERSION.is_empty());
        assert_eq!(NAME, "neural-comm");
    }

    #[test]
    fn features_check() {
        // Verify feature flags are accessible
        let _ = features::ASYNC;
        let _ = features::PYTHON;
        let _ = features::STD;
    }

    #[test]
    fn security_constants() {
        assert_eq!(security::DEFAULT_KEY_SIZE, 32);
        assert_eq!(security::DEFAULT_NONCE_SIZE, 12);
        assert_eq!(security::DEFAULT_TAG_SIZE, 16);
        assert!(security::MAX_MESSAGE_SIZE > 0);
    }
}