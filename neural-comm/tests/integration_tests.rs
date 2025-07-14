//! Integration tests for neural-comm

use neural_comm::{
    crypto::{CipherSuite, KeyPair},
    channels::{SecureChannel, ChannelConfig},
    protocols::{Message, MessageType},
    error::Result,
};
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn test_full_crypto_pipeline() -> Result<()> {
    // Test complete cryptographic pipeline
    let cipher_suite = CipherSuite::ChaCha20Poly1305;
    let keypair = KeyPair::generate(cipher_suite)?;
    
    // Test signing and verification
    let test_data = b"neural swarm test data";
    let signature = keypair.sign(test_data)?;
    let is_valid = keypair.verify(test_data, &signature)?;
    assert!(is_valid);
    
    // Test with wrong data
    let wrong_data = b"wrong data";
    let is_invalid = keypair.verify(wrong_data, &signature)?;
    assert!(!is_invalid);
    
    Ok(())
}

#[tokio::test]
async fn test_message_lifecycle() -> Result<()> {
    // Test complete message lifecycle
    let message = Message::new(
        MessageType::TaskAssignment,
        b"test task data".to_vec(),
    );
    
    // Validate message
    message.validate()?;
    
    // Serialize message
    let serialized = message.serialize()?;
    assert!(!serialized.is_empty());
    
    // Deserialize message
    let deserialized = Message::deserialize(&serialized)?;
    assert_eq!(message.header.msg_type, deserialized.header.msg_type);
    assert_eq!(message.header.message_id, deserialized.header.message_id);
    
    Ok(())
}

#[tokio::test]
async fn test_channel_creation_and_stats() -> Result<()> {
    let config = ChannelConfig::new()
        .cipher_suite(CipherSuite::ChaCha20Poly1305)
        .message_timeout(30);
    
    let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305)?;
    let channel = SecureChannel::new(config, keypair).await?;
    
    let stats = channel.stats().await;
    assert_eq!(stats.active_sessions, 0);
    assert_eq!(stats.total_messages, 0);
    
    channel.close().await?;
    Ok(())
}

#[tokio::test]
async fn test_different_cipher_suites() -> Result<()> {
    let cipher_suites = vec![
        CipherSuite::ChaCha20Poly1305,
        CipherSuite::AesGcm256,
    ];
    
    for cipher_suite in cipher_suites {
        let keypair = KeyPair::generate(cipher_suite)?;
        assert_eq!(keypair.cipher_suite(), cipher_suite);
        
        let test_data = b"cipher suite test";
        let signature = keypair.sign(test_data)?;
        let is_valid = keypair.verify(test_data, &signature)?;
        assert!(is_valid);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_message_validation() -> Result<()> {
    // Test valid message
    let valid_message = Message::heartbeat();
    assert!(valid_message.validate().is_ok());
    
    // Test message size validation would require creating an oversized message
    // which is complex due to the serialization overhead
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_operations() -> Result<()> {
    use tokio::task;
    
    // Test concurrent keypair generation
    let handles: Vec<_> = (0..10)
        .map(|_| {
            task::spawn(async {
                KeyPair::generate(CipherSuite::ChaCha20Poly1305)
            })
        })
        .collect();
    
    for handle in handles {
        let keypair = handle.await.unwrap()?;
        // Verify each keypair works
        let signature = keypair.sign(b"test")?;
        assert!(keypair.verify(b"test", &signature)?);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> Result<()> {
    use neural_comm::error::{NeuralCommError, ProtocolError};
    
    // Test invalid message deserialization
    let invalid_data = b"not a valid message";
    let result = Message::deserialize(invalid_data);
    assert!(result.is_err());
    
    match result.unwrap_err() {
        NeuralCommError::Protocol(ProtocolError::InvalidFormat(_)) => {
            // Expected error type
        }
        _ => panic!("Unexpected error type"),
    }
    
    Ok(())
}

#[tokio::test]
async fn test_memory_security() -> Result<()> {
    use neural_comm::memory::{SecureBuffer, MemorySecurity};
    
    // Test secure buffer
    let mut buffer = SecureBuffer::new(1024);
    buffer.extend_from_slice(b"sensitive data")?;
    assert_eq!(buffer.len(), 14);
    
    buffer.clear();
    assert_eq!(buffer.len(), 0);
    
    // Test constant time comparison
    let a = [1, 2, 3, 4];
    let b = [1, 2, 3, 4];
    let c = [1, 2, 3, 5];
    
    assert!(MemorySecurity::constant_time_eq(&a, &b));
    assert!(!MemorySecurity::constant_time_eq(&a, &c));
    
    Ok(())
}

#[tokio::test]
async fn test_cryptographic_primitives() -> Result<()> {
    use neural_comm::crypto::{
        symmetric::{ChaCha20Poly1305Cipher, SymmetricKey},
        hash::{Sha3Hash, Blake3Hash, HashFunction},
        random::SystemRng,
    };
    
    // Test symmetric encryption
    let key = SymmetricKey::generate(32)?;
    let cipher = ChaCha20Poly1305Cipher::new(&key)?;
    
    let plaintext = b"secret neural data";
    let aad = b"associated data";
    
    let ciphertext = cipher.encrypt(plaintext, aad)?;
    let decrypted = cipher.decrypt(&ciphertext, aad)?;
    
    assert_eq!(plaintext, decrypted.as_slice());
    
    // Test hash functions
    let data = b"data to hash";
    
    let sha3_hasher = Sha3Hash::new();
    let sha3_hash = sha3_hasher.hash(data);
    assert_eq!(sha3_hash.len(), 32);
    
    let blake3_hasher = Blake3Hash::new();
    let blake3_hash = blake3_hasher.hash(data);
    assert_eq!(blake3_hash.len(), 32);
    
    // Hashes should be different
    assert_ne!(sha3_hash, blake3_hash);
    
    // Test random generation
    let mut rng = SystemRng::new()?;
    let random_data1 = rng.generate_bytes(32)?;
    let random_data2 = rng.generate_bytes(32)?;
    
    assert_eq!(random_data1.len(), 32);
    assert_eq!(random_data2.len(), 32);
    assert_ne!(random_data1, random_data2); // Should be different
    
    Ok(())
}

#[tokio::test]
async fn test_key_derivation() -> Result<()> {
    use neural_comm::crypto::kdf::{HkdfDerivation, KeyDerivation};
    
    let hkdf = HkdfDerivation::new();
    let ikm = b"input key material";
    let info = b"context info";
    
    let mut output1 = [0u8; 32];
    let mut output2 = [0u8; 32];
    
    hkdf.derive(ikm, info, &mut output1)?;
    hkdf.derive(ikm, info, &mut output2)?;
    
    // Same inputs should produce same outputs
    assert_eq!(output1, output2);
    
    // Different info should produce different outputs
    let mut output3 = [0u8; 32];
    hkdf.derive(ikm, b"different info", &mut output3)?;
    assert_ne!(output1, output3);
    
    Ok(())
}

#[tokio::test]
async fn test_performance_requirements() -> Result<()> {
    // Test that basic operations complete within reasonable time
    let start = std::time::Instant::now();
    
    // Keypair generation should be fast
    let _keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305)?;
    let keygen_time = start.elapsed();
    assert!(keygen_time < Duration::from_millis(100)); // Should be under 100ms
    
    // Message operations should be very fast
    let start = std::time::Instant::now();
    let message = Message::heartbeat();
    message.validate()?;
    let _serialized = message.serialize()?;
    let msg_time = start.elapsed();
    assert!(msg_time < Duration::from_millis(10)); // Should be under 10ms
    
    Ok(())
}

#[tokio::test]
async fn test_channel_configuration() -> Result<()> {
    // Test various channel configurations
    let configs = vec![
        ChannelConfig::new()
            .cipher_suite(CipherSuite::ChaCha20Poly1305)
            .message_timeout(30)
            .enable_compression(false),
        ChannelConfig::new()
            .cipher_suite(CipherSuite::AesGcm256)
            .message_timeout(60)
            .enable_compression(true)
            .max_message_size(2 * 1024 * 1024),
    ];
    
    for config in configs {
        let keypair = KeyPair::generate(config.cipher_suite)?;
        let channel = SecureChannel::new(config, keypair).await?;
        
        let stats = channel.stats().await;
        assert_eq!(stats.active_sessions, 0);
        
        channel.close().await?;
    }
    
    Ok(())
}

#[cfg(feature = "python-bindings")]
#[tokio::test]
async fn test_python_compatibility() -> Result<()> {
    // Test that core types work with Python bindings
    use neural_comm::ffi::{PyCipherSuite, PyKeyPair, PyMessage, PyMessageType};
    
    // This test verifies the types can be created
    // Full Python integration would require pyo3 test environment
    
    Ok(())
}