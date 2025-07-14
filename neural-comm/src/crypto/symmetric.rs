//! Symmetric encryption implementations

use crate::error::{CryptoError, Result};
use crate::crypto::random::{SecureRng, SystemRng};
use rand::{CryptoRng, RngCore};
use zeroize::{Zeroize, ZeroizeOnDrop};
use serde::{Deserialize, Serialize};

use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce, Key as ChaChaKey,
};
use aes_gcm::{
    aead::{Aead as AesAead, AeadCore as AesAeadCore, KeyInit as AesKeyInit},
    Aes256Gcm, Nonce as AesNonce, Key as AesKey,
};

/// Trait for symmetric encryption operations
pub trait SymmetricCipher: Send + Sync {
    /// Encrypt plaintext with associated data
    fn encrypt(&self, plaintext: &[u8], associated_data: &[u8]) -> Result<Vec<u8>>;
    
    /// Decrypt ciphertext with associated data
    fn decrypt(&self, ciphertext: &[u8], associated_data: &[u8]) -> Result<Vec<u8>>;
    
    /// Get the key size in bytes
    fn key_size(&self) -> usize;
    
    /// Get the nonce size in bytes
    fn nonce_size(&self) -> usize;
    
    /// Get the authentication tag size in bytes
    fn tag_size(&self) -> usize;
}

/// Symmetric key for encryption/decryption
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct SymmetricKey {
    bytes: Vec<u8>,
}

impl SymmetricKey {
    /// Create a new symmetric key
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Generate a random key of the specified size
    pub fn generate(size: usize) -> Result<Self> {
        let mut rng = SystemRng::new()?;
        Self::generate_with_rng(size, &mut rng)
    }

    /// Generate a random key with a specific RNG
    pub fn generate_with_rng<R: CryptoRng + RngCore>(size: usize, rng: &mut R) -> Result<Self> {
        let mut bytes = vec![0u8; size];
        rng.try_fill_bytes(&mut bytes).map_err(|e| CryptoError::RandomGeneration(e.to_string()))?;
        Ok(Self { bytes })
    }

    /// Get the key bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the key size
    pub fn size(&self) -> usize {
        self.bytes.len()
    }
}

/// ChaCha20-Poly1305 cipher implementation
pub struct ChaCha20Poly1305Cipher {
    cipher: ChaCha20Poly1305,
}

impl ChaCha20Poly1305Cipher {
    /// Create a new ChaCha20-Poly1305 cipher with the given key
    pub fn new(key: &SymmetricKey) -> Result<Self> {
        if key.size() != 32 {
            return Err(CryptoError::InvalidKey("ChaCha20 requires 32-byte key".to_string()).into());
        }

        let chacha_key = ChaChaKey::from_slice(key.as_bytes());
        let cipher = ChaCha20Poly1305::new(chacha_key);
        
        Ok(Self { cipher })
    }

    /// Generate a random nonce
    pub fn generate_nonce<R: CryptoRng + RngCore>(&self, rng: &mut R) -> Result<[u8; 12]> {
        let mut nonce = [0u8; 12];
        rng.try_fill_bytes(&mut nonce).map_err(|e| CryptoError::RandomGeneration(e.to_string()))?;
        Ok(nonce)
    }
}

impl SymmetricCipher for ChaCha20Poly1305Cipher {
    fn encrypt(&self, plaintext: &[u8], associated_data: &[u8]) -> Result<Vec<u8> {
        let mut rng = SystemRng::new()?;
        let nonce_bytes = self.generate_nonce(&mut rng)?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        let mut ciphertext = self.cipher
            .encrypt(nonce, chacha20poly1305::aead::Payload {
                msg: plaintext,
                aad: associated_data,
            })
            .map_err(CryptoError::from)?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.append(&mut ciphertext);
        
        Ok(result)
    }

    fn decrypt(&self, ciphertext: &[u8], associated_data: &[u8]) -> Result<Vec<u8>> {
        if ciphertext.len() < 12 {
            return Err(CryptoError::Decryption("Ciphertext too short".to_string()).into());
        }

        let (nonce_bytes, actual_ciphertext) = ciphertext.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let plaintext = self.cipher
            .decrypt(nonce, chacha20poly1305::aead::Payload {
                msg: actual_ciphertext,
                aad: associated_data,
            })
            .map_err(CryptoError::from)?;

        Ok(plaintext)
    }

    fn key_size(&self) -> usize {
        32
    }

    fn nonce_size(&self) -> usize {
        12
    }

    fn tag_size(&self) -> usize {
        16
    }
}

/// AES-256-GCM cipher implementation
pub struct AesGcmCipher {
    cipher: Aes256Gcm,
}

impl AesGcmCipher {
    /// Create a new AES-GCM cipher with the given key
    pub fn new(key: &SymmetricKey) -> Result<Self> {
        if key.size() != 32 {
            return Err(CryptoError::InvalidKey("AES-256 requires 32-byte key".to_string()).into());
        }

        let aes_key = AesKey::from_slice(key.as_bytes());
        let cipher = Aes256Gcm::new(aes_key);
        
        Ok(Self { cipher })
    }

    /// Generate a random nonce
    pub fn generate_nonce<R: CryptoRng + RngCore>(&self, rng: &mut R) -> Result<[u8; 12]> {
        let mut nonce = [0u8; 12];
        rng.try_fill_bytes(&mut nonce).map_err(|e| CryptoError::RandomGeneration(e.to_string()))?;
        Ok(nonce)
    }
}

impl SymmetricCipher for AesGcmCipher {
    fn encrypt(&self, plaintext: &[u8], associated_data: &[u8]) -> Result<Vec<u8>> {
        let mut rng = SystemRng::new()?;
        let nonce_bytes = self.generate_nonce(&mut rng)?;
        let nonce = AesNonce::from_slice(&nonce_bytes);

        let mut ciphertext = self.cipher
            .encrypt(nonce, aes_gcm::aead::Payload {
                msg: plaintext,
                aad: associated_data,
            })
            .map_err(|e| CryptoError::Encryption(format!("AES-GCM encryption failed: {:?}", e)))?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.append(&mut ciphertext);
        
        Ok(result)
    }

    fn decrypt(&self, ciphertext: &[u8], associated_data: &[u8]) -> Result<Vec<u8>> {
        if ciphertext.len() < 12 {
            return Err(CryptoError::Decryption("Ciphertext too short".to_string()).into());
        }

        let (nonce_bytes, actual_ciphertext) = ciphertext.split_at(12);
        let nonce = AesNonce::from_slice(nonce_bytes);

        let plaintext = self.cipher
            .decrypt(nonce, aes_gcm::aead::Payload {
                msg: actual_ciphertext,
                aad: associated_data,
            })
            .map_err(|e| CryptoError::Decryption(format!("AES-GCM decryption failed: {:?}", e)))?;

        Ok(plaintext)
    }

    fn key_size(&self) -> usize {
        32
    }

    fn nonce_size(&self) -> usize {
        12
    }

    fn tag_size(&self) -> usize {
        16
    }
}

/// Encrypted data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// The encrypted data (includes nonce + ciphertext + tag)
    pub data: Vec<u8>,
    /// Associated authenticated data
    pub aad: Vec<u8>,
    /// Cipher suite used for encryption
    pub cipher_suite: crate::crypto::CipherSuite,
}

impl EncryptedData {
    /// Create new encrypted data
    pub fn new(data: Vec<u8>, aad: Vec<u8>, cipher_suite: crate::crypto::CipherSuite) -> Self {
        Self { data, aad, cipher_suite }
    }

    /// Get the encrypted data size
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_key_generation() {
        let key = SymmetricKey::generate(32).unwrap();
        assert_eq!(key.size(), 32);
    }

    #[test]
    fn test_chacha20_poly1305_encrypt_decrypt() {
        let key = SymmetricKey::generate(32).unwrap();
        let cipher = ChaCha20Poly1305Cipher::new(&key).unwrap();
        
        let plaintext = b"Hello, neural swarm!";
        let aad = b"associated data";
        
        let ciphertext = cipher.encrypt(plaintext, aad).unwrap();
        let decrypted = cipher.decrypt(&ciphertext, aad).unwrap();
        
        assert_eq!(plaintext, decrypted.as_slice());
    }

    #[test]
    fn test_aes_gcm_encrypt_decrypt() {
        let key = SymmetricKey::generate(32).unwrap();
        let cipher = AesGcmCipher::new(&key).unwrap();
        
        let plaintext = b"Hello, neural swarm!";
        let aad = b"associated data";
        
        let ciphertext = cipher.encrypt(plaintext, aad).unwrap();
        let decrypted = cipher.decrypt(&ciphertext, aad).unwrap();
        
        assert_eq!(plaintext, decrypted.as_slice());
    }

    #[test]
    fn test_wrong_aad_fails() {
        let key = SymmetricKey::generate(32).unwrap();
        let cipher = ChaCha20Poly1305Cipher::new(&key).unwrap();
        
        let plaintext = b"Hello, neural swarm!";
        let aad = b"associated data";
        let wrong_aad = b"wrong aad";
        
        let ciphertext = cipher.encrypt(plaintext, aad).unwrap();
        let result = cipher.decrypt(&ciphertext, wrong_aad);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_cipher_properties() {
        let key = SymmetricKey::generate(32).unwrap();
        let cipher = ChaCha20Poly1305Cipher::new(&key).unwrap();
        
        assert_eq!(cipher.key_size(), 32);
        assert_eq!(cipher.nonce_size(), 12);
        assert_eq!(cipher.tag_size(), 16);
    }

    #[test]
    fn test_invalid_key_size() {
        let key = SymmetricKey::generate(16).unwrap(); // Wrong size
        let result = ChaCha20Poly1305Cipher::new(&key);
        assert!(result.is_err());
    }
}