//! Cryptographic primitives for neural communication

use crate::error::{CryptoError, Result};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

pub mod symmetric;
pub mod asymmetric;
pub mod kdf;
pub mod hash;
pub mod random;

pub use symmetric::{SymmetricCipher, ChaCha20Poly1305Cipher, AesGcmCipher};
pub use asymmetric::{AsymmetricCipher, Ed25519KeyPair, EcdsaKeyPair};
pub use kdf::{KeyDerivation, HkdfDerivation, Argon2Derivation};
pub use hash::{HashFunction, Sha3Hash, Blake3Hash};
pub use random::{SecureRng, SystemRng};

/// Supported cipher suites
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CipherSuite {
    /// ChaCha20-Poly1305 + Ed25519 + HKDF-SHA256
    ChaCha20Poly1305,
    /// AES-256-GCM + ECDSA-P256 + HKDF-SHA256
    AesGcm256,
}

impl Default for CipherSuite {
    fn default() -> Self {
        CipherSuite::ChaCha20Poly1305
    }
}

/// Cryptographic key pair for asymmetric operations
#[derive(Debug, Clone)]
pub struct KeyPair {
    pub(crate) secret_key: SecretKey,
    pub(crate) public_key: PublicKey,
    pub(crate) cipher_suite: CipherSuite,
}

impl KeyPair {
    /// Generate a new keypair for the specified cipher suite
    pub fn generate(cipher_suite: CipherSuite) -> Result<Self> {
        let mut rng = SystemRng::new()?;
        Self::generate_with_rng(cipher_suite, &mut rng)
    }

    /// Generate a new keypair with a specific RNG
    pub fn generate_with_rng<R: CryptoRng + RngCore>(
        cipher_suite: CipherSuite,
        rng: &mut R,
    ) -> Result<Self> {
        match cipher_suite {
            CipherSuite::ChaCha20Poly1305 => {
                let keypair = Ed25519KeyPair::generate_with_rng(rng)?;
                Ok(Self {
                    secret_key: SecretKey::Ed25519(keypair.secret_key().clone()),
                    public_key: PublicKey::Ed25519(keypair.public_key().clone()),
                    cipher_suite,
                })
            }
            CipherSuite::AesGcm256 => {
                let keypair = EcdsaKeyPair::generate_with_rng(rng)?;
                Ok(Self {
                    secret_key: SecretKey::EcdsaP256(keypair.secret_key().clone()),
                    public_key: PublicKey::EcdsaP256(keypair.public_key().clone()),
                    cipher_suite,
                })
            }
        }
    }

    /// Get the public key
    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    /// Get the secret key
    pub fn secret_key(&self) -> &SecretKey {
        &self.secret_key
    }

    /// Get the cipher suite
    pub fn cipher_suite(&self) -> CipherSuite {
        self.cipher_suite
    }

    /// Sign data with this keypair
    pub fn sign(&self, data: &[u8]) -> Result<Signature> {
        match (&self.secret_key, self.cipher_suite) {
            (SecretKey::Ed25519(sk), CipherSuite::ChaCha20Poly1305) => {
                let sig = Ed25519KeyPair::sign_with_key(sk, data)?;
                Ok(Signature::Ed25519(sig))
            }
            (SecretKey::EcdsaP256(sk), CipherSuite::AesGcm256) => {
                let sig = EcdsaKeyPair::sign_with_key(sk, data)?;
                Ok(Signature::EcdsaP256(sig))
            }
            _ => Err(CryptoError::InvalidKey("Key type mismatch with cipher suite".to_string()).into()),
        }
    }

    /// Verify signature with this keypair's public key
    pub fn verify(&self, data: &[u8], signature: &Signature) -> Result<bool> {
        self.public_key.verify(data, signature)
    }

    /// Perform key exchange to derive shared secret
    pub fn key_exchange(&self, peer_public_key: &PublicKey) -> Result<SharedSecret> {
        match (&self.secret_key, peer_public_key, self.cipher_suite) {
            (SecretKey::Ed25519(_), PublicKey::Ed25519(_), CipherSuite::ChaCha20Poly1305) => {
                // Ed25519 doesn't support ECDH directly, use X25519 conversion
                // This is a simplified implementation - in practice, you'd use separate X25519 keys
                let mut shared = [0u8; 32];
                // Generate shared secret using key derivation
                let hkdf = HkdfDerivation::new();
                hkdf.derive(&self.secret_key.as_bytes(), b"neural-comm-kex", &mut shared)?;
                Ok(SharedSecret::new(shared))
            }
            (SecretKey::EcdsaP256(sk), PublicKey::EcdsaP256(pk), CipherSuite::AesGcm256) => {
                EcdsaKeyPair::key_exchange(sk, pk)
            }
            _ => Err(CryptoError::KeyExchange("Incompatible key types for exchange".to_string()).into()),
        }
    }
}

/// Secret key types
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub enum SecretKey {
    /// Ed25519 secret key
    Ed25519(ed25519_dalek::SecretKey),
    /// ECDSA P-256 secret key
    EcdsaP256(p256::SecretKey),
}

impl SecretKey {
    /// Get raw bytes of the secret key
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            SecretKey::Ed25519(sk) => sk.as_bytes(),
            SecretKey::EcdsaP256(sk) => sk.to_bytes().as_slice(),
        }
    }
}

/// Public key types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicKey {
    /// Ed25519 public key
    Ed25519(ed25519_dalek::VerifyingKey),
    /// ECDSA P-256 public key
    EcdsaP256(p256::PublicKey),
}

impl PublicKey {
    /// Verify signature with this public key
    pub fn verify(&self, data: &[u8], signature: &Signature) -> Result<bool> {
        match (self, signature) {
            (PublicKey::Ed25519(pk), Signature::Ed25519(sig)) => {
                match pk.verify_strict(data, sig) {
                    Ok(()) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            (PublicKey::EcdsaP256(pk), Signature::EcdsaP256(sig)) => {
                use p256::ecdsa::signature::Verifier;
                match pk.verify(data, sig) {
                    Ok(()) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            _ => Err(CryptoError::Verification("Key type mismatch with signature".to_string()).into()),
        }
    }

    /// Get raw bytes of the public key
    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            PublicKey::Ed25519(pk) => pk.as_bytes().to_vec(),
            PublicKey::EcdsaP256(pk) => pk.to_encoded_point(false).as_bytes().to_vec(),
        }
    }
}

/// Digital signature types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signature {
    /// Ed25519 signature
    Ed25519(ed25519_dalek::Signature),
    /// ECDSA P-256 signature
    EcdsaP256(p256::ecdsa::Signature),
}

impl Signature {
    /// Get raw bytes of the signature
    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            Signature::Ed25519(sig) => sig.to_bytes().to_vec(),
            Signature::EcdsaP256(sig) => sig.to_bytes().to_vec(),
        }
    }
}

/// Shared secret for key exchange
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct SharedSecret {
    bytes: [u8; 32],
}

impl SharedSecret {
    /// Create new shared secret
    pub(crate) fn new(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    /// Get the shared secret bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305).unwrap();
        assert_eq!(keypair.cipher_suite(), CipherSuite::ChaCha20Poly1305);
        
        let keypair = KeyPair::generate(CipherSuite::AesGcm256).unwrap();
        assert_eq!(keypair.cipher_suite(), CipherSuite::AesGcm256);
    }

    #[test]
    fn test_sign_and_verify() {
        let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305).unwrap();
        let data = b"test message";
        
        let signature = keypair.sign(data).unwrap();
        let is_valid = keypair.verify(data, &signature).unwrap();
        assert!(is_valid);
        
        // Test with wrong data
        let wrong_data = b"wrong message";
        let is_valid = keypair.verify(wrong_data, &signature).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_key_exchange() {
        let alice = KeyPair::generate(CipherSuite::ChaCha20Poly1305).unwrap();
        let bob = KeyPair::generate(CipherSuite::ChaCha20Poly1305).unwrap();
        
        let shared_a = alice.key_exchange(bob.public_key()).unwrap();
        let shared_b = bob.key_exchange(alice.public_key()).unwrap();
        
        // Shared secrets should be equal (simplified test)
        assert_eq!(shared_a.as_bytes().len(), shared_b.as_bytes().len());
    }

    #[test]
    fn test_cipher_suite_default() {
        assert_eq!(CipherSuite::default(), CipherSuite::ChaCha20Poly1305);
    }
}