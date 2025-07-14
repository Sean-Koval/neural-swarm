//! Asymmetric cryptography implementations

use crate::error::{CryptoError, Result};
use crate::crypto::{SharedSecret, random::{SecureRng, SystemRng}};
use rand::{CryptoRng, RngCore};
use zeroize::{Zeroize, ZeroizeOnDrop};

use ed25519_dalek::{
    Signature as Ed25519Signature, Signer, SigningKey, Verifier, VerifyingKey,
};
use p256::{
    ecdsa::{SigningKey as EcdsaSigningKey, Signature as EcdsaSignature, signature::Signer as EcdsaSigner},
    PublicKey as EcdsaPublicKey, SecretKey as EcdsaSecretKey,
};

/// Trait for asymmetric cryptographic operations
pub trait AsymmetricCipher: Send + Sync {
    type SecretKey: Clone + Zeroize;
    type PublicKey: Clone + PartialEq;
    type Signature: Clone + PartialEq;

    /// Generate a new keypair
    fn generate() -> Result<Self>
    where
        Self: Sized;

    /// Generate a keypair with a specific RNG
    fn generate_with_rng<R: CryptoRng + RngCore>(rng: &mut R) -> Result<Self>
    where
        Self: Sized;

    /// Get the secret key
    fn secret_key(&self) -> &Self::SecretKey;

    /// Get the public key
    fn public_key(&self) -> &Self::PublicKey;

    /// Sign data with the secret key
    fn sign(&self, data: &[u8]) -> Result<Self::Signature>;

    /// Verify signature with the public key
    fn verify(public_key: &Self::PublicKey, data: &[u8], signature: &Self::Signature) -> Result<bool>;

    /// Sign data with a specific secret key
    fn sign_with_key(secret_key: &Self::SecretKey, data: &[u8]) -> Result<Self::Signature>;
}

/// Ed25519 keypair implementation
#[derive(Debug, Clone)]
pub struct Ed25519KeyPair {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl Ed25519KeyPair {
    /// Create a new Ed25519 keypair from a signing key
    pub fn from_signing_key(signing_key: SigningKey) -> Self {
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Create keypair from raw bytes
    pub fn from_bytes(secret_bytes: &[u8]) -> Result<Self> {
        if secret_bytes.len() != 32 {
            return Err(CryptoError::InvalidKey("Ed25519 secret key must be 32 bytes".to_string()).into());
        }

        let signing_key = SigningKey::from_bytes(
            secret_bytes.try_into()
                .map_err(|_| CryptoError::InvalidKey("Invalid Ed25519 key bytes".to_string()))?
        );
        Ok(Self::from_signing_key(signing_key))
    }

    /// Get the secret key bytes
    pub fn secret_bytes(&self) -> &[u8] {
        self.signing_key.as_bytes()
    }

    /// Get the public key bytes
    pub fn public_bytes(&self) -> &[u8] {
        self.verifying_key.as_bytes()
    }
}

impl AsymmetricCipher for Ed25519KeyPair {
    type SecretKey = ed25519_dalek::SecretKey;
    type PublicKey = VerifyingKey;
    type Signature = Ed25519Signature;

    fn generate() -> Result<Self> {
        let mut rng = SystemRng::new()?;
        Self::generate_with_rng(&mut rng)
    }

    fn generate_with_rng<R: CryptoRng + RngCore>(rng: &mut R) -> Result<Self> {
        let signing_key = SigningKey::generate(rng);
        Ok(Self::from_signing_key(signing_key))
    }

    fn secret_key(&self) -> &Self::SecretKey {
        self.signing_key.as_bytes()
    }

    fn public_key(&self) -> &Self::PublicKey {
        &self.verifying_key
    }

    fn sign(&self, data: &[u8]) -> Result<Self::Signature> {
        Ok(self.signing_key.sign(data))
    }

    fn verify(public_key: &Self::PublicKey, data: &[u8], signature: &Self::Signature) -> Result<bool> {
        match public_key.verify_strict(data, signature) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn sign_with_key(secret_key: &Self::SecretKey, data: &[u8]) -> Result<Self::Signature> {
        let signing_key = SigningKey::from_bytes(secret_key);
        Ok(signing_key.sign(data))
    }
}

/// ECDSA P-256 keypair implementation
#[derive(Debug, Clone)]
pub struct EcdsaKeyPair {
    secret_key: EcdsaSecretKey,
    public_key: EcdsaPublicKey,
}

impl EcdsaKeyPair {
    /// Create a new ECDSA keypair from a secret key
    pub fn from_secret_key(secret_key: EcdsaSecretKey) -> Self {
        let public_key = secret_key.public_key();
        Self {
            secret_key,
            public_key,
        }
    }

    /// Create keypair from raw bytes
    pub fn from_bytes(secret_bytes: &[u8]) -> Result<Self> {
        let secret_key = EcdsaSecretKey::from_slice(secret_bytes)
            .map_err(|e| CryptoError::InvalidKey(format!("Invalid ECDSA key: {}", e)))?;
        Ok(Self::from_secret_key(secret_key))
    }

    /// Get the secret key bytes
    pub fn secret_bytes(&self) -> Vec<u8> {
        self.secret_key.to_bytes().to_vec()
    }

    /// Get the public key bytes
    pub fn public_bytes(&self) -> Vec<u8> {
        self.public_key.to_encoded_point(false).as_bytes().to_vec()
    }

    /// Perform ECDH key exchange
    pub fn key_exchange(secret_key: &EcdsaSecretKey, peer_public_key: &EcdsaPublicKey) -> Result<SharedSecret> {
        use p256::ecdh::EphemeralSecret;
        use p256::PublicKey;
        
        // Convert to the right types for ECDH
        let secret = EphemeralSecret::from(secret_key.clone());
        let shared_point = secret.diffie_hellman(peer_public_key);
        
        // Extract the x-coordinate as the shared secret
        let shared_bytes: [u8; 32] = shared_point.raw_secret_bytes().try_into()
            .map_err(|_| CryptoError::KeyExchange("Failed to extract shared secret".to_string()))?;
        
        Ok(SharedSecret::new(shared_bytes))
    }
}

impl AsymmetricCipher for EcdsaKeyPair {
    type SecretKey = EcdsaSecretKey;
    type PublicKey = EcdsaPublicKey;
    type Signature = EcdsaSignature;

    fn generate() -> Result<Self> {
        let mut rng = SystemRng::new()?;
        Self::generate_with_rng(&mut rng)
    }

    fn generate_with_rng<R: CryptoRng + RngCore>(rng: &mut R) -> Result<Self> {
        let secret_key = EcdsaSecretKey::random(rng);
        Ok(Self::from_secret_key(secret_key))
    }

    fn secret_key(&self) -> &Self::SecretKey {
        &self.secret_key
    }

    fn public_key(&self) -> &Self::PublicKey {
        &self.public_key
    }

    fn sign(&self, data: &[u8]) -> Result<Self::Signature> {
        let signing_key = EcdsaSigningKey::from(&self.secret_key);
        Ok(signing_key.sign(data))
    }

    fn verify(public_key: &Self::PublicKey, data: &[u8], signature: &Self::Signature) -> Result<bool> {
        use p256::ecdsa::signature::Verifier;
        match public_key.verify(data, signature) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn sign_with_key(secret_key: &Self::SecretKey, data: &[u8]) -> Result<Self::Signature> {
        let signing_key = EcdsaSigningKey::from(secret_key);
        Ok(signing_key.sign(data))
    }
}

/// Key exchange implementation for neural network agents
pub struct KeyExchange;

impl KeyExchange {
    /// Perform ephemeral key exchange between two agents
    pub fn ephemeral_exchange<C: AsymmetricCipher>(
        local_keypair: &C,
        peer_public_key: &C::PublicKey,
    ) -> Result<Vec<u8>> 
    where
        C::SecretKey: AsRef<[u8]>,
        C::PublicKey: AsRef<[u8]>,
    {
        // This is a simplified key exchange - in practice you'd use proper ECDH
        let local_secret = local_keypair.secret_key().as_ref();
        let peer_public = peer_public_key.as_ref();
        
        // Simple XOR-based key mixing (not cryptographically secure)
        // In practice, use proper KDF like HKDF
        let mut shared = Vec::with_capacity(32);
        for i in 0..32 {
            let local_byte = local_secret.get(i % local_secret.len()).unwrap_or(&0);
            let peer_byte = peer_public.get(i % peer_public.len()).unwrap_or(&0);
            shared.push(local_byte ^ peer_byte);
        }
        
        Ok(shared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ed25519_keypair_generation() {
        let keypair = Ed25519KeyPair::generate().unwrap();
        assert_eq!(keypair.secret_bytes().len(), 32);
        assert_eq!(keypair.public_bytes().len(), 32);
    }

    #[test]
    fn test_ed25519_sign_verify() {
        let keypair = Ed25519KeyPair::generate().unwrap();
        let data = b"test message for signing";
        
        let signature = keypair.sign(data).unwrap();
        let is_valid = Ed25519KeyPair::verify(keypair.public_key(), data, &signature).unwrap();
        assert!(is_valid);
        
        // Test with wrong data
        let wrong_data = b"wrong message";
        let is_valid = Ed25519KeyPair::verify(keypair.public_key(), wrong_data, &signature).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_ecdsa_keypair_generation() {
        let keypair = EcdsaKeyPair::generate().unwrap();
        assert_eq!(keypair.secret_bytes().len(), 32);
        assert!(!keypair.public_bytes().is_empty());
    }

    #[test]
    fn test_ecdsa_sign_verify() {
        let keypair = EcdsaKeyPair::generate().unwrap();
        let data = b"test message for ECDSA signing";
        
        let signature = keypair.sign(data).unwrap();
        let is_valid = EcdsaKeyPair::verify(keypair.public_key(), data, &signature).unwrap();
        assert!(is_valid);
        
        // Test with wrong data
        let wrong_data = b"wrong message";
        let is_valid = EcdsaKeyPair::verify(keypair.public_key(), wrong_data, &signature).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_ecdsa_key_exchange() {
        let alice = EcdsaKeyPair::generate().unwrap();
        let bob = EcdsaKeyPair::generate().unwrap();
        
        let shared_a = EcdsaKeyPair::key_exchange(alice.secret_key(), bob.public_key()).unwrap();
        let shared_b = EcdsaKeyPair::key_exchange(bob.secret_key(), alice.public_key()).unwrap();
        
        assert_eq!(shared_a.as_bytes(), shared_b.as_bytes());
    }

    #[test]
    fn test_keypair_from_bytes() {
        let original = Ed25519KeyPair::generate().unwrap();
        let secret_bytes = original.secret_bytes();
        
        let restored = Ed25519KeyPair::from_bytes(secret_bytes).unwrap();
        assert_eq!(original.secret_bytes(), restored.secret_bytes());
        assert_eq!(original.public_bytes(), restored.public_bytes());
    }

    #[test]
    fn test_invalid_key_bytes() {
        let wrong_size = [0u8; 16]; // Wrong size
        let result = Ed25519KeyPair::from_bytes(&wrong_size);
        assert!(result.is_err());
    }
}