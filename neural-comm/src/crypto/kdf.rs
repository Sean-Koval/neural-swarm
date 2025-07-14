//! Key derivation functions

use crate::error::{CryptoError, Result};
use hkdf::Hkdf;
use sha3::Sha3_256;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier, Salt};
use zeroize::Zeroize;

/// Trait for key derivation operations
pub trait KeyDerivation: Send + Sync {
    /// Derive key material from input key material
    fn derive(&self, ikm: &[u8], info: &[u8], output: &mut [u8]) -> Result<()>;
    
    /// Get the recommended output length
    fn recommended_length(&self) -> usize;
}

/// HKDF-based key derivation
pub struct HkdfDerivation {
    salt: Option<Vec<u8>>,
}

impl HkdfDerivation {
    /// Create a new HKDF derivation without salt
    pub fn new() -> Self {
        Self { salt: None }
    }

    /// Create a new HKDF derivation with salt
    pub fn with_salt(salt: Vec<u8>) -> Self {
        Self { salt: Some(salt) }
    }

    /// Derive a specific amount of key material
    pub fn derive_key(&self, ikm: &[u8], info: &[u8], length: usize) -> Result<Vec<u8>> {
        let mut output = vec![0u8; length];
        self.derive(ikm, info, &mut output)?;
        Ok(output)
    }
}

impl Default for HkdfDerivation {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyDerivation for HkdfDerivation {
    fn derive(&self, ikm: &[u8], info: &[u8], output: &mut [u8]) -> Result<()> {
        let salt = self.salt.as_deref().unwrap_or(&[]);
        let hkdf = Hkdf::<Sha3_256>::new(Some(salt), ikm);
        
        hkdf.expand(info, output)
            .map_err(|e| CryptoError::KeyGeneration(format!("HKDF derivation failed: {:?}", e)))?;
        
        Ok(())
    }

    fn recommended_length(&self) -> usize {
        32 // 256 bits
    }
}

/// Argon2-based password derivation
pub struct Argon2Derivation {
    argon2: Argon2<'static>,
}

impl Argon2Derivation {
    /// Create a new Argon2 derivation with default parameters
    pub fn new() -> Self {
        Self {
            argon2: Argon2::default(),
        }
    }

    /// Create Argon2 with custom parameters
    pub fn with_params(
        memory_cost: u32,
        time_cost: u32,
        parallelism: u32,
    ) -> Result<Self> {
        use argon2::Params;
        
        let params = Params::new(memory_cost, time_cost, parallelism, None)
            .map_err(|e| CryptoError::KeyGeneration(format!("Invalid Argon2 params: {}", e)))?;
        
        let argon2 = Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
        
        Ok(Self { argon2 })
    }

    /// Hash a password with a random salt
    pub fn hash_password(&self, password: &[u8]) -> Result<String> {
        use rand::RngCore;
        
        let mut salt_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut salt_bytes);
        
        let salt = Salt::from_b64(&base64::encode(&salt_bytes))
            .map_err(|e| CryptoError::KeyGeneration(format!("Salt generation failed: {}", e)))?;

        let password_hash = self.argon2
            .hash_password(password, &salt)
            .map_err(|e| CryptoError::KeyGeneration(format!("Password hashing failed: {}", e)))?;

        Ok(password_hash.to_string())
    }

    /// Verify a password against a hash
    pub fn verify_password(&self, password: &[u8], hash: &str) -> Result<bool> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| CryptoError::Verification(format!("Invalid hash format: {}", e)))?;

        match self.argon2.verify_password(password, &parsed_hash) {
            Ok(()) => Ok(true),
            Err(argon2::password_hash::Error::Password) => Ok(false),
            Err(e) => Err(CryptoError::Verification(format!("Verification error: {}", e)).into()),
        }
    }

    /// Derive key from password and salt
    pub fn derive_from_password(&self, password: &[u8], salt: &[u8], output: &mut [u8]) -> Result<()> {
        use argon2::password_hash::SaltString;
        
        let salt_string = SaltString::encode_b64(salt)
            .map_err(|e| CryptoError::KeyGeneration(format!("Salt encoding failed: {}", e)))?;
        
        let salt = Salt::from(&salt_string);
        
        self.argon2
            .hash_password_into(password, salt.as_bytes(), output)
            .map_err(|e| CryptoError::KeyGeneration(format!("Argon2 derivation failed: {}", e)))?;
        
        Ok(())
    }
}

impl Default for Argon2Derivation {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyDerivation for Argon2Derivation {
    fn derive(&self, ikm: &[u8], info: &[u8], output: &mut [u8]) -> Result<()> {
        // Use info as salt for Argon2
        let salt_len = std::cmp::min(info.len(), 64);
        let salt = &info[..salt_len];
        
        self.derive_from_password(ikm, salt, output)
    }

    fn recommended_length(&self) -> usize {
        32 // 256 bits
    }
}

/// Key stretching for low-entropy keys
pub struct KeyStretcher {
    iterations: u32,
}

impl KeyStretcher {
    /// Create a new key stretcher with default iterations
    pub fn new() -> Self {
        Self {
            iterations: 100_000,
        }
    }

    /// Create with custom iteration count
    pub fn with_iterations(iterations: u32) -> Self {
        Self { iterations }
    }

    /// Stretch a key using PBKDF2
    pub fn stretch_key(&self, key: &[u8], salt: &[u8], output: &mut [u8]) -> Result<()> {
        use sha3::Sha3_256;
        use pbkdf2::pbkdf2;
        
        pbkdf2::<hmac::Hmac<Sha3_256>>(key, salt, self.iterations, output)
            .map_err(|e| CryptoError::KeyGeneration(format!("PBKDF2 failed: {:?}", e)))?;
        
        Ok(())
    }
}

impl Default for KeyStretcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Secure key erasure utility
pub struct SecureKey {
    data: Vec<u8>,
}

impl SecureKey {
    /// Create a new secure key
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Get key data
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable key data
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get key length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if key is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Drop for SecureKey {
    fn drop(&mut self) {
        self.data.zeroize();
    }
}

impl Zeroize for SecureKey {
    fn zeroize(&mut self) {
        self.data.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hkdf_derivation() {
        let hkdf = HkdfDerivation::new();
        let ikm = b"input key material";
        let info = b"context info";
        let mut output = [0u8; 32];
        
        let result = hkdf.derive(ikm, info, &mut output);
        assert!(result.is_ok());
        assert_ne!(output, [0u8; 32]); // Should not be all zeros
    }

    #[test]
    fn test_hkdf_with_salt() {
        let salt = b"random salt".to_vec();
        let hkdf = HkdfDerivation::with_salt(salt);
        let ikm = b"input key material";
        let info = b"context info";
        let mut output = [0u8; 32];
        
        let result = hkdf.derive(ikm, info, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_argon2_password_hash() {
        let argon2 = Argon2Derivation::new();
        let password = b"secure password";
        
        let hash = argon2.hash_password(password).unwrap();
        let is_valid = argon2.verify_password(password, &hash).unwrap();
        assert!(is_valid);
        
        let is_invalid = argon2.verify_password(b"wrong password", &hash).unwrap();
        assert!(!is_invalid);
    }

    #[test]
    fn test_argon2_key_derivation() {
        let argon2 = Argon2Derivation::new();
        let password = b"password";
        let salt = b"salt";
        let mut output = [0u8; 32];
        
        let result = argon2.derive_from_password(password, salt, &mut output);
        assert!(result.is_ok());
        assert_ne!(output, [0u8; 32]);
    }

    #[test]
    fn test_key_stretcher() {
        let stretcher = KeyStretcher::with_iterations(1000);
        let key = b"weak key";
        let salt = b"random salt";
        let mut output = [0u8; 32];
        
        let result = stretcher.stretch_key(key, salt, &mut output);
        assert!(result.is_ok());
        assert_ne!(output, [0u8; 32]);
    }

    #[test]
    fn test_secure_key_zeroization() {
        let data = vec![1, 2, 3, 4, 5];
        let mut key = SecureKey::new(data);
        
        assert_eq!(key.len(), 5);
        assert!(!key.is_empty());
        
        key.zeroize();
        assert_eq!(key.as_bytes(), &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_recommended_lengths() {
        let hkdf = HkdfDerivation::new();
        assert_eq!(hkdf.recommended_length(), 32);
        
        let argon2 = Argon2Derivation::new();
        assert_eq!(argon2.recommended_length(), 32);
    }
}