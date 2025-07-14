//! Secure random number generation

use crate::error::{CryptoError, Result};
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_core::OsRng;
use getrandom::getrandom;

/// Trait for secure random number generation
pub trait SecureRng: CryptoRng + RngCore + Send + Sync {
    /// Fill buffer with random bytes
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<()>;
    
    /// Generate random bytes
    fn generate_bytes(&mut self, len: usize) -> Result<Vec<u8>>;
    
    /// Generate random u64
    fn generate_u64(&mut self) -> Result<u64>;
    
    /// Generate random u32  
    fn generate_u32(&mut self) -> Result<u32>;
}

/// System random number generator
pub struct SystemRng {
    rng: OsRng,
}

impl SystemRng {
    /// Create a new system RNG
    pub fn new() -> Result<Self> {
        Ok(Self { rng: OsRng })
    }

    /// Generate cryptographically secure random bytes
    pub fn random_bytes(len: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; len];
        getrandom(&mut bytes)
            .map_err(|e| CryptoError::RandomGeneration(format!("System RNG failed: {}", e)))?;
        Ok(bytes)
    }

    /// Generate a random nonce
    pub fn random_nonce() -> Result<[u8; 12]> {
        let mut nonce = [0u8; 12];
        getrandom(&mut nonce)
            .map_err(|e| CryptoError::RandomGeneration(format!("Nonce generation failed: {}", e)))?;
        Ok(nonce)
    }

    /// Generate a random key
    pub fn random_key(size: usize) -> Result<Vec<u8>> {
        Self::random_bytes(size)
    }

    /// Generate a random salt
    pub fn random_salt() -> Result<[u8; 32]> {
        let mut salt = [0u8; 32];
        getrandom(&mut salt)
            .map_err(|e| CryptoError::RandomGeneration(format!("Salt generation failed: {}", e)))?;
        Ok(salt)
    }
}

impl Default for SystemRng {
    fn default() -> Self {
        Self::new().expect("System RNG should always be available")
    }
}

impl RngCore for SystemRng {
    fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), rand_core::Error> {
        self.rng.try_fill_bytes(dest)
    }
}

impl CryptoRng for SystemRng {}

impl SecureRng for SystemRng {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<()> {
        RngCore::fill_bytes(self, dest);
        Ok(())
    }

    fn generate_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; len];
        self.fill_bytes(&mut bytes)?;
        Ok(bytes)
    }

    fn generate_u64(&mut self) -> Result<u64> {
        Ok(self.next_u64())
    }

    fn generate_u32(&mut self) -> Result<u32> {
        Ok(self.next_u32())
    }
}

/// Deterministic RNG for testing (NOT cryptographically secure)
#[cfg(test)]
pub struct TestRng {
    rng: rand_chacha::ChaCha20Rng,
}

#[cfg(test)]
impl TestRng {
    /// Create a new test RNG with fixed seed
    pub fn new() -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand_chacha::ChaCha20Rng::from_seed([42u8; 32]),
        }
    }

    /// Create test RNG with custom seed
    pub fn with_seed(seed: [u8; 32]) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand_chacha::ChaCha20Rng::from_seed(seed),
        }
    }
}

#[cfg(test)]
impl RngCore for TestRng {
    fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), rand_core::Error> {
        self.rng.try_fill_bytes(dest)
    }
}

#[cfg(test)]
impl CryptoRng for TestRng {}

#[cfg(test)]
impl SecureRng for TestRng {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<()> {
        RngCore::fill_bytes(self, dest);
        Ok(())
    }

    fn generate_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; len];
        self.fill_bytes(&mut bytes)?;
        Ok(bytes)
    }

    fn generate_u64(&mut self) -> Result<u64> {
        Ok(self.next_u64())
    }

    fn generate_u32(&mut self) -> Result<u32> {
        Ok(self.next_u32())
    }
}

/// Random utilities for common operations
pub struct RandomUtils;

impl RandomUtils {
    /// Generate a cryptographically secure UUID
    pub fn secure_uuid() -> Result<uuid::Uuid> {
        let mut bytes = [0u8; 16];
        getrandom(&mut bytes)
            .map_err(|e| CryptoError::RandomGeneration(format!("UUID generation failed: {}", e)))?;
        Ok(uuid::Uuid::from_bytes(bytes))
    }

    /// Generate a random session ID
    pub fn session_id() -> Result<[u8; 32]> {
        let mut id = [0u8; 32];
        getrandom(&mut id)
            .map_err(|e| CryptoError::RandomGeneration(format!("Session ID generation failed: {}", e)))?;
        Ok(id)
    }

    /// Generate a random challenge for authentication
    pub fn challenge() -> Result<[u8; 32]> {
        Self::session_id() // Same as session ID
    }

    /// Generate random padding of specified length
    pub fn random_padding(min_len: usize, max_len: usize) -> Result<Vec<u8>> {
        if min_len > max_len {
            return Err(CryptoError::RandomGeneration("Invalid padding range".to_string()).into());
        }

        let mut rng = SystemRng::new()?;
        let len = if min_len == max_len {
            min_len
        } else {
            min_len + (rng.generate_u32()? as usize % (max_len - min_len + 1))
        };

        rng.generate_bytes(len)
    }

    /// Generate random delay for timing attack mitigation (in milliseconds)
    pub fn random_delay_ms(min_ms: u64, max_ms: u64) -> Result<u64> {
        if min_ms > max_ms {
            return Err(CryptoError::RandomGeneration("Invalid delay range".to_string()).into());
        }

        let mut rng = SystemRng::new()?;
        let range = max_ms - min_ms + 1;
        let delay = min_ms + (rng.generate_u64()? % range);
        Ok(delay)
    }
}

/// Entropy source management
pub struct EntropySource;

impl EntropySource {
    /// Estimate available entropy (simplified implementation)
    pub fn estimate_entropy() -> Result<f64> {
        // This is a simplified entropy estimation
        // In practice, you'd query the system entropy pool
        let mut sample = [0u8; 1024];
        getrandom(&mut sample)
            .map_err(|e| CryptoError::RandomGeneration(format!("Entropy sampling failed: {}", e)))?;

        // Simple entropy calculation based on byte frequency
        let mut counts = [0u32; 256];
        for &byte in &sample {
            counts[byte as usize] += 1;
        }

        let mut entropy = 0.0;
        for &count in &counts {
            if count > 0 {
                let p = count as f64 / sample.len() as f64;
                entropy -= p * p.log2();
            }
        }

        Ok(entropy)
    }

    /// Check if sufficient entropy is available
    pub fn has_sufficient_entropy() -> Result<bool> {
        let entropy = Self::estimate_entropy()?;
        // Require at least 6 bits of entropy per byte
        Ok(entropy >= 6.0)
    }

    /// Wait for sufficient entropy (simplified)
    pub async fn wait_for_entropy() -> Result<()> {
        use tokio::time::{sleep, Duration};

        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 10;

        while attempts < MAX_ATTEMPTS {
            if Self::has_sufficient_entropy()? {
                return Ok(());
            }

            attempts += 1;
            sleep(Duration::from_millis(100)).await;
        }

        Err(CryptoError::RandomGeneration("Insufficient entropy after waiting".to_string()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_rng() {
        let mut rng = SystemRng::new().unwrap();
        
        let bytes1 = rng.generate_bytes(32).unwrap();
        let bytes2 = rng.generate_bytes(32).unwrap();
        
        assert_eq!(bytes1.len(), 32);
        assert_eq!(bytes2.len(), 32);
        assert_ne!(bytes1, bytes2); // Should be different
    }

    #[test]
    fn test_random_utilities() {
        let bytes = SystemRng::random_bytes(16).unwrap();
        assert_eq!(bytes.len(), 16);

        let nonce = SystemRng::random_nonce().unwrap();
        assert_eq!(nonce.len(), 12);

        let salt = SystemRng::random_salt().unwrap();
        assert_eq!(salt.len(), 32);
    }

    #[test]
    fn test_uuid_generation() {
        let uuid1 = RandomUtils::secure_uuid().unwrap();
        let uuid2 = RandomUtils::secure_uuid().unwrap();
        
        assert_ne!(uuid1, uuid2);
    }

    #[test]
    fn test_session_id() {
        let id1 = RandomUtils::session_id().unwrap();
        let id2 = RandomUtils::session_id().unwrap();
        
        assert_ne!(id1, id2);
        assert_eq!(id1.len(), 32);
    }

    #[test]
    fn test_random_padding() {
        let padding = RandomUtils::random_padding(10, 20).unwrap();
        assert!(padding.len() >= 10);
        assert!(padding.len() <= 20);

        let fixed_padding = RandomUtils::random_padding(15, 15).unwrap();
        assert_eq!(fixed_padding.len(), 15);
    }

    #[test]
    fn test_random_delay() {
        let delay = RandomUtils::random_delay_ms(100, 200).unwrap();
        assert!(delay >= 100);
        assert!(delay <= 200);
    }

    #[test]
    fn test_entropy_estimation() {
        let entropy = EntropySource::estimate_entropy().unwrap();
        assert!(entropy > 0.0);
        assert!(entropy <= 8.0); // Maximum possible entropy per byte
    }

    #[test]
    fn test_test_rng() {
        let mut rng1 = TestRng::new();
        let mut rng2 = TestRng::new();
        
        let bytes1 = rng1.generate_bytes(16).unwrap();
        let bytes2 = rng2.generate_bytes(16).unwrap();
        
        // With same seed, should generate same bytes
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn test_invalid_padding_range() {
        let result = RandomUtils::random_padding(20, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_delay_range() {
        let result = RandomUtils::random_delay_ms(200, 100);
        assert!(result.is_err());
    }
}