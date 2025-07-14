//! Hash function implementations

use crate::error::{CryptoError, Result};
use sha3::{Digest, Sha3_256, Sha3_512};
use blake3::{Hash, Hasher};

/// Trait for cryptographic hash functions
pub trait HashFunction: Send + Sync {
    /// Compute hash of input data
    fn hash(&self, data: &[u8]) -> Vec<u8>;
    
    /// Compute hash with incremental updates
    fn hash_incremental(&self, data_chunks: &[&[u8]]) -> Vec<u8>;
    
    /// Get the output size in bytes
    fn output_size(&self) -> usize;
    
    /// Get the algorithm name
    fn algorithm_name(&self) -> &'static str;
}

/// SHA-3 (256-bit) hash function
pub struct Sha3Hash;

impl Sha3Hash {
    /// Create a new SHA-3 hasher
    pub fn new() -> Self {
        Self
    }

    /// Compute SHA-3-256 hash
    pub fn sha3_256(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    /// Compute SHA-3-512 hash
    pub fn sha3_512(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha3_512::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    /// Create incremental hasher
    pub fn incremental() -> Sha3IncrementalHasher {
        Sha3IncrementalHasher::new()
    }
}

impl Default for Sha3Hash {
    fn default() -> Self {
        Self::new()
    }
}

impl HashFunction for Sha3Hash {
    fn hash(&self, data: &[u8]) -> Vec<u8> {
        Self::sha3_256(data)
    }

    fn hash_incremental(&self, data_chunks: &[&[u8]]) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        for chunk in data_chunks {
            hasher.update(chunk);
        }
        hasher.finalize().to_vec()
    }

    fn output_size(&self) -> usize {
        32 // SHA-3-256 output size
    }

    fn algorithm_name(&self) -> &'static str {
        "SHA3-256"
    }
}

/// BLAKE3 hash function
pub struct Blake3Hash;

impl Blake3Hash {
    /// Create a new BLAKE3 hasher
    pub fn new() -> Self {
        Self
    }

    /// Compute BLAKE3 hash
    pub fn blake3(data: &[u8]) -> Vec<u8> {
        blake3::hash(data).as_bytes().to_vec()
    }

    /// Compute keyed BLAKE3 hash
    pub fn blake3_keyed(key: &[u8; 32], data: &[u8]) -> Vec<u8> {
        blake3::keyed_hash(key, data).as_bytes().to_vec()
    }

    /// Compute BLAKE3 with custom output length
    pub fn blake3_custom_length(data: &[u8], length: usize) -> Vec<u8> {
        let mut hasher = Hasher::new();
        hasher.update(data);
        let mut output = vec![0u8; length];
        hasher.finalize_xof().fill(&mut output);
        output
    }

    /// Create incremental hasher
    pub fn incremental() -> Blake3IncrementalHasher {
        Blake3IncrementalHasher::new()
    }
}

impl Default for Blake3Hash {
    fn default() -> Self {
        Self::new()
    }
}

impl HashFunction for Blake3Hash {
    fn hash(&self, data: &[u8]) -> Vec<u8> {
        Self::blake3(data)
    }

    fn hash_incremental(&self, data_chunks: &[&[u8]]) -> Vec<u8> {
        let mut hasher = Hasher::new();
        for chunk in data_chunks {
            hasher.update(chunk);
        }
        hasher.finalize().as_bytes().to_vec()
    }

    fn output_size(&self) -> usize {
        32 // BLAKE3 default output size
    }

    fn algorithm_name(&self) -> &'static str {
        "BLAKE3"
    }
}

/// Incremental SHA-3 hasher
pub struct Sha3IncrementalHasher {
    hasher: Sha3_256,
}

impl Sha3IncrementalHasher {
    /// Create new incremental hasher
    pub fn new() -> Self {
        Self {
            hasher: Sha3_256::new(),
        }
    }

    /// Update hasher with data
    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// Finalize and get hash result
    pub fn finalize(self) -> Vec<u8> {
        self.hasher.finalize().to_vec()
    }

    /// Reset the hasher
    pub fn reset(&mut self) {
        self.hasher = Sha3_256::new();
    }
}

impl Default for Sha3IncrementalHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental BLAKE3 hasher
pub struct Blake3IncrementalHasher {
    hasher: Hasher,
}

impl Blake3IncrementalHasher {
    /// Create new incremental hasher
    pub fn new() -> Self {
        Self {
            hasher: Hasher::new(),
        }
    }

    /// Create keyed incremental hasher
    pub fn new_keyed(key: &[u8; 32]) -> Self {
        Self {
            hasher: Hasher::new_keyed(key),
        }
    }

    /// Update hasher with data
    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// Finalize and get hash result
    pub fn finalize(&self) -> Vec<u8> {
        self.hasher.finalize().as_bytes().to_vec()
    }

    /// Finalize with custom output length
    pub fn finalize_with_length(&self, length: usize) -> Vec<u8> {
        let mut output = vec![0u8; length];
        self.hasher.finalize_xof().fill(&mut output);
        output
    }

    /// Reset the hasher
    pub fn reset(&mut self) {
        self.hasher = Hasher::new();
    }
}

impl Default for Blake3IncrementalHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash-based Message Authentication Code (HMAC) utilities
pub struct HmacUtils;

impl HmacUtils {
    /// Compute HMAC-SHA3-256
    pub fn hmac_sha3_256(key: &[u8], message: &[u8]) -> Vec<u8> {
        use hmac::{Hmac, Mac};
        
        let mut mac = Hmac::<Sha3_256>::new_from_slice(key)
            .expect("HMAC can take key of any size");
        mac.update(message);
        mac.finalize().into_bytes().to_vec()
    }

    /// Verify HMAC-SHA3-256
    pub fn verify_hmac_sha3_256(key: &[u8], message: &[u8], expected: &[u8]) -> bool {
        use hmac::{Hmac, Mac};
        
        let mut mac = Hmac::<Sha3_256>::new_from_slice(key)
            .expect("HMAC can take key of any size");
        mac.update(message);
        
        mac.verify_slice(expected).is_ok()
    }
}

/// Merkle tree utilities for integrity verification
pub struct MerkleTree {
    leaves: Vec<Vec<u8>>,
    hash_function: Box<dyn HashFunction>,
}

impl MerkleTree {
    /// Create new Merkle tree with BLAKE3
    pub fn new_blake3() -> Self {
        Self {
            leaves: Vec::new(),
            hash_function: Box::new(Blake3Hash::new()),
        }
    }

    /// Create new Merkle tree with SHA-3
    pub fn new_sha3() -> Self {
        Self {
            leaves: Vec::new(),
            hash_function: Box::new(Sha3Hash::new()),
        }
    }

    /// Add leaf to the tree
    pub fn add_leaf(&mut self, data: &[u8]) {
        let hash = self.hash_function.hash(data);
        self.leaves.push(hash);
    }

    /// Compute root hash
    pub fn root(&self) -> Option<Vec<u8>> {
        if self.leaves.is_empty() {
            return None;
        }

        let mut level = self.leaves.clone();
        
        while level.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in level.chunks(2) {
                let combined = if chunk.len() == 2 {
                    [chunk[0].as_slice(), chunk[1].as_slice()].concat()
                } else {
                    chunk[0].clone()
                };
                
                let hash = self.hash_function.hash(&combined);
                next_level.push(hash);
            }
            
            level = next_level;
        }
        
        level.into_iter().next()
    }

    /// Get number of leaves
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha3_hash() {
        let hasher = Sha3Hash::new();
        let data = b"test data";
        let hash = hasher.hash(data);
        
        assert_eq!(hash.len(), 32);
        assert_eq!(hasher.output_size(), 32);
        assert_eq!(hasher.algorithm_name(), "SHA3-256");
    }

    #[test]
    fn test_blake3_hash() {
        let hasher = Blake3Hash::new();
        let data = b"test data";
        let hash = hasher.hash(data);
        
        assert_eq!(hash.len(), 32);
        assert_eq!(hasher.output_size(), 32);
        assert_eq!(hasher.algorithm_name(), "BLAKE3");
    }

    #[test]
    fn test_incremental_hashing() {
        let data1 = b"hello ";
        let data2 = b"world";
        let combined = b"hello world";
        
        // SHA-3
        let sha3 = Sha3Hash::new();
        let hash_combined = sha3.hash(combined);
        let hash_incremental = sha3.hash_incremental(&[data1, data2]);
        assert_eq!(hash_combined, hash_incremental);
        
        // BLAKE3
        let blake3 = Blake3Hash::new();
        let hash_combined = blake3.hash(combined);
        let hash_incremental = blake3.hash_incremental(&[data1, data2]);
        assert_eq!(hash_combined, hash_incremental);
    }

    #[test]
    fn test_incremental_hasher_sha3() {
        let mut hasher = Sha3IncrementalHasher::new();
        hasher.update(b"hello ");
        hasher.update(b"world");
        let hash = hasher.finalize();
        
        let direct_hash = Sha3Hash::sha3_256(b"hello world");
        assert_eq!(hash, direct_hash);
    }

    #[test]
    fn test_incremental_hasher_blake3() {
        let mut hasher = Blake3IncrementalHasher::new();
        hasher.update(b"hello ");
        hasher.update(b"world");
        let hash = hasher.finalize();
        
        let direct_hash = Blake3Hash::blake3(b"hello world");
        assert_eq!(hash, direct_hash);
    }

    #[test]
    fn test_hmac() {
        let key = b"secret key";
        let message = b"important message";
        
        let mac = HmacUtils::hmac_sha3_256(key, message);
        assert_eq!(mac.len(), 32);
        
        let is_valid = HmacUtils::verify_hmac_sha3_256(key, message, &mac);
        assert!(is_valid);
        
        let is_invalid = HmacUtils::verify_hmac_sha3_256(b"wrong key", message, &mac);
        assert!(!is_invalid);
    }

    #[test]
    fn test_merkle_tree() {
        let mut tree = MerkleTree::new_blake3();
        
        tree.add_leaf(b"leaf1");
        tree.add_leaf(b"leaf2");
        tree.add_leaf(b"leaf3");
        tree.add_leaf(b"leaf4");
        
        assert_eq!(tree.len(), 4);
        
        let root = tree.root().unwrap();
        assert_eq!(root.len(), 32);
    }

    #[test]
    fn test_empty_merkle_tree() {
        let tree = MerkleTree::new_sha3();
        assert!(tree.is_empty());
        assert!(tree.root().is_none());
    }

    #[test]
    fn test_single_leaf_merkle_tree() {
        let mut tree = MerkleTree::new_blake3();
        tree.add_leaf(b"single leaf");
        
        let root = tree.root().unwrap();
        let expected = Blake3Hash::blake3(b"single leaf");
        assert_eq!(root, expected);
    }
}