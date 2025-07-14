//! Secure memory management utilities

use crate::error::{CryptoError, Result};
use zeroize::{Zeroize, ZeroizeOnDrop};
use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::{self, NonNull},
    sync::atomic::{AtomicUsize, Ordering},
};

/// Global memory allocation counter for tracking
static SECURE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// Secure memory buffer that zeros on drop
#[derive(Debug, Zeroize, ZeroizeOnDrop)]
pub struct SecureBuffer {
    data: Vec<u8>,
    capacity: usize,
}

impl SecureBuffer {
    /// Create a new secure buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        SECURE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Create a secure buffer from existing data
    pub fn from_vec(mut data: Vec<u8>) -> Self {
        SECURE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        let capacity = data.capacity();
        Self { data, capacity }
    }

    /// Get the buffer data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable buffer data
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Extend buffer with data
    pub fn extend_from_slice(&mut self, data: &[u8]) -> Result<()> {
        if self.data.len() + data.len() > self.capacity {
            return Err(CryptoError::RandomGeneration("Buffer capacity exceeded".to_string()).into());
        }
        self.data.extend_from_slice(data);
        Ok(())
    }

    /// Clear the buffer (zeroing data)
    pub fn clear(&mut self) {
        self.data.zeroize();
        self.data.clear();
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Resize buffer (filling with zeros if expanding)
    pub fn resize(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.capacity {
            return Err(CryptoError::RandomGeneration("Cannot resize beyond capacity".to_string()).into());
        }
        self.data.resize(new_len, 0);
        Ok(())
    }
}

impl Drop for SecureBuffer {
    fn drop(&mut self) {
        SECURE_ALLOCATIONS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Locked memory page for sensitive data
#[cfg(unix)]
pub struct LockedMemory {
    ptr: NonNull<u8>,
    layout: Layout,
    locked: bool,
}

#[cfg(unix)]
impl LockedMemory {
    /// Create new locked memory region
    pub fn new(size: usize) -> Result<Self> {
        use libc::{mlock, mmap, MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE};
        
        let layout = Layout::from_size_align(size, std::mem::align_of::<u8>())
            .map_err(|e| CryptoError::RandomGeneration(format!("Invalid layout: {}", e)))?;

        unsafe {
            let ptr = mmap(
                ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return Err(CryptoError::RandomGeneration("Failed to allocate memory".to_string()).into());
            }

            let locked = mlock(ptr, size) == 0;
            let non_null = NonNull::new(ptr as *mut u8)
                .ok_or_else(|| CryptoError::RandomGeneration("Null pointer".to_string()))?;

            SECURE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);

            Ok(Self {
                ptr: non_null,
                layout,
                locked,
            })
        }
    }

    /// Get memory as slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }

    /// Get mutable memory as slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }

    /// Check if memory is locked
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Get memory size
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Zero the memory
    pub fn zero(&mut self) {
        self.as_mut_slice().zeroize();
    }
}

#[cfg(unix)]
impl Drop for LockedMemory {
    fn drop(&mut self) {
        use libc::{munlock, munmap};
        
        unsafe {
            // Zero before unlocking
            self.as_mut_slice().zeroize();
            
            if self.locked {
                munlock(self.ptr.as_ptr() as *mut libc::c_void, self.layout.size());
            }
            
            munmap(self.ptr.as_ptr() as *mut libc::c_void, self.layout.size());
        }
        
        SECURE_ALLOCATIONS.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(unix)]
unsafe impl Send for LockedMemory {}
#[cfg(unix)]
unsafe impl Sync for LockedMemory {}

/// Memory-mapped secure region for Windows
#[cfg(windows)]
pub struct LockedMemory {
    ptr: NonNull<u8>,
    size: usize,
    locked: bool,
}

#[cfg(windows)]
impl LockedMemory {
    /// Create new locked memory region
    pub fn new(size: usize) -> Result<Self> {
        use winapi::um::{
            memoryapi::{VirtualAlloc, VirtualLock},
            winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE},
        };

        unsafe {
            let ptr = VirtualAlloc(
                ptr::null_mut(),
                size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            );

            if ptr.is_null() {
                return Err(CryptoError::RandomGeneration("Failed to allocate memory".to_string()).into());
            }

            let locked = VirtualLock(ptr, size) != 0;
            let non_null = NonNull::new(ptr as *mut u8)
                .ok_or_else(|| CryptoError::RandomGeneration("Null pointer".to_string()))?;

            SECURE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);

            Ok(Self {
                ptr: non_null,
                size,
                locked,
            })
        }
    }

    /// Get memory as slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get mutable memory as slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Check if memory is locked
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Get memory size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Zero the memory
    pub fn zero(&mut self) {
        self.as_mut_slice().zeroize();
    }
}

#[cfg(windows)]
impl Drop for LockedMemory {
    fn drop(&mut self) {
        use winapi::um::{
            memoryapi::{VirtualFree, VirtualUnlock},
            winnt::MEM_RELEASE,
        };

        unsafe {
            // Zero before unlocking
            self.as_mut_slice().zeroize();
            
            if self.locked {
                VirtualUnlock(self.ptr.as_ptr() as *mut winapi::ctypes::c_void, self.size);
            }
            
            VirtualFree(
                self.ptr.as_ptr() as *mut winapi::ctypes::c_void,
                0,
                MEM_RELEASE,
            );
        }
        
        SECURE_ALLOCATIONS.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(windows)]
unsafe impl Send for LockedMemory {}
#[cfg(windows)]
unsafe impl Sync for LockedMemory {}

/// Fallback for other platforms
#[cfg(not(any(unix, windows)))]
pub struct LockedMemory {
    buffer: SecureBuffer,
}

#[cfg(not(any(unix, windows)))]
impl LockedMemory {
    /// Create new locked memory region (fallback implementation)
    pub fn new(size: usize) -> Result<Self> {
        Ok(Self {
            buffer: SecureBuffer::new(size),
        })
    }

    /// Get memory as slice
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// Get mutable memory as slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    /// Check if memory is locked (always false on unsupported platforms)
    pub fn is_locked(&self) -> bool {
        false
    }

    /// Get memory size
    pub fn size(&self) -> usize {
        self.buffer.capacity()
    }

    /// Zero the memory
    pub fn zero(&mut self) {
        self.buffer.clear();
    }
}

/// Secure memory pool for frequent allocations
pub struct SecureMemoryPool {
    buffers: Vec<SecureBuffer>,
    max_size: usize,
}

impl SecureMemoryPool {
    /// Create a new memory pool
    pub fn new(max_size: usize) -> Self {
        Self {
            buffers: Vec::new(),
            max_size,
        }
    }

    /// Get a buffer from the pool
    pub fn get_buffer(&mut self, size: usize) -> SecureBuffer {
        // Try to find a suitable buffer
        for i in 0..self.buffers.len() {
            if self.buffers[i].capacity() >= size {
                let mut buffer = self.buffers.swap_remove(i);
                buffer.clear();
                return buffer;
            }
        }

        // Create new buffer if pool not full
        SecureBuffer::new(size.max(1024)) // Minimum 1KB
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, mut buffer: SecureBuffer) {
        buffer.clear();
        
        if self.buffers.len() < self.max_size {
            self.buffers.push(buffer);
        }
        // Otherwise let it drop and be zeroized
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            buffers_in_pool: self.buffers.len(),
            max_size: self.max_size,
            total_capacity: self.buffers.iter().map(|b| b.capacity()).sum(),
        }
    }

    /// Clear all buffers in pool
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of buffers currently in pool
    pub buffers_in_pool: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Total capacity of all buffers
    pub total_capacity: usize,
}

/// Memory security utilities
pub struct MemorySecurity;

impl MemorySecurity {
    /// Get current secure allocation count
    pub fn allocation_count() -> usize {
        SECURE_ALLOCATIONS.load(Ordering::Relaxed)
    }

    /// Perform constant-time memory comparison
    pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let mut result = 0u8;
        for i in 0..a.len() {
            result |= a[i] ^ b[i];
        }
        result == 0
    }

    /// Secure memory copy with timing attack resistance
    pub fn secure_copy(src: &[u8], dst: &mut [u8]) -> Result<()> {
        if src.len() != dst.len() {
            return Err(CryptoError::RandomGeneration("Buffer size mismatch".to_string()).into());
        }

        for i in 0..src.len() {
            dst[i] = src[i];
        }
        
        Ok(())
    }

    /// Generate memory canary for buffer overflow detection
    pub fn generate_canary() -> [u8; 8] {
        use crate::crypto::random::SystemRng;
        let mut rng = SystemRng::new().expect("System RNG should be available");
        let mut canary = [0u8; 8];
        rng.fill_bytes(&mut canary).expect("Random generation should succeed");
        canary
    }

    /// Verify memory canary
    pub fn verify_canary(canary: &[u8; 8], expected: &[u8; 8]) -> bool {
        Self::constant_time_eq(canary, expected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_buffer() {
        let mut buffer = SecureBuffer::new(1024);
        assert_eq!(buffer.capacity(), 1024);
        assert!(buffer.is_empty());

        buffer.extend_from_slice(b"test data").unwrap();
        assert_eq!(buffer.len(), 9);
        assert_eq!(buffer.as_slice(), b"test data");

        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_secure_buffer_from_vec() {
        let data = vec![1, 2, 3, 4, 5];
        let buffer = SecureBuffer::from_vec(data);
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = SecureMemoryPool::new(5);
        
        let buffer1 = pool.get_buffer(512);
        let buffer2 = pool.get_buffer(1024);
        
        assert!(buffer1.capacity() >= 512);
        assert!(buffer2.capacity() >= 1024);
        
        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);
        
        let stats = pool.stats();
        assert_eq!(stats.buffers_in_pool, 2);
    }

    #[test]
    fn test_locked_memory() {
        let mut locked = LockedMemory::new(4096).unwrap();
        assert_eq!(locked.size(), 4096);
        
        let slice = locked.as_mut_slice();
        slice[0] = 42;
        assert_eq!(slice[0], 42);
        
        locked.zero();
        assert_eq!(slice[0], 0);
    }

    #[test]
    fn test_constant_time_comparison() {
        let a = [1, 2, 3, 4];
        let b = [1, 2, 3, 4];
        let c = [1, 2, 3, 5];
        
        assert!(MemorySecurity::constant_time_eq(&a, &b));
        assert!(!MemorySecurity::constant_time_eq(&a, &c));
        assert!(!MemorySecurity::constant_time_eq(&a, &[1, 2, 3]));
    }

    #[test]
    fn test_secure_copy() {
        let src = [1, 2, 3, 4];
        let mut dst = [0; 4];
        
        MemorySecurity::secure_copy(&src, &mut dst).unwrap();
        assert_eq!(src, dst);
        
        let mut wrong_size = [0; 3];
        assert!(MemorySecurity::secure_copy(&src, &mut wrong_size).is_err());
    }

    #[test]
    fn test_canary_generation() {
        let canary1 = MemorySecurity::generate_canary();
        let canary2 = MemorySecurity::generate_canary();
        
        assert_ne!(canary1, canary2); // Should be different
        assert!(MemorySecurity::verify_canary(&canary1, &canary1));
        assert!(!MemorySecurity::verify_canary(&canary1, &canary2));
    }

    #[test]
    fn test_allocation_tracking() {
        let initial_count = MemorySecurity::allocation_count();
        
        let _buffer = SecureBuffer::new(1024);
        assert_eq!(MemorySecurity::allocation_count(), initial_count + 1);
        
        drop(_buffer);
        assert_eq!(MemorySecurity::allocation_count(), initial_count);
    }
}