use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Aligned vector for SIMD operations
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    alignment: usize,
}

impl<T> AlignedVec<T> {
    /// Create a new aligned vector with specified alignment
    pub fn new(alignment: usize) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            alignment,
        }
    }
    
    /// Create a new aligned vector filled with zeros
    pub fn new_zeroed(len: usize, alignment: usize) -> Self 
    where 
        T: Clone + Default,
    {
        let mut vec = Self::with_capacity(len, alignment);
        vec.resize(len, T::default());
        vec
    }
    
    /// Create a new aligned vector from an iterator
    pub fn from_iter<I>(iter: I, alignment: usize) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Clone,
    {
        let items: Vec<T> = iter.into_iter().collect();
        let mut vec = Self::with_capacity(items.len(), alignment);
        for item in items {
            vec.push(item);
        }
        vec
    }
    
    /// Create with specified capacity
    pub fn with_capacity(capacity: usize, alignment: usize) -> Self {
        if capacity == 0 {
            return Self::new(alignment);
        }
        
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            alignment,
        ).expect("Invalid layout");
        
        let ptr = unsafe { alloc(layout) as *mut T };
        let ptr = NonNull::new(ptr).expect("Allocation failed");
        
        Self {
            ptr,
            len: 0,
            capacity,
            alignment,
        }
    }
    
    /// Push an element to the vector
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            self.grow();
        }
        
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
    }
    
    /// Resize the vector
    pub fn resize(&mut self, new_len: usize, value: T) 
    where 
        T: Clone,
    {
        if new_len > self.capacity {
            self.reserve(new_len - self.capacity);
        }
        
        if new_len > self.len {
            for i in self.len..new_len {
                unsafe {
                    self.ptr.as_ptr().add(i).write(value.clone());
                }
            }
        } else if new_len < self.len {
            for i in new_len..self.len {
                unsafe {
                    self.ptr.as_ptr().add(i).drop_in_place();
                }
            }
        }
        
        self.len = new_len;
    }
    
    /// Reserve additional capacity
    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.capacity + additional;
        if new_capacity <= self.capacity {
            return;
        }
        
        let new_layout = Layout::from_size_align(
            new_capacity * std::mem::size_of::<T>(),
            self.alignment,
        ).expect("Invalid layout");
        
        let new_ptr = unsafe { alloc(new_layout) as *mut T };
        let new_ptr = NonNull::new(new_ptr).expect("Allocation failed");
        
        if self.capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ptr.as_ptr(),
                    new_ptr.as_ptr(),
                    self.len,
                );
                
                let old_layout = Layout::from_size_align(
                    self.capacity * std::mem::size_of::<T>(),
                    self.alignment,
                ).expect("Invalid layout");
                dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            }
        }
        
        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }
    
    /// Grow the vector capacity
    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 { 4 } else { self.capacity * 2 };
        self.reserve(new_capacity - self.capacity);
    }
    
    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    
    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            for i in 0..self.len {
                unsafe {
                    self.ptr.as_ptr().add(i).drop_in_place();
                }
            }
            
            let layout = Layout::from_size_align(
                self.capacity * std::mem::size_of::<T>(),
                self.alignment,
            ).expect("Invalid layout");
            
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    block_size: usize,
    alignment: usize,
    free_blocks: Vec<*mut u8>,
    allocated_blocks: Vec<*mut u8>,
}

impl MemoryPool {
    pub fn new(block_size: usize, alignment: usize) -> Self {
        Self {
            block_size,
            alignment,
            free_blocks: Vec::new(),
            allocated_blocks: Vec::new(),
        }
    }
    
    pub fn allocate(&mut self) -> Option<*mut u8> {
        if self.free_blocks.is_empty() {
            self.expand_pool(8);
        }
        
        self.free_blocks.pop().map(|ptr| {
            self.allocated_blocks.push(ptr);
            ptr
        })
    }
    
    pub fn deallocate(&mut self, ptr: *mut u8) {
        if let Some(pos) = self.allocated_blocks.iter().position(|&p| p == ptr) {
            self.allocated_blocks.remove(pos);
            self.free_blocks.push(ptr);
        }
    }
    
    fn expand_pool(&mut self, count: usize) {
        for _ in 0..count {
            let layout = Layout::from_size_align(self.block_size, self.alignment)
                .expect("Invalid layout");
            
            let ptr = unsafe { alloc(layout) };
            if !ptr.is_null() {
                self.free_blocks.push(ptr);
            }
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.block_size, self.alignment)
            .expect("Invalid layout");
        
        for &ptr in &self.free_blocks {
            unsafe { dealloc(ptr, layout); }
        }
        
        for &ptr in &self.allocated_blocks {
            unsafe { dealloc(ptr, layout); }
        }
    }
}

/// Allocation patterns for testing
#[derive(Debug, Clone, Copy)]
pub enum AllocationPattern {
    Sequential,
    Random,
    Batch,
    Fragmented,
}

/// Memory analyzer for testing allocation patterns
pub struct MemoryAnalyzer;

impl MemoryAnalyzer {
    pub fn new() -> Self {
        Self
    }
    
    pub fn simulate_allocation_pattern(
        &self,
        pool: &mut MemoryPool,
        pattern: &AllocationPattern,
        operations: usize,
    ) -> f32 {
        // Simulate allocation pattern and return fragmentation ratio
        match pattern {
            AllocationPattern::Sequential => {
                let mut allocations = Vec::new();
                for _ in 0..operations {
                    if let Some(ptr) = pool.allocate() {
                        allocations.push(ptr);
                    }
                }
                for ptr in allocations {
                    pool.deallocate(ptr);
                }
                0.1 // Low fragmentation
            },
            AllocationPattern::Random => {
                let mut allocations = Vec::new();
                for i in 0..operations {
                    if i % 3 == 0 && !allocations.is_empty() {
                        let idx = i % allocations.len();
                        let ptr = allocations.remove(idx);
                        pool.deallocate(ptr);
                    } else if let Some(ptr) = pool.allocate() {
                        allocations.push(ptr);
                    }
                }
                for ptr in allocations {
                    pool.deallocate(ptr);
                }
                0.5 // Medium fragmentation
            },
            AllocationPattern::Fragmented => {
                let mut allocations = Vec::new();
                for i in 0..operations {
                    if i % 2 == 0 {
                        if let Some(ptr) = pool.allocate() {
                            allocations.push(ptr);
                        }
                    } else if !allocations.is_empty() {
                        let ptr = allocations.remove(0);
                        pool.deallocate(ptr);
                    }
                }
                for ptr in allocations {
                    pool.deallocate(ptr);
                }
                0.8 // High fragmentation
            },
            AllocationPattern::Batch => {
                for batch in 0..operations / 10 {
                    let mut batch_allocations = Vec::new();
                    for _ in 0..10 {
                        if let Some(ptr) = pool.allocate() {
                            batch_allocations.push(ptr);
                        }
                    }
                    if batch % 2 == 1 {
                        for ptr in batch_allocations {
                            pool.deallocate(ptr);
                        }
                    }
                }
                0.3 // Medium-low fragmentation
            },
        }
    }
}

impl Default for MemoryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}