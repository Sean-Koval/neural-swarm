//! Performance profiling and analysis utilities

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Memory profiler for tracking allocations and performance
#[derive(Debug)]
pub struct MemoryProfiler {
    allocations: HashMap<usize, AllocationInfo>,
    total_allocated: usize,
    total_deallocated: usize,
    peak_usage: usize,
    current_usage: usize,
    allocation_timeline: Vec<AllocationEvent>,
    start_time: Instant,
}

/// Information about a specific allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub size: usize,
    pub timestamp: Instant,
    pub location: Option<String>,
}

/// Allocation event for timeline tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    pub timestamp: Duration,
    pub event_type: AllocationEventType,
    pub size: usize,
    pub address: usize,
}

/// Types of allocation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationEventType {
    Allocation,
    Deallocation,
    Reallocation,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
            total_deallocated: 0,
            peak_usage: 0,
            current_usage: 0,
            allocation_timeline: Vec::new(),
            start_time: Instant::now(),
        }
    }
    
    /// Record a memory allocation
    pub fn before_allocation(&mut self, size: usize) {
        // Preparation for allocation tracking
    }
    
    /// Record completion of memory allocation
    pub fn after_allocation(&mut self, address: usize, size: usize) {
        let timestamp = Instant::now();
        
        self.allocations.insert(address, AllocationInfo {
            size,
            timestamp,
            location: None,
        });
        
        self.total_allocated += size;
        self.current_usage += size;
        
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
        
        self.allocation_timeline.push(AllocationEvent {
            timestamp: timestamp.duration_since(self.start_time),
            event_type: AllocationEventType::Allocation,
            size,
            address,
        });
    }
    
    /// Record a memory deallocation
    pub fn before_deallocation(&mut self, address: usize) {
        if let Some(info) = self.allocations.remove(&address) {
            self.total_deallocated += info.size;
            self.current_usage -= info.size;
            
            self.allocation_timeline.push(AllocationEvent {
                timestamp: Instant::now().duration_since(self.start_time),
                event_type: AllocationEventType::Deallocation,
                size: info.size,
                address,
            });
        }
    }
    
    /// Generate a comprehensive memory report
    pub fn generate_report(&self) -> MemoryReport {
        let fragmentation_ratio = self.calculate_fragmentation_ratio();
        let allocation_frequency = self.calculate_allocation_frequency();
        let memory_efficiency = self.calculate_memory_efficiency();
        
        MemoryReport {
            total_allocated: self.total_allocated,
            total_deallocated: self.total_deallocated,
            peak_usage: self.peak_usage,
            current_usage: self.current_usage,
            active_allocations: self.allocations.len(),
            fragmentation_ratio,
            allocation_frequency,
            memory_efficiency,
            timeline: self.allocation_timeline.clone(),
        }
    }
    
    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation_ratio(&self) -> f32 {
        if self.allocations.is_empty() {
            return 0.0;
        }
        
        let total_allocated_blocks = self.allocations.len();
        let ideal_blocks = if self.current_usage > 0 { 1 } else { 0 };
        
        if ideal_blocks == 0 {
            0.0
        } else {
            (total_allocated_blocks as f32 - ideal_blocks as f32) / total_allocated_blocks as f32
        }
    }
    
    /// Calculate allocation frequency (allocations per second)
    fn calculate_allocation_frequency(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.allocation_timeline.len() as f32 / elapsed
        } else {
            0.0
        }
    }
    
    /// Calculate memory efficiency (useful memory / total allocated)
    fn calculate_memory_efficiency(&self) -> f32 {
        if self.total_allocated > 0 {
            self.current_usage as f32 / self.total_allocated as f32
        } else {
            1.0
        }
    }
}

/// Memory profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    pub total_allocated: usize,
    pub total_deallocated: usize,
    pub peak_usage: usize,
    pub current_usage: usize,
    pub active_allocations: usize,
    pub fragmentation_ratio: f32,
    pub allocation_frequency: f32,
    pub memory_efficiency: f32,
    pub timeline: Vec<AllocationEvent>,
}

/// Simple allocation tracker for lightweight monitoring
#[derive(Debug)]
pub struct AllocationTracker {
    total_allocations: usize,
    total_deallocations: usize,
    total_bytes_allocated: usize,
    total_bytes_deallocated: usize,
    peak_allocations: usize,
    peak_bytes: usize,
    current_allocations: usize,
    current_bytes: usize,
}

impl AllocationTracker {
    /// Create a new allocation tracker
    pub fn new() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            total_bytes_allocated: 0,
            total_bytes_deallocated: 0,
            peak_allocations: 0,
            peak_bytes: 0,
            current_allocations: 0,
            current_bytes: 0,
        }
    }
    
    /// Track a new allocation
    pub fn track_allocation(&mut self, size: usize) {
        self.total_allocations += 1;
        self.total_bytes_allocated += size;
        self.current_allocations += 1;
        self.current_bytes += size;
        
        if self.current_allocations > self.peak_allocations {
            self.peak_allocations = self.current_allocations;
        }
        
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }
    }
    
    /// Track a deallocation
    pub fn track_deallocation(&mut self, size: usize) {
        self.total_deallocations += 1;
        self.total_bytes_deallocated += size;
        
        if self.current_allocations > 0 {
            self.current_allocations -= 1;
        }
        
        if self.current_bytes >= size {
            self.current_bytes -= size;
        }
    }
    
    /// Get current statistics
    pub fn get_statistics(&self) -> AllocationStatistics {
        AllocationStatistics {
            total_allocations: self.total_allocations,
            total_deallocations: self.total_deallocations,
            total_bytes_allocated: self.total_bytes_allocated,
            total_bytes_deallocated: self.total_bytes_deallocated,
            peak_allocations: self.peak_allocations,
            peak_bytes: self.peak_bytes,
            current_allocations: self.current_allocations,
            current_bytes: self.current_bytes,
            allocation_efficiency: self.calculate_allocation_efficiency(),
            memory_turnover: self.calculate_memory_turnover(),
        }
    }
    
    /// Calculate allocation efficiency
    fn calculate_allocation_efficiency(&self) -> f32 {
        if self.total_allocations > 0 {
            self.total_deallocations as f32 / self.total_allocations as f32
        } else {
            0.0
        }
    }
    
    /// Calculate memory turnover rate
    fn calculate_memory_turnover(&self) -> f32 {
        if self.total_bytes_allocated > 0 {
            self.total_bytes_deallocated as f32 / self.total_bytes_allocated as f32
        } else {
            0.0
        }
    }
}

/// Allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub total_bytes_allocated: usize,
    pub total_bytes_deallocated: usize,
    pub peak_allocations: usize,
    pub peak_bytes: usize,
    pub current_allocations: usize,
    pub current_bytes: usize,
    pub allocation_efficiency: f32,
    pub memory_turnover: f32,
}

/// Performance profiler for timing operations
#[derive(Debug)]
pub struct PerformanceProfiler {
    timings: HashMap<String, Vec<Duration>>,
    active_timers: HashMap<String, Instant>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            active_timers: HashMap::new(),
        }
    }
    
    /// Start timing an operation
    pub fn start_timer(&mut self, operation: &str) {
        self.active_timers.insert(operation.to_string(), Instant::now());
    }
    
    /// Stop timing an operation and record the duration
    pub fn stop_timer(&mut self, operation: &str) {
        if let Some(start_time) = self.active_timers.remove(operation) {
            let duration = start_time.elapsed();
            self.timings.entry(operation.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }
    
    /// Record a single timing measurement
    pub fn record_timing(&mut self, operation: &str, duration: Duration) {
        self.timings.entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }
    
    /// Generate a performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut operation_stats = HashMap::new();
        
        for (operation, timings) in &self.timings {
            let stats = self.calculate_timing_statistics(timings);
            operation_stats.insert(operation.clone(), stats);
        }
        
        PerformanceReport {
            operation_stats,
            total_operations: self.timings.values().map(|v| v.len()).sum(),
        }
    }
    
    /// Calculate statistics for a set of timings
    fn calculate_timing_statistics(&self, timings: &[Duration]) -> TimingStatistics {
        if timings.is_empty() {
            return TimingStatistics::default();
        }
        
        let mut sorted_timings = timings.to_vec();
        sorted_timings.sort();
        
        let count = timings.len();
        let total: Duration = timings.iter().sum();
        let mean = total / count as u32;
        
        let median = if count % 2 == 0 {
            (sorted_timings[count / 2 - 1] + sorted_timings[count / 2]) / 2
        } else {
            sorted_timings[count / 2]
        };
        
        let min = *sorted_timings.first().unwrap();
        let max = *sorted_timings.last().unwrap();
        
        // Calculate percentiles
        let p95_idx = (count as f32 * 0.95) as usize;
        let p99_idx = (count as f32 * 0.99) as usize;
        let p95 = sorted_timings[p95_idx.min(count - 1)];
        let p99 = sorted_timings[p99_idx.min(count - 1)];
        
        // Calculate standard deviation
        let variance: f64 = timings.iter()
            .map(|&duration| {
                let diff = duration.as_nanos() as f64 - mean.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>() / count as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);
        
        TimingStatistics {
            count,
            total,
            mean,
            median,
            min,
            max,
            std_dev,
            p95,
            p99,
        }
    }
}

/// Performance profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub operation_stats: HashMap<String, TimingStatistics>,
    pub total_operations: usize,
}

/// Timing statistics for an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStatistics {
    pub count: usize,
    #[serde(with = "duration_serde")]
    pub total: Duration,
    #[serde(with = "duration_serde")]
    pub mean: Duration,
    #[serde(with = "duration_serde")]
    pub median: Duration,
    #[serde(with = "duration_serde")]
    pub min: Duration,
    #[serde(with = "duration_serde")]
    pub max: Duration,
    #[serde(with = "duration_serde")]
    pub std_dev: Duration,
    #[serde(with = "duration_serde")]
    pub p95: Duration,
    #[serde(with = "duration_serde")]
    pub p99: Duration,
}

impl Default for TimingStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            total: Duration::ZERO,
            mean: Duration::ZERO,
            median: Duration::ZERO,
            min: Duration::ZERO,
            max: Duration::ZERO,
            std_dev: Duration::ZERO,
            p95: Duration::ZERO,
            p99: Duration::ZERO,
        }
    }
}

/// Custom serialization for Duration
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;
    
    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_nanos().serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanos = u128::deserialize(deserializer)?;
        Ok(Duration::from_nanos(nanos as u64))
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AllocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}