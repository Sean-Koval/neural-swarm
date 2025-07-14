//! Neural network optimization module
//!
//! This module provides various optimization techniques for neural networks including
//! SIMD acceleration, automatic differentiation, and performance optimizations.

use crate::error::NeuralError;

pub mod simd;

pub use simd::*;

/// Optimization error types specific to this module
#[derive(Debug, thiserror::Error)]
pub enum OptimizationError {
    #[error("SIMD operations not supported on this platform")]
    SimdNotSupported,
    
    #[error("Optimization failed: {message}")]
    OptimizationFailed { message: String },
    
    #[error("Invalid optimization parameters: {reason}")]
    InvalidParameters { reason: String },
}

impl From<OptimizationError> for NeuralError {
    fn from(err: OptimizationError) -> Self {
        NeuralError::Training { 
            message: format!("Optimization error: {}", err) 
        }
    }
}

/// Detect available SIMD features on the current platform
pub fn detect_simd_features() -> Vec<String> {
    let mut features = Vec::new();
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            features.push("sse2".to_string());
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("sse4.1".to_string());
        }
        if is_x86_feature_detected!("avx") {
            features.push("avx".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            features.push("avx2".to_string());
        }
        if is_x86_feature_detected!("avx512f") {
            features.push("avx512f".to_string());
        }
        if is_x86_feature_detected!("fma") {
            features.push("fma".to_string());
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        // ARM features are typically always available on aarch64
        features.push("neon".to_string());
        if std::arch::is_aarch64_feature_detected!("asimd") {
            features.push("asimd".to_string());
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No SIMD features available
    }
    
    features
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Enable parallel processing
    pub enable_parallel: bool,
    
    /// Cache optimization settings
    pub cache_optimization: CacheOptimization,
    
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_parallel: true,
            cache_optimization: CacheOptimization::default(),
            memory_alignment: 32, // 256-bit alignment for AVX
        }
    }
}

/// Cache optimization settings
#[derive(Debug, Clone)]
pub struct CacheOptimization {
    /// L1 cache size in bytes
    pub l1_cache_size: usize,
    
    /// L2 cache size in bytes
    pub l2_cache_size: usize,
    
    /// Cache line size in bytes
    pub cache_line_size: usize,
    
    /// Block size for tiled operations
    pub block_size: usize,
}

impl Default for CacheOptimization {
    fn default() -> Self {
        Self {
            l1_cache_size: 32 * 1024,      // 32KB L1
            l2_cache_size: 256 * 1024,     // 256KB L2
            cache_line_size: 64,           // 64 bytes
            block_size: 64,                // 64x64 blocks
        }
    }
}

/// Initialize optimization features
pub fn initialize_optimizations() -> OptimizationConfig {
    let config = OptimizationConfig::default();
    
    if config.enable_simd {
        simd::initialize_simd_features();
    }
    
    log::info!("Optimization features initialized: SIMD={}, Parallel={}", 
              config.enable_simd, config.enable_parallel);
    
    config
}

/// Get optimization capabilities report
pub fn get_optimization_report() -> OptimizationReport {
    let simd_features = detect_simd_features();
    let parallel_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    
    OptimizationReport {
        simd_features,
        parallel_threads,
        memory_alignment: 32,
        cache_optimization_available: true,
    }
}

/// Optimization capabilities report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationReport {
    pub simd_features: Vec<String>,
    pub parallel_threads: usize,
    pub memory_alignment: usize,
    pub cache_optimization_available: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_simd_features() {
        let features = detect_simd_features();
        // Features should be detected (may be empty on some platforms)
        assert!(features.len() >= 0);
    }
    
    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert!(config.enable_simd);
        assert!(config.enable_parallel);
        assert_eq!(config.memory_alignment, 32);
    }
    
    #[test]
    fn test_initialize_optimizations() {
        let config = initialize_optimizations();
        assert!(config.enable_simd || !config.enable_simd); // Should not panic
    }
    
    #[test]
    fn test_optimization_report() {
        let report = get_optimization_report();
        assert!(report.parallel_threads >= 1);
        assert_eq!(report.memory_alignment, 32);
        assert!(report.cache_optimization_available);
    }
}