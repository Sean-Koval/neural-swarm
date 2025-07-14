//! Performance optimization utilities
//!
//! This module provides various optimization strategies including SIMD operations,
//! parallel processing, and memory layout optimizations.

pub mod simd;
pub mod parallel;
pub mod memory;

use serde::{Deserialize, Serialize};
use crate::error::{OptimizationError, Result};

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub use_simd: bool,
    pub use_parallel: bool,
    pub memory_optimization: bool,
    pub cache_optimization: bool,
    pub simd_features: Vec<String>,
    pub thread_count: Option<usize>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            use_simd: true,
            use_parallel: true,
            memory_optimization: true,
            cache_optimization: true,
            simd_features: detect_simd_features(),
            thread_count: None,
        }
    }
}

/// Detect available SIMD features
pub fn detect_simd_features() -> Vec<String> {
    let mut features = Vec::new();
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            features.push("sse".to_string());
        }
        if is_x86_feature_detected!("sse2") {
            features.push("sse2".to_string());
        }
        if is_x86_feature_detected!("sse3") {
            features.push("sse3".to_string());
        }
        if is_x86_feature_detected!("ssse3") {
            features.push("ssse3".to_string());
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("sse4.1".to_string());
        }
        if is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2".to_string());
        }
        if is_x86_feature_detected!("avx") {
            features.push("avx".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            features.push("avx2".to_string());
        }
        if is_x86_feature_detected!("fma") {
            features.push("fma".to_string());
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        features.push("neon".to_string());
    }
    
    features
}

/// SIMD matrix operations
pub struct SIMDMatrixOps;

impl SIMDMatrixOps {
    /// Matrix multiplication with automatic SIMD dispatch
    pub fn matrix_multiply(
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize
    ) -> Result<()> {
        #[cfg(feature = "simd")]
        {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe { Self::matrix_multiply_avx2(a, b, c, m, n, k) };
                    return Ok(());
                } else if is_x86_feature_detected!("sse4.1") {
                    unsafe { Self::matrix_multiply_sse(a, b, c, m, n, k) };
                    return Ok(());
                }
            }
            
            #[cfg(target_arch = "aarch64")]
            {
                Self::matrix_multiply_neon(a, b, c, m, n, k);
                return Ok(());
            }
        }
        
        // Fallback to scalar implementation
        Self::matrix_multiply_scalar(a, b, c, m, n, k);
        Ok(())
    }
    
    /// Scalar matrix multiplication
    fn matrix_multiply_scalar(
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
    
    /// AVX2-optimized matrix multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn matrix_multiply_avx2(
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize
    ) {
        use std::arch::x86_64::*;
        
        for i in 0..m {
            for j in (0..n).step_by(8) {
                let mut sum = _mm256_setzero_ps();
                
                for l in 0..k {
                    let a_val = _mm256_set1_ps(a[i * k + l]);
                    let b_vals = if j + 8 <= n {
                        _mm256_loadu_ps(&b[l * n + j])
                    } else {
                        // Handle edge case where n is not divisible by 8
                        let mut temp = [0.0f32; 8];
                        for idx in 0..8.min(n - j) {
                            temp[idx] = b[l * n + j + idx];
                        }
                        _mm256_loadu_ps(temp.as_ptr())
                    };
                    sum = _mm256_fmadd_ps(a_val, b_vals, sum);
                }
                
                // Store results
                if j + 8 <= n {
                    _mm256_storeu_ps(&mut c[i * n + j], sum);
                } else {
                    // Handle edge case
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
                    for idx in 0..8.min(n - j) {
                        c[i * n + j + idx] = temp[idx];
                    }
                }
            }
        }
    }
    
    /// SSE4.1-optimized matrix multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn matrix_multiply_sse(
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize
    ) {
        use std::arch::x86_64::*;
        
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let mut sum = _mm_setzero_ps();
                
                for l in 0..k {
                    let a_val = _mm_set1_ps(a[i * k + l]);
                    let b_vals = if j + 4 <= n {
                        _mm_loadu_ps(&b[l * n + j])
                    } else {
                        let mut temp = [0.0f32; 4];
                        for idx in 0..4.min(n - j) {
                            temp[idx] = b[l * n + j + idx];
                        }
                        _mm_loadu_ps(temp.as_ptr())
                    };
                    sum = _mm_add_ps(sum, _mm_mul_ps(a_val, b_vals));
                }
                
                // Store results
                if j + 4 <= n {
                    _mm_storeu_ps(&mut c[i * n + j], sum);
                } else {
                    let mut temp = [0.0f32; 4];
                    _mm_storeu_ps(temp.as_mut_ptr(), sum);
                    for idx in 0..4.min(n - j) {
                        c[i * n + j + idx] = temp[idx];
                    }
                }
            }
        }
    }
    
    /// NEON-optimized matrix multiplication for ARM
    #[cfg(target_arch = "aarch64")]
    fn matrix_multiply_neon(
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize
    ) {
        use std::arch::aarch64::*;
        
        unsafe {
            for i in 0..m {
                for j in (0..n).step_by(4) {
                    let mut sum = vdupq_n_f32(0.0);
                    
                    for l in 0..k {
                        let a_val = vdupq_n_f32(a[i * k + l]);
                        let b_vals = if j + 4 <= n {
                            vld1q_f32(&b[l * n + j])
                        } else {
                            let mut temp = [0.0f32; 4];
                            for idx in 0..4.min(n - j) {
                                temp[idx] = b[l * n + j + idx];
                            }
                            vld1q_f32(temp.as_ptr())
                        };
                        sum = vfmaq_f32(sum, a_val, b_vals);
                    }
                    
                    // Store results
                    if j + 4 <= n {
                        vst1q_f32(&mut c[i * n + j], sum);
                    } else {
                        let mut temp = [0.0f32; 4];
                        vst1q_f32(temp.as_mut_ptr(), sum);
                        for idx in 0..4.min(n - j) {
                            c[i * n + j + idx] = temp[idx];
                        }
                    }
                }
            }
        }
    }
    
    /// Vector addition with SIMD
    pub fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(OptimizationError::SimdNotSupported.into());
        }
        
        #[cfg(feature = "simd")]
        {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe { Self::vector_add_avx2(a, b, result) };
                    return Ok(());
                }
            }
        }
        
        // Scalar fallback
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
        Ok(())
    }
    
    /// AVX2-optimized vector addition
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len();
        let simd_len = len & !7; // Round down to nearest multiple of 8
        
        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(&a[i]);
            let b_vec = _mm256_loadu_ps(&b[i]);
            let sum = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(&mut result[i], sum);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }
    }
    
    /// Dot product with SIMD optimization
    pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(OptimizationError::SimdNotSupported.into());
        }
        
        #[cfg(feature = "simd")]
        {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    return Ok(unsafe { Self::dot_product_avx2(a, b) });
                }
            }
        }
        
        // Scalar fallback
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }
    
    /// AVX2-optimized dot product
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = a.len();
        let simd_len = len & !7;
        let mut sum_vec = _mm256_setzero_ps();
        
        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(&a[i]);
            let b_vec = _mm256_loadu_ps(&b[i]);
            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
        }
        
        // Horizontal sum of the vector
        let sum_array: [f32; 8] = std::mem::transmute(sum_vec);
        let mut scalar_sum = sum_array.iter().sum::<f32>();
        
        // Handle remaining elements
        for i in simd_len..len {
            scalar_sum += a[i] * b[i];
        }
        
        scalar_sum
    }
}

/// Performance measurement utilities
pub struct PerformanceMeasurement {
    start_time: std::time::Instant,
    operation_name: String,
}

impl PerformanceMeasurement {
    pub fn start(operation_name: &str) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            operation_name: operation_name.to_string(),
        }
    }
    
    pub fn end(self) -> PerformanceResult {
        let duration = self.start_time.elapsed();
        PerformanceResult {
            operation: self.operation_name,
            duration,
            throughput: None,
        }
    }
    
    pub fn end_with_throughput(self, operations_count: usize) -> PerformanceResult {
        let duration = self.start_time.elapsed();
        let throughput = operations_count as f64 / duration.as_secs_f64();
        
        PerformanceResult {
            operation: self.operation_name,
            duration,
            throughput: Some(throughput),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceResult {
    pub operation: String,
    pub duration: std::time::Duration,
    pub throughput: Option<f64>,
}

impl std::fmt::Display for PerformanceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(throughput) = self.throughput {
            write!(f, "{}: {:.3}ms ({:.2} ops/sec)", 
                   self.operation, 
                   self.duration.as_millis(), 
                   throughput)
        } else {
            write!(f, "{}: {:.3}ms", 
                   self.operation, 
                   self.duration.as_millis())
        }
    }
}

/// Optimization benchmarks
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    pub fn benchmark_matrix_multiply(size: usize, iterations: usize) -> Vec<PerformanceResult> {
        let mut results = Vec::new();
        
        // Generate test data
        let a: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size * size).map(|i| (i * 2) as f32).collect();
        let mut c = vec![0.0; size * size];
        
        // Benchmark scalar implementation
        let start = Instant::now();
        for _ in 0..iterations {
            SIMDMatrixOps::matrix_multiply_scalar(&a, &b, &mut c, size, size, size);
        }
        let scalar_time = start.elapsed();
        
        results.push(PerformanceResult {
            operation: "Matrix Multiply (Scalar)".to_string(),
            duration: scalar_time,
            throughput: Some(iterations as f64 / scalar_time.as_secs_f64()),
        });
        
        // Benchmark SIMD implementation
        c.fill(0.0);
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = SIMDMatrixOps::matrix_multiply(&a, &b, &mut c, size, size, size);
        }
        let simd_time = start.elapsed();
        
        results.push(PerformanceResult {
            operation: "Matrix Multiply (SIMD)".to_string(),
            duration: simd_time,
            throughput: Some(iterations as f64 / simd_time.as_secs_f64()),
        });
        
        // Calculate speedup
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        results.push(PerformanceResult {
            operation: format!("SIMD Speedup: {:.2}x", speedup),
            duration: std::time::Duration::ZERO,
            throughput: None,
        });
        
        results
    }
    
    pub fn benchmark_vector_operations(size: usize, iterations: usize) -> Vec<PerformanceResult> {
        let mut results = Vec::new();
        
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        let mut result = vec![0.0; size];
        
        // Vector addition benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = SIMDMatrixOps::vector_add(&a, &b, &mut result);
        }
        let add_time = start.elapsed();
        
        results.push(PerformanceResult {
            operation: "Vector Addition".to_string(),
            duration: add_time,
            throughput: Some((iterations * size) as f64 / add_time.as_secs_f64()),
        });
        
        // Dot product benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = SIMDMatrixOps::dot_product(&a, &b);
        }
        let dot_time = start.elapsed();
        
        results.push(PerformanceResult {
            operation: "Dot Product".to_string(),
            duration: dot_time,
            throughput: Some((iterations * size) as f64 / dot_time.as_secs_f64()),
        });
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_matrix_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![2.0, 0.0, 1.0, 2.0]; // 2x2 matrix
        let mut c = vec![0.0; 4];
        
        SIMDMatrixOps::matrix_multiply(&a, &b, &mut c, 2, 2, 2).unwrap();
        
        // Expected result: [[4, 4], [10, 8]]
        assert_relative_eq!(c[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(c[1], 4.0, epsilon = 1e-6);
        assert_relative_eq!(c[2], 10.0, epsilon = 1e-6);
        assert_relative_eq!(c[3], 8.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_vector_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];
        
        SIMDMatrixOps::vector_add(&a, &b, &mut result).unwrap();
        
        assert_relative_eq!(result[0], 6.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 8.0, epsilon = 1e-6);
        assert_relative_eq!(result[2], 10.0, epsilon = 1e-6);
        assert_relative_eq!(result[3], 12.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let result = SIMDMatrixOps::dot_product(&a, &b).unwrap();
        
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_relative_eq!(result, 32.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_simd_feature_detection() {
        let features = detect_simd_features();
        // Just ensure it returns without panicking
        println!("Available SIMD features: {:?}", features);
    }
}