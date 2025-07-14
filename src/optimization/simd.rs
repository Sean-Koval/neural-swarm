//! SIMD optimization implementations
//!
//! This module provides SIMD-optimized implementations for neural network operations,
//! with automatic fallback to scalar implementations when SIMD is not available.

use crate::error::{FannError, Result};
use super::OptimizationError;

/// Initialize SIMD features and capabilities
pub fn initialize_simd_features() {
    // Runtime SIMD feature detection and initialization
    log::info!("Available SIMD features: {:?}", get_simd_capabilities());
}

/// Get available SIMD capabilities
pub fn get_simd_capabilities() -> Vec<String> {
    crate::optimization::detect_simd_features()
}

/// SIMD-optimized ReLU activation
pub fn relu_simd(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { relu_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { relu_sse(input, output) };
            return;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        relu_neon(input, output);
        return;
    }
    
    // Scalar fallback
    relu_scalar(input, output);
}

/// SIMD-optimized sigmoid activation
pub fn sigmoid_simd(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sigmoid_avx2(input, output) };
            return;
        }
    }
    
    // Scalar fallback
    sigmoid_scalar(input, output);
}

/// SIMD-optimized tanh activation
pub fn tanh_simd(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { tanh_avx2(input, output) };
            return;
        }
    }
    
    // Scalar fallback
    tanh_scalar(input, output);
}

// Scalar implementations
fn relu_scalar(input: &[f32], output: &mut [f32]) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.max(0.0);
    }
}

fn sigmoid_scalar(input: &[f32], output: &mut [f32]) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = 1.0 / (1.0 + (-x).exp());
    }
}

fn tanh_scalar(input: &[f32], output: &mut [f32]) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.tanh();
    }
}

// x86_64 SIMD implementations
#[cfg(target_arch = "x86_64")]
mod x86_simd {
    use super::*;
    use std::arch::x86_64::*;
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !7; // Round down to nearest multiple of 8
        let zero = _mm256_setzero_ps();
        
        for i in (0..simd_len).step_by(8) {
            let x = _mm256_loadu_ps(&input[i]);
            let result = _mm256_max_ps(x, zero);
            _mm256_storeu_ps(&mut output[i], result);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].max(0.0);
        }
    }
    
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn relu_sse(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !3; // Round down to nearest multiple of 4
        let zero = _mm_setzero_ps();
        
        for i in (0..simd_len).step_by(4) {
            let x = _mm_loadu_ps(&input[i]);
            let result = _mm_max_ps(x, zero);
            _mm_storeu_ps(&mut output[i], result);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].max(0.0);
        }
    }
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn sigmoid_avx2(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !7;
        let one = _mm256_set1_ps(1.0);
        
        for i in (0..simd_len).step_by(8) {
            let x = _mm256_loadu_ps(&input[i]);
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            
            // Fast exp approximation using polynomial
            let exp_neg_x = exp_ps_avx2(neg_x);
            let denominator = _mm256_add_ps(one, exp_neg_x);
            let result = _mm256_div_ps(one, denominator);
            
            _mm256_storeu_ps(&mut output[i], result);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
    }
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn tanh_avx2(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !7;
        
        for i in (0..simd_len).step_by(8) {
            let x = _mm256_loadu_ps(&input[i]);
            
            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            let two_x = _mm256_add_ps(x, x);
            let exp_2x = exp_ps_avx2(two_x);
            let one = _mm256_set1_ps(1.0);
            
            let numerator = _mm256_sub_ps(exp_2x, one);
            let denominator = _mm256_add_ps(exp_2x, one);
            let result = _mm256_div_ps(numerator, denominator);
            
            _mm256_storeu_ps(&mut output[i], result);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].tanh();
        }
    }
    
    /// Fast exponential approximation using AVX2
    #[target_feature(enable = "avx2")]
    unsafe fn exp_ps_avx2(x: __m256) -> __m256 {
        // Clamp x to avoid overflow
        let min_x = _mm256_set1_ps(-87.0);
        let max_x = _mm256_set1_ps(87.0);
        let x_clamped = _mm256_max_ps(_mm256_min_ps(x, max_x), min_x);
        
        // Use polynomial approximation for speed
        // This is a simplified approximation - for better accuracy, use more terms
        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.16666666);
        let c4 = _mm256_set1_ps(0.041666666);
        
        let x2 = _mm256_mul_ps(x_clamped, x_clamped);
        let x3 = _mm256_mul_ps(x2, x_clamped);
        let x4 = _mm256_mul_ps(x3, x_clamped);
        
        let term1 = _mm256_mul_ps(c1, x_clamped);
        let term2 = _mm256_mul_ps(c2, x2);
        let term3 = _mm256_mul_ps(c3, x3);
        let term4 = _mm256_mul_ps(c4, x4);
        
        let result = _mm256_add_ps(c0, 
                        _mm256_add_ps(term1, 
                            _mm256_add_ps(term2, 
                                _mm256_add_ps(term3, term4))));
        
        result
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86_simd::*;

// ARM NEON implementations
#[cfg(target_arch = "aarch64")]
mod arm_simd {
    use super::*;
    use std::arch::aarch64::*;
    
    pub fn relu_neon(input: &[f32], output: &mut [f32]) {
        unsafe {
            let len = input.len();
            let simd_len = len & !3; // Round down to nearest multiple of 4
            let zero = vdupq_n_f32(0.0);
            
            for i in (0..simd_len).step_by(4) {
                let x = vld1q_f32(&input[i]);
                let result = vmaxq_f32(x, zero);
                vst1q_f32(&mut output[i], result);
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = input[i].max(0.0);
            }
        }
    }
    
    pub fn sigmoid_neon(input: &[f32], output: &mut [f32]) {
        unsafe {
            let len = input.len();
            let simd_len = len & !3;
            let one = vdupq_n_f32(1.0);
            
            for i in (0..simd_len).step_by(4) {
                let x = vld1q_f32(&input[i]);
                let neg_x = vnegq_f32(x);
                
                // Fast exp approximation
                let exp_neg_x = exp_neon(neg_x);
                let denominator = vaddq_f32(one, exp_neg_x);
                let result = vdivq_f32(one, denominator);
                
                vst1q_f32(&mut output[i], result);
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = 1.0 / (1.0 + (-input[i]).exp());
            }
        }
    }
    
    unsafe fn exp_neon(x: float32x4_t) -> float32x4_t {
        // Simplified exponential approximation for NEON
        let c0 = vdupq_n_f32(1.0);
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(0.5);
        let c3 = vdupq_n_f32(0.16666666);
        
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);
        
        let term1 = vmulq_f32(c1, x);
        let term2 = vmulq_f32(c2, x2);
        let term3 = vmulq_f32(c3, x3);
        
        vaddq_f32(c0, vaddq_f32(term1, vaddq_f32(term2, term3)))
    }
}

#[cfg(target_arch = "aarch64")]
pub use arm_simd::*;

/// SIMD-optimized matrix operations
pub struct SIMDOperations;

impl SIMDOperations {
    /// Element-wise multiplication with SIMD
    pub fn elementwise_multiply(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(OptimizationError::SimdNotSupported.into());
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::elementwise_multiply_avx2(a, b, result) };
                return Ok(());
            }
        }
        
        // Scalar fallback
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
        Ok(())
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn elementwise_multiply_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len();
        let simd_len = len & !7;
        
        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(&a[i]);
            let b_vec = _mm256_loadu_ps(&b[i]);
            let product = _mm256_mul_ps(a_vec, b_vec);
            _mm256_storeu_ps(&mut result[i], product);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }
    }
    
    /// Reduce sum with SIMD
    pub fn reduce_sum(input: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::reduce_sum_avx2(input) };
            }
        }
        
        // Scalar fallback
        input.iter().sum()
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn reduce_sum_avx2(input: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = input.len();
        let simd_len = len & !7;
        let mut sum_vec = _mm256_setzero_ps();
        
        for i in (0..simd_len).step_by(8) {
            let x = _mm256_loadu_ps(&input[i]);
            sum_vec = _mm256_add_ps(sum_vec, x);
        }
        
        // Horizontal sum
        let sum_array: [f32; 8] = std::mem::transmute(sum_vec);
        let mut scalar_sum = sum_array.iter().sum::<f32>();
        
        // Handle remaining elements
        for i in simd_len..len {
            scalar_sum += input[i];
        }
        
        scalar_sum
    }
    
    /// Scale vector by scalar with SIMD
    pub fn scale_vector(input: &[f32], scale: f32, output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(OptimizationError::SimdNotSupported.into());
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::scale_vector_avx2(input, scale, output) };
                return Ok(());
            }
        }
        
        // Scalar fallback
        for i in 0..input.len() {
            output[i] = input[i] * scale;
        }
        Ok(())
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_vector_avx2(input: &[f32], scale: f32, output: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = input.len();
        let simd_len = len & !7;
        let scale_vec = _mm256_set1_ps(scale);
        
        for i in (0..simd_len).step_by(8) {
            let x = _mm256_loadu_ps(&input[i]);
            let result = _mm256_mul_ps(x, scale_vec);
            _mm256_storeu_ps(&mut output[i], result);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i] * scale;
        }
    }
}

/// SIMD benchmarking utilities
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    pub struct SIMDBenchmark {
        pub operation: String,
        pub size: usize,
        pub iterations: usize,
    }
    
    impl SIMDBenchmark {
        pub fn new(operation: &str, size: usize, iterations: usize) -> Self {
            Self {
                operation: operation.to_string(),
                size,
                iterations,
            }
        }
        
        pub fn benchmark_relu(&self) -> (f64, f64) {
            let input: Vec<f32> = (0..self.size).map(|i| (i as f32 - self.size as f32 / 2.0) * 0.1).collect();
            let mut output = vec![0.0; self.size];
            
            // Scalar benchmark
            let start = Instant::now();
            for _ in 0..self.iterations {
                relu_scalar(&input, &mut output);
            }
            let scalar_time = start.elapsed().as_secs_f64();
            
            // SIMD benchmark
            output.fill(0.0);
            let start = Instant::now();
            for _ in 0..self.iterations {
                relu_simd(&input, &mut output);
            }
            let simd_time = start.elapsed().as_secs_f64();
            
            (scalar_time, simd_time)
        }
        
        pub fn benchmark_sigmoid(&self) -> (f64, f64) {
            let input: Vec<f32> = (0..self.size).map(|i| (i as f32 - self.size as f32 / 2.0) * 0.1).collect();
            let mut output = vec![0.0; self.size];
            
            // Scalar benchmark
            let start = Instant::now();
            for _ in 0..self.iterations {
                sigmoid_scalar(&input, &mut output);
            }
            let scalar_time = start.elapsed().as_secs_f64();
            
            // SIMD benchmark
            output.fill(0.0);
            let start = Instant::now();
            for _ in 0..self.iterations {
                sigmoid_simd(&input, &mut output);
            }
            let simd_time = start.elapsed().as_secs_f64();
            
            (scalar_time, simd_time)
        }
        
        pub fn run_all_benchmarks(&self) -> Vec<(String, f64)> {
            let mut results = Vec::new();
            
            let (relu_scalar, relu_simd) = self.benchmark_relu();
            results.push(("ReLU Speedup".to_string(), relu_scalar / relu_simd));
            
            let (sigmoid_scalar, sigmoid_simd) = self.benchmark_sigmoid();
            results.push(("Sigmoid Speedup".to_string(), sigmoid_scalar / sigmoid_simd));
            
            results
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_relu_simd() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        
        relu_simd(&input, &mut output);
        
        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[4], 2.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_sigmoid_simd() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];
        
        sigmoid_simd(&input, &mut output);
        
        assert_relative_eq!(output[0], 0.5, epsilon = 1e-6);
        assert!(output[1] > 0.7 && output[1] < 0.8); // sigmoid(1) ≈ 0.731
        assert!(output[2] > 0.2 && output[2] < 0.3); // sigmoid(-1) ≈ 0.269
    }
    
    #[test]
    fn test_elementwise_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];
        
        SIMDOperations::elementwise_multiply(&a, &b, &mut result).unwrap();
        
        assert_relative_eq!(result[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 6.0, epsilon = 1e-6);
        assert_relative_eq!(result[2], 12.0, epsilon = 1e-6);
        assert_relative_eq!(result[3], 20.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_reduce_sum() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = SIMDOperations::reduce_sum(&input);
        
        assert_relative_eq!(sum, 15.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_scale_vector() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        
        SIMDOperations::scale_vector(&input, 2.5, &mut output).unwrap();
        
        assert_relative_eq!(output[0], 2.5, epsilon = 1e-6);
        assert_relative_eq!(output[1], 5.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 7.5, epsilon = 1e-6);
        assert_relative_eq!(output[3], 10.0, epsilon = 1e-6);
    }
}