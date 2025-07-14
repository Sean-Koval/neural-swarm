use serde::{Deserialize, Serialize};

/// Available activation functions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// Rectified Linear Unit: f(x) = max(0, x)
    ReLU,
    /// Sigmoid: f(x) = 1 / (1 + e^(-x))
    Sigmoid,
    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,
    /// Gaussian Error Linear Unit: f(x) = x * Φ(x)
    GELU,
    /// Swish: f(x) = x * sigmoid(x)
    Swish,
    /// Mish: f(x) = x * tanh(softplus(x))
    Mish,
    /// Leaky ReLU: f(x) = max(αx, x)
    LeakyReLU(f32),
    /// Exponential Linear Unit: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
    ELU(f32),
    /// Linear activation: f(x) = x
    Linear,
}

impl ActivationFunction {
    /// Apply the activation function to a single value
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::GELU => {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                let sqrt_2_over_pi = 0.7978845608;
                let a = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + a.tanh())
            },
            Self::Swish => x * (1.0 / (1.0 + (-x).exp())),
            Self::Mish => {
                let softplus = (1.0 + x.exp()).ln();
                x * softplus.tanh()
            },
            Self::LeakyReLU(alpha) => {
                if x > 0.0 { x } else { alpha * x }
            },
            Self::ELU(alpha) => {
                if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
            },
            Self::Linear => x,
        }
    }
    
    /// Apply the derivative of the activation function
    pub fn derivative(self, x: f32) -> f32 {
        match self {
            Self::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Self::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            },
            Self::Tanh => 1.0 - x.tanh().powi(2),
            Self::GELU => {
                // Derivative approximation
                let sqrt_2_over_pi = 0.7978845608;
                let cdf = 0.5 * (1.0 + (sqrt_2_over_pi * x).tanh());
                let pdf = sqrt_2_over_pi * (1.0 - (sqrt_2_over_pi * x).tanh().powi(2));
                cdf + x * pdf
            },
            Self::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 + x * (1.0 - sigmoid))
            },
            Self::Mish => {
                let exp_x = x.exp();
                let exp_2x = exp_x * exp_x;
                let exp_3x = exp_2x * exp_x;
                let delta = 2.0 * exp_x + exp_2x + 2.0;
                let omega = 4.0 * (x + 1.0) + 4.0 * exp_2x + 6.0 * exp_x + 4.0 * x * exp_x;
                (omega / delta.powi(2)) * exp_x
            },
            Self::LeakyReLU(alpha) => {
                if x > 0.0 { 1.0 } else { alpha }
            },
            Self::ELU(alpha) => {
                if x > 0.0 { 1.0 } else { alpha * x.exp() }
            },
            Self::Linear => 1.0,
        }
    }
    
    /// Apply the activation function to a slice of values
    pub fn apply_slice(self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.apply(*inp);
        }
    }
    
    /// Apply the derivative to a slice of values
    pub fn derivative_slice(self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.derivative(*inp);
        }
    }
    
    /// Get the name of the activation function
    pub fn name(self) -> &'static str {
        match self {
            Self::ReLU => "ReLU",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            Self::GELU => "GELU",
            Self::Swish => "Swish",
            Self::Mish => "Mish",
            Self::LeakyReLU(_) => "LeakyReLU",
            Self::ELU(_) => "ELU",
            Self::Linear => "Linear",
        }
    }
    
    /// Check if the activation function is bounded
    pub fn is_bounded(self) -> bool {
        matches!(self, Self::Sigmoid | Self::Tanh)
    }
    
    /// Get the range of the activation function
    pub fn range(self) -> (f32, f32) {
        match self {
            Self::ReLU => (0.0, f32::INFINITY),
            Self::Sigmoid => (0.0, 1.0),
            Self::Tanh => (-1.0, 1.0),
            Self::GELU => (f32::NEG_INFINITY, f32::INFINITY),
            Self::Swish => (f32::NEG_INFINITY, f32::INFINITY),
            Self::Mish => (f32::NEG_INFINITY, f32::INFINITY),
            Self::LeakyReLU(_) => (f32::NEG_INFINITY, f32::INFINITY),
            Self::ELU(alpha) => (-alpha, f32::INFINITY),
            Self::Linear => (f32::NEG_INFINITY, f32::INFINITY),
        }
    }
}

impl Default for ActivationFunction {
    fn default() -> Self {
        Self::ReLU
    }
}

/// Scalar activation function implementations
pub struct ScalarActivations;

impl ScalarActivations {
    pub fn new() -> Self {
        Self
    }
    
    pub fn relu(&self, input: &[f32], output: &mut [f32]) {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = inp.max(0.0);
        }
    }
    
    pub fn sigmoid(&self, input: &[f32], output: &mut [f32]) {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = 1.0 / (1.0 + (-inp).exp());
        }
    }
    
    pub fn tanh(&self, input: &[f32], output: &mut [f32]) {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = inp.tanh();
        }
    }
    
    pub fn gelu(&self, input: &[f32], output: &mut [f32]) {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = ActivationFunction::GELU.apply(*inp);
        }
    }
    
    pub fn swish(&self, input: &[f32], output: &mut [f32]) {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = ActivationFunction::Swish.apply(*inp);
        }
    }
}

/// SIMD-optimized activation function implementations
pub struct SIMDActivations;

impl SIMDActivations {
    pub fn new() -> Self {
        Self
    }
    
    pub fn relu(&self, input: &[f32], output: &mut [f32]) {
        // For now, fallback to scalar implementation
        // In a real implementation, this would use SIMD instructions
        ScalarActivations::new().relu(input, output);
    }
    
    pub fn sigmoid(&self, input: &[f32], output: &mut [f32]) {
        // For now, fallback to scalar implementation
        // In a real implementation, this would use SIMD instructions
        ScalarActivations::new().sigmoid(input, output);
    }
    
    pub fn tanh(&self, input: &[f32], output: &mut [f32]) {
        // For now, fallback to scalar implementation
        // In a real implementation, this would use SIMD instructions
        ScalarActivations::new().tanh(input, output);
    }
    
    pub fn gelu(&self, input: &[f32], output: &mut [f32]) {
        // For now, fallback to scalar implementation
        // In a real implementation, this would use SIMD instructions
        ScalarActivations::new().gelu(input, output);
    }
    
    pub fn swish(&self, input: &[f32], output: &mut [f32]) {
        // For now, fallback to scalar implementation
        // In a real implementation, this would use SIMD instructions
        ScalarActivations::new().swish(input, output);
    }
}

impl Default for ScalarActivations {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SIMDActivations {
    fn default() -> Self {
        Self::new()
    }
}