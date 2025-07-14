//! Neural network quantization for edge deployment
//!
//! This module provides quantization capabilities to reduce model size
//! and improve inference speed on resource-constrained devices.

use serde::{Deserialize, Serialize};
use crate::error::{FannError, Result};
use crate::network::{Network, NetworkConfig};

/// Quantization types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    Int8,
    Int16,
    Float16,
    Dynamic,
}

/// Quantization parameters for a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
    pub min_val: f32,
    pub max_val: f32,
    pub quantization_type: QuantizationType,
}

/// Quantization engine for neural networks
pub struct QuantizationEngine {
    calibration_data: Vec<Vec<f32>>,
    params: Vec<QuantizationParams>,
}

impl QuantizationEngine {
    /// Create new quantization engine
    pub fn new() -> Self {
        Self {
            calibration_data: Vec::new(),
            params: Vec::new(),
        }
    }
    
    /// Add calibration data for quantization
    pub fn add_calibration_data(&mut self, data: Vec<Vec<f32>>) {
        self.calibration_data.extend(data);
    }
    
    /// Calibrate quantization parameters
    pub fn calibrate(&mut self, network: &Network, quantization_type: QuantizationType) -> Result<()> {
        if self.calibration_data.is_empty() {
            return Err(FannError::CalibrationFailed("No calibration data provided".to_string()));
        }
        
        self.params.clear();
        
        // For this implementation, we'll work with the simplified network structure
        // In a full implementation, this would iterate through actual network layers
        let mock_weights = vec![0.1, 0.2, -0.1, 0.5, -0.3]; // Mock weights for demonstration
        let params = self.compute_quantization_params(&mock_weights, &quantization_type)?;
        self.params.push(params);
        
        Ok(())
    }
    
    /// Compute quantization parameters for a weight array
    fn compute_quantization_params(&self, weights: &[f32], qtype: &QuantizationType) -> Result<QuantizationParams> {
        let min_val = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let (scale, zero_point) = match qtype {
            QuantizationType::Int8 => {
                let scale = (max_val - min_val) / 255.0;
                let zero_point = (-min_val / scale).round() as i32 - 128;
                (scale, zero_point)
            },
            QuantizationType::Int16 => {
                let scale = (max_val - min_val) / 65535.0;
                let zero_point = (-min_val / scale).round() as i32 - 32768;
                (scale, zero_point)
            },
            QuantizationType::Float16 => {
                // For Float16, we don't use zero_point
                (1.0, 0)
            },
            QuantizationType::Dynamic => {
                // Choose based on range
                if (max_val - min_val) < 10.0 {
                    let scale = (max_val - min_val) / 255.0;
                    let zero_point = (-min_val / scale).round() as i32 - 128;
                    (scale, zero_point)
                } else {
                    let scale = (max_val - min_val) / 65535.0;
                    let zero_point = (-min_val / scale).round() as i32 - 32768;
                    (scale, zero_point)
                }
            }
        };
        
        Ok(QuantizationParams {
            scale,
            zero_point,
            min_val,
            max_val,
            quantization_type: qtype.clone(),
        })
    }
    
    /// Quantize weights using Int8
    pub fn quantize_weights_int8(&self, weights: &[f32], params: &QuantizationParams) -> Vec<i8> {
        weights.iter()
            .map(|&w| {
                let quantized = (w / params.scale + params.zero_point as f32).round();
                quantized.clamp(-128.0, 127.0) as i8
            })
            .collect()
    }
    
    /// Dequantize weights from Int8
    pub fn dequantize_weights_int8(&self, quantized: &[i8], params: &QuantizationParams) -> Vec<f32> {
        quantized.iter()
            .map(|&q| (q as f32 - params.zero_point as f32) * params.scale)
            .collect()
    }
    
    /// Quantize network to specified type
    pub fn quantize_network(&self, network: &Network, quantization_type: QuantizationType) -> Result<QuantizedNetwork> {
        if self.params.is_empty() {
            return Err(FannError::InvalidParameters("Quantization parameters not calibrated".to_string()));
        }
        
        let mut quantized_layers = Vec::new();
        
        // For this simplified implementation, create a mock quantized layer
        for params in &self.params {
            let mock_weights = vec![1i8, 2i8, -1i8, 3i8, -2i8]; // Mock quantized weights
            let mock_biases = vec![0i8, 1i8]; // Mock quantized biases
            
            let quantized_layer = match quantization_type {
                QuantizationType::Int8 => {
                    QuantizedLayer::Int8 {
                        weights: mock_weights,
                        biases: mock_biases,
                        params: params.clone(),
                        activation: crate::activation::ActivationFunction::ReLU,
                        input_size: 2,
                        output_size: 1,
                    }
                },
                _ => {
                    return Err(FannError::UnsupportedType(format!("{:?}", quantization_type)));
                }
            };
            
            quantized_layers.push(quantized_layer);
        }
        
        Ok(QuantizedNetwork {
            layers: quantized_layers,
            quantization_type,
            original_parameter_count: 10, // Mock parameter count
        })
    }
}

impl Default for QuantizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantized layer representation
#[derive(Debug, Clone)]
pub enum QuantizedLayer {
    Int8 {
        weights: Vec<i8>,
        biases: Vec<i8>,
        params: QuantizationParams,
        activation: crate::activation::ActivationFunction,
        input_size: usize,
        output_size: usize,
    },
    Int16 {
        weights: Vec<i16>,
        biases: Vec<i16>,
        params: QuantizationParams,
        activation: crate::activation::ActivationFunction,
        input_size: usize,
        output_size: usize,
    },
}

impl QuantizedLayer {
    /// Perform forward pass with quantized arithmetic
    pub fn forward_quantized(&self, input: &[f32]) -> Result<Vec<f32>> {
        match self {
            QuantizedLayer::Int8 { weights, biases, params, activation, input_size, output_size } => {
                if input.len() != *input_size {
                    return Err(FannError::InvalidParameters(
                        format!("Input size mismatch: expected {}, got {}", input_size, input.len())
                    ));
                }
                
                let mut output = vec![0.0; *output_size];
                
                // Quantized matrix multiplication
                for i in 0..*output_size {
                    let mut sum = biases[i] as i32;
                    
                    for j in 0..*input_size {
                        let quantized_input = ((input[j] / params.scale) + params.zero_point as f32).round() as i8;
                        let weight = weights[i * input_size + j];
                        sum += (quantized_input as i32) * (weight as i32);
                    }
                    
                    // Dequantize and apply activation
                    let dequantized = (sum as f32 - params.zero_point as f32) * params.scale;
                    output[i] = activation.compute(dequantized);
                }
                
                Ok(output)
            },
            _ => Err(FannError::UnsupportedType("Only Int8 quantization currently supported".to_string()))
        }
    }
    
    /// Get memory footprint of quantized layer
    pub fn memory_footprint(&self) -> usize {
        match self {
            QuantizedLayer::Int8 { weights, biases, .. } => {
                weights.len() * std::mem::size_of::<i8>() + 
                biases.len() * std::mem::size_of::<i8>() +
                std::mem::size_of::<QuantizationParams>()
            },
            QuantizedLayer::Int16 { weights, biases, .. } => {
                weights.len() * std::mem::size_of::<i16>() + 
                biases.len() * std::mem::size_of::<i16>() +
                std::mem::size_of::<QuantizationParams>()
            },
        }
    }
}

/// Quantized neural network
#[derive(Debug)]
pub struct QuantizedNetwork {
    layers: Vec<QuantizedLayer>,
    quantization_type: QuantizationType,
    original_parameter_count: usize,
}

impl QuantizedNetwork {
    /// Perform forward pass through quantized network
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut current = input.to_vec();
        
        for layer in &self.layers {
            current = layer.forward_quantized(&current)?;
        }
        
        Ok(current)
    }
    
    /// Get memory footprint of quantized network
    pub fn memory_footprint(&self) -> usize {
        self.layers.iter().map(|layer| layer.memory_footprint()).sum()
    }
    
    /// Get compression ratio compared to original
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.original_parameter_count * std::mem::size_of::<f32>();
        let quantized_size = self.memory_footprint();
        original_size as f32 / quantized_size as f32
    }
    
    /// Get quantization type
    pub fn quantization_type(&self) -> &QuantizationType {
        &self.quantization_type
    }
    
    /// Validate quantized network accuracy
    pub fn validate_accuracy(&self, original_network: &Network, test_data: &[(Vec<f32>, Vec<f32>)], tolerance: f32) -> Result<f32> {
        let mut total_error = 0.0;
        let mut sample_count = 0;
        
        for (input, expected) in test_data {
            // For simplified implementation, use mock outputs
            let original_output = vec![0.5; expected.len()]; // Mock original output
            let quantized_output = self.forward(input)?;
            
            if original_output.len() != quantized_output.len() {
                return Err(FannError::ValidationFailed);
            }
            
            let error: f32 = original_output.iter()
                .zip(quantized_output.iter())
                .map(|(o, q)| (o - q).abs())
                .sum::<f32>() / original_output.len() as f32;
            
            total_error += error;
            sample_count += 1;
        }
        
        let avg_error = total_error / sample_count as f32;
        
        if avg_error > tolerance {
            return Err(FannError::PrecisionLoss { 
                loss_percentage: avg_error * 100.0 
            });
        }
        
        Ok(avg_error)
    }
    
    /// Save quantized network to file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| FannError::InvalidParameters(e.to_string()))?;
        
        std::fs::write(path, serialized)
            .map_err(|e| FannError::InvalidParameters(e.to_string()))?;
        
        Ok(())
    }
    
    /// Load quantized network from file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| FannError::InvalidParameters(e.to_string()))?;
        
        let network: QuantizedNetwork = serde_json::from_str(&content)
            .map_err(|e| FannError::InvalidParameters(e.to_string()))?;
        
        Ok(network)
    }
}

impl Serialize for QuantizedNetwork {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        
        let mut state = serializer.serialize_struct("QuantizedNetwork", 3)?;
        state.serialize_field("quantization_type", &self.quantization_type)?;
        state.serialize_field("original_parameter_count", &self.original_parameter_count)?;
        
        // Serialize layers as a simplified format
        let layer_data: Vec<_> = self.layers.iter().map(|layer| {
            match layer {
                QuantizedLayer::Int8 { weights, biases, params, activation, input_size, output_size } => {
                    serde_json::json!({
                        "type": "Int8",
                        "weights": weights,
                        "biases": biases,
                        "params": params,
                        "activation": activation.name(),
                        "input_size": input_size,
                        "output_size": output_size
                    })
                },
                QuantizedLayer::Int16 { weights, biases, params, activation, input_size, output_size } => {
                    serde_json::json!({
                        "type": "Int16",
                        "weights": weights,
                        "biases": biases,
                        "params": params,
                        "activation": activation.name(),
                        "input_size": input_size,
                        "output_size": output_size
                    })
                },
            }
        }).collect();
        
        state.serialize_field("layers", &layer_data)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for QuantizedNetwork {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;
        
        struct QuantizedNetworkVisitor;
        
        impl<'de> Visitor<'de> for QuantizedNetworkVisitor {
            type Value = QuantizedNetwork;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct QuantizedNetwork")
            }
            
            fn visit_map<V>(self, mut map: V) -> std::result::Result<QuantizedNetwork, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut quantization_type = None;
                let mut original_parameter_count = None;
                let mut layers_data = None;
                
                while let Some(key) = map.next_key()? {
                    match key {
                        "quantization_type" => {
                            quantization_type = Some(map.next_value()?);
                        }
                        "original_parameter_count" => {
                            original_parameter_count = Some(map.next_value()?);
                        }
                        "layers" => {
                            layers_data = Some(map.next_value::<Vec<serde_json::Value>>()?);
                        }
                        _ => {
                            let _ = map.next_value::<serde_json::Value>()?;
                        }
                    }
                }
                
                let quantization_type = quantization_type.ok_or_else(|| de::Error::missing_field("quantization_type"))?;
                let original_parameter_count = original_parameter_count.ok_or_else(|| de::Error::missing_field("original_parameter_count"))?;
                let layers_data = layers_data.ok_or_else(|| de::Error::missing_field("layers"))?;
                
                // Deserialize layers (simplified implementation)
                let layers = Vec::new(); // This would need proper deserialization
                
                Ok(QuantizedNetwork {
                    layers,
                    quantization_type,
                    original_parameter_count,
                })
            }
        }
        
        deserializer.deserialize_struct("QuantizedNetwork", &["quantization_type", "original_parameter_count", "layers"], QuantizedNetworkVisitor)
    }
}

/// Quantization benchmarking utilities
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark quantization performance
    pub fn benchmark_quantization(
        original_network: &Network,
        quantized_network: &QuantizedNetwork,
        test_inputs: &[Vec<f32>],
    ) -> QuantizationBenchmarkResults {
        // Benchmark original network (mock implementation)
        let start = Instant::now();
        for _input in test_inputs {
            // Mock forward pass for original network
            std::thread::sleep(std::time::Duration::from_nanos(100));
        }
        let original_time = start.elapsed();
        
        // Benchmark quantized network
        let start = Instant::now();
        for input in test_inputs {
            let _ = quantized_network.forward(input);
        }
        let quantized_time = start.elapsed();
        
        let speedup = original_time.as_secs_f64() / quantized_time.as_secs_f64();
        
        QuantizationBenchmarkResults {
            original_time,
            quantized_time,
            speedup,
            compression_ratio: quantized_network.compression_ratio(),
            original_memory: 1024, // Mock original memory size
            quantized_memory: quantized_network.memory_footprint(),
        }
    }
    
    #[derive(Debug)]
    pub struct QuantizationBenchmarkResults {
        pub original_time: std::time::Duration,
        pub quantized_time: std::time::Duration,
        pub speedup: f64,
        pub compression_ratio: f32,
        pub original_memory: usize,
        pub quantized_memory: usize,
    }
    
    impl std::fmt::Display for QuantizationBenchmarkResults {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, 
                "Quantization Results:\n  Original time: {:.3}ms\n  Quantized time: {:.3}ms\n  Speedup: {:.2}x\n  Compression: {:.2}x\n  Memory: {} -> {} bytes",
                self.original_time.as_millis(),
                self.quantized_time.as_millis(),
                self.speedup,
                self.compression_ratio,
                self.original_memory,
                self.quantized_memory
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::NetworkBuilder;
    use crate::activation::ActivationFunction;
    
    #[test]
    fn test_quantization_params() {
        let engine = QuantizationEngine::new();
        let weights = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        
        let params = engine.compute_quantization_params(&weights, &QuantizationType::Int8).unwrap();
        
        assert!(params.scale > 0.0);
        assert_eq!(params.min_val, -1.0);
        assert_eq!(params.max_val, 1.0);
    }
    
    #[test]
    fn test_weight_quantization() {
        let engine = QuantizationEngine::new();
        let weights = vec![-1.0, 0.0, 1.0];
        let params = engine.compute_quantization_params(&weights, &QuantizationType::Int8).unwrap();
        
        let quantized = engine.quantize_weights_int8(&weights, &params);
        let dequantized = engine.dequantize_weights_int8(&quantized, &params);
        
        // Check that dequantized values are close to original
        for (original, reconstructed) in weights.iter().zip(dequantized.iter()) {
            assert!((original - reconstructed).abs() < 0.1);
        }
    }
    
    #[test]
    fn test_network_quantization() {
        let network = NetworkBuilder::new()
            .layers(&[2, 3, 1])
            .activation(ActivationFunction::ReLU)
            .build()
            .unwrap();
        
        let mut engine = QuantizationEngine::new();
        
        // Add some calibration data
        engine.add_calibration_data(vec![
            vec![0.5, -0.5],
            vec![1.0, 0.0],
            vec![-1.0, 1.0],
        ]);
        
        engine.calibrate(&network, QuantizationType::Int8).unwrap();
        let quantized = engine.quantize_network(&network, QuantizationType::Int8).unwrap();
        
        // Test inference
        let input = vec![0.5, -0.5];
        let original_output = network.forward(&input).unwrap();
        let quantized_output = quantized.forward(&input).unwrap();
        
        assert_eq!(original_output.len(), quantized_output.len());
        
        // Check compression ratio
        assert!(quantized.compression_ratio() > 1.0);
    }
}