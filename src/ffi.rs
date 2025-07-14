//! Foreign Function Interface (FFI) for Python bindings
//!
//! This module provides C-compatible functions for Python bindings using PyO3.

use crate::{
    activation::ActivationType,
    network::{LayerConfig, NeuralNetwork},
    training::{TrainingAlgorithm, TrainingData, TrainingParams, Trainer},
    inference::{ExecutionMode, InferenceConfig, InferenceEngine},
    utils::{DataNormalizer, NormalizationMethod},
    NeuralFloat, Result,
};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// FFI-compatible neural network wrapper
#[cfg_attr(feature = "python", pyclass)]
pub struct PyNeuralNetwork {
    network: NeuralNetwork,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyNeuralNetwork {
    #[new]
    fn new(layer_sizes: Vec<usize>, activation_names: Vec<String>) -> PyResult<Self> {
        if layer_sizes.len() != activation_names.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Layer sizes and activation names must have the same length"
            ));
        }

        let mut layer_configs = Vec::new();
        for (size, activation_name) in layer_sizes.iter().zip(activation_names.iter()) {
            let activation = match activation_name.as_str() {
                "linear" => ActivationType::Linear,
                "sigmoid" => ActivationType::Sigmoid,
                "tanh" => ActivationType::Tanh,
                "relu" => ActivationType::ReLU,
                "leaky_relu" => ActivationType::LeakyReLU,
                "elu" => ActivationType::ELU,
                "swish" => ActivationType::Swish,
                "gelu" => ActivationType::GELU,
                "sine" => ActivationType::Sine,
                "threshold" => ActivationType::Threshold,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown activation function: {}", activation_name)
                )),
            };
            layer_configs.push(LayerConfig::new(*size, activation));
        }

        let network = NeuralNetwork::new_feedforward(&layer_configs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self { network })
    }

    fn initialize_weights(&mut self, seed: Option<u64>) -> PyResult<()> {
        self.network.initialize_weights(seed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn predict(&mut self, input: Vec<NeuralFloat>) -> PyResult<Vec<NeuralFloat>> {
        self.network.predict(&input)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn get_input_size(&self) -> usize {
        self.network.input_size()
    }

    fn get_output_size(&self) -> usize {
        self.network.output_size()
    }

    fn get_num_weights(&self) -> usize {
        self.network.num_weights()
    }

    fn get_num_biases(&self) -> usize {
        self.network.num_biases()
    }

    fn calculate_mse(&mut self, inputs: Vec<Vec<NeuralFloat>>, targets: Vec<Vec<NeuralFloat>>) -> PyResult<NeuralFloat> {
        self.network.calculate_mse(&inputs, &targets)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn save_to_file(&self, path: &str) -> PyResult<()> {
        // Serialize network to JSON
        let serialized = serde_json::to_string(&self.network)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        std::fs::write(path, serialized)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load_from_file(path: &str) -> PyResult<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        
        let network: NeuralNetwork = serde_json::from_str(&data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(Self { network })
    }
}

/// FFI-compatible trainer wrapper
#[cfg_attr(feature = "python", pyclass)]
pub struct PyTrainer {
    trainer: Trainer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTrainer {
    #[new]
    fn new(algorithm_name: String) -> PyResult<Self> {
        let algorithm = match algorithm_name.as_str() {
            "backpropagation" => TrainingAlgorithm::Backpropagation,
            "momentum" => TrainingAlgorithm::BackpropagationMomentum,
            "rprop" => TrainingAlgorithm::Rprop,
            "quickprop" => TrainingAlgorithm::Quickprop,
            "batch" => TrainingAlgorithm::Batch,
            "incremental" => TrainingAlgorithm::Incremental,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown training algorithm: {}", algorithm_name)
            )),
        };

        let params = TrainingParams::default();
        let trainer = Trainer::new(algorithm, params);
        
        Ok(Self { trainer })
    }

    fn set_learning_rate(&mut self, learning_rate: NeuralFloat) {
        let mut params = self.trainer.get_params().clone();
        params.learning_rate = learning_rate;
        self.trainer.set_params(params);
    }

    fn set_momentum(&mut self, momentum: NeuralFloat) {
        let mut params = self.trainer.get_params().clone();
        params.momentum = momentum;
        self.trainer.set_params(params);
    }

    fn set_max_epochs(&mut self, max_epochs: usize) {
        let mut params = self.trainer.get_params().clone();
        params.max_epochs = max_epochs;
        self.trainer.set_params(params);
    }

    fn set_desired_error(&mut self, desired_error: NeuralFloat) {
        let mut params = self.trainer.get_params().clone();
        params.desired_error = desired_error;
        self.trainer.set_params(params);
    }

    fn train(&mut self, network: &mut PyNeuralNetwork, inputs: Vec<Vec<NeuralFloat>>, targets: Vec<Vec<NeuralFloat>>) -> PyResult<PyTrainingResult> {
        if inputs.len() != targets.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of inputs must equal number of targets"
            ));
        }

        let mut training_data = TrainingData::new();
        for (input, target) in inputs.into_iter().zip(targets.into_iter()) {
            training_data.add_sample(input, target);
        }

        let progress = self.trainer.train(&mut network.network, &training_data, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyTrainingResult {
            epochs: progress.epoch,
            final_error: progress.error,
            best_error: progress.best_error,
            training_time_seconds: progress.elapsed_time.as_secs_f64(),
        })
    }

    fn get_best_error(&self) -> NeuralFloat {
        self.trainer.get_best_error()
    }

    fn get_error_history(&self) -> Vec<NeuralFloat> {
        self.trainer.get_error_history().iter().cloned().collect()
    }
}

/// Training result for Python interface
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct PyTrainingResult {
    #[cfg_attr(feature = "python", pyo3(get))]
    pub epochs: usize,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub final_error: NeuralFloat,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub best_error: NeuralFloat,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub training_time_seconds: f64,
}

/// FFI-compatible inference engine wrapper
#[cfg_attr(feature = "python", pyclass)]
pub struct PyInferenceEngine {
    engine: InferenceEngine,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyInferenceEngine {
    #[new]
    fn new(network: &PyNeuralNetwork, mode: String, batch_size: usize) -> PyResult<Self> {
        let execution_mode = match mode.as_str() {
            "sequential" => ExecutionMode::Sequential,
            "parallel" => ExecutionMode::Parallel,
            "simd" => ExecutionMode::Simd,
            "simd_parallel" => ExecutionMode::SimdParallel,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown execution mode: {}", mode)
            )),
        };

        let config = InferenceConfig {
            mode: execution_mode,
            batch_size,
            ..Default::default()
        };

        let network_arc = Arc::new(network.network.clone());
        let engine = InferenceEngine::new(network_arc, config);

        Ok(Self { engine })
    }

    fn predict_one(&self, input: Vec<NeuralFloat>) -> PyResult<Vec<NeuralFloat>> {
        self.engine.predict_one(&input)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn predict_batch(&mut self, inputs: Vec<Vec<NeuralFloat>>) -> PyResult<PyInferenceResult> {
        let result = self.engine.predict_batch(&inputs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyInferenceResult {
            outputs: result.outputs,
            inference_time_us: result.inference_time_us,
            throughput: result.throughput,
            memory_usage: result.memory_usage,
        })
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        let mut config = self.engine.get_config().clone();
        config.batch_size = batch_size;
        self.engine.set_config(config);
    }

    fn get_stats(&self) -> PyResult<HashMap<String, serde_json::Value>> {
        let stats = self.engine.get_stats();
        let mut result = HashMap::new();
        
        result.insert("input_size".to_string(), serde_json::json!(stats.network_input_size));
        result.insert("output_size".to_string(), serde_json::json!(stats.network_output_size));
        result.insert("num_layers".to_string(), serde_json::json!(stats.num_layers));
        result.insert("num_weights".to_string(), serde_json::json!(stats.num_weights));
        result.insert("num_biases".to_string(), serde_json::json!(stats.num_biases));
        result.insert("batch_size".to_string(), serde_json::json!(stats.batch_size));
        result.insert("memory_optimized".to_string(), serde_json::json!(stats.memory_optimized));
        
        Ok(result)
    }
}

/// Inference result for Python interface
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct PyInferenceResult {
    #[cfg_attr(feature = "python", pyo3(get))]
    pub outputs: Vec<Vec<NeuralFloat>>,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub inference_time_us: u64,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub throughput: f64,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub memory_usage: usize,
}

/// FFI-compatible data normalizer wrapper
#[cfg_attr(feature = "python", pyclass)]
pub struct PyDataNormalizer {
    normalizer: DataNormalizer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDataNormalizer {
    #[new]
    fn new(method: String) -> PyResult<Self> {
        let normalization_method = match method.as_str() {
            "minmax" => NormalizationMethod::MinMax,
            "minmax_symmetric" => NormalizationMethod::MinMaxSymmetric,
            "zscore" => NormalizationMethod::ZScore,
            "robust" => NormalizationMethod::Robust,
            "unit_vector" => NormalizationMethod::UnitVector,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown normalization method: {}", method)
            )),
        };

        let normalizer = DataNormalizer::new(normalization_method);
        Ok(Self { normalizer })
    }

    fn fit(&mut self, data: Vec<Vec<NeuralFloat>>) -> PyResult<()> {
        self.normalizer.fit(&data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn transform(&self, data: Vec<Vec<NeuralFloat>>) -> PyResult<Vec<Vec<NeuralFloat>>> {
        self.normalizer.transform(&data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn fit_transform(&mut self, data: Vec<Vec<NeuralFloat>>) -> PyResult<Vec<Vec<NeuralFloat>>> {
        self.normalizer.fit_transform(&data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn inverse_transform(&self, data: Vec<Vec<NeuralFloat>>) -> PyResult<Vec<Vec<NeuralFloat>>> {
        self.normalizer.inverse_transform(&data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn is_fitted(&self) -> bool {
        self.normalizer.is_fitted()
    }
}

/// Utility functions for Python interface
#[cfg(feature = "python")]
#[pyfunction]
fn calculate_mse(predictions: Vec<NeuralFloat>, targets: Vec<NeuralFloat>) -> PyResult<NeuralFloat> {
    crate::utils::metrics::mse(&predictions, &targets)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn calculate_mae(predictions: Vec<NeuralFloat>, targets: Vec<NeuralFloat>) -> PyResult<NeuralFloat> {
    crate::utils::metrics::mae(&predictions, &targets)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn calculate_r2_score(predictions: Vec<NeuralFloat>, targets: Vec<NeuralFloat>) -> PyResult<NeuralFloat> {
    crate::utils::metrics::r2_score(&predictions, &targets)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn train_test_split(
    data: Vec<Vec<NeuralFloat>>,
    train_ratio: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> PyResult<(Vec<Vec<NeuralFloat>>, Vec<Vec<NeuralFloat>>)> {
    let test_ratio = 1.0 - train_ratio;
    let (train_data, _, test_data) = crate::utils::cross_validation::train_val_test_split(
        &data, train_ratio, 0.0, shuffle, seed
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    
    Ok((train_data, test_data))
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn neural_swarm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeuralNetwork>()?;
    m.add_class::<PyTrainer>()?;
    m.add_class::<PyTrainingResult>()?;
    m.add_class::<PyInferenceEngine>()?;
    m.add_class::<PyInferenceResult>()?;
    m.add_class::<PyDataNormalizer>()?;
    
    m.add_function(wrap_pyfunction!(calculate_mse, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mae, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(train_test_split, m)?)?;
    
    Ok(())
}

/// C-compatible API for other language bindings
#[no_mangle]
pub extern "C" fn neural_network_create(
    layer_sizes: *const usize,
    layer_count: usize,
    activation_types: *const u32,
) -> *mut NeuralNetwork {
    if layer_sizes.is_null() || activation_types.is_null() || layer_count < 2 {
        return std::ptr::null_mut();
    }

    let sizes = unsafe { std::slice::from_raw_parts(layer_sizes, layer_count) };
    let activations = unsafe { std::slice::from_raw_parts(activation_types, layer_count) };

    let mut layer_configs = Vec::new();
    for (&size, &activation_id) in sizes.iter().zip(activations.iter()) {
        let activation = match activation_id {
            0 => ActivationType::Linear,
            1 => ActivationType::Sigmoid,
            2 => ActivationType::Tanh,
            3 => ActivationType::ReLU,
            4 => ActivationType::LeakyReLU,
            5 => ActivationType::ELU,
            6 => ActivationType::Swish,
            7 => ActivationType::GELU,
            _ => return std::ptr::null_mut(),
        };
        layer_configs.push(LayerConfig::new(size, activation));
    }

    match NeuralNetwork::new_feedforward(&layer_configs) {
        Ok(mut network) => {
            if network.initialize_weights(None).is_ok() {
                Box::into_raw(Box::new(network))
            } else {
                std::ptr::null_mut()
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn neural_network_destroy(network: *mut NeuralNetwork) {
    if !network.is_null() {
        unsafe {
            let _ = Box::from_raw(network);
        }
    }
}

#[no_mangle]
pub extern "C" fn neural_network_predict(
    network: *mut NeuralNetwork,
    input: *const NeuralFloat,
    input_size: usize,
    output: *mut NeuralFloat,
    output_size: usize,
) -> i32 {
    if network.is_null() || input.is_null() || output.is_null() {
        return -1;
    }

    let network = unsafe { &mut *network };
    let input_slice = unsafe { std::slice::from_raw_parts(input, input_size) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, output_size) };

    match network.predict(input_slice) {
        Ok(result) => {
            if result.len() != output_size {
                return -2;
            }
            output_slice.copy_from_slice(&result);
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub extern "C" fn get_version() -> *const std::os::raw::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const std::os::raw::c_char
}