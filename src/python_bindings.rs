//! Python bindings for the FANN Rust Core library
//!
//! This module provides comprehensive Python bindings using PyO3, allowing
//! seamless integration between the high-performance Rust neural network
//! implementation and the Python ecosystem.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numpy::{PyArray1, PyArray2, ToPyArray};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use parking_lot::RwLock;

use crate::network::{NetworkBuilder, FeedforwardNetwork, NeuralNetwork, MutableNeuralNetwork};
use crate::training::{TrainingData, TrainingConfig, TrainingAlgorithm, BackpropagationTrainer};
use crate::activation::ActivationFunction;
use crate::error::{NetworkError, TrainingError};
use crate::optimization::SIMDMatrixOps;
use crate::utils::{MathUtils, DataUtils};

/// Python wrapper for FeedforwardNetwork
#[pyclass(name = "NeuralNetwork")]
pub struct PyNeuralNetwork {
    inner: Arc<RwLock<FeedforwardNetwork>>,
}

#[pymethods]
impl PyNeuralNetwork {
    /// Create a new neural network
    #[new]
    #[pyo3(signature = (layers, activation="relu", output_activation=None))]
    fn new(
        layers: Vec<usize>,
        activation: &str,
        output_activation: Option<&str>,
    ) -> PyResult<Self> {
        let activation_fn = parse_activation(activation)?;
        
        let mut activations = vec![activation_fn; layers.len() - 1];
        if let Some(output_act) = output_activation {
            let output_activation_fn = parse_activation(output_act)?;
            if let Some(last) = activations.last_mut() {
                *last = output_activation_fn;
            }
        }
        
        let network = NetworkBuilder::new()
            .layers(&layers)
            .activations(&activations)
            .training_algorithm(BackpropagationTrainer::new(0.001))
            .build()
            .map_err(|e| PyErr::from(e))?;
        
        Ok(Self {
            inner: Arc::new(RwLock::new(network)),
        })
    }
    
    /// Perform forward pass
    #[pyo3(signature = (input))]
    fn predict<'py>(&self, py: Python<'py>, input: &PyArray1<f32>) -> PyResult<&'py PyArray1<f32>> {
        let input_array = input.as_array();
        let input_slice = input_array.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array must be contiguous")
        })?;
        
        let network = self.inner.read();
        let output = network.forward(input_slice)
            .map_err(|e| PyErr::from(e))?;
        
        Ok(output.to_pyarray(py))
    }
    
    /// Train the network
    #[pyo3(signature = (inputs, targets, epochs=100, batch_size=32, learning_rate=0.001, validation_split=0.0, verbose=false))]
    fn train(
        &mut self,
        inputs: &PyArray2<f32>,
        targets: &PyArray2<f32>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        validation_split: f32,
        verbose: bool,
    ) -> PyResult<PyTrainingResults> {
        // Convert NumPy arrays to Rust vectors
        let input_array = inputs.as_array();
        let target_array = targets.as_array();
        
        let input_vecs: Vec<Vec<f32>> = input_array.outer_iter()
            .map(|row| row.to_vec())
            .collect();
        
        let target_vecs: Vec<Vec<f32>> = target_array.outer_iter()
            .map(|row| row.to_vec())
            .collect();
        
        let training_data = TrainingData::new(input_vecs, target_vecs)
            .map_err(|e| PyErr::from(e))?;
        
        let config = TrainingConfig {
            epochs,
            batch_size,
            learning_rate,
            validation_split,
            shuffle_data: true,
            verbose,
            ..Default::default()
        };
        
        let mut network = self.inner.write();
        let results = network.train(&training_data, config)
            .map_err(|e| PyErr::from(e))?;
        
        Ok(PyTrainingResults::from(results))
    }
    
    /// Get network weights
    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f32>> {
        let network = self.inner.read();
        let weights = network.get_weights();
        Ok(weights.to_pyarray(py))
    }
    
    /// Set network weights
    fn set_weights(&mut self, weights: &PyArray1<f32>) -> PyResult<()> {
        let weights_array = weights.as_array();
        let weights_slice = weights_array.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Weights array must be contiguous")
        })?;
        
        let mut network = self.inner.write();
        network.set_weights(weights_slice)
            .map_err(|e| PyErr::from(e))?;
        
        Ok(())
    }
    
    /// Save network to file
    fn save(&self, path: &str) -> PyResult<()> {
        let network = self.inner.read();
        network.save_to_file(path)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }
    
    /// Load network from file
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let network = FeedforwardNetwork::load_from_file(path)
            .map_err(|e| PyErr::from(e))?;
        
        Ok(Self {
            inner: Arc::new(RwLock::new(network)),
        })
    }
    
    /// Get network architecture information
    fn architecture(&self) -> PyResult<PyDict> {
        let network = self.inner.read();
        let arch = network.architecture();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("layers", arch.layers.len())?;
            dict.set_item("total_parameters", arch.total_parameters)?;
            dict.set_item("network_type", format!("{:?}", arch.network_type))?;
            
            let layer_info: Vec<(usize, String)> = arch.layers.iter()
                .map(|layer| (layer.size, layer.activation.name().to_string()))
                .collect();
            dict.set_item("layer_info", layer_info)?;
            
            Ok(dict.into())
        })
    }
    
    /// Get input size
    #[getter]
    fn input_size(&self) -> usize {
        let network = self.inner.read();
        network.input_size()
    }
    
    /// Get output size
    #[getter]
    fn output_size(&self) -> usize {
        let network = self.inner.read();
        network.output_size()
    }
    
    /// Get parameter count
    #[getter]
    fn parameter_count(&self) -> usize {
        let network = self.inner.read();
        network.parameter_count()
    }
    
    /// Get memory footprint in bytes
    #[getter]
    fn memory_footprint(&self) -> usize {
        let network = self.inner.read();
        network.memory_footprint()
    }
    
    /// Reset network weights
    fn reset_weights(&mut self) -> PyResult<()> {
        let mut network = self.inner.write();
        network.reset_weights()
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }
    
    /// Evaluate network on test data
    #[pyo3(signature = (inputs, targets))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        inputs: &PyArray2<f32>,
        targets: &PyArray2<f32>,
    ) -> PyResult<PyDict> {
        let input_array = inputs.as_array();
        let target_array = targets.as_array();
        
        let network = self.inner.read();
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_samples = 0;
        
        for (input_row, target_row) in input_array.outer_iter().zip(target_array.outer_iter()) {
            let input_slice = input_row.as_slice().unwrap();
            let target_slice = target_row.as_slice().unwrap();
            
            let prediction = network.forward(input_slice)
                .map_err(|e| PyErr::from(e))?;
            
            // Compute loss (MSE)
            let loss: f32 = prediction.iter().zip(target_slice.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>() / prediction.len() as f32;
            total_loss += loss;
            
            // Check accuracy for classification
            if prediction.len() == target_slice.len() {
                let pred_class = prediction.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                
                let true_class = target_slice.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                
                if pred_class == true_class {
                    correct_predictions += 1;
                }
            }
            total_samples += 1;
        }
        
        let avg_loss = total_loss / total_samples as f32;
        let accuracy = correct_predictions as f32 / total_samples as f32;
        
        let results = PyDict::new(py);
        results.set_item("loss", avg_loss)?;
        results.set_item("accuracy", accuracy)?;
        results.set_item("samples", total_samples)?;
        
        Ok(results)
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        let network = self.inner.read();
        format!(
            "NeuralNetwork(input_size={}, output_size={}, parameters={})",
            network.input_size(),
            network.output_size(),
            network.parameter_count()
        )
    }
}

/// Python wrapper for training results
#[pyclass(name = "TrainingResults")]
#[derive(Clone)]
pub struct PyTrainingResults {
    #[pyo3(get)]
    epochs_completed: usize,
    #[pyo3(get)]
    final_loss: f32,
    #[pyo3(get)]
    final_accuracy: f32,
    #[pyo3(get)]
    training_loss_history: Vec<f32>,
    #[pyo3(get)]
    validation_loss_history: Vec<f32>,
    #[pyo3(get)]
    training_accuracy_history: Vec<f32>,
    #[pyo3(get)]
    validation_accuracy_history: Vec<f32>,
    #[pyo3(get)]
    best_epoch: usize,
    #[pyo3(get)]
    best_validation_loss: f32,
    #[pyo3(get)]
    training_time_ms: f64,
    #[pyo3(get)]
    converged: bool,
}

impl From<crate::training::TrainingResults> for PyTrainingResults {
    fn from(results: crate::training::TrainingResults) -> Self {
        Self {
            epochs_completed: results.epochs_completed,
            final_loss: results.final_loss,
            final_accuracy: results.final_accuracy,
            training_loss_history: results.training_loss_history,
            validation_loss_history: results.validation_loss_history,
            training_accuracy_history: results.training_accuracy_history,
            validation_accuracy_history: results.validation_accuracy_history,
            best_epoch: results.best_epoch,
            best_validation_loss: results.best_validation_loss,
            training_time_ms: results.training_time.as_secs_f64() * 1000.0,
            converged: results.converged,
        }
    }
}

#[pymethods]
impl PyTrainingResults {
    fn __repr__(&self) -> String {
        format!(
            "TrainingResults(epochs={}, final_loss={:.6}, final_accuracy={:.4}, converged={})",
            self.epochs_completed, self.final_loss, self.final_accuracy, self.converged
        )
    }
}

/// Python wrapper for training data
#[pyclass(name = "TrainingData")]
pub struct PyTrainingData {
    inner: TrainingData,
}

#[pymethods]
impl PyTrainingData {
    #[new]
    fn new(inputs: &PyArray2<f32>, targets: &PyArray2<f32>) -> PyResult<Self> {
        let input_array = inputs.as_array();
        let target_array = targets.as_array();
        
        let input_vecs: Vec<Vec<f32>> = input_array.outer_iter()
            .map(|row| row.to_vec())
            .collect();
        
        let target_vecs: Vec<Vec<f32>> = target_array.outer_iter()
            .map(|row| row.to_vec())
            .collect();
        
        let training_data = TrainingData::new(input_vecs, target_vecs)
            .map_err(|e| PyErr::from(e))?;
        
        Ok(Self { inner: training_data })
    }
    
    /// Load training data from CSV file
    #[staticmethod]
    fn from_csv(path: &str, input_columns: usize, target_columns: usize) -> PyResult<Self> {
        let training_data = TrainingData::from_csv(path, input_columns, target_columns)
            .map_err(|e| PyErr::from(e))?;
        
        Ok(Self { inner: training_data })
    }
    
    /// Split data into training and validation sets
    fn split(&self, validation_ratio: f32) -> PyResult<(Self, Self)> {
        let (train_data, val_data) = self.inner.split(validation_ratio)
            .map_err(|e| PyErr::from(e))?;
        
        Ok((
            Self { inner: train_data },
            Self { inner: val_data },
        ))
    }
    
    /// Get data length
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Get input size
    #[getter]
    fn input_size(&self) -> usize {
        self.inner.input_size()
    }
    
    /// Get target size
    #[getter]
    fn target_size(&self) -> usize {
        self.inner.target_size()
    }
    
    /// Get inputs as NumPy array
    fn get_inputs<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f32>> {
        let inputs = self.inner.inputs();
        let rows = inputs.len();
        let cols = inputs[0].len();
        
        let flat_data: Vec<f32> = inputs.iter().flat_map(|row| row.iter().cloned()).collect();
        let array = Array2::from_shape_vec((rows, cols), flat_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(array.to_pyarray(py))
    }
    
    /// Get targets as NumPy array
    fn get_targets<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f32>> {
        let targets = self.inner.targets();
        let rows = targets.len();
        let cols = targets[0].len();
        
        let flat_data: Vec<f32> = targets.iter().flat_map(|row| row.iter().cloned()).collect();
        let array = Array2::from_shape_vec((rows, cols), flat_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(array.to_pyarray(py))
    }
}

/// Utility functions exposed to Python
#[pymodule]
fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Compute softmax
    #[pyfn(m)]
    fn softmax<'py>(py: Python<'py>, input: &PyArray1<f32>) -> PyResult<&'py PyArray1<f32>> {
        let input_array = input.as_array();
        let input_slice = input_array.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array must be contiguous")
        })?;
        
        let output = MathUtils::softmax(input_slice);
        Ok(output.to_pyarray(py))
    }
    
    /// One-hot encode
    #[pyfn(m)]
    fn one_hot_encode<'py>(py: Python<'py>, class_index: usize, num_classes: usize) -> PyResult<&'py PyArray1<f32>> {
        let output = DataUtils::one_hot_encode(class_index, num_classes);
        Ok(output.to_pyarray(py))
    }
    
    /// One-hot decode
    #[pyfn(m)]
    fn one_hot_decode(one_hot: &PyArray1<f32>) -> PyResult<usize> {
        let one_hot_array = one_hot.as_array();
        let one_hot_slice = one_hot_array.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array must be contiguous")
        })?;
        
        Ok(DataUtils::one_hot_decode(one_hot_slice))
    }
    
    /// Matrix multiplication with SIMD optimization
    #[pyfn(m)]
    fn matrix_multiply<'py>(
        py: Python<'py>,
        a: &PyArray2<f32>,
        b: &PyArray2<f32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let a_array = a.as_array();
        let b_array = b.as_array();
        
        let (m, k1) = a_array.dim();
        let (k2, n) = b_array.dim();
        
        if k1 != k2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Matrix dimensions don't match for multiplication"
            ));
        }
        
        let a_slice = a_array.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix A must be contiguous")
        })?;
        let b_slice = b_array.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix B must be contiguous")
        })?;
        
        let mut c = vec![0.0; m * n];
        SIMDMatrixOps::matrix_multiply(a_slice, b_slice, &mut c, m, n, k1)
            .map_err(|e| PyErr::from(e))?;
        
        let result_array = Array2::from_shape_vec((m, n), c)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(result_array.to_pyarray(py))
    }
    
    Ok(())
}

/// Performance benchmarking functions
#[pymodule]
fn benchmarks(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Run matrix multiplication benchmark
    #[pyfn(m)]
    fn benchmark_matrix_multiply(size: usize, iterations: usize) -> PyResult<PyDict> {
        let results = crate::optimization::benchmarks::benchmark_matrix_multiply(size, iterations);
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            for result in results {
                let entry = PyDict::new(py);
                entry.set_item("duration_ms", result.duration.as_millis())?;
                if let Some(throughput) = result.throughput {
                    entry.set_item("throughput", throughput)?;
                }
                dict.set_item(result.operation, entry)?;
            }
            
            Ok(dict.into())
        })
    }
    
    /// Run activation function benchmarks
    #[pyfn(m)]
    fn benchmark_activations(size: usize, iterations: usize) -> PyResult<PyDict> {
        use crate::activation::benchmarks::ActivationBenchmark;
        use crate::activation::ActivationFunction;
        
        let functions = vec![
            ActivationFunction::ReLU,
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
        ];
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            for function in functions {
                let benchmark = ActivationBenchmark::new(function.clone(), size, iterations);
                let results = benchmark.compare_implementations();
                
                let entry = PyDict::new(py);
                entry.set_item("scalar_time", results.scalar_time)?;
                entry.set_item("simd_time", results.simd_time)?;
                entry.set_item("speedup", results.speedup)?;
                
                dict.set_item(function.name(), entry)?;
            }
            
            Ok(dict.into())
        })
    }
    
    Ok(())
}

/// Library information and capabilities
#[pyfunction]
fn get_library_info() -> PyResult<PyDict> {
    let info = crate::get_info();
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("name", info.name)?;
        dict.set_item("version", info.version)?;
        dict.set_item("description", info.description)?;
        dict.set_item("features", info.features)?;
        dict.set_item("simd_support", info.simd_support)?;
        dict.set_item("thread_count", info.thread_count)?;
        
        Ok(dict.into())
    })
}

/// Detect available SIMD features
#[pyfunction]
fn get_simd_features() -> Vec<String> {
    crate::optimization::detect_simd_features()
}

/// Initialize the library
#[pyfunction]
fn initialize() -> PyResult<()> {
    crate::initialize().map_err(|e| PyErr::from(e))
}

/// Helper function to parse activation function names
fn parse_activation(name: &str) -> PyResult<ActivationFunction> {
    match name.to_lowercase().as_str() {
        "linear" => Ok(ActivationFunction::Linear),
        "relu" => Ok(ActivationFunction::ReLU),
        "sigmoid" => Ok(ActivationFunction::Sigmoid),
        "tanh" => Ok(ActivationFunction::Tanh),
        "swish" => Ok(ActivationFunction::Swish),
        "gelu" => Ok(ActivationFunction::GELU),
        "softmax" => Ok(ActivationFunction::Softmax),
        "softplus" => Ok(ActivationFunction::Softplus),
        "mish" => Ok(ActivationFunction::Mish),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown activation function: {}", name)
        )),
    }
}

/// Main Python module
#[pymodule]
fn fann_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add classes
    m.add_class::<PyNeuralNetwork>()?;
    m.add_class::<PyTrainingResults>()?;
    m.add_class::<PyTrainingData>()?;
    
    // Add utility functions
    m.add_function(wrap_pyfunction!(get_library_info, m)?)?;
    m.add_function(wrap_pyfunction!(get_simd_features, m)?)?;
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    
    // Add submodules
    m.add_submodule(utils(_py, PyModule::new(_py, "utils")?)?)?;
    m.add_submodule(benchmarks(_py, PyModule::new(_py, "benchmarks")?)?)?;
    
    // Add version information
    m.add("__version__", crate::VERSION)?;
    m.add("__name__", crate::NAME)?;
    m.add("__description__", crate::DESCRIPTION)?;
    
    Ok(())
}