use serde::{Deserialize, Serialize};

/// Training sample containing input and target output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

/// Collection of training samples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    samples: Vec<TrainingSample>,
}

/// Available training algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrainingAlgorithm {
    /// Standard backpropagation
    Backpropagation,
    /// Stochastic gradient descent
    StochasticGradientDescent,
}

impl TrainingData {
    /// Create a new empty training dataset
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }
    
    /// Add a training sample
    pub fn add_sample(&mut self, input: Vec<f32>, target: Vec<f32>) {
        self.samples.push(TrainingSample { input, target });
    }
    
    /// Add a training sample from a TrainingSample struct
    pub fn add_training_sample(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
    }
    
    /// Get all training samples
    pub fn samples(&self) -> &[TrainingSample] {
        &self.samples
    }
    
    /// Get the number of training samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    
    /// Check if the training data is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    
    /// Get a specific training sample by index
    pub fn get_sample(&self, index: usize) -> Option<&TrainingSample> {
        self.samples.get(index)
    }
    
    /// Clear all training samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }
    
    /// Create training data from separate input and target vectors
    pub fn from_vectors(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Self {
        let samples = inputs
            .into_iter()
            .zip(targets.into_iter())
            .map(|(input, target)| TrainingSample { input, target })
            .collect();
        
        Self { samples }
    }
    
    /// Split the training data into training and validation sets
    pub fn split(&self, train_ratio: f32) -> (TrainingData, TrainingData) {
        let split_point = (self.samples.len() as f32 * train_ratio) as usize;
        
        let train_samples = self.samples[..split_point].to_vec();
        let val_samples = self.samples[split_point..].to_vec();
        
        (
            TrainingData { samples: train_samples },
            TrainingData { samples: val_samples },
        )
    }
    
    /// Shuffle the training data
    pub fn shuffle(&mut self) {
        use crate::network::rand::seq::SliceRandom;
        self.samples.shuffle(&mut crate::network::rand::thread_rng());
    }
    
    /// Create a batch iterator over the training data
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = &[TrainingSample]> {
        self.samples.chunks(batch_size)
    }
    
    /// Get statistics about the training data
    pub fn statistics(&self) -> TrainingStatistics {
        if self.samples.is_empty() {
            return TrainingStatistics::default();
        }
        
        let input_size = self.samples[0].input.len();
        let output_size = self.samples[0].target.len();
        
        // Calculate input statistics
        let mut input_sum = vec![0.0; input_size];
        let mut input_sum_sq = vec![0.0; input_size];
        
        // Calculate output statistics
        let mut output_sum = vec![0.0; output_size];
        let mut output_sum_sq = vec![0.0; output_size];
        
        for sample in &self.samples {
            for (i, &val) in sample.input.iter().enumerate() {
                input_sum[i] += val;
                input_sum_sq[i] += val * val;
            }
            
            for (i, &val) in sample.target.iter().enumerate() {
                output_sum[i] += val;
                output_sum_sq[i] += val * val;
            }
        }
        
        let n = self.samples.len() as f32;
        
        let input_mean: Vec<f32> = input_sum.iter().map(|&s| s / n).collect();
        let input_std: Vec<f32> = input_sum_sq
            .iter()
            .zip(input_mean.iter())
            .map(|(&sum_sq, &mean)| ((sum_sq / n) - (mean * mean)).sqrt())
            .collect();
        
        let output_mean: Vec<f32> = output_sum.iter().map(|&s| s / n).collect();
        let output_std: Vec<f32> = output_sum_sq
            .iter()
            .zip(output_mean.iter())
            .map(|(&sum_sq, &mean)| ((sum_sq / n) - (mean * mean)).sqrt())
            .collect();
        
        TrainingStatistics {
            num_samples: self.samples.len(),
            input_size,
            output_size,
            input_mean,
            input_std,
            output_mean,
            output_std,
        }
    }
}

/// Statistics about the training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatistics {
    pub num_samples: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub input_mean: Vec<f32>,
    pub input_std: Vec<f32>,
    pub output_mean: Vec<f32>,
    pub output_std: Vec<f32>,
}

impl Default for TrainingStatistics {
    fn default() -> Self {
        Self {
            num_samples: 0,
            input_size: 0,
            output_size: 0,
            input_mean: Vec::new(),
            input_std: Vec::new(),
            output_mean: Vec::new(),
            output_std: Vec::new(),
        }
    }
}

impl Default for TrainingData {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingSample {
    /// Create a new training sample
    pub fn new(input: Vec<f32>, target: Vec<f32>) -> Self {
        Self { input, target }
    }
    
    /// Get the input size
    pub fn input_size(&self) -> usize {
        self.input.len()
    }
    
    /// Get the output size
    pub fn output_size(&self) -> usize {
        self.target.len()
    }
}

/// Training configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub algorithm: TrainingAlgorithm,
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub validation_split: f32,
    pub early_stopping: bool,
    pub early_stopping_patience: usize,
    pub early_stopping_threshold: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            algorithm: TrainingAlgorithm::Backpropagation,
            learning_rate: 0.1,
            epochs: 1000,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping: true,
            early_stopping_patience: 10,
            early_stopping_threshold: 1e-6,
        }
    }
}

/// Training results and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResults {
    pub final_error: f32,
    pub epochs_completed: usize,
    pub training_time: std::time::Duration,
    pub validation_error: Option<f32>,
    pub early_stopped: bool,
    pub error_history: Vec<f32>,
}

impl Default for TrainingResults {
    fn default() -> Self {
        Self {
            final_error: 0.0,
            epochs_completed: 0,
            training_time: std::time::Duration::from_secs(0),
            validation_error: None,
            early_stopped: false,
            error_history: Vec::new(),
        }
    }
}