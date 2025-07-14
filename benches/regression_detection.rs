use criterion::{criterion_group, criterion_main, Criterion};
use fann_rust_core::{NetworkBuilder, TrainingData, TrainingAlgorithm};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Baseline performance data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselinePerformance {
    pub version: String,
    pub timestamp: u64,
    pub benchmarks: HashMap<String, BenchmarkBaseline>,
}

/// Individual benchmark baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    pub mean_duration: f64,  // nanoseconds
    pub std_deviation: f64,
    pub throughput: Option<f64>,
    pub memory_usage: Option<usize>,
    pub samples: usize,
}

/// Regression detection thresholds
#[derive(Debug, Clone)]
pub struct RegressionThresholds {
    pub performance_degradation: f64,  // 5% = 0.05
    pub memory_increase: f64,          // 10% = 0.10
    pub throughput_decrease: f64,      // 5% = 0.05
    pub statistical_confidence: f64,  // 95% = 0.95
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            performance_degradation: 0.05,
            memory_increase: 0.10,
            throughput_decrease: 0.05,
            statistical_confidence: 0.95,
        }
    }
}

/// Regression test results
#[derive(Debug, Clone)]
pub struct RegressionResults {
    pub regressions_detected: Vec<PerformanceRegression>,
    pub improvements_detected: Vec<PerformanceImprovement>,
    pub baseline_updated: bool,
}

/// Performance regression details
#[derive(Debug, Clone)]
pub struct PerformanceRegression {
    pub benchmark_name: String,
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub baseline_value: f64,
    pub current_value: f64,
    pub change_percentage: f64,
    pub statistical_significance: f64,
}

/// Performance improvement details
#[derive(Debug, Clone)]
pub struct PerformanceImprovement {
    pub benchmark_name: String,
    pub improvement_type: ImprovementType,
    pub baseline_value: f64,
    pub current_value: f64,
    pub improvement_percentage: f64,
}

/// Types of performance regressions
#[derive(Debug, Clone)]
pub enum RegressionType {
    Latency,
    Throughput,
    Memory,
    Accuracy,
}

/// Types of performance improvements
#[derive(Debug, Clone)]
pub enum ImprovementType {
    Latency,
    Throughput,
    Memory,
    Accuracy,
}

/// Regression severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegressionSeverity {
    Minor,    // 5-10% degradation
    Moderate, // 10-20% degradation
    Major,    // 20-50% degradation
    Critical, // >50% degradation
}

/// Regression detection system
pub struct RegressionDetector {
    baseline_path: String,
    thresholds: RegressionThresholds,
}

impl RegressionDetector {
    pub fn new(baseline_path: &str) -> Self {
        Self {
            baseline_path: baseline_path.to_string(),
            thresholds: RegressionThresholds::default(),
        }
    }
    
    pub fn with_thresholds(mut self, thresholds: RegressionThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }
    
    /// Load baseline performance data
    pub fn load_baseline(&self) -> Option<BaselinePerformance> {
        if Path::new(&self.baseline_path).exists() {
            fs::read_to_string(&self.baseline_path)
                .ok()
                .and_then(|content| serde_json::from_str(&content).ok())
        } else {
            None
        }
    }
    
    /// Save baseline performance data
    pub fn save_baseline(&self, baseline: &BaselinePerformance) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(baseline)?;
        fs::write(&self.baseline_path, content)?;
        Ok(())
    }
    
    /// Check for performance regressions
    pub fn check_regressions(
        &self,
        current_results: &HashMap<String, BenchmarkBaseline>
    ) -> RegressionResults {
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        let mut baseline_updated = false;
        
        if let Some(baseline) = self.load_baseline() {
            for (benchmark_name, current_result) in current_results {
                if let Some(baseline_result) = baseline.benchmarks.get(benchmark_name) {
                    // Check for latency regression
                    let latency_change = (current_result.mean_duration - baseline_result.mean_duration) 
                        / baseline_result.mean_duration;
                    
                    if latency_change > self.thresholds.performance_degradation {
                        let significance = self.calculate_statistical_significance(
                            baseline_result, current_result
                        );
                        
                        if significance >= self.thresholds.statistical_confidence {
                            regressions.push(PerformanceRegression {
                                benchmark_name: benchmark_name.clone(),
                                regression_type: RegressionType::Latency,
                                severity: self.determine_severity(latency_change),
                                baseline_value: baseline_result.mean_duration,
                                current_value: current_result.mean_duration,
                                change_percentage: latency_change * 100.0,
                                statistical_significance: significance,
                            });
                        }
                    } else if latency_change < -0.02 { // 2% improvement threshold
                        improvements.push(PerformanceImprovement {
                            benchmark_name: benchmark_name.clone(),
                            improvement_type: ImprovementType::Latency,
                            baseline_value: baseline_result.mean_duration,
                            current_value: current_result.mean_duration,
                            improvement_percentage: -latency_change * 100.0,
                        });
                    }
                    
                    // Check for throughput regression
                    if let (Some(baseline_throughput), Some(current_throughput)) = 
                        (baseline_result.throughput, current_result.throughput) {
                        let throughput_change = (current_throughput - baseline_throughput) / baseline_throughput;
                        
                        if throughput_change < -self.thresholds.throughput_decrease {
                            regressions.push(PerformanceRegression {
                                benchmark_name: benchmark_name.clone(),
                                regression_type: RegressionType::Throughput,
                                severity: self.determine_severity(-throughput_change),
                                baseline_value: baseline_throughput,
                                current_value: current_throughput,
                                change_percentage: throughput_change * 100.0,
                                statistical_significance: 0.95, // Simplified
                            });
                        } else if throughput_change > 0.02 {
                            improvements.push(PerformanceImprovement {
                                benchmark_name: benchmark_name.clone(),
                                improvement_type: ImprovementType::Throughput,
                                baseline_value: baseline_throughput,
                                current_value: current_throughput,
                                improvement_percentage: throughput_change * 100.0,
                            });
                        }
                    }
                    
                    // Check for memory regression
                    if let (Some(baseline_memory), Some(current_memory)) = 
                        (baseline_result.memory_usage, current_result.memory_usage) {
                        let memory_change = (current_memory as f64 - baseline_memory as f64) 
                            / baseline_memory as f64;
                        
                        if memory_change > self.thresholds.memory_increase {
                            regressions.push(PerformanceRegression {
                                benchmark_name: benchmark_name.clone(),
                                regression_type: RegressionType::Memory,
                                severity: self.determine_severity(memory_change),
                                baseline_value: baseline_memory as f64,
                                current_value: current_memory as f64,
                                change_percentage: memory_change * 100.0,
                                statistical_significance: 0.95, // Simplified
                            });
                        } else if memory_change < -0.02 {
                            improvements.push(PerformanceImprovement {
                                benchmark_name: benchmark_name.clone(),
                                improvement_type: ImprovementType::Memory,
                                baseline_value: baseline_memory as f64,
                                current_value: current_memory as f64,
                                improvement_percentage: -memory_change * 100.0,
                            });
                        }
                    }
                }
            }
        } else {
            // No baseline exists, create one
            let new_baseline = BaselinePerformance {
                version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                benchmarks: current_results.clone(),
            };
            
            if self.save_baseline(&new_baseline).is_ok() {
                baseline_updated = true;
            }
        }
        
        RegressionResults {
            regressions_detected: regressions,
            improvements_detected: improvements,
            baseline_updated,
        }
    }
    
    /// Calculate statistical significance using t-test approximation
    fn calculate_statistical_significance(
        &self,
        baseline: &BenchmarkBaseline,
        current: &BenchmarkBaseline
    ) -> f64 {
        // Simplified statistical significance calculation
        // In practice, this would use proper t-test or Mann-Whitney U test
        
        let pooled_std = ((baseline.std_deviation.powi(2) / baseline.samples as f64) +
                         (current.std_deviation.powi(2) / current.samples as f64)).sqrt();
        
        if pooled_std > 0.0 {
            let t_statistic = ((current.mean_duration - baseline.mean_duration) / pooled_std).abs();
            
            // Simplified p-value approximation (for demonstration)
            // Real implementation would use proper statistical functions
            let degrees_of_freedom = baseline.samples + current.samples - 2;
            let critical_value = 2.0; // Approximate for 95% confidence
            
            if t_statistic > critical_value {
                0.95
            } else {
                0.5
            }
        } else {
            0.5
        }
    }
    
    /// Determine regression severity based on change percentage
    fn determine_severity(&self, change_percentage: f64) -> RegressionSeverity {
        let abs_change = change_percentage.abs();
        
        if abs_change >= 0.5 {
            RegressionSeverity::Critical
        } else if abs_change >= 0.2 {
            RegressionSeverity::Major
        } else if abs_change >= 0.1 {
            RegressionSeverity::Moderate
        } else {
            RegressionSeverity::Minor
        }
    }
}

/// Benchmark a simple neural network operation for regression testing
fn benchmark_neural_network_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_neural_inference");
    
    // Create a standard test network
    let network = NetworkBuilder::new()
        .add_layer(784, 128)
        .add_layer(128, 64)
        .add_layer(64, 10)
        .build()
        .expect("Failed to create test network");
    
    let test_input = vec![0.5f32; 784];
    
    group.bench_function("mnist_like_inference", |b| {
        b.iter(|| {
            let output = network.forward(&test_input);
            criterion::black_box(output);
        });
    });
    
    group.finish();
}

/// Benchmark matrix multiplication for regression testing
fn benchmark_matrix_operations_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_matrix_ops");
    
    let sizes = vec![(64, 64, 64), (128, 128, 128), (256, 256, 256)];
    
    for (m, n, k) in sizes {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        
        group.bench_function(
            &format!("matrix_mult_{}x{}x{}", m, n, k),
            |bench| {
                bench.iter(|| {
                    // Simple matrix multiplication
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[i * k + l] * b[l * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                    criterion::black_box(&c);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark training performance for regression testing
fn benchmark_training_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_training");
    group.measurement_time(Duration::from_secs(30));
    
    group.bench_function("xor_training", |b| {
        b.iter_batched(
            || {
                let mut network = NetworkBuilder::new()
                    .add_layer(2, 4)
                    .add_layer(4, 1)
                    .build()
                    .expect("Failed to create network");
                
                let mut training_data = TrainingData::new();
                training_data.add_sample(vec![0.0, 0.0], vec![0.0]);
                training_data.add_sample(vec![0.0, 1.0], vec![1.0]);
                training_data.add_sample(vec![1.0, 0.0], vec![1.0]);
                training_data.add_sample(vec![1.0, 1.0], vec![0.0]);
                
                (network, training_data)
            },
            |(mut network, training_data)| {
                network.train(
                    &training_data,
                    100, // Reduced epochs for benchmarking
                    0.1,
                    TrainingAlgorithm::Backpropagation
                ).expect("Training failed");
                criterion::black_box(network);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Automated regression detection benchmark
fn benchmark_regression_detection_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_detection_system");
    
    // Setup regression detector
    let detector = RegressionDetector::new("benches/baseline_performance.json")
        .with_thresholds(RegressionThresholds {
            performance_degradation: 0.05,
            memory_increase: 0.10,
            throughput_decrease: 0.05,
            statistical_confidence: 0.95,
        });
    
    // Create mock current results
    let mut current_results = HashMap::new();
    current_results.insert("test_benchmark".to_string(), BenchmarkBaseline {
        mean_duration: 1000000.0, // 1ms in nanoseconds
        std_deviation: 50000.0,
        throughput: Some(1000.0),
        memory_usage: Some(1024 * 1024), // 1MB
        samples: 100,
    });
    
    group.bench_function("regression_check", |b| {
        b.iter(|| {
            let results = detector.check_regressions(&current_results);
            criterion::black_box(results);
        });
    });
    
    // Benchmark with simulated regression
    let mut regressed_results = HashMap::new();
    regressed_results.insert("test_benchmark".to_string(), BenchmarkBaseline {
        mean_duration: 1200000.0, // 20% slower
        std_deviation: 60000.0,
        throughput: Some(800.0),   // 20% lower throughput
        memory_usage: Some(1200 * 1024), // 20% more memory
        samples: 100,
    });
    
    group.bench_function("regression_detection_with_issues", |b| {
        b.iter(|| {
            let results = detector.check_regressions(&regressed_results);
            criterion::black_box(results);
        });
    });
    
    group.finish();
}

criterion_group! {
    name = regression_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(50);
    targets = 
        benchmark_neural_network_inference,
        benchmark_matrix_operations_regression,
        benchmark_training_regression,
        benchmark_regression_detection_system
}

criterion_main!(regression_benches);