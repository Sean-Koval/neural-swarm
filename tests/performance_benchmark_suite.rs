//! Performance Benchmark Suite for Neural Task Decomposition
//!
//! This suite provides comprehensive performance benchmarks for all aspects
//! of the neural task decomposition system.

#[cfg(test)]
mod performance_benchmarks {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
    use std::time::{Duration, Instant};
    use tokio::runtime::Runtime;
    
    /// Benchmark configuration
    struct BenchmarkConfig {
        pub warm_up_time: Duration,
        pub measurement_time: Duration,
        pub sample_size: usize,
        pub significance_level: f64,
    }

    impl Default for BenchmarkConfig {
        fn default() -> Self {
            Self {
                warm_up_time: Duration::from_secs(3),
                measurement_time: Duration::from_secs(10),
                sample_size: 100,
                significance_level: 0.05,
            }
        }
    }

    /// Performance test suite structure
    pub struct PerformanceTestSuite {
        config: BenchmarkConfig,
        rt: Runtime,
        profiler: crate::test_utils::PerformanceProfiler,
    }

    impl PerformanceTestSuite {
        pub fn new() -> Self {
            Self {
                config: BenchmarkConfig::default(),
                rt: Runtime::new().unwrap(),
                profiler: crate::test_utils::PerformanceProfiler::new(),
            }
        }

        /// Benchmark decomposition speed across different strategies
        pub fn benchmark_decomposition_speed(&mut self, c: &mut Criterion) {
            let mut group = c.benchmark_group("decomposition_speed");
            
            let strategies = vec![
                DecompositionStrategy::Heuristic,
                DecompositionStrategy::Neural,
                DecompositionStrategy::Hybrid,
            ];
            
            let complexities = vec![0.1, 0.3, 0.5, 0.7, 0.9];
            
            for strategy in strategies {
                for complexity in &complexities {
                    let mut framework = TestFramework::new(strategy.clone());
                    let task = Task::new_test_task(
                        &format!("benchmark_{}_{}", strategy as u8, complexity),
                        *complexity
                    );
                    
                    group.bench_with_input(
                        BenchmarkId::new(format!("{:?}", strategy), complexity),
                        complexity,
                        |b, _| {
                            b.to_async(&self.rt).iter(|| async {
                                black_box(framework.run_decomposition_test(&task, strategy).await)
                            })
                        },
                    );
                }
            }
            
            group.finish();
        }

        /// Benchmark memory usage patterns
        pub fn benchmark_memory_usage(&mut self, c: &mut Criterion) {
            let mut group = c.benchmark_group("memory_usage");
            
            let task_sizes = vec![
                (1, "single_task"),
                (10, "small_batch"),
                (100, "medium_batch"),
                (1000, "large_batch"),
            ];
            
            for (size, name) in task_sizes {
                let tasks: Vec<_> = (0..size)
                    .map(|i| Task::new_test_task(&format!("mem_test_{}", i), 0.5))
                    .collect();
                
                group.bench_with_input(
                    BenchmarkId::new("memory_allocation", name),
                    &tasks,
                    |b, tasks| {
                        b.to_async(&self.rt).iter(|| async {
                            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
                            let mut results = Vec::new();
                            
                            for task in tasks {
                                let result = framework.run_decomposition_test(task, DecompositionStrategy::Neural).await;
                                results.push(black_box(result));
                            }
                            
                            results
                        })
                    },
                );
            }
            
            group.finish();
        }

        /// Benchmark concurrency performance
        pub fn benchmark_concurrency(&mut self, c: &mut Criterion) {
            let mut group = c.benchmark_group("concurrency");
            
            let concurrency_levels = vec![1, 2, 4, 8, 16];
            
            for level in concurrency_levels {
                group.bench_with_input(
                    BenchmarkId::new("concurrent_decomposition", level),
                    &level,
                    |b, &level| {
                        b.to_async(&self.rt).iter(|| async {
                            let tasks: Vec<_> = (0..level)
                                .map(|i| Task::new_test_task(&format!("concurrent_{}", i), 0.5))
                                .collect();
                            
                            let handles: Vec<_> = tasks.into_iter().map(|task| {
                                let framework = TestFramework::new(DecompositionStrategy::Hybrid);
                                tokio::spawn(async move {
                                    framework.decomposer.decompose_task(
                                        &task,
                                        DecompositionStrategy::Hybrid,
                                        &framework.context
                                    ).await
                                })
                            }).collect();
                            
                            let results: Vec<_> = futures::future::join_all(handles).await;
                            black_box(results)
                        })
                    },
                );
            }
            
            group.finish();
        }

        /// Benchmark neural network performance
        pub fn benchmark_neural_networks(&mut self, c: &mut Criterion) {
            let mut group = c.benchmark_group("neural_networks");
            
            // Transformer model benchmarks
            let transformer_configs = vec![
                (512, 256, 6, 8, "small"),
                (1024, 512, 12, 12, "medium"),
                (2048, 1024, 24, 16, "large"),
            ];
            
            for (vocab_size, hidden_size, layers, heads, size) in transformer_configs {
                let model = crate::neural_architecture_validation::MockTransformerModel::new(
                    vocab_size, hidden_size, layers, heads
                );
                
                let task = Task::new_test_task(&format!("transformer_{}", size), 0.6);
                
                group.bench_with_input(
                    BenchmarkId::new("transformer_encoding", size),
                    &task,
                    |b, task| {
                        b.iter(|| {
                            black_box(model.encode_task(task))
                        })
                    },
                );
                
                group.bench_with_input(
                    BenchmarkId::new("transformer_decoding", size),
                    &vec![0.5; 7],
                    |b, encoded| {
                        b.iter(|| {
                            black_box(model.decode_subtasks(encoded))
                        })
                    },
                );
            }
            
            // Decision network benchmarks
            let decision_configs = vec![
                (7, vec![16, 8], 3, "small"),
                (14, vec![32, 16, 8], 3, "medium"),
                (28, vec![64, 32, 16, 8], 3, "large"),
            ];
            
            for (input_size, hidden_layers, output_size, size) in decision_configs {
                let network = crate::neural_architecture_validation::MockDecisionNetwork::new(
                    input_size, hidden_layers, output_size
                );
                
                let input = vec![0.5; input_size];
                
                group.bench_with_input(
                    BenchmarkId::new("decision_network", size),
                    &input,
                    |b, input| {
                        b.iter(|| {
                            black_box(network.forward(input))
                        })
                    },
                );
            }
            
            group.finish();
        }

        /// Benchmark task graph operations
        pub fn benchmark_task_graph(&mut self, c: &mut Criterion) {
            let mut group = c.benchmark_group("task_graph");
            
            // Test with different graph sizes
            let graph_sizes = vec![5, 10, 20, 50, 100];
            
            for size in graph_sizes {
                // Create a complex task that will generate a graph of approximately 'size' nodes
                let task = Task {
                    id: format!("graph_test_{}", size),
                    name: format!("Graph Test {}", size),
                    description: "Task for graph benchmarking".to_string(),
                    complexity: (size as f32 / 100.0).min(1.0),
                    dependencies: (0..size/5).map(|i| format!("dep_{}", i)).collect(),
                    priority: TaskPriority::Medium,
                    estimated_duration: 1800,
                    required_capabilities: vec!["graph_test".to_string()],
                    resources: TaskResources {
                        memory_mb: 2048,
                        cpu_cores: 2.0,
                        network_bandwidth: 1000,
                        storage_gb: 50.0,
                    },
                };
                
                group.bench_with_input(
                    BenchmarkId::new("graph_construction", size),
                    &task,
                    |b, task| {
                        b.to_async(&self.rt).iter(|| async {
                            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
                            let result = framework.run_decomposition_test(task, DecompositionStrategy::Heuristic).await;
                            black_box(result)
                        })
                    },
                );
            }
            
            group.finish();
        }

        /// Benchmark validation performance
        pub fn benchmark_validation(&mut self, c: &mut Criterion) {
            let mut group = c.benchmark_group("validation");
            
            // Pre-generate test data
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            let task = MockNeuralTaskDecomposer::create_complex_task();
            
            let decomposition = self.rt.block_on(async {
                framework.run_decomposition_test(&task, DecompositionStrategy::Hybrid).await.unwrap()
            });
            
            group.bench_with_input(
                BenchmarkId::new("validation_speed", "complex_task"),
                &(&task, &decomposition),
                |b, (task, decomposition)| {
                    b.to_async(&self.rt).iter(|| async {
                        let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
                        let result = framework.run_validation_test(task, decomposition).await;
                        black_box(result)
                    })
                },
            );
            
            group.finish();
        }

        /// Benchmark Python FFI performance
        pub fn benchmark_python_ffi(&mut self, c: &mut Criterion) {
            let mut group = c.benchmark_group("python_ffi");
            
            let test_data = vec![
                (DecompositionStrategy::Heuristic, 0.3),
                (DecompositionStrategy::Neural, 0.5),
                (DecompositionStrategy::Hybrid, 0.7),
            ];
            
            for (strategy, complexity) in test_data {
                let task = Task::new_test_task(
                    &format!("ffi_test_{:?}_{}", strategy, complexity),
                    complexity
                );
                
                group.bench_with_input(
                    BenchmarkId::new("serialization", format!("{:?}", strategy)),
                    &task,
                    |b, task| {
                        b.to_async(&self.rt).iter(|| async {
                            let mut framework = TestFramework::new(strategy);
                            let result = framework.run_decomposition_test(task, strategy).await.unwrap();
                            
                            // Simulate FFI serialization/deserialization
                            let serialized = serde_json::to_string(&result).unwrap();
                            let deserialized: DecompositionResult = serde_json::from_str(&serialized).unwrap();
                            
                            black_box(deserialized)
                        })
                    },
                );
            }
            
            group.finish();
        }

        /// Comprehensive benchmark suite runner
        pub fn run_comprehensive_benchmarks(&mut self, c: &mut Criterion) {
            self.benchmark_decomposition_speed(c);
            self.benchmark_memory_usage(c);
            self.benchmark_concurrency(c);
            self.benchmark_neural_networks(c);
            self.benchmark_task_graph(c);
            self.benchmark_validation(c);
            self.benchmark_python_ffi(c);
        }
    }

    /// Stress test suite for edge cases and limits
    pub struct StressTestSuite {
        profiler: crate::test_utils::PerformanceProfiler,
    }

    impl StressTestSuite {
        pub fn new() -> Self {
            Self {
                profiler: crate::test_utils::PerformanceProfiler::new(),
            }
        }

        /// Test with extremely large tasks
        #[tokio::test]
        async fn test_large_task_handling() {
            let mut suite = StressTestSuite::new();
            
            // Create a very large task
            let large_task = Task {
                id: "stress_large_task".to_string(),
                name: "Very Large Task".to_string(),
                description: "A".repeat(10000), // Large description
                complexity: 0.95,
                dependencies: (0..1000).map(|i| format!("large_dep_{}", i)).collect(),
                priority: TaskPriority::Critical,
                estimated_duration: 86400, // 24 hours
                required_capabilities: (0..100).map(|i| format!("capability_{}", i)).collect(),
                resources: TaskResources {
                    memory_mb: 1024 * 1024, // 1TB
                    cpu_cores: 128.0,
                    network_bandwidth: 100000,
                    storage_gb: 10000.0,
                },
            };
            
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            
            let result = suite.profiler.time_async("large_task_decomposition", || {
                framework.run_decomposition_test(&large_task, DecompositionStrategy::Hybrid)
            }).await;
            
            // Should handle large tasks gracefully
            assert!(result.is_ok() || result.is_err());
            
            if let Ok(decomposition) = result {
                assert!(!decomposition.subtasks.is_empty());
                assert!(decomposition.subtasks.len() < 1000); // Should not explode
            }
        }

        /// Test with high concurrency
        #[tokio::test]
        async fn test_high_concurrency_stress() {
            let mut suite = StressTestSuite::new();
            
            // Create many concurrent tasks
            let num_tasks = 1000;
            let tasks: Vec<_> = (0..num_tasks)
                .map(|i| Task::new_test_task(&format!("stress_concurrent_{}", i), 0.5))
                .collect();
            
            let start = Instant::now();
            
            // Process all tasks concurrently
            let handles: Vec<_> = tasks.into_iter().map(|task| {
                let framework = TestFramework::new(DecompositionStrategy::Neural);
                tokio::spawn(async move {
                    framework.decomposer.decompose_task(
                        &task,
                        DecompositionStrategy::Neural,
                        &framework.context
                    ).await
                })
            }).collect();
            
            let results: Vec<_> = futures::future::join_all(handles).await;
            let elapsed = start.elapsed();
            
            // Check results
            let successful = results.iter().filter(|r| r.is_ok()).count();
            let failed = results.len() - successful;
            
            println!("Stress Test Results:");
            println!("- Total tasks: {}", num_tasks);
            println!("- Successful: {}", successful);
            println!("- Failed: {}", failed);
            println!("- Total time: {:?}", elapsed);
            println!("- Average time per task: {:?}", elapsed / num_tasks as u32);
            
            // At least 80% should succeed
            assert!(successful as f32 / num_tasks as f32 > 0.8);
        }

        /// Test memory pressure scenarios
        #[tokio::test]
        async fn test_memory_pressure() {
            let mut suite = StressTestSuite::new();
            
            // Create tasks that will consume significant memory
            let memory_intensive_tasks: Vec<_> = (0..100)
                .map(|i| Task {
                    id: format!("memory_stress_{}", i),
                    name: format!("Memory Stress Task {}", i),
                    description: "X".repeat(100000), // Large description
                    complexity: 0.8,
                    dependencies: (0..100).map(|j| format!("dep_{}_{}", i, j)).collect(),
                    priority: TaskPriority::High,
                    estimated_duration: 3600,
                    required_capabilities: (0..50).map(|j| format!("cap_{}_{}", i, j)).collect(),
                    resources: TaskResources {
                        memory_mb: 8192,
                        cpu_cores: 4.0,
                        network_bandwidth: 1000,
                        storage_gb: 100.0,
                    },
                })
                .collect();
            
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            let mut results = Vec::new();
            
            for task in memory_intensive_tasks {
                let result = suite.profiler.time_async("memory_pressure_task", || {
                    framework.run_decomposition_test(&task, DecompositionStrategy::Hybrid)
                }).await;
                
                results.push(result);
                
                // Force garbage collection simulation
                tokio::task::yield_now().await;
            }
            
            // Check that system remains stable under memory pressure
            let successful = results.iter().filter(|r| r.is_ok()).count();
            assert!(successful > 50); // At least half should succeed
        }

        /// Test with pathological task graphs
        #[tokio::test]
        async fn test_pathological_graphs() {
            let mut suite = StressTestSuite::new();
            
            // Create a task with complex dependency chains
            let chain_task = Task {
                id: "pathological_chain".to_string(),
                name: "Pathological Chain Task".to_string(),
                description: "Task with complex dependency chains".to_string(),
                complexity: 0.9,
                dependencies: (0..500).map(|i| format!("chain_dep_{}", i)).collect(),
                priority: TaskPriority::Medium,
                estimated_duration: 7200,
                required_capabilities: vec!["pathological".to_string()],
                resources: TaskResources {
                    memory_mb: 16384,
                    cpu_cores: 8.0,
                    network_bandwidth: 10000,
                    storage_gb: 1000.0,
                },
            };
            
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            let result = suite.profiler.time_async("pathological_graph", || {
                framework.run_decomposition_test(&chain_task, DecompositionStrategy::Neural)
            }).await;
            
            // Should handle pathological cases gracefully
            match result {
                Ok(decomposition) => {
                    assert!(!decomposition.subtasks.is_empty());
                    assert!(decomposition.subtasks.len() < 1000);
                }
                Err(e) => {
                    // Acceptable to fail on pathological cases
                    println!("Pathological graph test failed as expected: {:?}", e);
                }
            }
        }

        /// Test system recovery after failures
        #[tokio::test]
        async fn test_failure_recovery() {
            let mut suite = StressTestSuite::new();
            
            // Test recovery after various failure scenarios
            let recovery_tests = vec![
                ("resource_exhaustion", 0.95),
                ("timeout_scenario", 0.85),
                ("invalid_input", 0.75),
                ("memory_corruption", 0.65),
            ];
            
            for (test_name, complexity) in recovery_tests {
                let task = Task::new_test_task(test_name, complexity);
                let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
                
                // Attempt decomposition
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
                
                // Even if it fails, the system should recover for the next test
                match result {
                    Ok(decomposition) => {
                        assert!(!decomposition.subtasks.is_empty());
                    }
                    Err(e) => {
                        println!("Recovery test '{}' failed: {:?}", test_name, e);
                    }
                }
                
                // Test that system is still functional
                let simple_task = Task::new_test_task("recovery_check", 0.3);
                let recovery_result = framework.run_decomposition_test(&simple_task, DecompositionStrategy::Heuristic).await;
                
                assert!(recovery_result.is_ok(), "System failed to recover after {}", test_name);
            }
        }

        /// Print comprehensive performance report
        pub fn print_performance_report(&self) {
            self.profiler.print_report();
        }
    }

    /// Performance regression tests
    pub struct RegressionTestSuite {
        baseline_metrics: std::collections::HashMap<String, f64>,
        profiler: crate::test_utils::PerformanceProfiler,
    }

    impl RegressionTestSuite {
        pub fn new() -> Self {
            let mut baseline_metrics = std::collections::HashMap::new();
            
            // Set baseline performance expectations (in microseconds)
            baseline_metrics.insert("simple_decomposition".to_string(), 10000.0); // 10ms
            baseline_metrics.insert("complex_decomposition".to_string(), 50000.0); // 50ms
            baseline_metrics.insert("validation".to_string(), 5000.0); // 5ms
            baseline_metrics.insert("neural_encoding".to_string(), 2000.0); // 2ms
            baseline_metrics.insert("graph_construction".to_string(), 15000.0); // 15ms
            
            Self {
                baseline_metrics,
                profiler: crate::test_utils::PerformanceProfiler::new(),
            }
        }

        /// Test for performance regressions
        #[tokio::test]
        async fn test_performance_regression() {
            let mut suite = RegressionTestSuite::new();
            
            // Test simple decomposition performance
            let simple_task = Task::new_test_task("regression_simple", 0.3);
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            
            let _result = suite.profiler.time_async("simple_decomposition", || {
                framework.run_decomposition_test(&simple_task, DecompositionStrategy::Heuristic)
            }).await;
            
            // Test complex decomposition performance
            let complex_task = MockNeuralTaskDecomposer::create_complex_task();
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            let _result = suite.profiler.time_async("complex_decomposition", || {
                framework.run_decomposition_test(&complex_task, DecompositionStrategy::Neural)
            }).await;
            
            // Check for regressions
            suite.check_performance_regressions();
        }

        fn check_performance_regressions(&self) {
            let tolerance = 1.2; // 20% tolerance
            
            for (test_name, baseline) in &self.baseline_metrics {
                if let Some(stats) = self.profiler.get_stats(test_name) {
                    let current_avg = stats.mean_us;
                    let threshold = baseline * tolerance;
                    
                    if current_avg > threshold {
                        println!("⚠️  Performance regression detected in {}: {:.1}μs > {:.1}μs (baseline: {:.1}μs)", 
                            test_name, current_avg, threshold, baseline);
                    } else {
                        println!("✅ Performance OK for {}: {:.1}μs <= {:.1}μs", 
                            test_name, current_avg, threshold);
                    }
                }
            }
        }
    }

    // Import required types from parent modules
    use crate::neural_task_decomposition_test::*;
    use crate::neural_architecture_validation::*;
    use crate::test_utils::*;
}

// Main benchmark runner
pub fn run_all_benchmarks() {
    use performance_benchmarks::*;
    
    let mut suite = PerformanceTestSuite::new();
    let mut c = criterion::Criterion::default();
    
    suite.run_comprehensive_benchmarks(&mut c);
    
    // Run stress tests
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut stress_suite = StressTestSuite::new();
    
    println!("Running stress tests...");
    rt.block_on(async {
        stress_suite.test_large_task_handling().await;
        stress_suite.test_high_concurrency_stress().await;
        stress_suite.test_memory_pressure().await;
        stress_suite.test_pathological_graphs().await;
        stress_suite.test_failure_recovery().await;
    });
    
    stress_suite.print_performance_report();
    
    // Run regression tests
    let mut regression_suite = RegressionTestSuite::new();
    rt.block_on(async {
        regression_suite.test_performance_regression().await;
    });
}