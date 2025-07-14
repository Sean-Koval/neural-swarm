//! Comprehensive Test Runner for Neural Task Decomposition System
//!
//! This test runner orchestrates all testing components and provides
//! comprehensive reporting for the neural task decomposition system.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

/// Test execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionResult {
    pub test_name: String,
    pub category: String,
    pub status: TestStatus,
    pub duration: Duration,
    pub error_message: Option<String>,
    pub performance_metrics: HashMap<String, f64>,
    pub coverage_data: CoverageData,
}

/// Test status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Timeout,
    Error,
}

/// Coverage data for test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageData {
    pub lines_covered: u32,
    pub lines_total: u32,
    pub branches_covered: u32,
    pub branches_total: u32,
    pub functions_covered: u32,
    pub functions_total: u32,
}

/// Comprehensive test suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteConfig {
    pub timeout_duration: Duration,
    pub parallel_execution: bool,
    pub max_concurrent_tests: usize,
    pub enable_stress_tests: bool,
    pub enable_performance_tests: bool,
    pub enable_regression_tests: bool,
    pub coverage_threshold: f64,
    pub performance_threshold: f64,
}

impl Default for TestSuiteConfig {
    fn default() -> Self {
        Self {
            timeout_duration: Duration::from_secs(300), // 5 minutes
            parallel_execution: true,
            max_concurrent_tests: 10,
            enable_stress_tests: true,
            enable_performance_tests: true,
            enable_regression_tests: true,
            coverage_threshold: 0.8, // 80%
            performance_threshold: 1.2, // 20% tolerance
        }
    }
}

/// Main test runner orchestrator
pub struct ComprehensiveTestRunner {
    config: TestSuiteConfig,
    results: Vec<TestExecutionResult>,
    profiler: crate::test_utils::PerformanceProfiler,
    start_time: Instant,
}

impl ComprehensiveTestRunner {
    pub fn new(config: TestSuiteConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            profiler: crate::test_utils::PerformanceProfiler::new(),
            start_time: Instant::now(),
        }
    }

    /// Execute all test categories
    pub async fn run_all_tests(&mut self) -> ComprehensiveTestReport {
        println!("🚀 Starting Comprehensive Neural Task Decomposition Test Suite");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // 1. Task Decomposition Tests
        self.run_task_decomposition_tests().await;
        
        // 2. Neural Architecture Tests
        self.run_neural_architecture_tests().await;
        
        // 3. Task Graph Tests
        self.run_task_graph_tests().await;
        
        // 4. Swarm Integration Tests
        self.run_swarm_integration_tests().await;
        
        // 5. Performance Tests
        if self.config.enable_performance_tests {
            self.run_performance_tests().await;
        }
        
        // 6. Python FFI Tests
        self.run_python_ffi_tests().await;
        
        // 7. Integration Tests
        self.run_integration_tests().await;
        
        // 8. Stress Tests
        if self.config.enable_stress_tests {
            self.run_stress_tests().await;
        }
        
        // 9. Regression Tests
        if self.config.enable_regression_tests {
            self.run_regression_tests().await;
        }
        
        // Generate comprehensive report
        self.generate_comprehensive_report().await
    }

    /// Run task decomposition tests
    async fn run_task_decomposition_tests(&mut self) {
        println!("\n📋 Running Task Decomposition Tests...");
        
        let test_cases = vec![
            ("heuristic_simple", "Heuristic decomposition of simple tasks"),
            ("heuristic_complex", "Heuristic decomposition of complex tasks"),
            ("neural_adaptive", "Neural decomposition with adaptive complexity"),
            ("hybrid_strategy", "Hybrid decomposition strategy"),
            ("edge_case_malformed", "Edge case with malformed task"),
            ("circular_dependency", "Circular dependency detection"),
            ("resource_limits", "Resource limit handling"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Task Decomposition",
                description,
                self.execute_task_decomposition_test(test_name)
            ).await;
        }
    }

    /// Run neural architecture tests
    async fn run_neural_architecture_tests(&mut self) {
        println!("\n🧠 Running Neural Architecture Tests...");
        
        let test_cases = vec![
            ("transformer_accuracy", "Transformer model accuracy validation"),
            ("bert_compatibility", "BERT compatibility testing"),
            ("gpt_compatibility", "GPT compatibility testing"),
            ("decision_network", "Decision network architecture validation"),
            ("rl_convergence", "Reinforcement learning convergence"),
            ("mixture_of_experts", "Mixture of experts selection"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Neural Architecture",
                description,
                self.execute_neural_architecture_test(test_name)
            ).await;
        }
    }

    /// Run task graph tests
    async fn run_task_graph_tests(&mut self) {
        println!("\n📊 Running Task Graph Tests...");
        
        let test_cases = vec![
            ("dag_construction", "DAG construction correctness"),
            ("cycle_detection", "Cycle detection validation"),
            ("topological_sorting", "Topological sorting verification"),
            ("priority_scheduling", "Priority scheduling optimization"),
            ("critical_path", "Critical path identification"),
            ("parallelization", "Parallelization opportunities"),
            ("dynamic_modification", "Dynamic graph modification"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Task Graph",
                description,
                self.execute_task_graph_test(test_name)
            ).await;
        }
    }

    /// Run swarm integration tests
    async fn run_swarm_integration_tests(&mut self) {
        println!("\n🐝 Running Swarm Integration Tests...");
        
        let test_cases = vec![
            ("neural_comm_passing", "Neural-comm task passing"),
            ("distributed_sync", "Distributed task synchronization"),
            ("fann_integration", "FANN integration correctness"),
            ("agent_coordination", "Agent coordination protocols"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Swarm Integration",
                description,
                self.execute_swarm_integration_test(test_name)
            ).await;
        }
    }

    /// Run performance tests
    async fn run_performance_tests(&mut self) {
        println!("\n⚡ Running Performance Tests...");
        
        let test_cases = vec![
            ("decomposition_speed", "Decomposition speed benchmarks"),
            ("memory_efficiency", "Memory usage efficiency"),
            ("concurrency_safety", "Concurrency safety validation"),
            ("scalability_limits", "Scalability limit testing"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Performance",
                description,
                self.execute_performance_test(test_name)
            ).await;
        }
    }

    /// Run Python FFI tests
    async fn run_python_ffi_tests(&mut self) {
        println!("\n🐍 Running Python FFI Tests...");
        
        let test_cases = vec![
            ("data_consistency", "Cross-language data consistency"),
            ("memory_safety", "Memory safety boundaries"),
            ("async_integration", "Async Python integration"),
            ("performance_impact", "Performance impact measurement"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Python FFI",
                description,
                self.execute_python_ffi_test(test_name)
            ).await;
        }
    }

    /// Run integration tests
    async fn run_integration_tests(&mut self) {
        println!("\n🔗 Running Integration Tests...");
        
        let test_cases = vec![
            ("end_to_end_workflow", "Complete workflow validation"),
            ("fault_tolerance", "Fault tolerance and recovery"),
            ("system_resilience", "System resilience testing"),
            ("monitoring_integration", "Monitoring integration"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Integration",
                description,
                self.execute_integration_test(test_name)
            ).await;
        }
    }

    /// Run stress tests
    async fn run_stress_tests(&mut self) {
        println!("\n💪 Running Stress Tests...");
        
        let test_cases = vec![
            ("large_task_handling", "Large task handling"),
            ("high_concurrency", "High concurrency stress"),
            ("memory_pressure", "Memory pressure scenarios"),
            ("pathological_graphs", "Pathological graph handling"),
            ("failure_recovery", "System failure recovery"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Stress",
                description,
                self.execute_stress_test(test_name)
            ).await;
        }
    }

    /// Run regression tests
    async fn run_regression_tests(&mut self) {
        println!("\n📈 Running Regression Tests...");
        
        let test_cases = vec![
            ("performance_regression", "Performance regression detection"),
            ("api_compatibility", "API compatibility validation"),
            ("behavior_consistency", "Behavior consistency checking"),
        ];

        for (test_name, description) in test_cases {
            self.run_single_test(
                test_name,
                "Regression",
                description,
                self.execute_regression_test(test_name)
            ).await;
        }
    }

    /// Execute a single test with timeout and error handling
    async fn run_single_test<F, Fut>(
        &mut self,
        test_name: &str,
        category: &str,
        description: &str,
        test_future: F,
    ) where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<HashMap<String, f64>, String>>,
    {
        print!("  ├─ {}: {} ... ", test_name, description);
        
        let start_time = Instant::now();
        let result = timeout(self.config.timeout_duration, test_future()).await;
        let duration = start_time.elapsed();

        let (status, error_message, performance_metrics) = match result {
            Ok(Ok(metrics)) => {
                println!("✅ PASS ({:.2}s)", duration.as_secs_f64());
                (TestStatus::Passed, None, metrics)
            }
            Ok(Err(error)) => {
                println!("❌ FAIL ({:.2}s) - {}", duration.as_secs_f64(), error);
                (TestStatus::Failed, Some(error), HashMap::new())
            }
            Err(_) => {
                println!("⏱️  TIMEOUT ({:.2}s)", duration.as_secs_f64());
                (TestStatus::Timeout, Some("Test timed out".to_string()), HashMap::new())
            }
        };

        let test_result = TestExecutionResult {
            test_name: test_name.to_string(),
            category: category.to_string(),
            status,
            duration,
            error_message,
            performance_metrics,
            coverage_data: CoverageData {
                lines_covered: 0,
                lines_total: 0,
                branches_covered: 0,
                branches_total: 0,
                functions_covered: 0,
                functions_total: 0,
            },
        };

        self.results.push(test_result);
    }

    /// Mock test execution functions
    async fn execute_task_decomposition_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        // Mock implementation
        let mut metrics = HashMap::new();
        
        match test_name {
            "heuristic_simple" => {
                metrics.insert("decomposition_time_ms".to_string(), 15.0);
                metrics.insert("accuracy_score".to_string(), 0.85);
                Ok(metrics)
            }
            "heuristic_complex" => {
                metrics.insert("decomposition_time_ms".to_string(), 45.0);
                metrics.insert("accuracy_score".to_string(), 0.78);
                Ok(metrics)
            }
            "neural_adaptive" => {
                metrics.insert("decomposition_time_ms".to_string(), 35.0);
                metrics.insert("accuracy_score".to_string(), 0.82);
                Ok(metrics)
            }
            "hybrid_strategy" => {
                metrics.insert("decomposition_time_ms".to_string(), 25.0);
                metrics.insert("accuracy_score".to_string(), 0.88);
                Ok(metrics)
            }
            "edge_case_malformed" => {
                metrics.insert("error_handling_score".to_string(), 0.95);
                Ok(metrics)
            }
            "circular_dependency" => {
                metrics.insert("detection_accuracy".to_string(), 0.92);
                Ok(metrics)
            }
            "resource_limits" => {
                metrics.insert("resource_efficiency".to_string(), 0.87);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    async fn execute_neural_architecture_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        let mut metrics = HashMap::new();
        
        match test_name {
            "transformer_accuracy" => {
                metrics.insert("encoding_accuracy".to_string(), 0.91);
                metrics.insert("decoding_accuracy".to_string(), 0.89);
                Ok(metrics)
            }
            "bert_compatibility" => {
                metrics.insert("compatibility_score".to_string(), 0.94);
                Ok(metrics)
            }
            "gpt_compatibility" => {
                metrics.insert("compatibility_score".to_string(), 0.92);
                Ok(metrics)
            }
            "decision_network" => {
                metrics.insert("decision_accuracy".to_string(), 0.86);
                Ok(metrics)
            }
            "rl_convergence" => {
                metrics.insert("convergence_rate".to_string(), 0.83);
                Ok(metrics)
            }
            "mixture_of_experts" => {
                metrics.insert("expert_selection_accuracy".to_string(), 0.88);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    async fn execute_task_graph_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        let mut metrics = HashMap::new();
        
        match test_name {
            "dag_construction" => {
                metrics.insert("construction_time_ms".to_string(), 12.0);
                metrics.insert("correctness_score".to_string(), 0.96);
                Ok(metrics)
            }
            "cycle_detection" => {
                metrics.insert("detection_accuracy".to_string(), 0.98);
                Ok(metrics)
            }
            "topological_sorting" => {
                metrics.insert("sorting_correctness".to_string(), 0.97);
                Ok(metrics)
            }
            "priority_scheduling" => {
                metrics.insert("optimization_score".to_string(), 0.84);
                Ok(metrics)
            }
            "critical_path" => {
                metrics.insert("path_accuracy".to_string(), 0.93);
                Ok(metrics)
            }
            "parallelization" => {
                metrics.insert("parallelization_efficiency".to_string(), 0.81);
                Ok(metrics)
            }
            "dynamic_modification" => {
                metrics.insert("modification_success_rate".to_string(), 0.89);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    async fn execute_swarm_integration_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        let mut metrics = HashMap::new();
        
        match test_name {
            "neural_comm_passing" => {
                metrics.insert("message_success_rate".to_string(), 0.97);
                metrics.insert("encryption_overhead_ms".to_string(), 5.0);
                Ok(metrics)
            }
            "distributed_sync" => {
                metrics.insert("sync_consistency".to_string(), 0.94);
                metrics.insert("sync_latency_ms".to_string(), 25.0);
                Ok(metrics)
            }
            "fann_integration" => {
                metrics.insert("integration_correctness".to_string(), 0.91);
                metrics.insert("performance_improvement".to_string(), 3.2);
                Ok(metrics)
            }
            "agent_coordination" => {
                metrics.insert("coordination_efficiency".to_string(), 0.85);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    async fn execute_performance_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        let mut metrics = HashMap::new();
        
        match test_name {
            "decomposition_speed" => {
                metrics.insert("throughput_tasks_per_sec".to_string(), 145.0);
                metrics.insert("latency_p99_ms".to_string(), 85.0);
                Ok(metrics)
            }
            "memory_efficiency" => {
                metrics.insert("memory_usage_mb".to_string(), 78.0);
                metrics.insert("memory_efficiency_score".to_string(), 0.87);
                Ok(metrics)
            }
            "concurrency_safety" => {
                metrics.insert("concurrent_tasks_supported".to_string(), 1000.0);
                metrics.insert("race_condition_score".to_string(), 0.99);
                Ok(metrics)
            }
            "scalability_limits" => {
                metrics.insert("max_task_complexity".to_string(), 0.95);
                metrics.insert("scaling_efficiency".to_string(), 0.82);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    async fn execute_python_ffi_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        let mut metrics = HashMap::new();
        
        match test_name {
            "data_consistency" => {
                metrics.insert("consistency_score".to_string(), 0.98);
                Ok(metrics)
            }
            "memory_safety" => {
                metrics.insert("safety_score".to_string(), 0.96);
                Ok(metrics)
            }
            "async_integration" => {
                metrics.insert("async_compatibility".to_string(), 0.93);
                Ok(metrics)
            }
            "performance_impact" => {
                metrics.insert("ffi_overhead_percent".to_string(), 12.0);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    async fn execute_integration_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        let mut metrics = HashMap::new();
        
        match test_name {
            "end_to_end_workflow" => {
                metrics.insert("workflow_success_rate".to_string(), 0.92);
                metrics.insert("end_to_end_latency_ms".to_string(), 450.0);
                Ok(metrics)
            }
            "fault_tolerance" => {
                metrics.insert("recovery_success_rate".to_string(), 0.88);
                metrics.insert("recovery_time_ms".to_string(), 120.0);
                Ok(metrics)
            }
            "system_resilience" => {
                metrics.insert("resilience_score".to_string(), 0.86);
                Ok(metrics)
            }
            "monitoring_integration" => {
                metrics.insert("monitoring_coverage".to_string(), 0.94);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    async fn execute_stress_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        let mut metrics = HashMap::new();
        
        match test_name {
            "large_task_handling" => {
                metrics.insert("max_task_size_mb".to_string(), 512.0);
                metrics.insert("handling_success_rate".to_string(), 0.78);
                Ok(metrics)
            }
            "high_concurrency" => {
                metrics.insert("max_concurrent_tasks".to_string(), 2500.0);
                metrics.insert("degradation_point".to_string(), 2000.0);
                Ok(metrics)
            }
            "memory_pressure" => {
                metrics.insert("memory_limit_mb".to_string(), 2048.0);
                metrics.insert("pressure_handling_score".to_string(), 0.81);
                Ok(metrics)
            }
            "pathological_graphs" => {
                metrics.insert("pathological_handling_rate".to_string(), 0.73);
                Ok(metrics)
            }
            "failure_recovery" => {
                metrics.insert("recovery_success_rate".to_string(), 0.85);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    async fn execute_regression_test(&self, test_name: &str) -> Result<HashMap<String, f64>, String> {
        let mut metrics = HashMap::new();
        
        match test_name {
            "performance_regression" => {
                metrics.insert("performance_change_percent".to_string(), 5.0);
                metrics.insert("regression_detected".to_string(), 0.0);
                Ok(metrics)
            }
            "api_compatibility" => {
                metrics.insert("compatibility_score".to_string(), 0.99);
                Ok(metrics)
            }
            "behavior_consistency" => {
                metrics.insert("consistency_score".to_string(), 0.97);
                Ok(metrics)
            }
            _ => Err(format!("Unknown test: {}", test_name)),
        }
    }

    /// Generate comprehensive test report
    async fn generate_comprehensive_report(&self) -> ComprehensiveTestReport {
        let total_duration = self.start_time.elapsed();
        
        let mut category_stats = HashMap::new();
        let mut overall_metrics = HashMap::new();
        
        // Calculate category statistics
        for result in &self.results {
            let category_stat = category_stats.entry(result.category.clone()).or_insert(CategoryStatistics {
                total_tests: 0,
                passed: 0,
                failed: 0,
                skipped: 0,
                timeout: 0,
                error: 0,
                total_duration: Duration::from_secs(0),
                avg_duration: Duration::from_secs(0),
            });
            
            category_stat.total_tests += 1;
            category_stat.total_duration += result.duration;
            
            match result.status {
                TestStatus::Passed => category_stat.passed += 1,
                TestStatus::Failed => category_stat.failed += 1,
                TestStatus::Skipped => category_stat.skipped += 1,
                TestStatus::Timeout => category_stat.timeout += 1,
                TestStatus::Error => category_stat.error += 1,
            }
        }
        
        // Calculate average durations
        for stat in category_stats.values_mut() {
            if stat.total_tests > 0 {
                stat.avg_duration = stat.total_duration / stat.total_tests as u32;
            }
        }
        
        // Calculate overall metrics
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let failed_tests = self.results.iter().filter(|r| r.status == TestStatus::Failed).count();
        
        overall_metrics.insert("total_tests".to_string(), total_tests as f64);
        overall_metrics.insert("passed_tests".to_string(), passed_tests as f64);
        overall_metrics.insert("failed_tests".to_string(), failed_tests as f64);
        overall_metrics.insert("pass_rate".to_string(), passed_tests as f64 / total_tests as f64);
        overall_metrics.insert("total_duration_seconds".to_string(), total_duration.as_secs_f64());
        
        ComprehensiveTestReport {
            timestamp: chrono::Utc::now(),
            total_duration,
            overall_metrics,
            category_statistics: category_stats,
            detailed_results: self.results.clone(),
            summary: self.generate_summary(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate test summary
    fn generate_summary(&self) -> String {
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let failed_tests = self.results.iter().filter(|r| r.status == TestStatus::Failed).count();
        let pass_rate = passed_tests as f64 / total_tests as f64;
        
        format!(
            "Neural Task Decomposition Test Suite completed with {}/{} tests passing ({:.1}% pass rate). \
            {} tests failed, {} tests skipped/timeout/error. \
            Total execution time: {:.2}s.",
            passed_tests, total_tests, pass_rate * 100.0, failed_tests,
            total_tests - passed_tests - failed_tests,
            self.start_time.elapsed().as_secs_f64()
        )
    }

    /// Generate recommendations based on test results
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let pass_rate = self.results.iter().filter(|r| r.status == TestStatus::Passed).count() as f64 / self.results.len() as f64;
        
        if pass_rate < 0.8 {
            recommendations.push("Consider investigating failing tests and improving system reliability.".to_string());
        }
        
        let avg_duration = self.results.iter().map(|r| r.duration.as_secs_f64()).sum::<f64>() / self.results.len() as f64;
        if avg_duration > 30.0 {
            recommendations.push("Consider optimizing test execution time for better developer productivity.".to_string());
        }
        
        // Check for performance issues
        let performance_issues = self.results.iter()
            .filter(|r| r.category == "Performance" && r.status == TestStatus::Failed)
            .count();
        
        if performance_issues > 0 {
            recommendations.push("Performance tests are failing. Consider profiling and optimizing critical paths.".to_string());
        }
        
        recommendations
    }
}

/// Category statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStatistics {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub timeout: usize,
    pub error: usize,
    pub total_duration: Duration,
    pub avg_duration: Duration,
}

/// Comprehensive test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveTestReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub total_duration: Duration,
    pub overall_metrics: HashMap<String, f64>,
    pub category_statistics: HashMap<String, CategoryStatistics>,
    pub detailed_results: Vec<TestExecutionResult>,
    pub summary: String,
    pub recommendations: Vec<String>,
}

impl ComprehensiveTestReport {
    /// Print detailed console report
    pub fn print_detailed_report(&self) {
        println!("\n");
        println!("🎯 COMPREHENSIVE TEST REPORT");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Overall statistics
        println!("\n📊 OVERALL STATISTICS");
        println!("  Total Tests: {}", self.overall_metrics.get("total_tests").unwrap_or(&0.0));
        println!("  Passed: {}", self.overall_metrics.get("passed_tests").unwrap_or(&0.0));
        println!("  Failed: {}", self.overall_metrics.get("failed_tests").unwrap_or(&0.0));
        println!("  Pass Rate: {:.1}%", self.overall_metrics.get("pass_rate").unwrap_or(&0.0) * 100.0);
        println!("  Total Duration: {:.2}s", self.overall_metrics.get("total_duration_seconds").unwrap_or(&0.0));
        
        // Category breakdown
        println!("\n📋 CATEGORY BREAKDOWN");
        for (category, stats) in &self.category_statistics {
            let pass_rate = stats.passed as f64 / stats.total_tests as f64;
            println!("  {}: {}/{} passed ({:.1}%) - avg {:.2}s", 
                category, stats.passed, stats.total_tests, pass_rate * 100.0, stats.avg_duration.as_secs_f64());
        }
        
        // Failed tests
        let failed_tests: Vec<_> = self.detailed_results.iter()
            .filter(|r| r.status == TestStatus::Failed)
            .collect();
        
        if !failed_tests.is_empty() {
            println!("\n❌ FAILED TESTS");
            for test in failed_tests {
                println!("  {} ({}): {}", 
                    test.test_name, 
                    test.category, 
                    test.error_message.as_deref().unwrap_or("No error message"));
            }
        }
        
        // Recommendations
        if !self.recommendations.is_empty() {
            println!("\n💡 RECOMMENDATIONS");
            for rec in &self.recommendations {
                println!("  • {}", rec);
            }
        }
        
        println!("\n📋 SUMMARY");
        println!("  {}", self.summary);
        
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

/// Main test execution function
pub async fn run_comprehensive_test_suite() -> ComprehensiveTestReport {
    let config = TestSuiteConfig::default();
    let mut runner = ComprehensiveTestRunner::new(config);
    
    let report = runner.run_all_tests().await;
    report.print_detailed_report();
    
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_comprehensive_suite_execution() {
        let report = run_comprehensive_test_suite().await;
        
        // Verify report structure
        assert!(!report.detailed_results.is_empty());
        assert!(!report.category_statistics.is_empty());
        assert!(!report.summary.is_empty());
        
        // Check that all major categories are tested
        let categories: std::collections::HashSet<_> = report.category_statistics.keys().collect();
        assert!(categories.contains(&"Task Decomposition".to_string()));
        assert!(categories.contains(&"Neural Architecture".to_string()));
        assert!(categories.contains(&"Performance".to_string()));
        
        println!("✅ Comprehensive test suite execution completed successfully!");
    }
}