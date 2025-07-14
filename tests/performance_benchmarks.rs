//! Performance Benchmarks for Distributed Neural Swarm
//!
//! This module provides comprehensive performance testing and benchmarking
//! for distributed neural network operations, consensus protocols, and
//! communication patterns under various load conditions.

use neural_swarm::{
    coordination::{SwarmCoordinator, CoordinationStrategy},
    communication::{Message, MessageType},
    memory::{MemoryManager, AgentMemory},
    agents::AgentId,
    NeuralNetwork, NeuralFloat,
    activation::ActivationType,
    network::{LayerConfig, NetworkBuilder},
    training::{TrainingData, TrainingParams, TrainingAlgorithm},
};
use neural_comm::{
    SecureChannel, ChannelConfig, CipherSuite, KeyPair,
    Message as CommMessage, MessageType as CommMessageType,
    memory::{SecureBuffer, SecureMemoryPool},
};
use std::{
    sync::{Arc, RwLock, Mutex, Barrier},
    thread,
    time::{Duration, Instant, SystemTime},
    collections::{HashMap, VecDeque},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};
use tokio::{
    runtime::Runtime,
    sync::{oneshot, broadcast, Semaphore},
    time::{sleep, interval},
};
use rand::{thread_rng, Rng, RngCore};
use serde::{Serialize, Deserialize};

/// Performance test configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Number of nodes in the test
    pub node_count: usize,
    /// Test duration in seconds
    pub duration_seconds: u64,
    /// Target throughput (operations/second)
    pub target_throughput: u64,
    /// Concurrent operations per node
    pub concurrent_ops: usize,
    /// Memory limit per node (bytes)
    pub memory_limit: usize,
    /// Network bandwidth limit (bytes/second)
    pub bandwidth_limit: u64,
    /// CPU limit (percentage)
    pub cpu_limit: f64,
    /// Enable stress testing
    pub stress_testing: bool,
    /// Enable memory pressure testing
    pub memory_pressure: bool,
    /// Enable latency testing
    pub latency_testing: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            node_count: 10,
            duration_seconds: 60,
            target_throughput: 10000,
            concurrent_ops: 100,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            bandwidth_limit: 100 * 1024 * 1024, // 100MB/s
            cpu_limit: 80.0,
            stress_testing: true,
            memory_pressure: true,
            latency_testing: true,
        }
    }
}

/// Performance metrics collector
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total operations processed
    pub total_operations: AtomicU64,
    /// Total time spent processing
    pub total_processing_time: AtomicU64,
    /// Peak memory usage
    pub peak_memory_usage: AtomicU64,
    /// Average latency (microseconds)
    pub average_latency: AtomicU64,
    /// Throughput (operations/second)
    pub throughput: AtomicU64,
    /// Error count
    pub error_count: AtomicU64,
    /// Network bytes sent
    pub network_bytes_sent: AtomicU64,
    /// Network bytes received
    pub network_bytes_received: AtomicU64,
    /// CPU usage percentage
    pub cpu_usage: AtomicU64,
    /// Memory allocations
    pub memory_allocations: AtomicU64,
    /// GC pauses (microseconds)
    pub gc_pause_time: AtomicU64,
    /// Consensus rounds
    pub consensus_rounds: AtomicU64,
    /// Failed consensus attempts
    pub failed_consensus: AtomicU64,
    /// Start time
    pub start_time: Instant,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_operations: AtomicU64::new(0),
            total_processing_time: AtomicU64::new(0),
            peak_memory_usage: AtomicU64::new(0),
            average_latency: AtomicU64::new(0),
            throughput: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            network_bytes_sent: AtomicU64::new(0),
            network_bytes_received: AtomicU64::new(0),
            cpu_usage: AtomicU64::new(0),
            memory_allocations: AtomicU64::new(0),
            gc_pause_time: AtomicU64::new(0),
            consensus_rounds: AtomicU64::new(0),
            failed_consensus: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn record_operation(&self, processing_time: Duration) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time.fetch_add(processing_time.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_network_traffic(&self, bytes_sent: u64, bytes_received: u64) {
        self.network_bytes_sent.fetch_add(bytes_sent, Ordering::Relaxed);
        self.network_bytes_received.fetch_add(bytes_received, Ordering::Relaxed);
    }

    pub fn record_memory_usage(&self, usage: u64) {
        let current_peak = self.peak_memory_usage.load(Ordering::Relaxed);
        if usage > current_peak {
            self.peak_memory_usage.store(usage, Ordering::Relaxed);
        }
    }

    pub fn record_consensus_round(&self, success: bool) {
        self.consensus_rounds.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.failed_consensus.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn calculate_throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let ops = self.total_operations.load(Ordering::Relaxed) as f64;
        if elapsed > 0.0 {
            ops / elapsed
        } else {
            0.0
        }
    }

    pub fn calculate_average_latency(&self) -> Duration {
        let total_time = self.total_processing_time.load(Ordering::Relaxed);
        let total_ops = self.total_operations.load(Ordering::Relaxed);
        if total_ops > 0 {
            Duration::from_micros(total_time / total_ops)
        } else {
            Duration::from_secs(0)
        }
    }

    pub fn error_rate(&self) -> f64 {
        let errors = self.error_count.load(Ordering::Relaxed) as f64;
        let total = self.total_operations.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            errors / total
        } else {
            0.0
        }
    }

    pub fn consensus_success_rate(&self) -> f64 {
        let total = self.consensus_rounds.load(Ordering::Relaxed) as f64;
        let failed = self.failed_consensus.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            (total - failed) / total
        } else {
            0.0
        }
    }
}

/// Performance test node
#[derive(Debug)]
pub struct PerformanceTestNode {
    pub id: AgentId,
    pub neural_network: Arc<RwLock<NeuralNetwork>>,
    pub memory_manager: Arc<RwLock<MemoryManager>>,
    pub secure_channel: Arc<RwLock<SecureChannel>>,
    pub metrics: Arc<PerformanceMetrics>,
    pub memory_pool: Arc<Mutex<SecureMemoryPool>>,
    pub coordinator: Arc<SwarmCoordinator>,
    pub workload_generator: Arc<WorkloadGenerator>,
    pub resource_monitor: Arc<ResourceMonitor>,
}

impl PerformanceTestNode {
    pub async fn new(
        id: AgentId,
        strategy: CoordinationStrategy,
        config: &PerformanceConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create neural network
        let layer_configs = vec![
            LayerConfig::new(100, ActivationType::Linear),
            LayerConfig::new(200, ActivationType::ReLU),
            LayerConfig::new(100, ActivationType::Sigmoid),
            LayerConfig::new(50, ActivationType::Tanh),
            LayerConfig::new(10, ActivationType::Linear),
        ];
        let mut network = NeuralNetwork::new_feedforward(&layer_configs)?;
        network.initialize_weights(Some(42))?;

        // Create secure channel
        let keypair = KeyPair::generate()?;
        let channel_config = ChannelConfig::new()
            .cipher_suite(CipherSuite::ChaCha20Poly1305)
            .enable_forward_secrecy(true)
            .message_timeout(30);
        let channel = SecureChannel::new(channel_config, keypair).await?;

        Ok(Self {
            id,
            neural_network: Arc::new(RwLock::new(network)),
            memory_manager: Arc::new(RwLock::new(MemoryManager::new())),
            secure_channel: Arc::new(RwLock::new(channel)),
            metrics: Arc::new(PerformanceMetrics::new()),
            memory_pool: Arc::new(Mutex::new(SecureMemoryPool::new(100))),
            coordinator: Arc::new(SwarmCoordinator {
                agents: vec![id],
                strategy,
            }),
            workload_generator: Arc::new(WorkloadGenerator::new(config.clone())),
            resource_monitor: Arc::new(ResourceMonitor::new()),
        })
    }

    /// Run performance benchmark
    pub async fn run_benchmark(&self, config: &PerformanceConfig) -> NodeBenchmarkResult {
        let mut result = NodeBenchmarkResult::new(self.id);
        
        // Start resource monitoring
        let monitor_handle = self.resource_monitor.start_monitoring(self.metrics.clone());
        
        // Create workload tasks
        let mut handles = Vec::new();
        
        for _ in 0..config.concurrent_ops {
            let node_clone = self.clone();
            let config_clone = config.clone();
            
            let handle = tokio::spawn(async move {
                node_clone.run_workload_task(config_clone).await
            });
            
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            if let Err(e) = handle.await {
                result.errors.push(format!("Task error: {}", e));
            }
        }

        // Stop monitoring
        self.resource_monitor.stop_monitoring().await;

        // Calculate final metrics
        result.throughput = self.metrics.calculate_throughput();
        result.average_latency = self.metrics.calculate_average_latency();
        result.error_rate = self.metrics.error_rate();
        result.peak_memory_usage = self.metrics.peak_memory_usage.load(Ordering::Relaxed);
        result.total_operations = self.metrics.total_operations.load(Ordering::Relaxed);
        result.network_bytes_sent = self.metrics.network_bytes_sent.load(Ordering::Relaxed);
        result.network_bytes_received = self.metrics.network_bytes_received.load(Ordering::Relaxed);
        result.consensus_success_rate = self.metrics.consensus_success_rate();
        
        result
    }

    /// Run individual workload task
    async fn run_workload_task(&self, config: PerformanceConfig) -> Result<(), Box<dyn std::error::Error>> {
        let end_time = Instant::now() + Duration::from_secs(config.duration_seconds);
        
        while Instant::now() < end_time {
            // Generate workload
            let workload = self.workload_generator.generate_workload().await;
            
            // Process workload
            let start_time = Instant::now();
            let result = self.process_workload(workload).await;
            let processing_time = start_time.elapsed();
            
            // Record metrics
            self.metrics.record_operation(processing_time);
            if result.is_err() {
                self.metrics.record_error();
            }
            
            // Memory pressure simulation
            if config.memory_pressure {
                self.simulate_memory_pressure().await;
            }
            
            // Rate limiting
            let target_interval = Duration::from_secs(1) / config.target_throughput as u32;
            if processing_time < target_interval {
                sleep(target_interval - processing_time).await;
            }
        }
        
        Ok(())
    }

    /// Process workload item
    async fn process_workload(&self, workload: WorkloadItem) -> Result<(), Box<dyn std::error::Error>> {
        match workload {
            WorkloadItem::NeuralInference { input } => {
                let network = self.neural_network.read().unwrap();
                let _output = network.predict(&input)?;
            }
            WorkloadItem::NeuralTraining { data } => {
                let mut network = self.neural_network.write().unwrap();
                // Simulate training step
                let _ = network.predict(&data.input)?;
            }
            WorkloadItem::MemoryOperation { key, value } => {
                let mut memory = self.memory_manager.write().unwrap();
                // Simulate memory operation
            }
            WorkloadItem::NetworkMessage { message } => {
                // Simulate network message processing
                let message_size = message.len() as u64;
                self.metrics.record_network_traffic(message_size, 0);
            }
            WorkloadItem::ConsensusProposal { proposal } => {
                // Simulate consensus proposal
                let success = thread_rng().gen_bool(0.9); // 90% success rate
                self.metrics.record_consensus_round(success);
            }
        }
        
        Ok(())
    }

    /// Simulate memory pressure
    async fn simulate_memory_pressure(&self) {
        let mut pool = self.memory_pool.lock().unwrap();
        let buffer = pool.get_buffer(1024 * 1024); // 1MB buffer
        // Use buffer briefly then return
        pool.return_buffer(buffer);
    }

    /// Run stress test
    pub async fn run_stress_test(&self, config: &PerformanceConfig) -> StressTestResult {
        let mut result = StressTestResult::new(self.id);
        
        // Gradually increase load
        for load_factor in [1.0, 2.0, 4.0, 8.0, 16.0] {
            let mut stress_config = config.clone();
            stress_config.target_throughput = (config.target_throughput as f64 * load_factor) as u64;
            stress_config.concurrent_ops = (config.concurrent_ops as f64 * load_factor) as usize;
            stress_config.duration_seconds = 30; // Shorter duration for stress test
            
            let benchmark_result = self.run_benchmark(&stress_config).await;
            result.load_test_results.push((load_factor, benchmark_result));
            
            // Check if system is breaking down
            if benchmark_result.error_rate > 0.1 {
                result.breakdown_load_factor = Some(load_factor);
                break;
            }
        }
        
        result
    }

    /// Run latency test
    pub async fn run_latency_test(&self, config: &PerformanceConfig) -> LatencyTestResult {
        let mut result = LatencyTestResult::new(self.id);
        
        // Measure latency under different percentiles
        let mut latencies = Vec::new();
        
        for _ in 0..10000 {
            let start = Instant::now();
            let workload = self.workload_generator.generate_workload().await;
            let _ = self.process_workload(workload).await;
            let latency = start.elapsed();
            latencies.push(latency);
        }
        
        latencies.sort();
        
        result.p50_latency = latencies[latencies.len() / 2];
        result.p95_latency = latencies[latencies.len() * 95 / 100];
        result.p99_latency = latencies[latencies.len() * 99 / 100];
        result.p999_latency = latencies[latencies.len() * 999 / 1000];
        result.max_latency = latencies[latencies.len() - 1];
        
        result
    }
}

impl Clone for PerformanceTestNode {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            neural_network: self.neural_network.clone(),
            memory_manager: self.memory_manager.clone(),
            secure_channel: self.secure_channel.clone(),
            metrics: self.metrics.clone(),
            memory_pool: self.memory_pool.clone(),
            coordinator: self.coordinator.clone(),
            workload_generator: self.workload_generator.clone(),
            resource_monitor: self.resource_monitor.clone(),
        }
    }
}

/// Workload generator for performance testing
#[derive(Debug)]
pub struct WorkloadGenerator {
    config: PerformanceConfig,
}

impl WorkloadGenerator {
    pub fn new(config: PerformanceConfig) -> Self {
        Self { config }
    }

    pub async fn generate_workload(&self) -> WorkloadItem {
        let mut rng = thread_rng();
        
        match rng.gen_range(0..5) {
            0 => WorkloadItem::NeuralInference {
                input: (0..100).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            },
            1 => WorkloadItem::NeuralTraining {
                data: TrainingData::new(),
            },
            2 => WorkloadItem::MemoryOperation {
                key: format!("key_{}", rng.gen::<u64>()),
                value: (0..1024).map(|_| rng.gen::<u8>()).collect(),
            },
            3 => WorkloadItem::NetworkMessage {
                message: (0..512).map(|_| rng.gen::<u8>()).collect(),
            },
            4 => WorkloadItem::ConsensusProposal {
                proposal: (0..256).map(|_| rng.gen::<u8>()).collect(),
            },
            _ => panic!("Invalid test configuration"),
        }
    }
}

/// Workload item types
#[derive(Debug, Clone)]
pub enum WorkloadItem {
    NeuralInference {
        input: Vec<NeuralFloat>,
    },
    NeuralTraining {
        data: TrainingData,
    },
    MemoryOperation {
        key: String,
        value: Vec<u8>,
    },
    NetworkMessage {
        message: Vec<u8>,
    },
    ConsensusProposal {
        proposal: Vec<u8>,
    },
}

/// Resource monitor for system metrics
#[derive(Debug)]
pub struct ResourceMonitor {
    monitoring: Arc<RwLock<bool>>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            monitoring: Arc::new(RwLock::new(false)),
        }
    }

    pub fn start_monitoring(&self, metrics: Arc<PerformanceMetrics>) -> tokio::task::JoinHandle<()> {
        *self.monitoring.write().unwrap() = true;
        let monitoring = self.monitoring.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));
            
            while *monitoring.read().unwrap() {
                interval.tick().await;
                
                // Monitor memory usage
                let memory_usage = Self::get_memory_usage();
                metrics.record_memory_usage(memory_usage);
                
                // Monitor CPU usage
                let cpu_usage = Self::get_cpu_usage();
                metrics.cpu_usage.store(cpu_usage as u64, Ordering::Relaxed);
            }
        })
    }

    pub async fn stop_monitoring(&self) {
        *self.monitoring.write().unwrap() = false;
    }

    fn get_memory_usage() -> u64 {
        // Simplified memory usage tracking
        // In a real implementation, this would use system APIs
        0
    }

    fn get_cpu_usage() -> f64 {
        // Simplified CPU usage tracking
        // In a real implementation, this would use system APIs
        0.0
    }
}

/// Performance benchmark cluster
pub struct PerformanceBenchmarkCluster {
    nodes: Vec<Arc<PerformanceTestNode>>,
    config: PerformanceConfig,
}

impl PerformanceBenchmarkCluster {
    pub async fn new(config: PerformanceConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut nodes = Vec::new();
        
        for i in 0..config.node_count {
            let mut agent_id = [0u8; 32];
            agent_id[0] = i as u8;
            
            let strategy = match i % 4 {
                0 => CoordinationStrategy::Hierarchical,
                1 => CoordinationStrategy::Mesh,
                2 => CoordinationStrategy::Ring,
                _ => CoordinationStrategy::Star,
            };
            
            let node = PerformanceTestNode::new(agent_id, strategy, &config).await?;
            nodes.push(Arc::new(node));
        }
        
        Ok(Self { nodes, config })
    }

    /// Run comprehensive performance benchmark
    pub async fn run_comprehensive_benchmark(&self) -> ComprehensiveBenchmarkResult {
        let mut result = ComprehensiveBenchmarkResult::new();
        
        // Run parallel benchmarks on all nodes
        let mut handles = Vec::new();
        
        for node in &self.nodes {
            let node_clone = node.clone();
            let config_clone = self.config.clone();
            
            let handle = tokio::spawn(async move {
                node_clone.run_benchmark(&config_clone).await
            });
            
            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            let node_result = handle.await.unwrap();
            result.node_results.push(node_result);
        }

        // Calculate aggregate metrics
        result.calculate_aggregate_metrics();
        
        result
    }

    /// Run stress test on cluster
    pub async fn run_cluster_stress_test(&self) -> ClusterStressTestResult {
        let mut result = ClusterStressTestResult::new();
        
        // Run stress tests on all nodes
        let mut handles = Vec::new();
        
        for node in &self.nodes {
            let node_clone = node.clone();
            let config_clone = self.config.clone();
            
            let handle = tokio::spawn(async move {
                node_clone.run_stress_test(&config_clone).await
            });
            
            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            let stress_result = handle.await.unwrap();
            result.node_stress_results.push(stress_result);
        }

        // Calculate cluster breakdown point
        result.calculate_cluster_breakdown();
        
        result
    }

    /// Run latency test on cluster
    pub async fn run_cluster_latency_test(&self) -> ClusterLatencyTestResult {
        let mut result = ClusterLatencyTestResult::new();
        
        // Run latency tests on all nodes
        let mut handles = Vec::new();
        
        for node in &self.nodes {
            let node_clone = node.clone();
            let config_clone = self.config.clone();
            
            let handle = tokio::spawn(async move {
                node_clone.run_latency_test(&config_clone).await
            });
            
            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            let latency_result = handle.await.unwrap();
            result.node_latency_results.push(latency_result);
        }

        // Calculate cluster latency statistics
        result.calculate_cluster_latency_stats();
        
        result
    }
}

/// Benchmark result structures
#[derive(Debug, Clone)]
pub struct NodeBenchmarkResult {
    pub node_id: AgentId,
    pub throughput: f64,
    pub average_latency: Duration,
    pub error_rate: f64,
    pub peak_memory_usage: u64,
    pub total_operations: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub consensus_success_rate: f64,
    pub errors: Vec<String>,
}

impl NodeBenchmarkResult {
    pub fn new(node_id: AgentId) -> Self {
        Self {
            node_id,
            throughput: 0.0,
            average_latency: Duration::from_secs(0),
            error_rate: 0.0,
            peak_memory_usage: 0,
            total_operations: 0,
            network_bytes_sent: 0,
            network_bytes_received: 0,
            consensus_success_rate: 0.0,
            errors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub node_id: AgentId,
    pub load_test_results: Vec<(f64, NodeBenchmarkResult)>,
    pub breakdown_load_factor: Option<f64>,
}

impl StressTestResult {
    pub fn new(node_id: AgentId) -> Self {
        Self {
            node_id,
            load_test_results: Vec::new(),
            breakdown_load_factor: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LatencyTestResult {
    pub node_id: AgentId,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub p999_latency: Duration,
    pub max_latency: Duration,
}

impl LatencyTestResult {
    pub fn new(node_id: AgentId) -> Self {
        Self {
            node_id,
            p50_latency: Duration::from_secs(0),
            p95_latency: Duration::from_secs(0),
            p99_latency: Duration::from_secs(0),
            p999_latency: Duration::from_secs(0),
            max_latency: Duration::from_secs(0),
        }
    }
}

#[derive(Debug)]
pub struct ComprehensiveBenchmarkResult {
    pub node_results: Vec<NodeBenchmarkResult>,
    pub aggregate_throughput: f64,
    pub average_latency: Duration,
    pub total_operations: u64,
    pub overall_error_rate: f64,
    pub peak_memory_usage: u64,
    pub network_throughput: f64,
    pub consensus_success_rate: f64,
}

impl ComprehensiveBenchmarkResult {
    pub fn new() -> Self {
        Self {
            node_results: Vec::new(),
            aggregate_throughput: 0.0,
            average_latency: Duration::from_secs(0),
            total_operations: 0,
            overall_error_rate: 0.0,
            peak_memory_usage: 0,
            network_throughput: 0.0,
            consensus_success_rate: 0.0,
        }
    }

    pub fn calculate_aggregate_metrics(&mut self) {
        if self.node_results.is_empty() {
            return;
        }

        // Calculate aggregate throughput
        self.aggregate_throughput = self.node_results.iter()
            .map(|r| r.throughput)
            .sum();

        // Calculate average latency
        let total_latency_micros: u64 = self.node_results.iter()
            .map(|r| r.average_latency.as_micros() as u64)
            .sum();
        self.average_latency = Duration::from_micros(total_latency_micros / self.node_results.len() as u64);

        // Calculate total operations
        self.total_operations = self.node_results.iter()
            .map(|r| r.total_operations)
            .sum();

        // Calculate overall error rate
        let total_errors: f64 = self.node_results.iter()
            .map(|r| r.error_rate * r.total_operations as f64)
            .sum();
        self.overall_error_rate = if self.total_operations > 0 {
            total_errors / self.total_operations as f64
        } else {
            0.0
        };

        // Calculate peak memory usage
        self.peak_memory_usage = self.node_results.iter()
            .map(|r| r.peak_memory_usage)
            .max()
            .unwrap_or(0);

        // Calculate network throughput
        let total_network_bytes: u64 = self.node_results.iter()
            .map(|r| r.network_bytes_sent + r.network_bytes_received)
            .sum();
        self.network_throughput = total_network_bytes as f64;

        // Calculate consensus success rate
        let avg_consensus_rate: f64 = self.node_results.iter()
            .map(|r| r.consensus_success_rate)
            .sum::<f64>() / self.node_results.len() as f64;
        self.consensus_success_rate = avg_consensus_rate;
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== COMPREHENSIVE PERFORMANCE BENCHMARK REPORT ===\n\n");
        report.push_str(&format!("Nodes Tested: {}\n", self.node_results.len()));
        report.push_str(&format!("Aggregate Throughput: {:.2} ops/sec\n", self.aggregate_throughput));
        report.push_str(&format!("Average Latency: {:.2} ms\n", self.average_latency.as_millis()));
        report.push_str(&format!("Total Operations: {}\n", self.total_operations));
        report.push_str(&format!("Overall Error Rate: {:.4}%\n", self.overall_error_rate * 100.0));
        report.push_str(&format!("Peak Memory Usage: {:.2} MB\n", self.peak_memory_usage as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("Network Throughput: {:.2} MB/s\n", self.network_throughput / 1024.0 / 1024.0));
        report.push_str(&format!("Consensus Success Rate: {:.2}%\n", self.consensus_success_rate * 100.0));
        
        report.push_str("\n=== PER-NODE RESULTS ===\n");
        for (i, result) in self.node_results.iter().enumerate() {
            report.push_str(&format!("Node {}: {:.2} ops/sec, {:.2} ms latency, {:.4}% error rate\n",
                i, result.throughput, result.average_latency.as_millis(), result.error_rate * 100.0));
        }
        
        report
    }
}

#[derive(Debug)]
pub struct ClusterStressTestResult {
    pub node_stress_results: Vec<StressTestResult>,
    pub cluster_breakdown_factor: Option<f64>,
}

impl ClusterStressTestResult {
    pub fn new() -> Self {
        Self {
            node_stress_results: Vec::new(),
            cluster_breakdown_factor: None,
        }
    }

    pub fn calculate_cluster_breakdown(&mut self) {
        let breakdown_factors: Vec<f64> = self.node_stress_results.iter()
            .filter_map(|r| r.breakdown_load_factor)
            .collect();
        
        if !breakdown_factors.is_empty() {
            self.cluster_breakdown_factor = Some(breakdown_factors.iter().sum::<f64>() / breakdown_factors.len() as f64);
        }
    }
}

#[derive(Debug)]
pub struct ClusterLatencyTestResult {
    pub node_latency_results: Vec<LatencyTestResult>,
    pub cluster_p50_latency: Duration,
    pub cluster_p95_latency: Duration,
    pub cluster_p99_latency: Duration,
    pub cluster_p999_latency: Duration,
    pub cluster_max_latency: Duration,
}

impl ClusterLatencyTestResult {
    pub fn new() -> Self {
        Self {
            node_latency_results: Vec::new(),
            cluster_p50_latency: Duration::from_secs(0),
            cluster_p95_latency: Duration::from_secs(0),
            cluster_p99_latency: Duration::from_secs(0),
            cluster_p999_latency: Duration::from_secs(0),
            cluster_max_latency: Duration::from_secs(0),
        }
    }

    pub fn calculate_cluster_latency_stats(&mut self) {
        if self.node_latency_results.is_empty() {
            return;
        }

        // Calculate average latencies across all nodes
        let p50_sum: u64 = self.node_latency_results.iter()
            .map(|r| r.p50_latency.as_micros() as u64)
            .sum();
        self.cluster_p50_latency = Duration::from_micros(p50_sum / self.node_latency_results.len() as u64);

        let p95_sum: u64 = self.node_latency_results.iter()
            .map(|r| r.p95_latency.as_micros() as u64)
            .sum();
        self.cluster_p95_latency = Duration::from_micros(p95_sum / self.node_latency_results.len() as u64);

        let p99_sum: u64 = self.node_latency_results.iter()
            .map(|r| r.p99_latency.as_micros() as u64)
            .sum();
        self.cluster_p99_latency = Duration::from_micros(p99_sum / self.node_latency_results.len() as u64);

        let p999_sum: u64 = self.node_latency_results.iter()
            .map(|r| r.p999_latency.as_micros() as u64)
            .sum();
        self.cluster_p999_latency = Duration::from_micros(p999_sum / self.node_latency_results.len() as u64);

        self.cluster_max_latency = self.node_latency_results.iter()
            .map(|r| r.max_latency)
            .max()
            .unwrap_or(Duration::from_secs(0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_performance_node_creation() {
        let config = PerformanceConfig::default();
        let mut agent_id = [0u8; 32];
        agent_id[0] = 1;
        
        let node = PerformanceTestNode::new(agent_id, CoordinationStrategy::Mesh, &config).await;
        assert!(node.is_ok());
    }

    #[tokio::test]
    async fn test_workload_generation() {
        let config = PerformanceConfig::default();
        let generator = WorkloadGenerator::new(config);
        
        for _ in 0..10 {
            let workload = generator.generate_workload().await;
            // Verify workload is valid
            match workload {
                WorkloadItem::NeuralInference { input } => assert!(!input.is_empty()),
                WorkloadItem::NeuralTraining { .. } => {},
                WorkloadItem::MemoryOperation { key, value } => {
                    assert!(!key.is_empty());
                    assert!(!value.is_empty());
                },
                WorkloadItem::NetworkMessage { message } => assert!(!message.is_empty()),
                WorkloadItem::ConsensusProposal { proposal } => assert!(!proposal.is_empty()),
            }
        }
    }

    #[tokio::test]
    async fn test_performance_metrics() {
        let metrics = PerformanceMetrics::new();
        
        // Record some operations
        metrics.record_operation(Duration::from_millis(10));
        metrics.record_operation(Duration::from_millis(20));
        metrics.record_error();
        metrics.record_network_traffic(1024, 512);
        metrics.record_memory_usage(1024 * 1024);
        
        assert_eq!(metrics.total_operations.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.error_count.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.network_bytes_sent.load(Ordering::Relaxed), 1024);
        assert_eq!(metrics.network_bytes_received.load(Ordering::Relaxed), 512);
        assert_eq!(metrics.peak_memory_usage.load(Ordering::Relaxed), 1024 * 1024);
        
        let avg_latency = metrics.calculate_average_latency();
        assert_eq!(avg_latency, Duration::from_millis(15));
        
        let error_rate = metrics.error_rate();
        assert_eq!(error_rate, 0.5);
    }

    #[tokio::test]
    async fn test_node_benchmark() {
        let mut config = PerformanceConfig::default();
        config.duration_seconds = 5; // Short test
        config.target_throughput = 100;
        config.concurrent_ops = 10;
        
        let mut agent_id = [0u8; 32];
        agent_id[0] = 1;
        
        let node = PerformanceTestNode::new(agent_id, CoordinationStrategy::Mesh, &config).await.unwrap();
        let result = node.run_benchmark(&config).await;
        
        assert!(result.throughput > 0.0);
        assert!(result.total_operations > 0);
        assert!(result.average_latency.as_millis() > 0);
    }

    #[tokio::test]
    async fn test_cluster_benchmark() {
        let mut config = PerformanceConfig::default();
        config.node_count = 3;
        config.duration_seconds = 5; // Short test
        config.target_throughput = 100;
        config.concurrent_ops = 5;
        
        let cluster = PerformanceBenchmarkCluster::new(config).await.unwrap();
        let result = cluster.run_comprehensive_benchmark().await;
        
        assert_eq!(result.node_results.len(), 3);
        assert!(result.aggregate_throughput > 0.0);
        assert!(result.total_operations > 0);
        
        println!("{}", result.generate_report());
    }

    #[tokio::test]
    async fn test_stress_test() {
        let mut config = PerformanceConfig::default();
        config.duration_seconds = 5; // Short test
        config.target_throughput = 50;
        config.concurrent_ops = 5;
        
        let mut agent_id = [0u8; 32];
        agent_id[0] = 1;
        
        let node = PerformanceTestNode::new(agent_id, CoordinationStrategy::Mesh, &config).await.unwrap();
        let result = node.run_stress_test(&config).await;
        
        assert!(!result.load_test_results.is_empty());
        
        // Check that load factors are increasing
        let mut prev_factor = 0.0;
        for (load_factor, _) in &result.load_test_results {
            assert!(*load_factor > prev_factor);
            prev_factor = *load_factor;
        }
    }

    #[tokio::test]
    async fn test_latency_test() {
        let config = PerformanceConfig::default();
        let mut agent_id = [0u8; 32];
        agent_id[0] = 1;
        
        let node = PerformanceTestNode::new(agent_id, CoordinationStrategy::Mesh, &config).await.unwrap();
        let result = node.run_latency_test(&config).await;
        
        // Check that latency percentiles are in order
        assert!(result.p50_latency <= result.p95_latency);
        assert!(result.p95_latency <= result.p99_latency);
        assert!(result.p99_latency <= result.p999_latency);
        assert!(result.p999_latency <= result.max_latency);
    }

    #[test]
    fn test_comprehensive_benchmark_result() {
        let mut result = ComprehensiveBenchmarkResult::new();
        
        // Add some node results
        let mut node_result1 = NodeBenchmarkResult::new([1u8; 32]);
        node_result1.throughput = 100.0;
        node_result1.total_operations = 1000;
        node_result1.error_rate = 0.01;
        node_result1.average_latency = Duration::from_millis(10);
        
        let mut node_result2 = NodeBenchmarkResult::new([2u8; 32]);
        node_result2.throughput = 200.0;
        node_result2.total_operations = 2000;
        node_result2.error_rate = 0.02;
        node_result2.average_latency = Duration::from_millis(20);
        
        result.node_results.push(node_result1);
        result.node_results.push(node_result2);
        
        result.calculate_aggregate_metrics();
        
        assert_eq!(result.aggregate_throughput, 300.0);
        assert_eq!(result.total_operations, 3000);
        assert_eq!(result.average_latency, Duration::from_millis(15));
        
        let expected_error_rate = (0.01 * 1000.0 + 0.02 * 2000.0) / 3000.0;
        assert!((result.overall_error_rate - expected_error_rate).abs() < 0.0001);
    }

    #[test]
    fn test_workload_item_variants() {
        let mut rng = thread_rng();
        
        let inference_workload = WorkloadItem::NeuralInference {
            input: vec![1.0, 2.0, 3.0],
        };
        
        let training_workload = WorkloadItem::NeuralTraining {
            data: TrainingData::new(),
        };
        
        let memory_workload = WorkloadItem::MemoryOperation {
            key: "test_key".to_string(),
            value: vec![1, 2, 3, 4],
        };
        
        let network_workload = WorkloadItem::NetworkMessage {
            message: vec![0xFF, 0xAB, 0xCD],
        };
        
        let consensus_workload = WorkloadItem::ConsensusProposal {
            proposal: vec![0x12, 0x34, 0x56],
        };
        
        // All workload items should be valid
        match inference_workload {
            WorkloadItem::NeuralInference { input } => assert_eq!(input.len(), 3),
            _ => panic!("Wrong workload type"),
        }
        
        match training_workload {
            WorkloadItem::NeuralTraining { .. } => {},
            _ => panic!("Wrong workload type"),
        }
        
        match memory_workload {
            WorkloadItem::MemoryOperation { key, value } => {
                assert_eq!(key, "test_key");
                assert_eq!(value.len(), 4);
            },
            _ => panic!("Wrong workload type"),
        }
        
        match network_workload {
            WorkloadItem::NetworkMessage { message } => assert_eq!(message.len(), 3),
            _ => panic!("Wrong workload type"),
        }
        
        match consensus_workload {
            WorkloadItem::ConsensusProposal { proposal } => assert_eq!(proposal.len(), 3),
            _ => panic!("Wrong workload type"),
        }
    }
}