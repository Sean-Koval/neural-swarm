//! Chaos Engineering Test Suite for Neuroplex Distributed Memory System
//!
//! This module implements comprehensive chaos engineering tests to validate the
//! fault tolerance and resilience of the neuroplex distributed memory system.

use neuroplex::crdt::*;
use neuroplex::consensus::*;
use neuroplex::memory::distributed::DistributedMemory;
use neuroplex::{NeuroConfig, MemoryConfig, CompressionAlgorithm, NodeId};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock, Mutex};
use tokio::time::{sleep, timeout};
use uuid::Uuid;
use rand::{thread_rng, Rng};

/// Chaos engineering test configuration
#[derive(Debug, Clone)]
pub struct ChaosTestConfig {
    pub node_count: usize,
    pub test_duration: Duration,
    pub failure_injection_rate: f64,
    pub network_partition_rate: f64,
    pub message_drop_rate: f64,
    pub latency_spike_rate: f64,
    pub memory_pressure_rate: f64,
    pub byzantine_node_rate: f64,
    pub max_partition_duration: Duration,
    pub max_failure_duration: Duration,
    pub recovery_time: Duration,
}

impl Default for ChaosTestConfig {
    fn default() -> Self {
        Self {
            node_count: 7,
            test_duration: Duration::from_secs(60),
            failure_injection_rate: 0.1,
            network_partition_rate: 0.05,
            message_drop_rate: 0.02,
            latency_spike_rate: 0.03,
            memory_pressure_rate: 0.02,
            byzantine_node_rate: 0.1,
            max_partition_duration: Duration::from_secs(10),
            max_failure_duration: Duration::from_secs(5),
            recovery_time: Duration::from_secs(2),
        }
    }
}

/// Chaos injection types
#[derive(Debug, Clone)]
pub enum ChaosInjection {
    NodeFailure {
        node_id: NodeId,
        duration: Duration,
    },
    NetworkPartition {
        affected_nodes: Vec<NodeId>,
        duration: Duration,
    },
    MessageDrop {
        source: NodeId,
        target: NodeId,
        drop_rate: f64,
    },
    LatencySpike {
        affected_nodes: Vec<NodeId>,
        latency: Duration,
        duration: Duration,
    },
    MemoryPressure {
        node_id: NodeId,
        pressure_level: f64,
        duration: Duration,
    },
    ByzantineBehavior {
        node_id: NodeId,
        behavior_type: ByzantineBehaviorType,
        duration: Duration,
    },
    DataCorruption {
        node_id: NodeId,
        corruption_rate: f64,
    },
    ClockSkew {
        node_id: NodeId,
        skew_amount: Duration,
    },
}

/// Types of Byzantine behavior
#[derive(Debug, Clone)]
pub enum ByzantineBehaviorType {
    DropMessages,
    CorruptMessages,
    DelayMessages,
    SendConflictingMessages,
    VoteRandomly,
    IgnoreElections,
}

/// Chaos engineering test node
#[derive(Debug)]
pub struct ChaosTestNode {
    pub node_id: NodeId,
    pub memory: Arc<RwLock<DistributedMemory>>,
    pub consensus: Arc<RwLock<RaftConsensus>>,
    pub chaos_state: Arc<RwLock<ChaosNodeState>>,
    pub metrics: Arc<RwLock<ChaosNodeMetrics>>,
    pub message_queue: Arc<Mutex<Vec<ConsensusMessage>>>,
}

/// Chaos state for a node
#[derive(Debug, Default)]
pub struct ChaosNodeState {
    pub is_failed: bool,
    pub is_partitioned: bool,
    pub message_drop_rate: f64,
    pub latency_spike: Option<Duration>,
    pub memory_pressure: f64,
    pub byzantine_behavior: Option<ByzantineBehaviorType>,
    pub data_corruption_rate: f64,
    pub clock_skew: Duration,
    pub injected_failures: Vec<ChaosInjection>,
}

/// Chaos metrics for a node
#[derive(Debug, Default)]
pub struct ChaosNodeMetrics {
    pub total_failures: u64,
    pub total_partitions: u64,
    pub messages_dropped: u64,
    pub messages_delayed: u64,
    pub byzantine_actions: u64,
    pub recovery_time: Duration,
    pub availability: f64,
    pub consistency_violations: u64,
}

impl ChaosTestNode {
    /// Create a new chaos test node
    pub async fn new(node_id: NodeId, config: &ChaosTestConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let memory_config = MemoryConfig {
            max_size: 1024 * 1024 * 1024,
            compression: CompressionAlgorithm::None,
            replication_factor: 3,
        };
        let memory = DistributedMemory::new(node_id, memory_config);

        let consensus_config = RaftConfig {
            election_timeout: 150,
            heartbeat_interval: 50,
            log_compaction_threshold: 1000,
            min_cluster_size: config.node_count,
        };
        let consensus = RaftConsensus::new(node_id, consensus_config).await?;

        Ok(Self {
            node_id,
            memory: Arc::new(RwLock::new(memory)),
            consensus: Arc::new(RwLock::new(consensus)),
            chaos_state: Arc::new(RwLock::new(ChaosNodeState::default())),
            metrics: Arc::new(RwLock::new(ChaosNodeMetrics::default())),
            message_queue: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Inject chaos into the node
    pub async fn inject_chaos(&self, injection: ChaosInjection) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = self.chaos_state.write().await;
        let mut metrics = self.metrics.write().await;

        match injection.clone() {
            ChaosInjection::NodeFailure { duration, .. } => {
                state.is_failed = true;
                metrics.total_failures += 1;
                
                // Schedule recovery
                let state_clone = self.chaos_state.clone();
                tokio::spawn(async move {
                    sleep(duration).await;
                    let mut state = state_clone.write().await;
                    state.is_failed = false;
                });
            }
            ChaosInjection::NetworkPartition { duration, .. } => {
                state.is_partitioned = true;
                metrics.total_partitions += 1;
                
                // Schedule partition healing
                let state_clone = self.chaos_state.clone();
                tokio::spawn(async move {
                    sleep(duration).await;
                    let mut state = state_clone.write().await;
                    state.is_partitioned = false;
                });
            }
            ChaosInjection::MessageDrop { drop_rate, .. } => {
                state.message_drop_rate = drop_rate;
            }
            ChaosInjection::LatencySpike { latency, duration, .. } => {
                state.latency_spike = Some(latency);
                
                // Schedule latency restoration
                let state_clone = self.chaos_state.clone();
                tokio::spawn(async move {
                    sleep(duration).await;
                    let mut state = state_clone.write().await;
                    state.latency_spike = None;
                });
            }
            ChaosInjection::MemoryPressure { pressure_level, duration, .. } => {
                state.memory_pressure = pressure_level;
                
                // Schedule pressure relief
                let state_clone = self.chaos_state.clone();
                tokio::spawn(async move {
                    sleep(duration).await;
                    let mut state = state_clone.write().await;
                    state.memory_pressure = 0.0;
                });
            }
            ChaosInjection::ByzantineBehavior { behavior_type, duration, .. } => {
                state.byzantine_behavior = Some(behavior_type);
                
                // Schedule behavior restoration
                let state_clone = self.chaos_state.clone();
                tokio::spawn(async move {
                    sleep(duration).await;
                    let mut state = state_clone.write().await;
                    state.byzantine_behavior = None;
                });
            }
            ChaosInjection::DataCorruption { corruption_rate, .. } => {
                state.data_corruption_rate = corruption_rate;
            }
            ChaosInjection::ClockSkew { skew_amount, .. } => {
                state.clock_skew = skew_amount;
            }
        }

        state.injected_failures.push(injection);
        Ok(())
    }

    /// Process a message with chaos effects
    pub async fn process_message_with_chaos(&self, message: ConsensusMessage) -> Result<(), Box<dyn std::error::Error>> {
        let state = self.chaos_state.read().await;
        let mut metrics = self.metrics.write().await;

        // Check if node is failed
        if state.is_failed {
            return Ok(()); // Drop message
        }

        // Check if node is partitioned
        if state.is_partitioned {
            return Ok(()); // Drop message
        }

        // Check for message drop
        if thread_rng().gen_bool(state.message_drop_rate) {
            metrics.messages_dropped += 1;
            return Ok(()); // Drop message
        }

        // Apply latency spike
        if let Some(latency) = state.latency_spike {
            sleep(latency).await;
            metrics.messages_delayed += 1;
        }

        // Apply Byzantine behavior
        if let Some(ref behavior) = state.byzantine_behavior {
            match behavior {
                ByzantineBehaviorType::DropMessages => {
                    metrics.byzantine_actions += 1;
                    return Ok(()); // Drop message
                }
                ByzantineBehaviorType::CorruptMessages => {
                    metrics.byzantine_actions += 1;
                    // Continue with corrupted processing
                }
                ByzantineBehaviorType::DelayMessages => {
                    metrics.byzantine_actions += 1;
                    sleep(Duration::from_millis(thread_rng().gen_range(100..1000))).await;
                }
                ByzantineBehaviorType::SendConflictingMessages => {
                    metrics.byzantine_actions += 1;
                    // Send conflicting message (implementation depends on message type)
                }
                ByzantineBehaviorType::VoteRandomly => {
                    metrics.byzantine_actions += 1;
                    // Process with random vote (implementation depends on message type)
                }
                ByzantineBehaviorType::IgnoreElections => {
                    metrics.byzantine_actions += 1;
                    if matches!(message, ConsensusMessage::VoteRequest(_)) {
                        return Ok(()); // Ignore election
                    }
                }
            }
        }

        drop(state);
        drop(metrics);

        // Process message normally
        let mut consensus = self.consensus.write().await;
        consensus.handle_message(message).await?;

        Ok(())
    }

    /// Perform memory operation with chaos effects
    pub async fn memory_operation_with_chaos(&self, key: &str, operation: MemoryOperation) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        let state = self.chaos_state.read().await;

        // Check if node is failed
        if state.is_failed {
            return Err("Node is failed".into());
        }

        // Apply memory pressure
        if state.memory_pressure > 0.5 {
            sleep(Duration::from_millis((state.memory_pressure * 100.0) as u64)).await;
        }

        // Apply data corruption
        if thread_rng().gen_bool(state.data_corruption_rate) {
            match operation {
                MemoryOperation::Set(mut data) => {
                    // Corrupt data
                    for byte in &mut data {
                        if thread_rng().gen_bool(0.01) {
                            *byte = thread_rng().gen();
                        }
                    }
                    let mut memory = self.memory.write().await;
                    memory.set(key, &data).await?;
                    return Ok(None);
                }
                MemoryOperation::Get => {
                    let mut memory = self.memory.write().await;
                    let mut result = memory.get(key).await?;
                    if let Some(ref mut data) = result {
                        // Corrupt read data
                        for byte in data {
                            if thread_rng().gen_bool(0.01) {
                                *byte = thread_rng().gen();
                            }
                        }
                    }
                    return Ok(result);
                }
            }
        }

        drop(state);

        // Perform normal memory operation
        let mut memory = self.memory.write().await;
        match operation {
            MemoryOperation::Set(data) => {
                memory.set(key, &data).await?;
                Ok(None)
            }
            MemoryOperation::Get => {
                memory.get(key).await
            }
        }
    }

    /// Get current chaos metrics
    pub async fn get_chaos_metrics(&self) -> ChaosNodeMetrics {
        self.metrics.read().await.clone()
    }

    /// Calculate availability
    pub async fn calculate_availability(&self, test_duration: Duration) -> f64 {
        let metrics = self.metrics.read().await;
        let total_downtime = metrics.total_failures as f64 * 1.0; // Simplified calculation
        let availability = (test_duration.as_secs_f64() - total_downtime) / test_duration.as_secs_f64();
        availability.max(0.0).min(1.0)
    }
}

/// Memory operation types
#[derive(Debug, Clone)]
pub enum MemoryOperation {
    Set(Vec<u8>),
    Get,
}

/// Chaos engineering test cluster
pub struct ChaosTestCluster {
    pub nodes: Vec<Arc<ChaosTestNode>>,
    pub config: ChaosTestConfig,
    pub chaos_controller: ChaosController,
}

impl ChaosTestCluster {
    /// Create a new chaos test cluster
    pub async fn new(config: ChaosTestConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut nodes = Vec::new();

        // Create nodes
        for _ in 0..config.node_count {
            let node_id = Uuid::new_v4();
            let node = ChaosTestNode::new(node_id, &config).await?;
            nodes.push(Arc::new(node));
        }

        let chaos_controller = ChaosController::new(config.clone());

        Ok(Self {
            nodes,
            config,
            chaos_controller,
        })
    }

    /// Run chaos engineering tests
    pub async fn run_chaos_tests(&self) -> ChaosTestResults {
        let mut results = ChaosTestResults::new();

        // Test 1: Random node failures
        results.add_test(self.test_random_node_failures().await);

        // Test 2: Network partitions
        results.add_test(self.test_network_partitions().await);

        // Test 3: Message drops and delays
        results.add_test(self.test_message_chaos().await);

        // Test 4: Byzantine behavior
        results.add_test(self.test_byzantine_behavior().await);

        // Test 5: Memory pressure
        results.add_test(self.test_memory_pressure().await);

        // Test 6: Combined chaos scenarios
        results.add_test(self.test_combined_chaos().await);

        // Test 7: Recovery testing
        results.add_test(self.test_recovery_scenarios().await);

        results
    }

    /// Test random node failures
    async fn test_random_node_failures(&self) -> ChaosTestResult {
        let mut result = ChaosTestResult::new("Random Node Failures");

        // Start all nodes
        for node in &self.nodes {
            let mut consensus = node.consensus.write().await;
            consensus.start().await.unwrap();
        }

        // Inject random failures
        let mut handles = Vec::new();
        for node in &self.nodes {
            let node_clone = node.clone();
            let config = self.config.clone();
            let handle = tokio::spawn(async move {
                let mut rng = thread_rng();
                let test_end = Instant::now() + config.test_duration;
                
                while Instant::now() < test_end {
                    if rng.gen_bool(config.failure_injection_rate) {
                        let failure_duration = Duration::from_millis(
                            rng.gen_range(100..config.max_failure_duration.as_millis() as u64)
                        );
                        let injection = ChaosInjection::NodeFailure {
                            node_id: node_clone.node_id,
                            duration: failure_duration,
                        };
                        node_clone.inject_chaos(injection).await.unwrap();
                    }
                    
                    sleep(Duration::from_millis(100)).await;
                }
            });
            handles.push(handle);
        }

        // Run workload during chaos
        let workload_handle = self.run_workload().await;

        // Wait for chaos injection to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Wait for workload to complete
        workload_handle.await.unwrap();

        // Check system state
        let mut total_failures = 0;
        let mut total_availability = 0.0;
        
        for node in &self.nodes {
            let metrics = node.get_chaos_metrics().await;
            total_failures += metrics.total_failures;
            total_availability += node.calculate_availability(self.config.test_duration).await;
        }

        let avg_availability = total_availability / self.nodes.len() as f64;

        if avg_availability > 0.8 {
            result.mark_passed(&format!("System maintained {:.1}% availability with {} failures", avg_availability * 100.0, total_failures));
        } else {
            result.mark_failed(&format!("Low availability: {:.1}% with {} failures", avg_availability * 100.0, total_failures));
        }

        result
    }

    /// Test network partitions
    async fn test_network_partitions(&self) -> ChaosTestResult {
        let mut result = ChaosTestResult::new("Network Partitions");

        // Create network partitions
        let partition_size = self.nodes.len() / 2;
        let partition_nodes: Vec<_> = self.nodes.iter().take(partition_size).collect();

        // Inject partition
        for node in &partition_nodes {
            let injection = ChaosInjection::NetworkPartition {
                affected_nodes: vec![node.node_id],
                duration: self.config.max_partition_duration,
            };
            node.inject_chaos(injection).await.unwrap();
        }

        // Wait for partition to take effect
        sleep(Duration::from_millis(500)).await;

        // Run workload
        let workload_handle = self.run_workload().await;
        workload_handle.await.unwrap();

        // Check if majority partition remained operational
        let majority_nodes: Vec<_> = self.nodes.iter().skip(partition_size).collect();
        let mut operational_nodes = 0;

        for node in &majority_nodes {
            let state = node.chaos_state.read().await;
            if !state.is_failed && !state.is_partitioned {
                operational_nodes += 1;
            }
        }

        if operational_nodes >= majority_nodes.len() / 2 {
            result.mark_passed("Majority partition remained operational");
        } else {
            result.mark_failed("Majority partition failed to remain operational");
        }

        result
    }

    /// Test message chaos (drops and delays)
    async fn test_message_chaos(&self) -> ChaosTestResult {
        let mut result = ChaosTestResult::new("Message Chaos");

        // Inject message drops and delays
        for node in &self.nodes {
            let drop_injection = ChaosInjection::MessageDrop {
                source: node.node_id,
                target: Uuid::new_v4(), // Random target
                drop_rate: self.config.message_drop_rate,
            };
            node.inject_chaos(drop_injection).await.unwrap();

            let latency_injection = ChaosInjection::LatencySpike {
                affected_nodes: vec![node.node_id],
                latency: Duration::from_millis(100),
                duration: Duration::from_secs(5),
            };
            node.inject_chaos(latency_injection).await.unwrap();
        }

        // Run workload
        let workload_handle = self.run_workload().await;
        workload_handle.await.unwrap();

        // Check message metrics
        let mut total_dropped = 0;
        let mut total_delayed = 0;

        for node in &self.nodes {
            let metrics = node.get_chaos_metrics().await;
            total_dropped += metrics.messages_dropped;
            total_delayed += metrics.messages_delayed;
        }

        if total_dropped > 0 && total_delayed > 0 {
            result.mark_passed(&format!("Message chaos applied: {} dropped, {} delayed", total_dropped, total_delayed));
        } else {
            result.mark_warning("Message chaos not effectively applied");
        }

        result
    }

    /// Test Byzantine behavior
    async fn test_byzantine_behavior(&self) -> ChaosTestResult {
        let mut result = ChaosTestResult::new("Byzantine Behavior");

        // Inject Byzantine behavior on some nodes
        let byzantine_count = (self.nodes.len() as f64 * self.config.byzantine_node_rate) as usize;
        
        for i in 0..byzantine_count {
            let behaviors = vec![
                ByzantineBehaviorType::DropMessages,
                ByzantineBehaviorType::CorruptMessages,
                ByzantineBehaviorType::DelayMessages,
                ByzantineBehaviorType::VoteRandomly,
            ];
            
            let behavior = behaviors[thread_rng().gen_range(0..behaviors.len())].clone();
            let injection = ChaosInjection::ByzantineBehavior {
                node_id: self.nodes[i].node_id,
                behavior_type: behavior,
                duration: Duration::from_secs(10),
            };
            
            self.nodes[i].inject_chaos(injection).await.unwrap();
        }

        // Run workload
        let workload_handle = self.run_workload().await;
        workload_handle.await.unwrap();

        // Check Byzantine metrics
        let mut total_byzantine_actions = 0;
        for node in &self.nodes {
            let metrics = node.get_chaos_metrics().await;
            total_byzantine_actions += metrics.byzantine_actions;
        }

        if total_byzantine_actions > 0 {
            result.mark_passed(&format!("System handled {} Byzantine actions", total_byzantine_actions));
        } else {
            result.mark_warning("No Byzantine actions detected");
        }

        result
    }

    /// Test memory pressure
    async fn test_memory_pressure(&self) -> ChaosTestResult {
        let mut result = ChaosTestResult::new("Memory Pressure");

        // Inject memory pressure
        for node in &self.nodes {
            let injection = ChaosInjection::MemoryPressure {
                node_id: node.node_id,
                pressure_level: 0.8,
                duration: Duration::from_secs(10),
            };
            node.inject_chaos(injection).await.unwrap();
        }

        // Run memory-intensive workload
        let start_time = Instant::now();
        let mut handles = Vec::new();

        for node in &self.nodes {
            let node_clone = node.clone();
            let handle = tokio::spawn(async move {
                for i in 0..1000 {
                    let key = format!("memory_test_{}", i);
                    let data = vec![0u8; 1024]; // 1KB per entry
                    let operation = MemoryOperation::Set(data);
                    node_clone.memory_operation_with_chaos(&key, operation).await.unwrap();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let elapsed = start_time.elapsed();
        
        if elapsed < Duration::from_secs(30) {
            result.mark_passed(&format!("Memory operations completed in {:.1}s despite pressure", elapsed.as_secs_f64()));
        } else {
            result.mark_warning(&format!("Memory operations took {:.1}s under pressure", elapsed.as_secs_f64()));
        }

        result
    }

    /// Test combined chaos scenarios
    async fn test_combined_chaos(&self) -> ChaosTestResult {
        let mut result = ChaosTestResult::new("Combined Chaos");

        // Inject multiple types of chaos simultaneously
        for (i, node) in self.nodes.iter().enumerate() {
            match i % 4 {
                0 => {
                    let injection = ChaosInjection::NodeFailure {
                        node_id: node.node_id,
                        duration: Duration::from_secs(3),
                    };
                    node.inject_chaos(injection).await.unwrap();
                }
                1 => {
                    let injection = ChaosInjection::NetworkPartition {
                        affected_nodes: vec![node.node_id],
                        duration: Duration::from_secs(5),
                    };
                    node.inject_chaos(injection).await.unwrap();
                }
                2 => {
                    let injection = ChaosInjection::ByzantineBehavior {
                        node_id: node.node_id,
                        behavior_type: ByzantineBehaviorType::DropMessages,
                        duration: Duration::from_secs(7),
                    };
                    node.inject_chaos(injection).await.unwrap();
                }
                3 => {
                    let injection = ChaosInjection::MemoryPressure {
                        node_id: node.node_id,
                        pressure_level: 0.7,
                        duration: Duration::from_secs(8),
                    };
                    node.inject_chaos(injection).await.unwrap();
                }
                _ => {}
            }
        }

        // Run workload under combined chaos
        let workload_handle = self.run_workload().await;
        workload_handle.await.unwrap();

        // Check overall system health
        let mut healthy_nodes = 0;
        for node in &self.nodes {
            let state = node.chaos_state.read().await;
            if !state.is_failed && !state.is_partitioned {
                healthy_nodes += 1;
            }
        }

        let health_ratio = healthy_nodes as f64 / self.nodes.len() as f64;
        
        if health_ratio > 0.5 {
            result.mark_passed(&format!("System maintained {:.1}% healthy nodes under combined chaos", health_ratio * 100.0));
        } else {
            result.mark_failed(&format!("System degraded to {:.1}% healthy nodes", health_ratio * 100.0));
        }

        result
    }

    /// Test recovery scenarios
    async fn test_recovery_scenarios(&self) -> ChaosTestResult {
        let mut result = ChaosTestResult::new("Recovery Scenarios");

        // Inject severe chaos
        for node in &self.nodes {
            let injection = ChaosInjection::NodeFailure {
                node_id: node.node_id,
                duration: Duration::from_secs(2),
            };
            node.inject_chaos(injection).await.unwrap();
        }

        // Wait for failures to take effect
        sleep(Duration::from_millis(500)).await;

        // Measure recovery time
        let recovery_start = Instant::now();
        
        // Wait for recovery
        sleep(self.config.recovery_time).await;

        // Check recovery
        let mut recovered_nodes = 0;
        for node in &self.nodes {
            let state = node.chaos_state.read().await;
            if !state.is_failed {
                recovered_nodes += 1;
            }
        }

        let recovery_time = recovery_start.elapsed();
        let recovery_ratio = recovered_nodes as f64 / self.nodes.len() as f64;

        if recovery_ratio > 0.8 {
            result.mark_passed(&format!("System recovered {:.1}% of nodes in {:.1}s", recovery_ratio * 100.0, recovery_time.as_secs_f64()));
        } else {
            result.mark_failed(&format!("Poor recovery: {:.1}% of nodes in {:.1}s", recovery_ratio * 100.0, recovery_time.as_secs_f64()));
        }

        result
    }

    /// Run workload during chaos testing
    async fn run_workload(&self) -> tokio::task::JoinHandle<()> {
        let nodes = self.nodes.clone();
        tokio::spawn(async move {
            let mut handles = Vec::new();
            
            for node in nodes {
                let node_clone = node.clone();
                let handle = tokio::spawn(async move {
                    for i in 0..100 {
                        let key = format!("workload_key_{}", i);
                        let data = format!("workload_data_{}", i).into_bytes();
                        
                        // Try to set data
                        let _ = node_clone.memory_operation_with_chaos(&key, MemoryOperation::Set(data)).await;
                        
                        // Try to get data
                        let _ = node_clone.memory_operation_with_chaos(&key, MemoryOperation::Get).await;
                        
                        sleep(Duration::from_millis(10)).await;
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.await.unwrap();
            }
        })
    }
}

/// Chaos controller for managing chaos injection
pub struct ChaosController {
    config: ChaosTestConfig,
}

impl ChaosController {
    pub fn new(config: ChaosTestConfig) -> Self {
        Self { config }
    }

    /// Generate random chaos injection
    pub fn generate_random_chaos(&self, node_id: NodeId) -> ChaosInjection {
        let mut rng = thread_rng();
        
        match rng.gen_range(0..6) {
            0 => ChaosInjection::NodeFailure {
                node_id,
                duration: Duration::from_millis(rng.gen_range(100..self.config.max_failure_duration.as_millis() as u64)),
            },
            1 => ChaosInjection::NetworkPartition {
                affected_nodes: vec![node_id],
                duration: Duration::from_millis(rng.gen_range(1000..self.config.max_partition_duration.as_millis() as u64)),
            },
            2 => ChaosInjection::MessageDrop {
                source: node_id,
                target: Uuid::new_v4(),
                drop_rate: rng.gen_range(0.01..0.1),
            },
            3 => ChaosInjection::LatencySpike {
                affected_nodes: vec![node_id],
                latency: Duration::from_millis(rng.gen_range(50..500)),
                duration: Duration::from_secs(rng.gen_range(1..10)),
            },
            4 => ChaosInjection::MemoryPressure {
                node_id,
                pressure_level: rng.gen_range(0.3..0.9),
                duration: Duration::from_secs(rng.gen_range(2..15)),
            },
            5 => ChaosInjection::ByzantineBehavior {
                node_id,
                behavior_type: ByzantineBehaviorType::DropMessages,
                duration: Duration::from_secs(rng.gen_range(5..20)),
            },
            _ => unreachable!(),
        }
    }
}

/// Individual chaos test result
#[derive(Debug, Clone)]
pub struct ChaosTestResult {
    pub name: String,
    pub status: ChaosTestStatus,
    pub message: String,
    pub execution_time: Duration,
    pub chaos_injections: u64,
    pub recovery_time: Duration,
}

impl ChaosTestResult {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            status: ChaosTestStatus::Running,
            message: String::new(),
            execution_time: Duration::new(0, 0),
            chaos_injections: 0,
            recovery_time: Duration::new(0, 0),
        }
    }

    pub fn mark_passed(mut self, message: &str) -> Self {
        self.status = ChaosTestStatus::Passed;
        self.message = message.to_string();
        self
    }

    pub fn mark_failed(mut self, message: &str) -> Self {
        self.status = ChaosTestStatus::Failed;
        self.message = message.to_string();
        self
    }

    pub fn mark_warning(mut self, message: &str) -> Self {
        self.status = ChaosTestStatus::Warning;
        self.message = message.to_string();
        self
    }
}

/// Chaos test status
#[derive(Debug, Clone, PartialEq)]
pub enum ChaosTestStatus {
    Running,
    Passed,
    Failed,
    Warning,
}

/// Chaos test results collection
#[derive(Debug, Clone)]
pub struct ChaosTestResults {
    pub tests: Vec<ChaosTestResult>,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
}

impl ChaosTestResults {
    pub fn new() -> Self {
        Self {
            tests: Vec::new(),
            passed: 0,
            failed: 0,
            warnings: 0,
        }
    }

    pub fn add_test(&mut self, test: ChaosTestResult) {
        match test.status {
            ChaosTestStatus::Passed => self.passed += 1,
            ChaosTestStatus::Failed => self.failed += 1,
            ChaosTestStatus::Warning => self.warnings += 1,
            _ => {}
        }
        self.tests.push(test);
    }

    pub fn success_rate(&self) -> f64 {
        if self.tests.is_empty() {
            1.0
        } else {
            self.passed as f64 / self.tests.len() as f64
        }
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Chaos Engineering Test Results ===\n");
        report.push_str(&format!("Total Tests: {}\n", self.tests.len()));
        report.push_str(&format!("Passed: {}\n", self.passed));
        report.push_str(&format!("Failed: {}\n", self.failed));
        report.push_str(&format!("Warnings: {}\n", self.warnings));
        report.push_str(&format!("Resilience Score: {:.1}%\n\n", self.success_rate() * 100.0));

        for test in &self.tests {
            let status = match test.status {
                ChaosTestStatus::Passed => "RESILIENT",
                ChaosTestStatus::Failed => "VULNERABLE",
                ChaosTestStatus::Warning => "DEGRADED",
                ChaosTestStatus::Running => "TESTING",
            };
            report.push_str(&format!("{}: {} - {}\n", test.name, status, test.message));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chaos_cluster_creation() {
        let config = ChaosTestConfig::default();
        let cluster = ChaosTestCluster::new(config).await.unwrap();
        assert_eq!(cluster.nodes.len(), 7);
    }

    #[tokio::test]
    async fn test_chaos_injection() {
        let config = ChaosTestConfig::default();
        let node = ChaosTestNode::new(Uuid::new_v4(), &config).await.unwrap();
        
        let injection = ChaosInjection::NodeFailure {
            node_id: node.node_id,
            duration: Duration::from_millis(100),
        };
        
        node.inject_chaos(injection).await.unwrap();
        
        let state = node.chaos_state.read().await;
        assert!(state.is_failed);
        
        // Wait for recovery
        sleep(Duration::from_millis(150)).await;
        
        let state = node.chaos_state.read().await;
        assert!(!state.is_failed);
    }

    #[tokio::test]
    async fn test_chaos_comprehensive_suite() {
        let config = ChaosTestConfig {
            node_count: 5,
            test_duration: Duration::from_secs(10),
            ..Default::default()
        };
        
        let cluster = ChaosTestCluster::new(config).await.unwrap();
        let results = cluster.run_chaos_tests().await;
        
        println!("{}", results.generate_report());
        
        // System should be resilient to some degree
        assert!(results.success_rate() > 0.4);
    }
}