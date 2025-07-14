// Distributed Memory Performance Benchmarks
// Comprehensive benchmarking suite for distributed neural network memory systems

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Benchmark distributed memory read/write performance
fn benchmark_distributed_memory_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("distributed_memory_throughput");
    
    let node_counts = vec![1, 3, 5, 10];
    let operation_counts = vec![100, 1000, 10000];
    
    for &node_count in &node_counts {
        for &op_count in &operation_counts {
            group.throughput(Throughput::Elements(op_count as u64));
            
            // Read-heavy workload (90% reads, 10% writes)
            group.bench_with_input(
                BenchmarkId::new("read_heavy", format!("{}nodes_{}ops", node_count, op_count)),
                &(node_count, op_count),
                |b, &(nodes, ops)| {
                    b.to_async(&rt).iter(|| async {
                        let cluster = MockDistributedMemory::new(nodes).await;
                        let mut read_count = 0;
                        let mut write_count = 0;
                        
                        for i in 0..ops {
                            if i % 10 == 0 {
                                // Write operation
                                let key = format!("key_{}", i);
                                let value = format!("value_{}", i);
                                black_box(cluster.write(key, value.into_bytes()).await);
                                write_count += 1;
                            } else {
                                // Read operation
                                let key = format!("key_{}", i / 10);
                                black_box(cluster.read(key).await);
                                read_count += 1;
                            }
                        }
                        
                        (read_count, write_count)
                    })
                },
            );
            
            // Write-heavy workload (10% reads, 90% writes)
            group.bench_with_input(
                BenchmarkId::new("write_heavy", format!("{}nodes_{}ops", node_count, op_count)),
                &(node_count, op_count),
                |b, &(nodes, ops)| {
                    b.to_async(&rt).iter(|| async {
                        let cluster = MockDistributedMemory::new(nodes).await;
                        let mut read_count = 0;
                        let mut write_count = 0;
                        
                        for i in 0..ops {
                            if i % 10 == 0 {
                                // Read operation
                                let key = format!("key_{}", i / 10);
                                black_box(cluster.read(key).await);
                                read_count += 1;
                            } else {
                                // Write operation
                                let key = format!("key_{}", i);
                                let value = format!("value_{}", i);
                                black_box(cluster.write(key, value.into_bytes()).await);
                                write_count += 1;
                            }
                        }
                        
                        (read_count, write_count)
                    })
                },
            );
            
            // Uniform workload (50% reads, 50% writes)
            group.bench_with_input(
                BenchmarkId::new("uniform", format!("{}nodes_{}ops", node_count, op_count)),
                &(node_count, op_count),
                |b, &(nodes, ops)| {
                    b.to_async(&rt).iter(|| async {
                        let cluster = MockDistributedMemory::new(nodes).await;
                        let mut read_count = 0;
                        let mut write_count = 0;
                        
                        for i in 0..ops {
                            if i % 2 == 0 {
                                // Read operation
                                let key = format!("key_{}", i / 2);
                                black_box(cluster.read(key).await);
                                read_count += 1;
                            } else {
                                // Write operation
                                let key = format!("key_{}", i);
                                let value = format!("value_{}", i);
                                black_box(cluster.write(key, value.into_bytes()).await);
                                write_count += 1;
                            }
                        }
                        
                        (read_count, write_count)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark consistency model performance
fn benchmark_consistency_models(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("consistency_models");
    
    let data_sizes = vec![1024, 4096, 16384, 65536]; // bytes
    
    for &size in &data_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        // Strong consistency (consensus-based)
        group.bench_with_input(
            BenchmarkId::new("strong_consistency", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let cluster = MockDistributedMemory::new(5).await;
                    let key = "test_key".to_string();
                    let value = vec![0u8; size];
                    
                    // Simulate strong consistency with consensus
                    let start = std::time::Instant::now();
                    cluster.write_with_consensus(key.clone(), value.clone()).await;
                    let consensus_time = start.elapsed();
                    
                    // Verify consistency across all nodes
                    let read_start = std::time::Instant::now();
                    let read_value = cluster.read_with_consistency_check(key).await;
                    let read_time = read_start.elapsed();
                    
                    black_box((consensus_time, read_time, read_value))
                })
            },
        );
        
        // Eventual consistency (CRDT-based)
        group.bench_with_input(
            BenchmarkId::new("eventual_consistency", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let cluster = MockDistributedMemory::new(5).await;
                    let key = "test_key".to_string();
                    let value = vec![0u8; size];
                    
                    // Simulate eventual consistency with CRDT
                    let start = std::time::Instant::now();
                    cluster.write_with_crdt(key.clone(), value.clone()).await;
                    let write_time = start.elapsed();
                    
                    // Trigger convergence
                    let convergence_start = std::time::Instant::now();
                    cluster.trigger_convergence().await;
                    let convergence_time = convergence_start.elapsed();
                    
                    black_box((write_time, convergence_time))
                })
            },
        );
        
        // Causal consistency (vector clock-based)
        group.bench_with_input(
            BenchmarkId::new("causal_consistency", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let cluster = MockDistributedMemory::new(5).await;
                    let key = "test_key".to_string();
                    let value = vec![0u8; size];
                    
                    // Simulate causal consistency with vector clocks
                    let start = std::time::Instant::now();
                    cluster.write_with_causal_consistency(key.clone(), value.clone()).await;
                    let write_time = start.elapsed();
                    
                    // Read with causal ordering
                    let read_start = std::time::Instant::now();
                    let read_value = cluster.read_with_causal_consistency(key).await;
                    let read_time = read_start.elapsed();
                    
                    black_box((write_time, read_time, read_value))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark CRDT performance characteristics
fn benchmark_crdt_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("crdt_operations");
    
    let element_counts = vec![100, 1000, 10000, 100000];
    
    for &count in &element_counts {
        group.throughput(Throughput::Elements(count as u64));
        
        // G-Counter (grow-only counter)
        group.bench_with_input(
            BenchmarkId::new("g_counter_merge", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let mut counter1 = GCounter::new(5);
                        let mut counter2 = GCounter::new(5);
                        
                        // Populate counters
                        for i in 0..count {
                            counter1.increment(i % 5);
                            counter2.increment((i + 1) % 5);
                        }
                        
                        (counter1, counter2)
                    },
                    |(mut counter1, counter2)| {
                        black_box(counter1.merge(&counter2));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        
        // OR-Set (observed-remove set)
        group.bench_with_input(
            BenchmarkId::new("or_set_merge", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let mut set1 = ORSet::new(1);
                        let mut set2 = ORSet::new(2);
                        
                        // Populate sets
                        for i in 0..count {
                            set1.add(format!("item_{}", i));
                            set2.add(format!("item_{}", i + count / 2));
                        }
                        
                        (set1, set2)
                    },
                    |(mut set1, set2)| {
                        black_box(set1.merge(&set2));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        
        // LWW-Register (last-write-wins register)
        group.bench_with_input(
            BenchmarkId::new("lww_register_merge", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let mut register1 = LWWRegister::new(1);
                        let mut register2 = LWWRegister::new(2);
                        
                        // Perform updates
                        for i in 0..count {
                            register1.update(format!("value_{}", i), i as u64);
                            register2.update(format!("value_{}", i + count), (i + count) as u64);
                        }
                        
                        (register1, register2)
                    },
                    |(mut register1, register2)| {
                        black_box(register1.merge(&register2));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark consensus protocol performance
fn benchmark_consensus_protocols(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("consensus_protocols");
    
    let cluster_sizes = vec![3, 5, 7, 9];
    let log_entry_counts = vec![10, 100, 1000];
    
    for &cluster_size in &cluster_sizes {
        for &entry_count in &log_entry_counts {
            group.throughput(Throughput::Elements(entry_count as u64));
            
            // Raft leader election
            group.bench_with_input(
                BenchmarkId::new("raft_leader_election", format!("{}nodes", cluster_size)),
                &cluster_size,
                |b, &cluster_size| {
                    b.to_async(&rt).iter(|| async {
                        let cluster = MockRaftCluster::new(cluster_size).await;
                        
                        // Simulate leader failure and re-election
                        let start = std::time::Instant::now();
                        cluster.simulate_leader_failure().await;
                        let new_leader = cluster.elect_leader().await;
                        let election_time = start.elapsed();
                        
                        black_box((new_leader, election_time))
                    })
                },
            );
            
            // Raft log replication
            group.bench_with_input(
                BenchmarkId::new("raft_log_replication", format!("{}nodes_{}entries", cluster_size, entry_count)),
                &(cluster_size, entry_count),
                |b, &(cluster_size, entry_count)| {
                    b.to_async(&rt).iter(|| async {
                        let cluster = MockRaftCluster::new(cluster_size).await;
                        let leader = cluster.get_leader().await;
                        
                        let start = std::time::Instant::now();
                        for i in 0..entry_count {
                            let entry = format!("log_entry_{}", i);
                            leader.append_entry(entry.into_bytes()).await;
                        }
                        let replication_time = start.elapsed();
                        
                        black_box(replication_time)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark network partition recovery
fn benchmark_network_partition_recovery(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("network_partition_recovery");
    
    let cluster_sizes = vec![5, 7, 9];
    let partition_types = vec!["symmetric", "asymmetric", "isolated_node"];
    
    for &cluster_size in &cluster_sizes {
        for partition_type in &partition_types {
            group.bench_with_input(
                BenchmarkId::new("partition_recovery", format!("{}nodes_{}", cluster_size, partition_type)),
                &(cluster_size, partition_type),
                |b, &(cluster_size, partition_type)| {
                    b.to_async(&rt).iter(|| async {
                        let cluster = MockDistributedMemory::new(cluster_size).await;
                        
                        // Create network partition
                        let start = std::time::Instant::now();
                        cluster.create_network_partition(partition_type).await;
                        let partition_time = start.elapsed();
                        
                        // Simulate some operations during partition
                        for i in 0..10 {
                            let key = format!("key_{}", i);
                            let value = format!("value_{}", i);
                            cluster.write(key, value.into_bytes()).await;
                        }
                        
                        // Heal network partition
                        let heal_start = std::time::Instant::now();
                        cluster.heal_network_partition().await;
                        let heal_time = heal_start.elapsed();
                        
                        // Measure convergence time
                        let convergence_start = std::time::Instant::now();
                        cluster.wait_for_convergence().await;
                        let convergence_time = convergence_start.elapsed();
                        
                        black_box((partition_time, heal_time, convergence_time))
                    })
                },
            );
        }
    }
    
    group.finish();
}

// Mock implementations for benchmarking

struct MockDistributedMemory {
    nodes: Vec<Arc<RwLock<HashMap<String, Vec<u8>>>>>,
    node_count: usize,
}

impl MockDistributedMemory {
    async fn new(node_count: usize) -> Arc<Self> {
        let nodes = (0..node_count)
            .map(|_| Arc::new(RwLock::new(HashMap::new())))
            .collect();
        
        Arc::new(Self { nodes, node_count })
    }
    
    async fn write(&self, key: String, value: Vec<u8>) -> Result<(), String> {
        // Simulate write to primary node
        let primary_node = self.hash_key(&key) % self.node_count;
        self.nodes[primary_node].write().await.insert(key, value);
        Ok(())
    }
    
    async fn read(&self, key: String) -> Result<Option<Vec<u8>>, String> {
        // Simulate read from primary node
        let primary_node = self.hash_key(&key) % self.node_count;
        let value = self.nodes[primary_node].read().await.get(&key).cloned();
        Ok(value)
    }
    
    async fn write_with_consensus(&self, key: String, value: Vec<u8>) -> Result<(), String> {
        // Simulate consensus write (simplified)
        let majority = (self.node_count / 2) + 1;
        
        // Write to majority of nodes
        for i in 0..majority {
            self.nodes[i].write().await.insert(key.clone(), value.clone());
        }
        
        // Simulate consensus delay
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }
    
    async fn read_with_consistency_check(&self, key: String) -> Result<Option<Vec<u8>>, String> {
        // Read from majority and check consistency
        let majority = (self.node_count / 2) + 1;
        let mut values = Vec::new();
        
        for i in 0..majority {
            let value = self.nodes[i].read().await.get(&key).cloned();
            values.push(value);
        }
        
        // Return most common value (simplified)
        Ok(values.into_iter().find(|v| v.is_some()).flatten())
    }
    
    async fn write_with_crdt(&self, key: String, value: Vec<u8>) -> Result<(), String> {
        // Simulate CRDT write (eventually consistent)
        let target_node = self.hash_key(&key) % self.node_count;
        self.nodes[target_node].write().await.insert(key, value);
        Ok(())
    }
    
    async fn trigger_convergence(&self) -> Result<(), String> {
        // Simulate CRDT convergence (simplified)
        tokio::time::sleep(Duration::from_millis(2)).await;
        Ok(())
    }
    
    async fn write_with_causal_consistency(&self, key: String, value: Vec<u8>) -> Result<(), String> {
        // Simulate causal consistency with vector clock
        let target_node = self.hash_key(&key) % self.node_count;
        self.nodes[target_node].write().await.insert(key, value);
        
        // Simulate vector clock update
        tokio::time::sleep(Duration::from_millis(1)).await;
        Ok(())
    }
    
    async fn read_with_causal_consistency(&self, key: String) -> Result<Option<Vec<u8>>, String> {
        // Read with causal ordering check
        let target_node = self.hash_key(&key) % self.node_count;
        let value = self.nodes[target_node].read().await.get(&key).cloned();
        Ok(value)
    }
    
    async fn create_network_partition(&self, _partition_type: &str) -> Result<(), String> {
        // Simulate network partition
        tokio::time::sleep(Duration::from_millis(1)).await;
        Ok(())
    }
    
    async fn heal_network_partition(&self) -> Result<(), String> {
        // Simulate network partition healing
        tokio::time::sleep(Duration::from_millis(2)).await;
        Ok(())
    }
    
    async fn wait_for_convergence(&self) -> Result<(), String> {
        // Simulate waiting for convergence after partition healing
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }
    
    fn hash_key(&self, key: &str) -> usize {
        // Simple hash function for key distribution
        key.len() % self.node_count
    }
}

struct MockRaftCluster {
    _node_count: usize,
    _leader: Option<usize>,
}

impl MockRaftCluster {
    async fn new(node_count: usize) -> Arc<Self> {
        Arc::new(Self {
            _node_count: node_count,
            _leader: Some(0),
        })
    }
    
    async fn simulate_leader_failure(&self) -> Result<(), String> {
        // Simulate leader failure
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }
    
    async fn elect_leader(&self) -> Result<usize, String> {
        // Simulate leader election
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(1) // New leader ID
    }
    
    async fn get_leader(&self) -> Result<Arc<MockRaftLeader>, String> {
        Ok(Arc::new(MockRaftLeader {}))
    }
}

struct MockRaftLeader;

impl MockRaftLeader {
    async fn append_entry(&self, _entry: Vec<u8>) -> Result<(), String> {
        // Simulate log entry appending and replication
        tokio::time::sleep(Duration::from_millis(1)).await;
        Ok(())
    }
}

// CRDT implementations for benchmarking

#[derive(Clone)]
struct GCounter {
    counters: Vec<u64>,
    node_id: usize,
}

impl GCounter {
    fn new(node_count: usize) -> Self {
        Self {
            counters: vec![0; node_count],
            node_id: 0,
        }
    }
    
    fn increment(&mut self, node_id: usize) {
        if node_id < self.counters.len() {
            self.counters[node_id] += 1;
        }
    }
    
    fn merge(&mut self, other: &GCounter) {
        for (i, &count) in other.counters.iter().enumerate() {
            if i < self.counters.len() {
                self.counters[i] = self.counters[i].max(count);
            }
        }
    }
}

#[derive(Clone)]
struct ORSet {
    added: HashMap<String, u64>,
    removed: HashMap<String, u64>,
    _node_id: usize,
}

impl ORSet {
    fn new(node_id: usize) -> Self {
        Self {
            added: HashMap::new(),
            removed: HashMap::new(),
            _node_id: node_id,
        }
    }
    
    fn add(&mut self, item: String) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.added.insert(item, timestamp);
    }
    
    fn merge(&mut self, other: &ORSet) {
        for (item, &timestamp) in &other.added {
            let current = self.added.get(item).copied().unwrap_or(0);
            self.added.insert(item.clone(), current.max(timestamp));
        }
        
        for (item, &timestamp) in &other.removed {
            let current = self.removed.get(item).copied().unwrap_or(0);
            self.removed.insert(item.clone(), current.max(timestamp));
        }
    }
}

#[derive(Clone)]
struct LWWRegister {
    value: Option<String>,
    timestamp: u64,
    _node_id: usize,
}

impl LWWRegister {
    fn new(node_id: usize) -> Self {
        Self {
            value: None,
            timestamp: 0,
            _node_id: node_id,
        }
    }
    
    fn update(&mut self, value: String, timestamp: u64) {
        if timestamp > self.timestamp {
            self.value = Some(value);
            self.timestamp = timestamp;
        }
    }
    
    fn merge(&mut self, other: &LWWRegister) {
        if other.timestamp > self.timestamp {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
        }
    }
}

criterion_group! {
    name = distributed_memory_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(50)
        .with_plots();
    targets = 
        benchmark_distributed_memory_throughput,
        benchmark_consistency_models,
        benchmark_crdt_operations,
        benchmark_consensus_protocols,
        benchmark_network_partition_recovery
}

criterion_main!(distributed_memory_benches);