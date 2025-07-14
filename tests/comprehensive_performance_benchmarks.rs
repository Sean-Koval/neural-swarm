//! Comprehensive Performance Benchmarks for Neuroplex Distributed Memory System
//!
//! This module provides extensive performance benchmarks for all components of the
//! neuroplex distributed memory system including CRDTs, consensus protocols, and
//! distributed memory operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuroplex::crdt::*;
use neuroplex::consensus::*;
use neuroplex::memory::distributed::DistributedMemory;
use neuroplex::{NeuroConfig, MemoryConfig, CompressionAlgorithm, NodeId};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Performance benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub warm_up_time: Duration,
    pub measurement_time: Duration,
    pub sample_size: usize,
    pub node_counts: Vec<usize>,
    pub operation_counts: Vec<usize>,
    pub data_sizes: Vec<usize>,
    pub concurrency_levels: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warm_up_time: Duration::from_secs(3),
            measurement_time: Duration::from_secs(10),
            sample_size: 100,
            node_counts: vec![1, 3, 5, 10, 20],
            operation_counts: vec![10, 100, 1000, 10000],
            data_sizes: vec![64, 256, 1024, 4096, 16384],
            concurrency_levels: vec![1, 2, 4, 8, 16, 32],
        }
    }
}

/// Performance benchmark suite
pub struct PerformanceBenchmarkSuite {
    config: BenchmarkConfig,
    runtime: Runtime,
}

impl PerformanceBenchmarkSuite {
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
            runtime: Runtime::new().unwrap(),
        }
    }

    /// Benchmark CRDT operations
    pub fn benchmark_crdt_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("crdt_operations");
        
        let operation_counts = vec![10, 100, 1000, 10000];
        let node_counts = vec![1, 5, 10, 50];

        // G-Counter benchmarks
        for &op_count in &operation_counts {
            group.throughput(Throughput::Elements(op_count as u64));
            group.bench_with_input(
                BenchmarkId::new("gcounter_increment", op_count),
                &op_count,
                |b, &op_count| {
                    b.iter(|| {
                        let mut counter = GCounter::new();
                        let node_id = Uuid::new_v4();
                        for i in 0..op_count {
                            counter.increment(node_id, i as u64);
                        }
                        black_box(counter.value())
                    })
                },
            );
        }

        // PN-Counter benchmarks
        for &op_count in &operation_counts {
            group.throughput(Throughput::Elements(op_count as u64));
            group.bench_with_input(
                BenchmarkId::new("pncounter_operations", op_count),
                &op_count,
                |b, &op_count| {
                    b.iter(|| {
                        let mut counter = PNCounter::new();
                        let node_id = Uuid::new_v4();
                        for i in 0..op_count {
                            if i % 2 == 0 {
                                counter.increment(node_id, i as u64);
                            } else {
                                counter.decrement(node_id, i as u64);
                            }
                        }
                        black_box(counter.value())
                    })
                },
            );
        }

        // OR-Set benchmarks
        for &op_count in &operation_counts {
            group.throughput(Throughput::Elements(op_count as u64));
            group.bench_with_input(
                BenchmarkId::new("orset_operations", op_count),
                &op_count,
                |b, &op_count| {
                    b.iter(|| {
                        let mut set = ORSet::new();
                        let node_id = Uuid::new_v4();
                        for i in 0..op_count {
                            let element = format!("element_{}", i);
                            if i % 3 == 0 {
                                set.add(element, node_id);
                            } else {
                                set.remove(&element, node_id);
                            }
                        }
                        black_box(set.contains_elements())
                    })
                },
            );
        }

        // LWW-Register benchmarks
        for &op_count in &operation_counts {
            group.throughput(Throughput::Elements(op_count as u64));
            group.bench_with_input(
                BenchmarkId::new("lww_register_operations", op_count),
                &op_count,
                |b, &op_count| {
                    b.iter(|| {
                        let mut register = LWWRegister::new();
                        let node_id = Uuid::new_v4();
                        for i in 0..op_count {
                            let value = format!("value_{}", i);
                            register.set(value, i as u64, node_id);
                        }
                        black_box(register.value())
                    })
                },
            );
        }

        // CRDT merge benchmarks
        for &node_count in &node_counts {
            group.bench_with_input(
                BenchmarkId::new("gcounter_merge", node_count),
                &node_count,
                |b, &node_count| {
                    b.iter(|| {
                        let mut counters = Vec::new();
                        for i in 0..node_count {
                            let mut counter = GCounter::new();
                            let node_id = Uuid::new_v4();
                            counter.increment(node_id, (i * 100) as u64);
                            counters.push(counter);
                        }
                        
                        let mut final_counter = GCounter::new();
                        for counter in counters {
                            final_counter.merge(&counter);
                        }
                        black_box(final_counter.value())
                    })
                },
            );
        }

        group.finish();
    }

    /// Benchmark distributed memory operations
    pub fn benchmark_distributed_memory(c: &mut Criterion) {
        let mut group = c.benchmark_group("distributed_memory");
        let rt = Runtime::new().unwrap();
        
        let data_sizes = vec![64, 256, 1024, 4096, 16384];
        let operation_counts = vec![100, 1000, 10000];

        // Memory creation and initialization
        group.bench_function("memory_creation", |b| {
            b.iter(|| {
                let node_id = Uuid::new_v4();
                let config = MemoryConfig {
                    max_size: 1024 * 1024 * 1024,
                    compression: CompressionAlgorithm::None,
                    replication_factor: 3,
                };
                let memory = DistributedMemory::new(node_id, config);
                black_box(memory)
            })
        });

        // Sequential write operations
        for &data_size in &data_sizes {
            group.throughput(Throughput::Bytes(data_size as u64));
            group.bench_with_input(
                BenchmarkId::new("sequential_write", data_size),
                &data_size,
                |b, &data_size| {
                    b.to_async(&rt).iter(|| async {
                        let node_id = Uuid::new_v4();
                        let config = MemoryConfig {
                            max_size: 1024 * 1024 * 1024,
                            compression: CompressionAlgorithm::None,
                            replication_factor: 3,
                        };
                        let memory = DistributedMemory::new(node_id, config);
                        
                        let data = vec![0u8; data_size];
                        memory.set("test_key", &data).await.unwrap();
                        black_box(())
                    })
                },
            );
        }

        // Sequential read operations
        for &data_size in &data_sizes {
            group.throughput(Throughput::Bytes(data_size as u64));
            group.bench_with_input(
                BenchmarkId::new("sequential_read", data_size),
                &data_size,
                |b, &data_size| {
                    b.to_async(&rt).iter_with_setup(
                        || {
                            let node_id = Uuid::new_v4();
                            let config = MemoryConfig {
                                max_size: 1024 * 1024 * 1024,
                                compression: CompressionAlgorithm::None,
                                replication_factor: 3,
                            };
                            let memory = DistributedMemory::new(node_id, config);
                            let data = vec![0u8; data_size];
                            rt.block_on(async {
                                memory.set("test_key", &data).await.unwrap();
                            });
                            memory
                        },
                        |memory| async move {
                            let result = memory.get("test_key").await.unwrap();
                            black_box(result)
                        },
                    )
                },
            );
        }

        // Random access operations
        for &op_count in &operation_counts {
            group.throughput(Throughput::Elements(op_count as u64));
            group.bench_with_input(
                BenchmarkId::new("random_access", op_count),
                &op_count,
                |b, &op_count| {
                    b.to_async(&rt).iter(|| async {
                        let node_id = Uuid::new_v4();
                        let config = MemoryConfig {
                            max_size: 1024 * 1024 * 1024,
                            compression: CompressionAlgorithm::None,
                            replication_factor: 3,
                        };
                        let memory = DistributedMemory::new(node_id, config);
                        
                        // Write phase
                        for i in 0..op_count {
                            let key = format!("key_{}", i);
                            let value = format!("value_{}", i).into_bytes();
                            memory.set(&key, &value).await.unwrap();
                        }
                        
                        // Read phase
                        for i in 0..op_count {
                            let key = format!("key_{}", i);
                            let _value = memory.get(&key).await.unwrap();
                        }
                        
                        black_box(())
                    })
                },
            );
        }

        // Memory synchronization benchmarks
        for &node_count in &[2, 5, 10] {
            group.bench_with_input(
                BenchmarkId::new("memory_sync", node_count),
                &node_count,
                |b, &node_count| {
                    b.to_async(&rt).iter(|| async {
                        let mut memories = Vec::new();
                        
                        // Create multiple memory instances
                        for i in 0..node_count {
                            let node_id = Uuid::new_v4();
                            let config = MemoryConfig {
                                max_size: 1024 * 1024 * 1024,
                                compression: CompressionAlgorithm::None,
                                replication_factor: 3,
                            };
                            let memory = DistributedMemory::new(node_id, config);
                            
                            // Add some data
                            for j in 0..10 {
                                let key = format!("key_{}_{}", i, j);
                                let value = format!("value_{}_{}", i, j).into_bytes();
                                memory.set(&key, &value).await.unwrap();
                            }
                            
                            memories.push(memory);
                        }
                        
                        // Simulate synchronization
                        for i in 0..node_count {
                            for j in 0..node_count {
                                if i != j {
                                    let entries = memories[i].get_all_entries().await;
                                    memories[j].merge_entries(entries).await.unwrap();
                                }
                            }
                        }
                        
                        black_box(())
                    })
                },
            );
        }

        group.finish();
    }

    /// Benchmark consensus operations
    pub fn benchmark_consensus_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("consensus_operations");
        let rt = Runtime::new().unwrap();
        
        let node_counts = vec![3, 5, 7, 10];
        let message_counts = vec![10, 100, 1000];

        // Consensus message creation
        group.bench_function("message_creation", |b| {
            b.iter(|| {
                let vote_request = VoteRequest {
                    term: 1,
                    candidate_id: Uuid::new_v4(),
                    last_log_index: 0,
                    last_log_term: 0,
                };
                let message = ConsensusMessage::VoteRequest(vote_request);
                black_box(message)
            })
        });

        // Message serialization/deserialization
        group.bench_function("message_serialization", |b| {
            b.iter(|| {
                let vote_request = VoteRequest {
                    term: 1,
                    candidate_id: Uuid::new_v4(),
                    last_log_index: 0,
                    last_log_term: 0,
                };
                let message = ConsensusMessage::VoteRequest(vote_request);
                
                let serialized = serde_json::to_string(&message).unwrap();
                let deserialized: ConsensusMessage = serde_json::from_str(&serialized).unwrap();
                black_box(deserialized)
            })
        });

        // Leader election simulation
        for &node_count in &node_counts {
            group.bench_with_input(
                BenchmarkId::new("leader_election", node_count),
                &node_count,
                |b, &node_count| {
                    b.iter(|| {
                        // Simulate leader election process
                        let mut votes = HashMap::new();
                        let candidate_id = Uuid::new_v4();
                        
                        for _ in 0..node_count {
                            let voter_id = Uuid::new_v4();
                            let vote_response = VoteResponse {
                                term: 1,
                                vote_granted: true,
                                voter_id,
                            };
                            votes.insert(voter_id, vote_response);
                        }
                        
                        // Count votes
                        let granted_votes = votes.values().filter(|v| v.vote_granted).count();
                        let is_leader = granted_votes > node_count / 2;
                        
                        black_box(is_leader)
                    })
                },
            );
        }

        // Log entry processing
        for &entry_count in &message_counts {
            group.throughput(Throughput::Elements(entry_count as u64));
            group.bench_with_input(
                BenchmarkId::new("log_entry_processing", entry_count),
                &entry_count,
                |b, &entry_count| {
                    b.iter(|| {
                        let mut log_entries = Vec::new();
                        
                        for i in 0..entry_count {
                            let entry = LogEntry {
                                term: 1,
                                index: i as u64,
                                command: format!("command_{}", i).into_bytes(),
                                timestamp: std::time::SystemTime::now(),
                            };
                            log_entries.push(entry);
                        }
                        
                        // Simulate log processing
                        for entry in &log_entries {
                            let _processed = entry.command.len();
                        }
                        
                        black_box(log_entries.len())
                    })
                },
            );
        }

        group.finish();
    }

    /// Benchmark concurrent operations
    pub fn benchmark_concurrent_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("concurrent_operations");
        let rt = Runtime::new().unwrap();
        
        let concurrency_levels = vec![1, 2, 4, 8, 16];
        let operations_per_thread = 1000;

        // Concurrent CRDT operations
        for &concurrency in &concurrency_levels {
            group.throughput(Throughput::Elements((concurrency * operations_per_thread) as u64));
            group.bench_with_input(
                BenchmarkId::new("concurrent_crdt", concurrency),
                &concurrency,
                |b, &concurrency| {
                    b.to_async(&rt).iter(|| async {
                        let counter = Arc::new(RwLock::new(GCounter::new()));
                        let mut handles = Vec::new();
                        
                        for i in 0..concurrency {
                            let counter_clone = counter.clone();
                            let handle = tokio::spawn(async move {
                                let node_id = Uuid::new_v4();
                                for j in 0..operations_per_thread {
                                    let mut counter = counter_clone.write().await;
                                    counter.increment(node_id, (i * operations_per_thread + j) as u64);
                                }
                            });
                            handles.push(handle);
                        }
                        
                        for handle in handles {
                            handle.await.unwrap();
                        }
                        
                        let final_counter = counter.read().await;
                        black_box(final_counter.value())
                    })
                },
            );
        }

        // Concurrent memory operations
        for &concurrency in &concurrency_levels {
            group.throughput(Throughput::Elements((concurrency * operations_per_thread) as u64));
            group.bench_with_input(
                BenchmarkId::new("concurrent_memory", concurrency),
                &concurrency,
                |b, &concurrency| {
                    b.to_async(&rt).iter(|| async {
                        let node_id = Uuid::new_v4();
                        let config = MemoryConfig {
                            max_size: 1024 * 1024 * 1024,
                            compression: CompressionAlgorithm::None,
                            replication_factor: 3,
                        };
                        let memory = Arc::new(DistributedMemory::new(node_id, config));
                        let mut handles = Vec::new();
                        
                        for i in 0..concurrency {
                            let memory_clone = memory.clone();
                            let handle = tokio::spawn(async move {
                                for j in 0..operations_per_thread {
                                    let key = format!("key_{}_{}", i, j);
                                    let value = format!("value_{}_{}", i, j).into_bytes();
                                    memory_clone.set(&key, &value).await.unwrap();
                                }
                            });
                            handles.push(handle);
                        }
                        
                        for handle in handles {
                            handle.await.unwrap();
                        }
                        
                        black_box(())
                    })
                },
            );
        }

        group.finish();
    }

    /// Benchmark memory usage patterns
    pub fn benchmark_memory_usage(c: &mut Criterion) {
        let mut group = c.benchmark_group("memory_usage");
        
        let data_sizes = vec![1024, 4096, 16384, 65536];
        let entry_counts = vec![100, 1000, 10000];

        // Memory allocation patterns
        for &data_size in &data_sizes {
            group.throughput(Throughput::Bytes(data_size as u64));
            group.bench_with_input(
                BenchmarkId::new("memory_allocation", data_size),
                &data_size,
                |b, &data_size| {
                    b.iter(|| {
                        let data = vec![0u8; data_size];
                        let mut storage = HashMap::new();
                        
                        for i in 0..100 {
                            let key = format!("key_{}", i);
                            storage.insert(key, data.clone());
                        }
                        
                        black_box(storage.len())
                    })
                },
            );
        }

        // Memory fragmentation test
        for &entry_count in &entry_counts {
            group.bench_with_input(
                BenchmarkId::new("memory_fragmentation", entry_count),
                &entry_count,
                |b, &entry_count| {
                    b.iter(|| {
                        let mut storage = HashMap::new();
                        
                        // Create entries of varying sizes
                        for i in 0..entry_count {
                            let key = format!("key_{}", i);
                            let size = (i % 10 + 1) * 1024; // 1KB to 10KB
                            let value = vec![0u8; size];
                            storage.insert(key, value);
                        }
                        
                        // Remove every other entry
                        for i in (0..entry_count).step_by(2) {
                            let key = format!("key_{}", i);
                            storage.remove(&key);
                        }
                        
                        black_box(storage.len())
                    })
                },
            );
        }

        group.finish();
    }

    /// Benchmark network simulation
    pub fn benchmark_network_simulation(c: &mut Criterion) {
        let mut group = c.benchmark_group("network_simulation");
        let rt = Runtime::new().unwrap();
        
        let message_sizes = vec![64, 256, 1024, 4096];
        let latencies = vec![1, 10, 50, 100]; // milliseconds

        // Network latency simulation
        for &latency in &latencies {
            group.bench_with_input(
                BenchmarkId::new("network_latency", latency),
                &latency,
                |b, &latency| {
                    b.to_async(&rt).iter(|| async {
                        let start = Instant::now();
                        tokio::time::sleep(Duration::from_millis(latency)).await;
                        let elapsed = start.elapsed();
                        black_box(elapsed)
                    })
                },
            );
        }

        // Message throughput simulation
        for &message_size in &message_sizes {
            group.throughput(Throughput::Bytes(message_size as u64));
            group.bench_with_input(
                BenchmarkId::new("message_throughput", message_size),
                &message_size,
                |b, &message_size| {
                    b.iter(|| {
                        let message = vec![0u8; message_size];
                        
                        // Simulate message processing
                        let checksum: u32 = message.iter().map(|&b| b as u32).sum();
                        
                        black_box(checksum)
                    })
                },
            );
        }

        group.finish();
    }

    /// Benchmark fault tolerance scenarios
    pub fn benchmark_fault_tolerance(c: &mut Criterion) {
        let mut group = c.benchmark_group("fault_tolerance");
        let rt = Runtime::new().unwrap();
        
        let failure_rates = vec![0.0, 0.1, 0.2, 0.3];
        let recovery_times = vec![100, 500, 1000, 2000]; // milliseconds

        // Failure detection simulation
        for &failure_rate in &failure_rates {
            group.bench_with_input(
                BenchmarkId::new("failure_detection", (failure_rate * 100.0) as u32),
                &failure_rate,
                |b, &failure_rate| {
                    b.iter(|| {
                        let nodes = 100;
                        let mut failed_nodes = 0;
                        
                        for _ in 0..nodes {
                            if rand::random::<f64>() < failure_rate {
                                failed_nodes += 1;
                            }
                        }
                        
                        let availability = (nodes - failed_nodes) as f64 / nodes as f64;
                        black_box(availability)
                    })
                },
            );
        }

        // Recovery time simulation
        for &recovery_time in &recovery_times {
            group.bench_with_input(
                BenchmarkId::new("recovery_time", recovery_time),
                &recovery_time,
                |b, &recovery_time| {
                    b.to_async(&rt).iter(|| async {
                        let start = Instant::now();
                        
                        // Simulate recovery process
                        tokio::time::sleep(Duration::from_millis(recovery_time)).await;
                        
                        let elapsed = start.elapsed();
                        black_box(elapsed)
                    })
                },
            );
        }

        group.finish();
    }

    /// Run all comprehensive benchmarks
    pub fn run_all_benchmarks(c: &mut Criterion) {
        Self::benchmark_crdt_operations(c);
        Self::benchmark_distributed_memory(c);
        Self::benchmark_consensus_operations(c);
        Self::benchmark_concurrent_operations(c);
        Self::benchmark_memory_usage(c);
        Self::benchmark_network_simulation(c);
        Self::benchmark_fault_tolerance(c);
    }
}

// Benchmark group definitions
criterion_group!(
    benches,
    PerformanceBenchmarkSuite::run_all_benchmarks
);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = PerformanceBenchmarkSuite::new();
        assert_eq!(suite.config.node_counts.len(), 5);
        assert_eq!(suite.config.operation_counts.len(), 4);
    }

    #[test]
    fn test_crdt_performance() {
        let mut counter = GCounter::new();
        let node_id = Uuid::new_v4();
        let start = Instant::now();
        
        for i in 0..1000 {
            counter.increment(node_id, i);
        }
        
        let elapsed = start.elapsed();
        let ops_per_sec = 1000.0 / elapsed.as_secs_f64();
        
        println!("CRDT performance: {:.0} ops/sec", ops_per_sec);
        assert!(ops_per_sec > 10000.0); // Should be very fast
    }

    #[tokio::test]
    async fn test_memory_performance() {
        let node_id = Uuid::new_v4();
        let config = MemoryConfig {
            max_size: 1024 * 1024 * 1024,
            compression: CompressionAlgorithm::None,
            replication_factor: 3,
        };
        let memory = DistributedMemory::new(node_id, config);
        
        let start = Instant::now();
        
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let value = format!("value_{}", i).into_bytes();
            memory.set(&key, &value).await.unwrap();
        }
        
        let elapsed = start.elapsed();
        let ops_per_sec = 1000.0 / elapsed.as_secs_f64();
        
        println!("Memory performance: {:.0} ops/sec", ops_per_sec);
        assert!(ops_per_sec > 1000.0); // Should be reasonably fast
    }

    #[test]
    fn test_consensus_message_performance() {
        let start = Instant::now();
        
        for i in 0..10000 {
            let vote_request = VoteRequest {
                term: i,
                candidate_id: Uuid::new_v4(),
                last_log_index: i,
                last_log_term: i,
            };
            let message = ConsensusMessage::VoteRequest(vote_request);
            let _serialized = serde_json::to_string(&message).unwrap();
        }
        
        let elapsed = start.elapsed();
        let ops_per_sec = 10000.0 / elapsed.as_secs_f64();
        
        println!("Consensus message performance: {:.0} ops/sec", ops_per_sec);
        assert!(ops_per_sec > 5000.0); // Should be fast enough
    }
}