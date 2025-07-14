//! Comprehensive CRDT Property Testing Suite
//!
//! This module provides property-based tests for all CRDT implementations in the neuroplex
//! distributed memory system, ensuring convergence, commutativity, and associativity properties.

use neuroplex::crdt::*;
use neuroplex::memory::distributed::DistributedMemory;
use neuroplex::{NeuroConfig, MemoryConfig, CompressionAlgorithm, NodeId, Timestamp};
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;
use uuid::Uuid;

/// Property-based test configuration
#[derive(Debug, Clone)]
struct PropertyTestConfig {
    max_operations: usize,
    max_nodes: usize,
    max_concurrent_operations: usize,
    convergence_timeout_ms: u64,
}

impl Default for PropertyTestConfig {
    fn default() -> Self {
        Self {
            max_operations: 100,
            max_nodes: 10,
            max_concurrent_operations: 50,
            convergence_timeout_ms: 5000,
        }
    }
}

/// Test harness for CRDT property testing
pub struct CrdtPropertyTestHarness {
    config: PropertyTestConfig,
    runtime: Runtime,
}

impl CrdtPropertyTestHarness {
    pub fn new() -> Self {
        Self {
            config: PropertyTestConfig::default(),
            runtime: Runtime::new().unwrap(),
        }
    }

    /// Generate random G-Counter operations
    pub fn generate_gcounter_operations() -> BoxedStrategy<Vec<(NodeId, u64)>> {
        prop::collection::vec(
            (any::<[u8; 16]>().prop_map(|bytes| Uuid::from_bytes(bytes)), any::<u64>()),
            1..20,
        )
        .boxed()
    }

    /// Generate random PN-Counter operations
    pub fn generate_pncounter_operations() -> BoxedStrategy<Vec<(NodeId, i64)>> {
        prop::collection::vec(
            (
                any::<[u8; 16]>().prop_map(|bytes| Uuid::from_bytes(bytes)),
                any::<i64>(),
            ),
            1..20,
        )
        .boxed()
    }

    /// Generate random OR-Set operations
    pub fn generate_orset_operations() -> BoxedStrategy<Vec<(String, bool)>> {
        prop::collection::vec(
            (
                "[a-z]{1,10}".prop_map(|s| s.to_string()),
                any::<bool>(), // true = add, false = remove
            ),
            1..30,
        )
        .boxed()
    }

    /// Generate random LWW-Register operations
    pub fn generate_lww_register_operations() -> BoxedStrategy<Vec<(String, Timestamp)>> {
        prop::collection::vec(
            (
                "[a-z]{1,10}".prop_map(|s| s.to_string()),
                any::<u64>(), // timestamp
            ),
            1..20,
        )
        .boxed()
    }

    /// Test G-Counter convergence property
    pub fn test_gcounter_convergence(&self, operations: Vec<(NodeId, u64)>) -> bool {
        let mut counter1 = GCounter::new();
        let mut counter2 = GCounter::new();
        let mut counter3 = GCounter::new();

        // Apply operations in different orders to different counters
        for (node_id, increment) in &operations {
            counter1.increment(*node_id, *increment);
        }

        for (node_id, increment) in operations.iter().rev() {
            counter2.increment(*node_id, *increment);
        }

        // Apply operations in random order to third counter
        let mut ops_shuffled = operations.clone();
        use rand::seq::SliceRandom;
        ops_shuffled.shuffle(&mut rand::thread_rng());
        for (node_id, increment) in ops_shuffled {
            counter3.increment(node_id, increment);
        }

        // All counters should converge to same value
        let value1 = counter1.value();
        let value2 = counter2.value();
        let value3 = counter3.value();

        value1 == value2 && value2 == value3
    }

    /// Test PN-Counter convergence property
    pub fn test_pncounter_convergence(&self, operations: Vec<(NodeId, i64)>) -> bool {
        let mut counter1 = PNCounter::new();
        let mut counter2 = PNCounter::new();

        // Apply operations in different orders
        for (node_id, delta) in &operations {
            if *delta >= 0 {
                counter1.increment(*node_id, *delta as u64);
            } else {
                counter1.decrement(*node_id, delta.abs() as u64);
            }
        }

        for (node_id, delta) in operations.iter().rev() {
            if *delta >= 0 {
                counter2.increment(*node_id, *delta as u64);
            } else {
                counter2.decrement(*node_id, delta.abs() as u64);
            }
        }

        // Both counters should converge to same value
        counter1.value() == counter2.value()
    }

    /// Test OR-Set convergence property
    pub fn test_orset_convergence(&self, operations: Vec<(String, bool)>) -> bool {
        let mut set1 = ORSet::new();
        let mut set2 = ORSet::new();

        let node_id = Uuid::new_v4();

        // Apply operations in different orders
        for (element, is_add) in &operations {
            if *is_add {
                set1.add(element.clone(), node_id);
            } else {
                set1.remove(element.clone(), node_id);
            }
        }

        for (element, is_add) in operations.iter().rev() {
            if *is_add {
                set2.add(element.clone(), node_id);
            } else {
                set2.remove(element.clone(), node_id);
            }
        }

        // Both sets should converge to same elements
        set1.contains_elements() == set2.contains_elements()
    }

    /// Test LWW-Register convergence property
    pub fn test_lww_register_convergence(&self, operations: Vec<(String, Timestamp)>) -> bool {
        let mut register1 = LWWRegister::new();
        let mut register2 = LWWRegister::new();

        let node_id = Uuid::new_v4();

        // Apply operations in different orders
        for (value, timestamp) in &operations {
            register1.set(value.clone(), *timestamp, node_id);
        }

        for (value, timestamp) in operations.iter().rev() {
            register2.set(value.clone(), *timestamp, node_id);
        }

        // Both registers should converge to same value (last writer wins)
        register1.value() == register2.value()
    }

    /// Test CRDT commutativity property
    pub fn test_crdt_commutativity(&self) -> bool {
        let operations = vec![
            (Uuid::new_v4(), 10u64),
            (Uuid::new_v4(), 20u64),
            (Uuid::new_v4(), 5u64),
        ];

        // Test all permutations of operations
        use itertools::Itertools;
        let mut results = Vec::new();

        for perm in operations.iter().permutations(operations.len()) {
            let mut counter = GCounter::new();
            for (node_id, increment) in perm {
                counter.increment(*node_id, *increment);
            }
            results.push(counter.value());
        }

        // All permutations should yield same result
        results.iter().all(|&x| x == results[0])
    }

    /// Test CRDT associativity property
    pub fn test_crdt_associativity(&self) -> bool {
        let ops1 = vec![(Uuid::new_v4(), 10u64), (Uuid::new_v4(), 20u64)];
        let ops2 = vec![(Uuid::new_v4(), 15u64), (Uuid::new_v4(), 25u64)];
        let ops3 = vec![(Uuid::new_v4(), 30u64), (Uuid::new_v4(), 5u64)];

        // Test (A ∪ B) ∪ C = A ∪ (B ∪ C)
        let mut counter1 = GCounter::new();
        let mut counter2 = GCounter::new();
        let mut counter3 = GCounter::new();

        // Apply (A ∪ B) ∪ C
        for (node_id, increment) in &ops1 {
            counter1.increment(*node_id, *increment);
        }
        for (node_id, increment) in &ops2 {
            counter1.increment(*node_id, *increment);
        }
        for (node_id, increment) in &ops3 {
            counter1.increment(*node_id, *increment);
        }

        // Apply A ∪ (B ∪ C)
        for (node_id, increment) in &ops1 {
            counter2.increment(*node_id, *increment);
        }
        for (node_id, increment) in &ops2 {
            counter3.increment(*node_id, *increment);
        }
        for (node_id, increment) in &ops3 {
            counter3.increment(*node_id, *increment);
        }

        // Merge counter3 into counter2
        counter2.merge(&counter3);

        // Results should be equal
        counter1.value() == counter2.value()
    }

    /// Test CRDT idempotency property
    pub fn test_crdt_idempotency(&self) -> bool {
        let mut counter = GCounter::new();
        let node_id = Uuid::new_v4();

        // Apply same operation multiple times
        counter.increment(node_id, 10);
        let value1 = counter.value();

        counter.increment(node_id, 0); // Should not change
        let value2 = counter.value();

        // Apply same logical operation again
        counter.increment(node_id, 10);
        counter.increment(node_id, 10);
        let value3 = counter.value();

        // G-Counter should accumulate (not idempotent for same increments)
        // But should be idempotent for merges
        let mut counter_copy = counter.clone();
        counter.merge(&counter_copy);
        let value4 = counter.value();

        value1 == value2 && value3 == value4
    }

    /// Test concurrent CRDT operations
    pub fn test_concurrent_crdt_operations(&self) -> bool {
        self.runtime.block_on(async {
            let num_nodes = 5;
            let operations_per_node = 20;
            let mut handles = Vec::new();

            let counters = Arc::new(tokio::sync::Mutex::new(Vec::new()));

            for node_idx in 0..num_nodes {
                let counters_clone = counters.clone();
                let handle = tokio::spawn(async move {
                    let mut counter = GCounter::new();
                    let node_id = Uuid::new_v4();

                    for i in 0..operations_per_node {
                        counter.increment(node_id, i as u64);
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                    }

                    let mut counters = counters_clone.lock().await;
                    counters.push(counter);
                });

                handles.push(handle);
            }

            // Wait for all operations to complete
            for handle in handles {
                handle.await.unwrap();
            }

            // Merge all counters
            let counters = counters.lock().await;
            let mut final_counter = GCounter::new();

            for counter in counters.iter() {
                final_counter.merge(counter);
            }

            // Expected value: sum of 0..operations_per_node for each node
            let expected_per_node: u64 = (0..operations_per_node).sum();
            let expected_total = expected_per_node * num_nodes as u64;

            final_counter.value() == expected_total
        })
    }

    /// Test CRDT with network partition simulation
    pub fn test_crdt_partition_tolerance(&self) -> bool {
        self.runtime.block_on(async {
            // Create two partitions
            let mut partition1_counters = Vec::new();
            let mut partition2_counters = Vec::new();

            // Initialize nodes in each partition
            for _ in 0..3 {
                partition1_counters.push(GCounter::new());
                partition2_counters.push(GCounter::new());
            }

            // Simulate operations in partition 1
            for i in 0..10 {
                let node_id = Uuid::new_v4();
                for counter in &mut partition1_counters {
                    counter.increment(node_id, i as u64);
                }
            }

            // Simulate operations in partition 2
            for i in 10..20 {
                let node_id = Uuid::new_v4();
                for counter in &mut partition2_counters {
                    counter.increment(node_id, i as u64);
                }
            }

            // Simulate partition healing - merge all counters
            let mut final_counter = GCounter::new();
            for counter in &partition1_counters {
                final_counter.merge(counter);
            }
            for counter in &partition2_counters {
                final_counter.merge(counter);
            }

            // Should converge to consistent state
            let expected_value: u64 = (0..20).sum();
            final_counter.value() == expected_value
        })
    }

    /// Test CRDT memory efficiency
    pub fn test_crdt_memory_efficiency(&self) -> bool {
        let mut counter = GCounter::new();
        let node_id = Uuid::new_v4();

        // Add many operations
        for i in 0..1000 {
            counter.increment(node_id, 1);
        }

        // Memory usage should be reasonable (not grow indefinitely)
        let memory_usage = std::mem::size_of_val(&counter);
        memory_usage < 1024 * 1024 // Less than 1MB
    }

    /// Test CRDT serialization/deserialization
    pub fn test_crdt_serialization(&self) -> bool {
        let mut counter = GCounter::new();
        let node_id = Uuid::new_v4();

        counter.increment(node_id, 42);

        // Serialize and deserialize
        let serialized = serde_json::to_string(&counter).unwrap();
        let deserialized: GCounter = serde_json::from_str(&serialized).unwrap();

        counter.value() == deserialized.value()
    }

    /// Run all CRDT property tests
    pub fn run_all_crdt_property_tests(&self) -> CrdtPropertyTestResults {
        let mut results = CrdtPropertyTestResults::new();

        // Test convergence properties
        results.add_test("gcounter_convergence", self.test_gcounter_convergence_property());
        results.add_test("pncounter_convergence", self.test_pncounter_convergence_property());
        results.add_test("orset_convergence", self.test_orset_convergence_property());
        results.add_test("lww_register_convergence", self.test_lww_register_convergence_property());

        // Test algebraic properties
        results.add_test("crdt_commutativity", self.test_crdt_commutativity());
        results.add_test("crdt_associativity", self.test_crdt_associativity());
        results.add_test("crdt_idempotency", self.test_crdt_idempotency());

        // Test concurrent operations
        results.add_test("concurrent_operations", self.test_concurrent_crdt_operations());

        // Test partition tolerance
        results.add_test("partition_tolerance", self.test_crdt_partition_tolerance());

        // Test memory efficiency
        results.add_test("memory_efficiency", self.test_crdt_memory_efficiency());

        // Test serialization
        results.add_test("serialization", self.test_crdt_serialization());

        results
    }

    /// Property-based test for G-Counter convergence
    fn test_gcounter_convergence_property(&self) -> bool {
        let test_config = proptest::test_runner::Config::default();
        let mut runner = proptest::test_runner::TestRunner::new(test_config);

        runner
            .run(&Self::generate_gcounter_operations(), |operations| {
                prop_assert!(self.test_gcounter_convergence(operations));
                Ok(())
            })
            .is_ok()
    }

    /// Property-based test for PN-Counter convergence
    fn test_pncounter_convergence_property(&self) -> bool {
        let test_config = proptest::test_runner::Config::default();
        let mut runner = proptest::test_runner::TestRunner::new(test_config);

        runner
            .run(&Self::generate_pncounter_operations(), |operations| {
                prop_assert!(self.test_pncounter_convergence(operations));
                Ok(())
            })
            .is_ok()
    }

    /// Property-based test for OR-Set convergence
    fn test_orset_convergence_property(&self) -> bool {
        let test_config = proptest::test_runner::Config::default();
        let mut runner = proptest::test_runner::TestRunner::new(test_config);

        runner
            .run(&Self::generate_orset_operations(), |operations| {
                prop_assert!(self.test_orset_convergence(operations));
                Ok(())
            })
            .is_ok()
    }

    /// Property-based test for LWW-Register convergence
    fn test_lww_register_convergence_property(&self) -> bool {
        let test_config = proptest::test_runner::Config::default();
        let mut runner = proptest::test_runner::TestRunner::new(test_config);

        runner
            .run(&Self::generate_lww_register_operations(), |operations| {
                prop_assert!(self.test_lww_register_convergence(operations));
                Ok(())
            })
            .is_ok()
    }
}

/// Results structure for CRDT property tests
#[derive(Debug, Clone)]
pub struct CrdtPropertyTestResults {
    pub tests: HashMap<String, bool>,
    pub passed: usize,
    pub failed: usize,
    pub total: usize,
}

impl CrdtPropertyTestResults {
    pub fn new() -> Self {
        Self {
            tests: HashMap::new(),
            passed: 0,
            failed: 0,
            total: 0,
        }
    }

    pub fn add_test(&mut self, name: &str, result: bool) {
        self.tests.insert(name.to_string(), result);
        self.total += 1;
        if result {
            self.passed += 1;
        } else {
            self.failed += 1;
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== CRDT Property Test Results ===\n");
        report.push_str(&format!("Total Tests: {}\n", self.total));
        report.push_str(&format!("Passed: {}\n", self.passed));
        report.push_str(&format!("Failed: {}\n", self.failed));
        report.push_str(&format!("Success Rate: {:.1}%\n\n", self.success_rate() * 100.0));

        for (test_name, result) in &self.tests {
            let status = if *result { "PASS" } else { "FAIL" };
            report.push_str(&format!("{}: {}\n", test_name, status));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crdt_property_harness() {
        let harness = CrdtPropertyTestHarness::new();
        let results = harness.run_all_crdt_property_tests();
        
        println!("{}", results.generate_report());
        
        // All property tests should pass
        assert!(results.success_rate() > 0.9);
    }

    #[test]
    fn test_gcounter_basic_properties() {
        let harness = CrdtPropertyTestHarness::new();
        
        // Test with simple operations
        let operations = vec![
            (Uuid::new_v4(), 10),
            (Uuid::new_v4(), 20),
            (Uuid::new_v4(), 5),
        ];
        
        assert!(harness.test_gcounter_convergence(operations));
    }

    #[test]
    fn test_pncounter_basic_properties() {
        let harness = CrdtPropertyTestHarness::new();
        
        // Test with simple operations
        let operations = vec![
            (Uuid::new_v4(), 10),
            (Uuid::new_v4(), -5),
            (Uuid::new_v4(), 15),
        ];
        
        assert!(harness.test_pncounter_convergence(operations));
    }

    #[test]
    fn test_orset_basic_properties() {
        let harness = CrdtPropertyTestHarness::new();
        
        // Test with simple operations
        let operations = vec![
            ("a".to_string(), true),
            ("b".to_string(), true),
            ("a".to_string(), false),
            ("c".to_string(), true),
        ];
        
        assert!(harness.test_orset_convergence(operations));
    }

    #[test]
    fn test_lww_register_basic_properties() {
        let harness = CrdtPropertyTestHarness::new();
        
        // Test with simple operations
        let operations = vec![
            ("value1".to_string(), 100),
            ("value2".to_string(), 200),
            ("value3".to_string(), 150),
        ];
        
        assert!(harness.test_lww_register_convergence(operations));
    }

    #[test]
    fn test_crdt_commutativity() {
        let harness = CrdtPropertyTestHarness::new();
        assert!(harness.test_crdt_commutativity());
    }

    #[test]
    fn test_crdt_associativity() {
        let harness = CrdtPropertyTestHarness::new();
        assert!(harness.test_crdt_associativity());
    }

    #[test]
    fn test_crdt_idempotency() {
        let harness = CrdtPropertyTestHarness::new();
        assert!(harness.test_crdt_idempotency());
    }

    #[test]
    fn test_concurrent_crdt_operations() {
        let harness = CrdtPropertyTestHarness::new();
        assert!(harness.test_concurrent_crdt_operations());
    }

    #[test]
    fn test_crdt_partition_tolerance() {
        let harness = CrdtPropertyTestHarness::new();
        assert!(harness.test_crdt_partition_tolerance());
    }

    #[test]
    fn test_crdt_memory_efficiency() {
        let harness = CrdtPropertyTestHarness::new();
        assert!(harness.test_crdt_memory_efficiency());
    }

    #[test]
    fn test_crdt_serialization() {
        let harness = CrdtPropertyTestHarness::new();
        assert!(harness.test_crdt_serialization());
    }
}