//! CRDT Correctness Tests for Neural Swarm
//!
//! This module provides comprehensive testing for Conflict-free Replicated Data Types (CRDTs)
//! ensuring convergence, commutativity, and associativity properties in distributed neural networks.

use neural_swarm::{
    memory::{MemoryManager, AgentMemory},
    agents::AgentId,
    NeuralFloat,
};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::{SystemTime, Duration},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use rand::{thread_rng, Rng};

/// CRDT operation types for neural network state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CRDTOperation {
    /// Update neural network weights
    WeightUpdate {
        layer_id: u32,
        neuron_id: u32,
        weight_id: u32,
        value: NeuralFloat,
        timestamp: SystemTime,
        origin: AgentId,
    },
    /// Add new neural connection
    ConnectionAdd {
        from_layer: u32,
        from_neuron: u32,
        to_layer: u32,
        to_neuron: u32,
        weight: NeuralFloat,
        timestamp: SystemTime,
        origin: AgentId,
    },
    /// Remove neural connection
    ConnectionRemove {
        from_layer: u32,
        from_neuron: u32,
        to_layer: u32,
        to_neuron: u32,
        timestamp: SystemTime,
        origin: AgentId,
    },
    /// Update learning rate
    LearningRateUpdate {
        layer_id: u32,
        rate: NeuralFloat,
        timestamp: SystemTime,
        origin: AgentId,
    },
    /// Batch gradient update
    GradientUpdate {
        layer_id: u32,
        gradients: Vec<NeuralFloat>,
        timestamp: SystemTime,
        origin: AgentId,
    },
}

impl CRDTOperation {
    /// Get the timestamp for ordering operations
    pub fn timestamp(&self) -> SystemTime {
        match self {
            CRDTOperation::WeightUpdate { timestamp, .. } => *timestamp,
            CRDTOperation::ConnectionAdd { timestamp, .. } => *timestamp,
            CRDTOperation::ConnectionRemove { timestamp, .. } => *timestamp,
            CRDTOperation::LearningRateUpdate { timestamp, .. } => *timestamp,
            CRDTOperation::GradientUpdate { timestamp, .. } => *timestamp,
        }
    }

    /// Get the origin agent for the operation
    pub fn origin(&self) -> AgentId {
        match self {
            CRDTOperation::WeightUpdate { origin, .. } => *origin,
            CRDTOperation::ConnectionAdd { origin, .. } => *origin,
            CRDTOperation::ConnectionRemove { origin, .. } => *origin,
            CRDTOperation::LearningRateUpdate { origin, .. } => *origin,
            CRDTOperation::GradientUpdate { origin, .. } => *origin,
        }
    }

    /// Get operation priority for conflict resolution
    pub fn priority(&self) -> u32 {
        match self {
            CRDTOperation::WeightUpdate { .. } => 3,
            CRDTOperation::ConnectionAdd { .. } => 2,
            CRDTOperation::ConnectionRemove { .. } => 1,
            CRDTOperation::LearningRateUpdate { .. } => 4,
            CRDTOperation::GradientUpdate { .. } => 5,
        }
    }
}

/// CRDT state for neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCRDTState {
    /// Network weights organized by layer -> neuron -> weight
    pub weights: HashMap<u32, HashMap<u32, HashMap<u32, NeuralFloat>>>,
    /// Neural connections
    pub connections: HashMap<(u32, u32), HashMap<(u32, u32), NeuralFloat>>,
    /// Learning rates per layer
    pub learning_rates: HashMap<u32, NeuralFloat>,
    /// Operation log for debugging
    pub operation_log: Vec<CRDTOperation>,
    /// Vector clock for causality tracking
    pub vector_clock: HashMap<AgentId, u64>,
    /// Last update timestamp
    pub last_update: SystemTime,
}

impl NeuralCRDTState {
    /// Create new CRDT state
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            connections: HashMap::new(),
            learning_rates: HashMap::new(),
            operation_log: Vec::new(),
            vector_clock: HashMap::new(),
            last_update: SystemTime::now(),
        }
    }

    /// Apply a CRDT operation to the state
    pub fn apply_operation(&mut self, operation: CRDTOperation) -> Result<(), String> {
        // Update vector clock
        let origin = operation.origin();
        let clock_value = self.vector_clock.get(&origin).unwrap_or(&0) + 1;
        self.vector_clock.insert(origin, clock_value);

        // Apply operation based on type
        match operation.clone() {
            CRDTOperation::WeightUpdate { layer_id, neuron_id, weight_id, value, .. } => {
                self.weights
                    .entry(layer_id)
                    .or_default()
                    .entry(neuron_id)
                    .or_default()
                    .insert(weight_id, value);
            }
            CRDTOperation::ConnectionAdd { from_layer, from_neuron, to_layer, to_neuron, weight, .. } => {
                self.connections
                    .entry((from_layer, from_neuron))
                    .or_default()
                    .insert((to_layer, to_neuron), weight);
            }
            CRDTOperation::ConnectionRemove { from_layer, from_neuron, to_layer, to_neuron, .. } => {
                if let Some(connections) = self.connections.get_mut(&(from_layer, from_neuron)) {
                    connections.remove(&(to_layer, to_neuron));
                }
            }
            CRDTOperation::LearningRateUpdate { layer_id, rate, .. } => {
                self.learning_rates.insert(layer_id, rate);
            }
            CRDTOperation::GradientUpdate { layer_id, gradients, .. } => {
                // Apply gradients to weights (simplified)
                if let Some(layer_weights) = self.weights.get_mut(&layer_id) {
                    for (neuron_id, neuron_weights) in layer_weights.iter_mut() {
                        for (weight_id, weight) in neuron_weights.iter_mut() {
                            if let Some(gradient) = gradients.get(*weight_id as usize) {
                                let learning_rate = self.learning_rates.get(&layer_id).unwrap_or(&0.01);
                                *weight -= learning_rate * gradient;
                            }
                        }
                    }
                }
            }
        }

        // Add to operation log
        self.operation_log.push(operation);
        self.last_update = SystemTime::now();

        Ok(())
    }

    /// Merge with another CRDT state
    pub fn merge(&mut self, other: &NeuralCRDTState) -> Result<(), String> {
        // Merge operations based on timestamps and causality
        let mut combined_ops = self.operation_log.clone();
        combined_ops.extend(other.operation_log.clone());

        // Sort operations by timestamp for deterministic merge
        combined_ops.sort_by(|a, b| {
            a.timestamp().cmp(&b.timestamp())
                .then_with(|| a.origin().cmp(&b.origin()))
                .then_with(|| a.priority().cmp(&b.priority()))
        });

        // Clear current state and reapply all operations
        self.weights.clear();
        self.connections.clear();
        self.learning_rates.clear();
        self.operation_log.clear();
        self.vector_clock.clear();

        // Reapply operations in correct order
        for operation in combined_ops {
            self.apply_operation(operation)?;
        }

        Ok(())
    }

    /// Check if this state is causally consistent with another
    pub fn is_causally_consistent(&self, other: &NeuralCRDTState) -> bool {
        // Check vector clocks for causality
        for (agent, clock) in &self.vector_clock {
            if let Some(other_clock) = other.vector_clock.get(agent) {
                if clock > other_clock {
                    return false;
                }
            }
        }
        true
    }

    /// Get state hash for consistency checking
    pub fn state_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        // Hash weights
        for (layer_id, layer) in &self.weights {
            layer_id.hash(&mut hasher);
            for (neuron_id, neuron) in layer {
                neuron_id.hash(&mut hasher);
                for (weight_id, weight) in neuron {
                    weight_id.hash(&mut hasher);
                    weight.to_bits().hash(&mut hasher);
                }
            }
        }

        // Hash connections
        for (from, to_map) in &self.connections {
            from.hash(&mut hasher);
            for (to, weight) in to_map {
                to.hash(&mut hasher);
                weight.to_bits().hash(&mut hasher);
            }
        }

        // Hash learning rates
        for (layer_id, rate) in &self.learning_rates {
            layer_id.hash(&mut hasher);
            rate.to_bits().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get operation count
    pub fn operation_count(&self) -> usize {
        self.operation_log.len()
    }
}

/// CRDT replica for testing
#[derive(Debug)]
pub struct CRDTReplica {
    pub id: AgentId,
    pub state: Arc<RwLock<NeuralCRDTState>>,
    pub pending_operations: Arc<RwLock<Vec<CRDTOperation>>>,
}

impl CRDTReplica {
    /// Create new CRDT replica
    pub fn new(id: AgentId) -> Self {
        Self {
            id,
            state: Arc::new(RwLock::new(NeuralCRDTState::new())),
            pending_operations: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Generate a random operation
    pub fn generate_random_operation(&self) -> CRDTOperation {
        let mut rng = thread_rng();
        let timestamp = SystemTime::now();

        match rng.gen_range(0..5) {
            0 => CRDTOperation::WeightUpdate {
                layer_id: rng.gen_range(0..5),
                neuron_id: rng.gen_range(0..10),
                weight_id: rng.gen_range(0..10),
                value: rng.gen_range(-1.0..1.0),
                timestamp,
                origin: self.id,
            },
            1 => CRDTOperation::ConnectionAdd {
                from_layer: rng.gen_range(0..4),
                from_neuron: rng.gen_range(0..10),
                to_layer: rng.gen_range(1..5),
                to_neuron: rng.gen_range(0..10),
                weight: rng.gen_range(-1.0..1.0),
                timestamp,
                origin: self.id,
            },
            2 => CRDTOperation::ConnectionRemove {
                from_layer: rng.gen_range(0..4),
                from_neuron: rng.gen_range(0..10),
                to_layer: rng.gen_range(1..5),
                to_neuron: rng.gen_range(0..10),
                timestamp,
                origin: self.id,
            },
            3 => CRDTOperation::LearningRateUpdate {
                layer_id: rng.gen_range(0..5),
                rate: rng.gen_range(0.001..0.1),
                timestamp,
                origin: self.id,
            },
            4 => CRDTOperation::GradientUpdate {
                layer_id: rng.gen_range(0..5),
                gradients: (0..10).map(|_| rng.gen_range(-0.1..0.1)).collect(),
                timestamp,
                origin: self.id,
            },
            _ => unreachable!(),
        }
    }

    /// Apply local operation
    pub fn apply_local_operation(&self, operation: CRDTOperation) -> Result<(), String> {
        let mut state = self.state.write().unwrap();
        state.apply_operation(operation.clone())?;
        
        let mut pending = self.pending_operations.write().unwrap();
        pending.push(operation);
        
        Ok(())
    }

    /// Receive and apply remote operation
    pub fn receive_operation(&self, operation: CRDTOperation) -> Result<(), String> {
        let mut state = self.state.write().unwrap();
        state.apply_operation(operation)
    }

    /// Sync with another replica
    pub fn sync_with(&self, other: &CRDTReplica) -> Result<(), String> {
        let other_state = other.state.read().unwrap();
        let mut my_state = self.state.write().unwrap();
        
        my_state.merge(&other_state)?;
        
        Ok(())
    }

    /// Get current state hash
    pub fn get_state_hash(&self) -> u64 {
        let state = self.state.read().unwrap();
        state.state_hash()
    }

    /// Get operation count
    pub fn get_operation_count(&self) -> usize {
        let state = self.state.read().unwrap();
        state.operation_count()
    }
}

/// CRDT test framework
pub struct CRDTTestFramework {
    pub replicas: Vec<CRDTReplica>,
    pub test_duration: Duration,
    pub operation_rate: u64,
}

impl CRDTTestFramework {
    /// Create new CRDT test framework
    pub fn new(replica_count: usize, test_duration: Duration, operation_rate: u64) -> Self {
        let mut replicas = Vec::new();
        
        for i in 0..replica_count {
            let mut agent_id = [0u8; 32];
            agent_id[0] = i as u8;
            replicas.push(CRDTReplica::new(agent_id));
        }

        Self {
            replicas,
            test_duration,
            operation_rate,
        }
    }

    /// Test CRDT convergence property
    pub async fn test_convergence(&self) -> CRDTTestResult {
        let mut result = CRDTTestResult::new("CRDT Convergence Test");
        
        // Phase 1: Generate concurrent operations
        let mut handles = Vec::new();
        
        for replica in &self.replicas {
            let replica_id = replica.id;
            let replica_clone = replica.clone();
            
            let handle = tokio::spawn(async move {
                let mut operations = Vec::new();
                for _ in 0..100 {
                    let operation = replica_clone.generate_random_operation();
                    operations.push(operation.clone());
                    replica_clone.apply_local_operation(operation).unwrap();
                    
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                operations
            });
            
            handles.push(handle);
        }

        // Wait for all operations to complete
        let mut all_operations = Vec::new();
        for handle in handles {
            let operations = handle.await.unwrap();
            all_operations.extend(operations);
        }

        // Phase 2: Synchronize all replicas
        for i in 0..self.replicas.len() {
            for j in i + 1..self.replicas.len() {
                self.replicas[i].sync_with(&self.replicas[j]).unwrap();
            }
        }

        // Phase 3: Check convergence
        let first_hash = self.replicas[0].get_state_hash();
        let mut converged = true;
        
        for replica in &self.replicas[1..] {
            if replica.get_state_hash() != first_hash {
                converged = false;
                break;
            }
        }

        if converged {
            result.mark_passed("All replicas converged to same state");
        } else {
            result.mark_failed("Replicas did not converge");
        }

        result.operation_count = all_operations.len();
        result.replica_count = self.replicas.len();
        result
    }

    /// Test CRDT commutativity property
    pub async fn test_commutativity(&self) -> CRDTTestResult {
        let mut result = CRDTTestResult::new("CRDT Commutativity Test");
        
        if self.replicas.len() < 2 {
            result.mark_failed("Need at least 2 replicas for commutativity test");
            return result;
        }

        // Generate operations
        let mut operations = Vec::new();
        for i in 0..50 {
            let operation = self.replicas[0].generate_random_operation();
            operations.push(operation);
        }

        // Apply operations in different orders
        let replica1 = &self.replicas[0];
        let replica2 = &self.replicas[1];

        // Reset states
        *replica1.state.write().unwrap() = NeuralCRDTState::new();
        *replica2.state.write().unwrap() = NeuralCRDTState::new();

        // Apply in original order to replica1
        for operation in &operations {
            replica1.apply_local_operation(operation.clone()).unwrap();
        }

        // Apply in reverse order to replica2
        for operation in operations.iter().rev() {
            replica2.apply_local_operation(operation.clone()).unwrap();
        }

        // Check if final states are equivalent
        let hash1 = replica1.get_state_hash();
        let hash2 = replica2.get_state_hash();

        if hash1 == hash2 {
            result.mark_passed("Operations are commutative");
        } else {
            result.mark_failed("Operations are not commutative");
        }

        result.operation_count = operations.len();
        result.replica_count = 2;
        result
    }

    /// Test CRDT associativity property
    pub async fn test_associativity(&self) -> CRDTTestResult {
        let mut result = CRDTTestResult::new("CRDT Associativity Test");
        
        if self.replicas.len() < 3 {
            result.mark_failed("Need at least 3 replicas for associativity test");
            return result;
        }

        // Generate operations
        let mut operations = Vec::new();
        for _ in 0..30 {
            let operation = self.replicas[0].generate_random_operation();
            operations.push(operation);
        }

        // Split operations into groups
        let group1 = &operations[0..10];
        let group2 = &operations[10..20];
        let group3 = &operations[20..30];

        // Test (A ∪ B) ∪ C = A ∪ (B ∪ C)
        let replica1 = &self.replicas[0];
        let replica2 = &self.replicas[1];
        let replica3 = &self.replicas[2];

        // Reset states
        *replica1.state.write().unwrap() = NeuralCRDTState::new();
        *replica2.state.write().unwrap() = NeuralCRDTState::new();
        *replica3.state.write().unwrap() = NeuralCRDTState::new();

        // Apply (A ∪ B) ∪ C to replica1
        for operation in group1 {
            replica1.apply_local_operation(operation.clone()).unwrap();
        }
        for operation in group2 {
            replica1.apply_local_operation(operation.clone()).unwrap();
        }
        for operation in group3 {
            replica1.apply_local_operation(operation.clone()).unwrap();
        }

        // Apply A ∪ (B ∪ C) to replica2
        for operation in group1 {
            replica2.apply_local_operation(operation.clone()).unwrap();
        }
        for operation in group2 {
            replica3.apply_local_operation(operation.clone()).unwrap();
        }
        for operation in group3 {
            replica3.apply_local_operation(operation.clone()).unwrap();
        }
        replica2.sync_with(replica3).unwrap();

        // Check if results are equivalent
        let hash1 = replica1.get_state_hash();
        let hash2 = replica2.get_state_hash();

        if hash1 == hash2 {
            result.mark_passed("Operations are associative");
        } else {
            result.mark_failed("Operations are not associative");
        }

        result.operation_count = operations.len();
        result.replica_count = 3;
        result
    }

    /// Test CRDT idempotency property
    pub async fn test_idempotency(&self) -> CRDTTestResult {
        let mut result = CRDTTestResult::new("CRDT Idempotency Test");
        
        let replica = &self.replicas[0];
        
        // Generate operation
        let operation = replica.generate_random_operation();
        
        // Apply operation once
        replica.apply_local_operation(operation.clone()).unwrap();
        let hash1 = replica.get_state_hash();
        
        // Apply same operation again
        replica.apply_local_operation(operation.clone()).unwrap();
        let hash2 = replica.get_state_hash();
        
        if hash1 == hash2 {
            result.mark_passed("Operations are idempotent");
        } else {
            result.mark_failed("Operations are not idempotent");
        }

        result.operation_count = 2;
        result.replica_count = 1;
        result
    }

    /// Test CRDT conflict resolution
    pub async fn test_conflict_resolution(&self) -> CRDTTestResult {
        let mut result = CRDTTestResult::new("CRDT Conflict Resolution Test");
        
        if self.replicas.len() < 2 {
            result.mark_failed("Need at least 2 replicas for conflict resolution test");
            return result;
        }

        let replica1 = &self.replicas[0];
        let replica2 = &self.replicas[1];

        // Reset states
        *replica1.state.write().unwrap() = NeuralCRDTState::new();
        *replica2.state.write().unwrap() = NeuralCRDTState::new();

        // Create conflicting operations (same weight update)
        let timestamp = SystemTime::now();
        let operation1 = CRDTOperation::WeightUpdate {
            layer_id: 0,
            neuron_id: 0,
            weight_id: 0,
            value: 0.5,
            timestamp,
            origin: replica1.id,
        };
        
        let operation2 = CRDTOperation::WeightUpdate {
            layer_id: 0,
            neuron_id: 0,
            weight_id: 0,
            value: 0.8,
            timestamp,
            origin: replica2.id,
        };

        // Apply operations to respective replicas
        replica1.apply_local_operation(operation1.clone()).unwrap();
        replica2.apply_local_operation(operation2.clone()).unwrap();

        // Exchange operations
        replica1.receive_operation(operation2).unwrap();
        replica2.receive_operation(operation1).unwrap();

        // Check if conflict was resolved consistently
        let hash1 = replica1.get_state_hash();
        let hash2 = replica2.get_state_hash();

        if hash1 == hash2 {
            result.mark_passed("Conflicts resolved consistently");
        } else {
            result.mark_failed("Conflict resolution inconsistent");
        }

        result.operation_count = 2;
        result.replica_count = 2;
        result
    }

    /// Run comprehensive CRDT tests
    pub async fn run_comprehensive_tests(&self) -> Vec<CRDTTestResult> {
        let mut results = Vec::new();
        
        results.push(self.test_convergence().await);
        results.push(self.test_commutativity().await);
        results.push(self.test_associativity().await);
        results.push(self.test_idempotency().await);
        results.push(self.test_conflict_resolution().await);
        
        results
    }
}

impl Clone for CRDTReplica {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            state: self.state.clone(),
            pending_operations: self.pending_operations.clone(),
        }
    }
}

/// CRDT test result
#[derive(Debug, Clone)]
pub struct CRDTTestResult {
    pub test_name: String,
    pub status: CRDTTestStatus,
    pub message: String,
    pub operation_count: usize,
    pub replica_count: usize,
    pub execution_time: Duration,
}

impl CRDTTestResult {
    pub fn new(name: &str) -> Self {
        Self {
            test_name: name.to_string(),
            status: CRDTTestStatus::Running,
            message: String::new(),
            operation_count: 0,
            replica_count: 0,
            execution_time: Duration::new(0, 0),
        }
    }

    pub fn mark_passed(mut self, message: &str) -> Self {
        self.status = CRDTTestStatus::Passed;
        self.message = message.to_string();
        self
    }

    pub fn mark_failed(mut self, message: &str) -> Self {
        self.status = CRDTTestStatus::Failed;
        self.message = message.to_string();
        self
    }

    pub fn mark_warning(mut self, message: &str) -> Self {
        self.status = CRDTTestStatus::Warning;
        self.message = message.to_string();
        self
    }
}

/// CRDT test status
#[derive(Debug, Clone, PartialEq)]
pub enum CRDTTestStatus {
    Running,
    Passed,
    Failed,
    Warning,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_crdt_convergence() {
        let framework = CRDTTestFramework::new(3, Duration::from_secs(10), 100);
        let result = framework.test_convergence().await;
        
        println!("Convergence test: {:?}", result);
        assert_eq!(result.status, CRDTTestStatus::Passed);
    }

    #[tokio::test]
    async fn test_crdt_commutativity() {
        let framework = CRDTTestFramework::new(2, Duration::from_secs(5), 50);
        let result = framework.test_commutativity().await;
        
        println!("Commutativity test: {:?}", result);
        assert_eq!(result.status, CRDTTestStatus::Passed);
    }

    #[tokio::test]
    async fn test_crdt_associativity() {
        let framework = CRDTTestFramework::new(3, Duration::from_secs(5), 30);
        let result = framework.test_associativity().await;
        
        println!("Associativity test: {:?}", result);
        assert_eq!(result.status, CRDTTestStatus::Passed);
    }

    #[tokio::test]
    async fn test_crdt_idempotency() {
        let framework = CRDTTestFramework::new(1, Duration::from_secs(1), 2);
        let result = framework.test_idempotency().await;
        
        println!("Idempotency test: {:?}", result);
        assert_eq!(result.status, CRDTTestStatus::Passed);
    }

    #[tokio::test]
    async fn test_crdt_conflict_resolution() {
        let framework = CRDTTestFramework::new(2, Duration::from_secs(5), 2);
        let result = framework.test_conflict_resolution().await;
        
        println!("Conflict resolution test: {:?}", result);
        assert_eq!(result.status, CRDTTestStatus::Passed);
    }

    #[tokio::test]
    async fn test_comprehensive_crdt_suite() {
        let framework = CRDTTestFramework::new(4, Duration::from_secs(20), 200);
        let results = framework.run_comprehensive_tests().await;
        
        println!("Comprehensive CRDT test results:");
        for result in &results {
            println!("  {} - {:?}: {}", result.test_name, result.status, result.message);
        }
        
        let passed_count = results.iter().filter(|r| r.status == CRDTTestStatus::Passed).count();
        assert!(passed_count >= 4, "Expected at least 4 tests to pass");
    }

    #[test]
    fn test_crdt_operation_creation() {
        let mut agent_id = [0u8; 32];
        agent_id[0] = 1;
        
        let replica = CRDTReplica::new(agent_id);
        let operation = replica.generate_random_operation();
        
        assert_eq!(operation.origin(), agent_id);
        assert!(operation.timestamp() <= SystemTime::now());
        assert!(operation.priority() > 0);
    }

    #[test]
    fn test_crdt_state_operations() {
        let mut state = NeuralCRDTState::new();
        
        let mut agent_id = [0u8; 32];
        agent_id[0] = 1;
        
        let operation = CRDTOperation::WeightUpdate {
            layer_id: 0,
            neuron_id: 0,
            weight_id: 0,
            value: 0.5,
            timestamp: SystemTime::now(),
            origin: agent_id,
        };
        
        assert!(state.apply_operation(operation).is_ok());
        assert_eq!(state.operation_count(), 1);
        
        let weight = state.weights
            .get(&0)
            .and_then(|layer| layer.get(&0))
            .and_then(|neuron| neuron.get(&0));
        assert_eq!(weight, Some(&0.5));
    }

    #[test]
    fn test_crdt_state_merge() {
        let mut state1 = NeuralCRDTState::new();
        let mut state2 = NeuralCRDTState::new();
        
        let mut agent_id = [0u8; 32];
        agent_id[0] = 1;
        
        let operation1 = CRDTOperation::WeightUpdate {
            layer_id: 0,
            neuron_id: 0,
            weight_id: 0,
            value: 0.5,
            timestamp: SystemTime::now(),
            origin: agent_id,
        };
        
        let operation2 = CRDTOperation::WeightUpdate {
            layer_id: 0,
            neuron_id: 0,
            weight_id: 1,
            value: 0.8,
            timestamp: SystemTime::now(),
            origin: agent_id,
        };
        
        state1.apply_operation(operation1).unwrap();
        state2.apply_operation(operation2).unwrap();
        
        assert!(state1.merge(&state2).is_ok());
        assert_eq!(state1.operation_count(), 2);
    }

    #[test]
    fn test_crdt_replica_sync() {
        let mut agent_id1 = [0u8; 32];
        agent_id1[0] = 1;
        let mut agent_id2 = [0u8; 32];
        agent_id2[0] = 2;
        
        let replica1 = CRDTReplica::new(agent_id1);
        let replica2 = CRDTReplica::new(agent_id2);
        
        let operation = replica1.generate_random_operation();
        replica1.apply_local_operation(operation.clone()).unwrap();
        
        let hash1_before = replica1.get_state_hash();
        let hash2_before = replica2.get_state_hash();
        
        assert_ne!(hash1_before, hash2_before);
        
        replica2.sync_with(&replica1).unwrap();
        
        let hash1_after = replica1.get_state_hash();
        let hash2_after = replica2.get_state_hash();
        
        assert_eq!(hash1_after, hash2_after);
    }
}