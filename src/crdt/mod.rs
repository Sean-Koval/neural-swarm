//! Conflict-free Replicated Data Types (CRDTs)
//!
//! Implementation of various CRDT types for distributed consistency
//! without requiring coordination between nodes.

pub mod counter;
pub mod set;
pub mod register;
pub mod sequence;

pub use counter::{GCounter, PNCounter};
pub use set::ORSet;
pub use register::LWWRegister;
pub use sequence::RGASequence;

use crate::{NodeId, Timestamp, VersionVector, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// CRDT operation trait
pub trait CrdtOperation: Send + Sync + Clone + std::fmt::Debug {
    type State: Send + Sync + Clone + std::fmt::Debug;
    
    /// Apply this operation to a state
    fn apply(&self, state: &mut Self::State) -> Result<()>;
    
    /// Get the timestamp of this operation
    fn timestamp(&self) -> Timestamp;
    
    /// Get the node ID that created this operation
    fn node_id(&self) -> NodeId;
}

/// CRDT state trait
pub trait CrdtState: Send + Sync + Clone + std::fmt::Debug {
    type Operation: CrdtOperation;
    type Value: Send + Sync + Clone + std::fmt::Debug;
    
    /// Create a new empty state
    fn new() -> Self;
    
    /// Apply an operation to this state
    fn apply(&mut self, op: &Self::Operation) -> Result<()>;
    
    /// Merge another state into this one
    fn merge(&mut self, other: &Self) -> Result<()>;
    
    /// Get the current value
    fn value(&self) -> Self::Value;
    
    /// Get the version vector
    fn version_vector(&self) -> VersionVector;
    
    /// Check if this state is causally dependent on another
    fn is_concurrent(&self, other: &Self) -> bool;
}

/// Generic CRDT container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crdt<S: CrdtState> {
    state: S,
    operations: Vec<S::Operation>,
    version_vector: VersionVector,
}

impl<S: CrdtState> Crdt<S> {
    /// Create a new CRDT instance
    pub fn new() -> Self {
        Self {
            state: S::new(),
            operations: Vec::new(),
            version_vector: HashMap::new(),
        }
    }
    
    /// Apply an operation
    pub fn apply(&mut self, op: S::Operation) -> Result<()> {
        // Update version vector
        let node_id = op.node_id();
        let timestamp = op.timestamp();
        let current = self.version_vector.entry(node_id).or_insert(0);
        *current = (*current).max(timestamp);
        
        // Apply to state
        self.state.apply(&op)?;
        
        // Store operation for replay
        self.operations.push(op);
        
        Ok(())
    }
    
    /// Merge another CRDT into this one
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        // Merge version vectors
        for (node_id, timestamp) in &other.version_vector {
            let current = self.version_vector.entry(*node_id).or_insert(0);
            *current = (*current).max(*timestamp);
        }
        
        // Apply missing operations
        for op in &other.operations {
            let node_id = op.node_id();
            let timestamp = op.timestamp();
            let our_timestamp = self.version_vector.get(&node_id).unwrap_or(&0);
            
            if timestamp > *our_timestamp {
                self.state.apply(op)?;
                self.operations.push(op.clone());
            }
        }
        
        Ok(())
    }
    
    /// Get the current value
    pub fn value(&self) -> S::Value {
        self.state.value()
    }
    
    /// Get the version vector
    pub fn version_vector(&self) -> &VersionVector {
        &self.version_vector
    }
    
    /// Get all operations
    pub fn operations(&self) -> &[S::Operation] {
        &self.operations
    }
    
    /// Get operations newer than given version vector
    pub fn delta_operations(&self, version_vector: &VersionVector) -> Vec<S::Operation> {
        self.operations
            .iter()
            .filter(|op| {
                let node_id = op.node_id();
                let timestamp = op.timestamp();
                let remote_timestamp = version_vector.get(&node_id).unwrap_or(&0);
                timestamp > *remote_timestamp
            })
            .cloned()
            .collect()
    }
    
    /// Compact operations (remove redundant ones)
    pub fn compact(&mut self) -> Result<usize> {
        let initial_count = self.operations.len();
        
        // For now, just keep the operations as-is
        // In a real implementation, we would:
        // 1. Remove operations that are superseded by newer ones
        // 2. Merge operations that can be combined
        // 3. Remove operations older than a certain threshold
        
        Ok(initial_count - self.operations.len())
    }
}

impl<S: CrdtState> Default for Crdt<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// CRDT delta for efficient synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtDelta<S: CrdtState> {
    pub operations: Vec<S::Operation>,
    pub version_vector: VersionVector,
}

impl<S: CrdtState> CrdtDelta<S> {
    /// Create a new delta
    pub fn new(operations: Vec<S::Operation>, version_vector: VersionVector) -> Self {
        Self {
            operations,
            version_vector,
        }
    }
    
    /// Check if this delta is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
    
    /// Get the size of this delta
    pub fn size(&self) -> usize {
        self.operations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    // Mock CRDT state for testing
    #[derive(Debug, Clone)]
    struct MockState {
        value: i32,
        version_vector: VersionVector,
    }

    #[derive(Debug, Clone)]
    struct MockOperation {
        delta: i32,
        timestamp: Timestamp,
        node_id: NodeId,
    }

    impl CrdtOperation for MockOperation {
        type State = MockState;
        
        fn apply(&self, state: &mut Self::State) -> Result<()> {
            state.value += self.delta;
            state.version_vector.insert(self.node_id, self.timestamp);
            Ok(())
        }
        
        fn timestamp(&self) -> Timestamp {
            self.timestamp
        }
        
        fn node_id(&self) -> NodeId {
            self.node_id
        }
    }

    impl CrdtState for MockState {
        type Operation = MockOperation;
        type Value = i32;
        
        fn new() -> Self {
            Self {
                value: 0,
                version_vector: HashMap::new(),
            }
        }
        
        fn apply(&mut self, op: &Self::Operation) -> Result<()> {
            op.apply(self)
        }
        
        fn merge(&mut self, other: &Self) -> Result<()> {
            self.value += other.value;
            for (node_id, timestamp) in &other.version_vector {
                let current = self.version_vector.entry(*node_id).or_insert(0);
                *current = (*current).max(*timestamp);
            }
            Ok(())
        }
        
        fn value(&self) -> Self::Value {
            self.value
        }
        
        fn version_vector(&self) -> VersionVector {
            self.version_vector.clone()
        }
        
        fn is_concurrent(&self, other: &Self) -> bool {
            // Simplified concurrency check
            for (node_id, timestamp) in &self.version_vector {
                if let Some(other_timestamp) = other.version_vector.get(node_id) {
                    if timestamp > other_timestamp {
                        return true;
                    }
                }
            }
            false
        }
    }

    #[test]
    fn test_crdt_basic_operations() {
        let mut crdt = Crdt::<MockState>::new();
        let node_id = Uuid::new_v4();
        
        let op1 = MockOperation {
            delta: 5,
            timestamp: 1,
            node_id,
        };
        
        let op2 = MockOperation {
            delta: 3,
            timestamp: 2,
            node_id,
        };
        
        crdt.apply(op1).unwrap();
        crdt.apply(op2).unwrap();
        
        assert_eq!(crdt.value(), 8);
        assert_eq!(crdt.operations().len(), 2);
        assert_eq!(crdt.version_vector().get(&node_id), Some(&2));
    }
    
    #[test]
    fn test_crdt_merge() {
        let mut crdt1 = Crdt::<MockState>::new();
        let mut crdt2 = Crdt::<MockState>::new();
        
        let node_id1 = Uuid::new_v4();
        let node_id2 = Uuid::new_v4();
        
        let op1 = MockOperation {
            delta: 5,
            timestamp: 1,
            node_id: node_id1,
        };
        
        let op2 = MockOperation {
            delta: 3,
            timestamp: 1,
            node_id: node_id2,
        };
        
        crdt1.apply(op1).unwrap();
        crdt2.apply(op2).unwrap();
        
        crdt1.merge(&crdt2).unwrap();
        
        assert_eq!(crdt1.value(), 8);
        assert_eq!(crdt1.operations().len(), 2);
    }
    
    #[test]
    fn test_crdt_delta_operations() {
        let mut crdt = Crdt::<MockState>::new();
        let node_id = Uuid::new_v4();
        
        let op1 = MockOperation {
            delta: 5,
            timestamp: 1,
            node_id,
        };
        
        let op2 = MockOperation {
            delta: 3,
            timestamp: 2,
            node_id,
        };
        
        crdt.apply(op1).unwrap();
        crdt.apply(op2).unwrap();
        
        let mut version_vector = HashMap::new();
        version_vector.insert(node_id, 1);
        
        let delta_ops = crdt.delta_operations(&version_vector);
        assert_eq!(delta_ops.len(), 1);
        assert_eq!(delta_ops[0].delta, 3);
    }
}