//! Counter CRDTs
//!
//! Implementation of G-Counter (grow-only) and PN-Counter (increment/decrement)

use super::{CrdtOperation, CrdtState};
use crate::{NodeId, Timestamp, VersionVector, Result, NeuroError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// G-Counter operation (grow-only)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCounterOp {
    pub node_id: NodeId,
    pub timestamp: Timestamp,
    pub increment: u64,
}

impl CrdtOperation for GCounterOp {
    type State = GCounterState;
    
    fn apply(&self, state: &mut Self::State) -> Result<()> {
        state.apply(self)
    }
    
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
    
    fn node_id(&self) -> NodeId {
        self.node_id
    }
}

/// G-Counter state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCounterState {
    counters: HashMap<NodeId, u64>,
    version_vector: VersionVector,
}

impl CrdtState for GCounterState {
    type Operation = GCounterOp;
    type Value = u64;
    
    fn new() -> Self {
        Self {
            counters: HashMap::new(),
            version_vector: HashMap::new(),
        }
    }
    
    fn apply(&mut self, op: &Self::Operation) -> Result<()> {
        // Update counter for this node
        let current = self.counters.entry(op.node_id).or_insert(0);
        *current = (*current).max(*current + op.increment);
        
        // Update version vector
        let version = self.version_vector.entry(op.node_id).or_insert(0);
        *version = (*version).max(op.timestamp);
        
        Ok(())
    }
    
    fn merge(&mut self, other: &Self) -> Result<()> {
        // Merge counters (take maximum for each node)
        for (node_id, count) in &other.counters {
            let current = self.counters.entry(*node_id).or_insert(0);
            *current = (*current).max(*count);
        }
        
        // Merge version vectors
        for (node_id, timestamp) in &other.version_vector {
            let current = self.version_vector.entry(*node_id).or_insert(0);
            *current = (*current).max(*timestamp);
        }
        
        Ok(())
    }
    
    fn value(&self) -> Self::Value {
        self.counters.values().sum()
    }
    
    fn version_vector(&self) -> VersionVector {
        self.version_vector.clone()
    }
    
    fn is_concurrent(&self, other: &Self) -> bool {
        // Check if version vectors are concurrent
        let mut self_greater = false;
        let mut other_greater = false;
        
        let all_nodes: std::collections::HashSet<_> = self.version_vector.keys()
            .chain(other.version_vector.keys())
            .collect();
        
        for node_id in all_nodes {
            let self_ts = self.version_vector.get(node_id).unwrap_or(&0);
            let other_ts = other.version_vector.get(node_id).unwrap_or(&0);
            
            if self_ts > other_ts {
                self_greater = true;
            } else if other_ts > self_ts {
                other_greater = true;
            }
        }
        
        self_greater && other_greater
    }
}

/// G-Counter (grow-only counter)
pub type GCounter = super::Crdt<GCounterState>;

impl GCounter {
    /// Increment the counter
    pub fn increment(&mut self, node_id: NodeId, amount: u64) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let op = GCounterOp {
            node_id,
            timestamp,
            increment: amount,
        };
        
        self.apply(op)
    }
    
    /// Get the current count
    pub fn count(&self) -> u64 {
        self.value()
    }
}

/// PN-Counter operation (increment/decrement)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PNCounterOp {
    pub node_id: NodeId,
    pub timestamp: Timestamp,
    pub increment: i64,
}

impl CrdtOperation for PNCounterOp {
    type State = PNCounterState;
    
    fn apply(&self, state: &mut Self::State) -> Result<()> {
        state.apply(self)
    }
    
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
    
    fn node_id(&self) -> NodeId {
        self.node_id
    }
}

/// PN-Counter state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PNCounterState {
    positive: GCounterState,
    negative: GCounterState,
    version_vector: VersionVector,
}

impl CrdtState for PNCounterState {
    type Operation = PNCounterOp;
    type Value = i64;
    
    fn new() -> Self {
        Self {
            positive: GCounterState::new(),
            negative: GCounterState::new(),
            version_vector: HashMap::new(),
        }
    }
    
    fn apply(&mut self, op: &Self::Operation) -> Result<()> {
        // Update version vector
        let version = self.version_vector.entry(op.node_id).or_insert(0);
        *version = (*version).max(op.timestamp);
        
        // Apply to appropriate counter
        if op.increment >= 0 {
            let pos_op = GCounterOp {
                node_id: op.node_id,
                timestamp: op.timestamp,
                increment: op.increment as u64,
            };
            self.positive.apply(&pos_op)?;
        } else {
            let neg_op = GCounterOp {
                node_id: op.node_id,
                timestamp: op.timestamp,
                increment: (-op.increment) as u64,
            };
            self.negative.apply(&neg_op)?;
        }
        
        Ok(())
    }
    
    fn merge(&mut self, other: &Self) -> Result<()> {
        self.positive.merge(&other.positive)?;
        self.negative.merge(&other.negative)?;
        
        // Merge version vectors
        for (node_id, timestamp) in &other.version_vector {
            let current = self.version_vector.entry(*node_id).or_insert(0);
            *current = (*current).max(*timestamp);
        }
        
        Ok(())
    }
    
    fn value(&self) -> Self::Value {
        self.positive.value() as i64 - self.negative.value() as i64
    }
    
    fn version_vector(&self) -> VersionVector {
        self.version_vector.clone()
    }
    
    fn is_concurrent(&self, other: &Self) -> bool {
        self.positive.is_concurrent(&other.positive) || 
        self.negative.is_concurrent(&other.negative)
    }
}

/// PN-Counter (increment/decrement counter)
pub type PNCounter = super::Crdt<PNCounterState>;

impl PNCounter {
    /// Increment the counter
    pub fn increment(&mut self, node_id: NodeId, amount: i64) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let op = PNCounterOp {
            node_id,
            timestamp,
            increment: amount,
        };
        
        self.apply(op)
    }
    
    /// Decrement the counter
    pub fn decrement(&mut self, node_id: NodeId, amount: i64) -> Result<()> {
        self.increment(node_id, -amount)
    }
    
    /// Get the current count
    pub fn count(&self) -> i64 {
        self.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_g_counter_basic() {
        let mut counter = GCounter::new();
        let node_id = Uuid::new_v4();
        
        counter.increment(node_id, 5).unwrap();
        assert_eq!(counter.count(), 5);
        
        counter.increment(node_id, 3).unwrap();
        assert_eq!(counter.count(), 8);
    }
    
    #[test]
    fn test_g_counter_merge() {
        let mut counter1 = GCounter::new();
        let mut counter2 = GCounter::new();
        
        let node_id1 = Uuid::new_v4();
        let node_id2 = Uuid::new_v4();
        
        counter1.increment(node_id1, 5).unwrap();
        counter2.increment(node_id2, 3).unwrap();
        
        counter1.merge(&counter2).unwrap();
        
        assert_eq!(counter1.count(), 8);
    }
    
    #[test]
    fn test_pn_counter_basic() {
        let mut counter = PNCounter::new();
        let node_id = Uuid::new_v4();
        
        counter.increment(node_id, 10).unwrap();
        assert_eq!(counter.count(), 10);
        
        counter.decrement(node_id, 3).unwrap();
        assert_eq!(counter.count(), 7);
        
        counter.increment(node_id, -2).unwrap();
        assert_eq!(counter.count(), 5);
    }
    
    #[test]
    fn test_pn_counter_merge() {
        let mut counter1 = PNCounter::new();
        let mut counter2 = PNCounter::new();
        
        let node_id1 = Uuid::new_v4();
        let node_id2 = Uuid::new_v4();
        
        counter1.increment(node_id1, 10).unwrap();
        counter1.decrement(node_id1, 3).unwrap();
        
        counter2.increment(node_id2, 5).unwrap();
        counter2.decrement(node_id2, 2).unwrap();
        
        counter1.merge(&counter2).unwrap();
        
        assert_eq!(counter1.count(), 10); // (10-3) + (5-2) = 7 + 3 = 10
    }
    
    #[test]
    fn test_counter_idempotent_merge() {
        let mut counter1 = GCounter::new();
        let counter2 = counter1.clone();
        
        let node_id = Uuid::new_v4();
        counter1.increment(node_id, 5).unwrap();
        
        let count_before = counter1.count();
        counter1.merge(&counter2).unwrap();
        let count_after = counter1.count();
        
        assert_eq!(count_before, count_after);
    }
}