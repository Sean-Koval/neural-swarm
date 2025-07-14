//! G-Counter (Grow-only Counter) CRDT implementation
//!
//! A G-Counter is a grow-only counter that can only increment.
//! It maintains a vector of counters, one per node, and the total
//! value is the sum of all node counters.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::{Result, NodeId};
use super::{Crdt, CrdtOperation, CrdtError};

/// G-Counter (Grow-only Counter) implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GCounter {
    /// Per-node counters
    counters: HashMap<NodeId, u64>,
    /// Node ID for this counter instance
    node_id: NodeId,
}

impl GCounter {
    /// Create a new G-Counter for the given node
    pub fn new(node_id: NodeId) -> Self {
        let mut counters = HashMap::new();
        counters.insert(node_id, 0);
        
        Self {
            counters,
            node_id,
        }
    }
    
    /// Increment the counter for this node
    pub fn increment(&mut self, amount: u64) {
        let current = self.counters.get(&self.node_id).unwrap_or(&0);
        self.counters.insert(self.node_id, current + amount);
    }
    
    /// Increment the counter for a specific node
    pub fn increment_node(&mut self, node_id: NodeId, amount: u64) {
        let current = self.counters.get(&node_id).unwrap_or(&0);
        self.counters.insert(node_id, current + amount);
    }
    
    /// Get the total value of the counter
    pub fn total(&self) -> u64 {
        self.counters.values().sum()
    }
    
    /// Get the counter value for a specific node
    pub fn get_node_count(&self, node_id: &NodeId) -> u64 {
        self.counters.get(node_id).unwrap_or(&0).clone()
    }
    
    /// Get all node counters
    pub fn get_all_counters(&self) -> &HashMap<NodeId, u64> {
        &self.counters
    }
    
    /// Compare this counter with another for causality
    pub fn compare(&self, other: &GCounter) -> CausalityRelation {
        let mut self_greater = false;
        let mut other_greater = false;
        
        // Get all nodes from both counters
        let all_nodes: std::collections::HashSet<NodeId> = self.counters.keys()
            .chain(other.counters.keys())
            .cloned()
            .collect();
        
        for node_id in all_nodes {
            let self_count = self.counters.get(&node_id).unwrap_or(&0);
            let other_count = other.counters.get(&node_id).unwrap_or(&0);
            
            match self_count.cmp(other_count) {
                std::cmp::Ordering::Greater => self_greater = true,
                std::cmp::Ordering::Less => other_greater = true,
                std::cmp::Ordering::Equal => {},
            }
        }
        
        match (self_greater, other_greater) {
            (true, false) => CausalityRelation::Greater,
            (false, true) => CausalityRelation::Less,
            (false, false) => CausalityRelation::Equal,
            (true, true) => CausalityRelation::Concurrent,
        }
    }
    
    /// Reset the counter (for testing purposes)
    pub fn reset(&mut self) {
        self.counters.clear();
        self.counters.insert(self.node_id, 0);
    }
}

/// Causality relation between two G-Counters
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CausalityRelation {
    Less,
    Greater,
    Equal,
    Concurrent,
}

impl Crdt for GCounter {
    fn merge(&self, other: &Self) -> Result<Self> {
        let mut merged_counters = self.counters.clone();
        
        // Take the maximum counter value for each node
        for (node_id, other_count) in &other.counters {
            let self_count = merged_counters.get(node_id).unwrap_or(&0);
            merged_counters.insert(*node_id, (*self_count).max(*other_count));
        }
        
        Ok(GCounter {
            counters: merged_counters,
            node_id: self.node_id,
        })
    }
    
    fn apply_operation(&mut self, operation: &CrdtOperation) -> Result<()> {
        match operation {
            CrdtOperation::GCounterIncrement { node_id, amount } => {
                self.increment_node(*node_id, *amount);
                Ok(())
            }
            _ => Err(CrdtError::InvalidOperation(format!(
                "Operation {:?} not applicable to G-Counter",
                operation
            )).into()),
        }
    }
    
    fn value(&self) -> serde_json::Value {
        serde_json::Value::Number(serde_json::Number::from(self.total()))
    }
    
    fn is_causally_dependent(&self, other: &Self) -> bool {
        matches!(self.compare(other), CausalityRelation::Greater)
    }
}

impl Default for GCounter {
    fn default() -> Self {
        Self::new(Uuid::new_v4())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_g_counter_creation() {
        let node_id = Uuid::new_v4();
        let counter = GCounter::new(node_id);
        
        assert_eq!(counter.total(), 0);
        assert_eq!(counter.get_node_count(&node_id), 0);
    }
    
    #[test]
    fn test_g_counter_increment() {
        let node_id = Uuid::new_v4();
        let mut counter = GCounter::new(node_id);
        
        counter.increment(5);
        assert_eq!(counter.total(), 5);
        assert_eq!(counter.get_node_count(&node_id), 5);
        
        counter.increment(3);
        assert_eq!(counter.total(), 8);
        assert_eq!(counter.get_node_count(&node_id), 8);
    }
    
    #[test]
    fn test_g_counter_merge() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let mut counter1 = GCounter::new(node1);
        let mut counter2 = GCounter::new(node2);
        
        counter1.increment(5);
        counter2.increment(3);
        
        let merged = counter1.merge(&counter2).unwrap();
        
        assert_eq!(merged.total(), 8);
        assert_eq!(merged.get_node_count(&node1), 5);
        assert_eq!(merged.get_node_count(&node2), 3);
    }
    
    #[test]
    fn test_g_counter_causality() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let mut counter1 = GCounter::new(node1);
        let mut counter2 = GCounter::new(node2);
        
        // Initially equal
        assert_eq!(counter1.compare(&counter2), CausalityRelation::Equal);
        
        // counter1 increments
        counter1.increment(5);
        assert_eq!(counter1.compare(&counter2), CausalityRelation::Greater);
        assert_eq!(counter2.compare(&counter1), CausalityRelation::Less);
        
        // counter2 increments
        counter2.increment(3);
        assert_eq!(counter1.compare(&counter2), CausalityRelation::Concurrent);
        assert_eq!(counter2.compare(&counter1), CausalityRelation::Concurrent);
    }
    
    #[test]
    fn test_g_counter_apply_operation() {
        let node_id = Uuid::new_v4();
        let mut counter = GCounter::new(node_id);
        
        let operation = CrdtOperation::GCounterIncrement { node_id, amount: 10 };
        counter.apply_operation(&operation).unwrap();
        
        assert_eq!(counter.total(), 10);
    }
    
    #[test]
    fn test_g_counter_serialization() {
        let node_id = Uuid::new_v4();
        let mut counter = GCounter::new(node_id);
        counter.increment(42);
        
        let serialized = serde_json::to_string(&counter).unwrap();
        let deserialized: GCounter = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(counter, deserialized);
        assert_eq!(deserialized.total(), 42);
    }
}