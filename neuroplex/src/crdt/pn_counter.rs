//! PN-Counter (Increment/Decrement Counter) CRDT implementation
//!
//! A PN-Counter combines two G-Counters: one for increments and one for decrements.
//! The total value is the difference between the two counters.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::{Result, NodeId};
use super::{Crdt, CrdtOperation, CrdtError, GCounter};

/// PN-Counter (Increment/Decrement Counter) implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PNCounter {
    /// Counter for increments
    increments: GCounter,
    /// Counter for decrements
    decrements: GCounter,
    /// Node ID for this counter instance
    node_id: NodeId,
}

impl PNCounter {
    /// Create a new PN-Counter for the given node
    pub fn new(node_id: NodeId) -> Self {
        Self {
            increments: GCounter::new(node_id),
            decrements: GCounter::new(node_id),
            node_id,
        }
    }
    
    /// Increment the counter for this node
    pub fn increment(&mut self, amount: i64) {
        if amount > 0 {
            self.increments.increment_node(self.node_id, amount as u64);
        } else if amount < 0 {
            self.decrements.increment_node(self.node_id, (-amount) as u64);
        }
    }
    
    /// Decrement the counter for this node
    pub fn decrement(&mut self, amount: i64) {
        if amount > 0 {
            self.decrements.increment_node(self.node_id, amount as u64);
        } else if amount < 0 {
            self.increments.increment_node(self.node_id, (-amount) as u64);
        }
    }
    
    /// Increment the counter for a specific node
    pub fn increment_node(&mut self, node_id: NodeId, amount: i64) {
        if amount > 0 {
            self.increments.increment_node(node_id, amount as u64);
        } else if amount < 0 {
            self.decrements.increment_node(node_id, (-amount) as u64);
        }
    }
    
    /// Decrement the counter for a specific node
    pub fn decrement_node(&mut self, node_id: NodeId, amount: i64) {
        if amount > 0 {
            self.decrements.increment_node(node_id, amount as u64);
        } else if amount < 0 {
            self.increments.increment_node(node_id, (-amount) as u64);
        }
    }
    
    /// Get the total value of the counter
    pub fn total(&self) -> i64 {
        self.increments.total() as i64 - self.decrements.total() as i64
    }
    
    /// Get the increment counter value for a specific node
    pub fn get_node_increments(&self, node_id: &NodeId) -> u64 {
        self.increments.get_node_count(node_id)
    }
    
    /// Get the decrement counter value for a specific node
    pub fn get_node_decrements(&self, node_id: &NodeId) -> u64 {
        self.decrements.get_node_count(node_id)
    }
    
    /// Get the net value for a specific node
    pub fn get_node_total(&self, node_id: &NodeId) -> i64 {
        self.get_node_increments(node_id) as i64 - self.get_node_decrements(node_id) as i64
    }
    
    /// Get all node increments
    pub fn get_all_increments(&self) -> &HashMap<NodeId, u64> {
        self.increments.get_all_counters()
    }
    
    /// Get all node decrements
    pub fn get_all_decrements(&self) -> &HashMap<NodeId, u64> {
        self.decrements.get_all_counters()
    }
    
    /// Compare this counter with another for causality
    pub fn compare(&self, other: &PNCounter) -> CausalityRelation {
        let inc_relation = self.increments.compare(&other.increments);
        let dec_relation = self.decrements.compare(&other.decrements);
        
        match (inc_relation, dec_relation) {
            (super::g_counter::CausalityRelation::Equal, super::g_counter::CausalityRelation::Equal) => CausalityRelation::Equal,
            (super::g_counter::CausalityRelation::Greater, super::g_counter::CausalityRelation::Greater) |
            (super::g_counter::CausalityRelation::Greater, super::g_counter::CausalityRelation::Equal) |
            (super::g_counter::CausalityRelation::Equal, super::g_counter::CausalityRelation::Greater) => CausalityRelation::Greater,
            (super::g_counter::CausalityRelation::Less, super::g_counter::CausalityRelation::Less) |
            (super::g_counter::CausalityRelation::Less, super::g_counter::CausalityRelation::Equal) |
            (super::g_counter::CausalityRelation::Equal, super::g_counter::CausalityRelation::Less) => CausalityRelation::Less,
            _ => CausalityRelation::Concurrent,
        }
    }
    
    /// Reset the counter (for testing purposes)
    pub fn reset(&mut self) {
        self.increments.reset();
        self.decrements.reset();
    }
}

/// Causality relation between two PN-Counters
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CausalityRelation {
    Less,
    Greater,
    Equal,
    Concurrent,
}

impl Crdt for PNCounter {
    fn merge(&self, other: &Self) -> Result<Self> {
        let merged_increments = self.increments.merge(&other.increments)?;
        let merged_decrements = self.decrements.merge(&other.decrements)?;
        
        Ok(PNCounter {
            increments: merged_increments,
            decrements: merged_decrements,
            node_id: self.node_id,
        })
    }
    
    fn apply_operation(&mut self, operation: &CrdtOperation) -> Result<()> {
        match operation {
            CrdtOperation::PNCounterIncrement { node_id, amount } => {
                self.increment_node(*node_id, *amount);
                Ok(())
            }
            CrdtOperation::PNCounterDecrement { node_id, amount } => {
                self.decrement_node(*node_id, *amount);
                Ok(())
            }
            _ => Err(CrdtError::InvalidOperation(format!(
                "Operation {:?} not applicable to PN-Counter",
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

impl Default for PNCounter {
    fn default() -> Self {
        Self::new(Uuid::new_v4())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pn_counter_creation() {
        let node_id = Uuid::new_v4();
        let counter = PNCounter::new(node_id);
        
        assert_eq!(counter.total(), 0);
        assert_eq!(counter.get_node_total(&node_id), 0);
    }
    
    #[test]
    fn test_pn_counter_increment() {
        let node_id = Uuid::new_v4();
        let mut counter = PNCounter::new(node_id);
        
        counter.increment(5);
        assert_eq!(counter.total(), 5);
        assert_eq!(counter.get_node_total(&node_id), 5);
        
        counter.increment(3);
        assert_eq!(counter.total(), 8);
        assert_eq!(counter.get_node_total(&node_id), 8);
    }
    
    #[test]
    fn test_pn_counter_decrement() {
        let node_id = Uuid::new_v4();
        let mut counter = PNCounter::new(node_id);
        
        counter.increment(10);
        counter.decrement(3);
        assert_eq!(counter.total(), 7);
        assert_eq!(counter.get_node_total(&node_id), 7);
        
        counter.decrement(5);
        assert_eq!(counter.total(), 2);
        assert_eq!(counter.get_node_total(&node_id), 2);
    }
    
    #[test]
    fn test_pn_counter_negative_values() {
        let node_id = Uuid::new_v4();
        let mut counter = PNCounter::new(node_id);
        
        counter.decrement(5);
        assert_eq!(counter.total(), -5);
        assert_eq!(counter.get_node_total(&node_id), -5);
        
        counter.increment(3);
        assert_eq!(counter.total(), -2);
        assert_eq!(counter.get_node_total(&node_id), -2);
    }
    
    #[test]
    fn test_pn_counter_merge() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let mut counter1 = PNCounter::new(node1);
        let mut counter2 = PNCounter::new(node2);
        
        counter1.increment(5);
        counter1.decrement(2);
        
        counter2.increment(3);
        counter2.decrement(1);
        
        let merged = counter1.merge(&counter2).unwrap();
        
        assert_eq!(merged.total(), 5); // (5-2) + (3-1) = 3 + 2 = 5
        assert_eq!(merged.get_node_total(&node1), 3);
        assert_eq!(merged.get_node_total(&node2), 2);
    }
    
    #[test]
    fn test_pn_counter_causality() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let mut counter1 = PNCounter::new(node1);
        let mut counter2 = PNCounter::new(node2);
        
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
    fn test_pn_counter_apply_operation() {
        let node_id = Uuid::new_v4();
        let mut counter = PNCounter::new(node_id);
        
        let inc_operation = CrdtOperation::PNCounterIncrement { node_id, amount: 10 };
        counter.apply_operation(&inc_operation).unwrap();
        assert_eq!(counter.total(), 10);
        
        let dec_operation = CrdtOperation::PNCounterDecrement { node_id, amount: 3 };
        counter.apply_operation(&dec_operation).unwrap();
        assert_eq!(counter.total(), 7);
    }
    
    #[test]
    fn test_pn_counter_serialization() {
        let node_id = Uuid::new_v4();
        let mut counter = PNCounter::new(node_id);
        counter.increment(42);
        counter.decrement(17);
        
        let serialized = serde_json::to_string(&counter).unwrap();
        let deserialized: PNCounter = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(counter, deserialized);
        assert_eq!(deserialized.total(), 25);
    }
}