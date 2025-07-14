//! Vector Clock implementation for causality tracking
//!
//! Vector clocks are used to determine the causal ordering of events
//! in a distributed system without requiring synchronized clocks.

use std::collections::HashMap;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::NodeId;

/// Vector Clock implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorClock {
    /// Clock values for each node
    clocks: HashMap<NodeId, u64>,
    /// The node ID this clock belongs to
    node_id: NodeId,
}

/// Causality relation between two vector clocks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CausalityRelation {
    HappensBefore,
    HappensAfter,
    Concurrent,
    Equal,
}

impl VectorClock {
    /// Create a new vector clock for the given node
    pub fn new(node_id: NodeId) -> Self {
        let mut clocks = HashMap::new();
        clocks.insert(node_id, 0);
        
        Self {
            clocks,
            node_id,
        }
    }
    
    /// Create a vector clock from a map of node clocks
    pub fn from_clocks(clocks: HashMap<NodeId, u64>, node_id: NodeId) -> Self {
        Self {
            clocks,
            node_id,
        }
    }
    
    /// Increment the clock for this node
    pub fn tick(&mut self) {
        let current = self.clocks.get(&self.node_id).unwrap_or(&0);
        self.clocks.insert(self.node_id, current + 1);
    }
    
    /// Increment the clock for this node by a specific amount
    pub fn tick_by(&mut self, amount: u64) {
        let current = self.clocks.get(&self.node_id).unwrap_or(&0);
        self.clocks.insert(self.node_id, current + amount);
    }
    
    /// Update the clock when receiving an event from another node
    pub fn update(&mut self, other: &VectorClock) {
        // Update our clock to be max of both clocks for all nodes
        for (node_id, other_time) in &other.clocks {
            let our_time = self.clocks.get(node_id).unwrap_or(&0);
            self.clocks.insert(*node_id, (*our_time).max(*other_time));
        }
        
        // Increment our own clock
        self.tick();
    }
    
    /// Get the clock value for a specific node
    pub fn get_time(&self, node_id: &NodeId) -> u64 {
        self.clocks.get(node_id).unwrap_or(&0).clone()
    }
    
    /// Get the clock value for this node
    pub fn get_own_time(&self) -> u64 {
        self.get_time(&self.node_id)
    }
    
    /// Get all clock values
    pub fn get_clocks(&self) -> &HashMap<NodeId, u64> {
        &self.clocks
    }
    
    /// Get the node ID this clock belongs to
    pub fn get_node_id(&self) -> NodeId {
        self.node_id
    }
    
    /// Compare this vector clock with another to determine causality
    pub fn compare(&self, other: &VectorClock) -> CausalityRelation {
        let mut self_less = false;
        let mut self_greater = false;
        
        // Get all nodes from both clocks
        let all_nodes: std::collections::HashSet<NodeId> = self.clocks.keys()
            .chain(other.clocks.keys())
            .cloned()
            .collect();
        
        for node_id in all_nodes {
            let self_time = self.clocks.get(&node_id).unwrap_or(&0);
            let other_time = other.clocks.get(&node_id).unwrap_or(&0);
            
            match self_time.cmp(other_time) {
                Ordering::Less => self_less = true,
                Ordering::Greater => self_greater = true,
                Ordering::Equal => {},
            }
        }
        
        match (self_less, self_greater) {
            (true, false) => CausalityRelation::HappensBefore,
            (false, true) => CausalityRelation::HappensAfter,
            (false, false) => CausalityRelation::Equal,
            (true, true) => CausalityRelation::Concurrent,
        }
    }
    
    /// Check if this clock happens before another
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        matches!(self.compare(other), CausalityRelation::HappensBefore)
    }
    
    /// Check if this clock happens after another
    pub fn happens_after(&self, other: &VectorClock) -> bool {
        matches!(self.compare(other), CausalityRelation::HappensAfter)
    }
    
    /// Check if this clock is concurrent with another
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        matches!(self.compare(other), CausalityRelation::Concurrent)
    }
    
    /// Check if this clock is equal to another
    pub fn is_equal(&self, other: &VectorClock) -> bool {
        matches!(self.compare(other), CausalityRelation::Equal)
    }
    
    /// Merge two vector clocks (takes the maximum time for each node)
    pub fn merge(&self, other: &VectorClock) -> VectorClock {
        let mut merged_clocks = self.clocks.clone();
        
        for (node_id, other_time) in &other.clocks {
            let our_time = merged_clocks.get(node_id).unwrap_or(&0);
            merged_clocks.insert(*node_id, (*our_time).max(*other_time));
        }
        
        VectorClock {
            clocks: merged_clocks,
            node_id: self.node_id,
        }
    }
    
    /// Reset the clock (for testing purposes)
    pub fn reset(&mut self) {
        self.clocks.clear();
        self.clocks.insert(self.node_id, 0);
    }
    
    /// Get the total number of events across all nodes
    pub fn total_events(&self) -> u64 {
        self.clocks.values().sum()
    }
    
    /// Get the number of nodes in the clock
    pub fn node_count(&self) -> usize {
        self.clocks.len()
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new(Uuid::new_v4())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_clock_creation() {
        let node_id = Uuid::new_v4();
        let clock = VectorClock::new(node_id);
        
        assert_eq!(clock.get_own_time(), 0);
        assert_eq!(clock.get_node_id(), node_id);
        assert_eq!(clock.node_count(), 1);
    }
    
    #[test]
    fn test_vector_clock_tick() {
        let node_id = Uuid::new_v4();
        let mut clock = VectorClock::new(node_id);
        
        clock.tick();
        assert_eq!(clock.get_own_time(), 1);
        
        clock.tick();
        assert_eq!(clock.get_own_time(), 2);
        
        clock.tick_by(5);
        assert_eq!(clock.get_own_time(), 7);
    }
    
    #[test]
    fn test_vector_clock_update() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let mut clock1 = VectorClock::new(node1);
        let mut clock2 = VectorClock::new(node2);
        
        // Node 1 has some events
        clock1.tick();
        clock1.tick();
        assert_eq!(clock1.get_own_time(), 2);
        
        // Node 2 has some events
        clock2.tick();
        assert_eq!(clock2.get_own_time(), 1);
        
        // Node 2 receives message from node 1
        clock2.update(&clock1);
        
        // Node 2 should have node 1's time and incremented its own
        assert_eq!(clock2.get_time(&node1), 2);
        assert_eq!(clock2.get_own_time(), 2); // 1 + 1 (from update)
    }
    
    #[test]
    fn test_vector_clock_causality() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let mut clock1 = VectorClock::new(node1);
        let mut clock2 = VectorClock::new(node2);
        
        // Initially concurrent (both at 0)
        assert_eq!(clock1.compare(&clock2), CausalityRelation::Equal);
        
        // Node 1 has an event
        clock1.tick();
        assert_eq!(clock1.compare(&clock2), CausalityRelation::HappensAfter);
        assert_eq!(clock2.compare(&clock1), CausalityRelation::HappensBefore);
        
        // Node 2 has an event
        clock2.tick();
        assert_eq!(clock1.compare(&clock2), CausalityRelation::Concurrent);
        assert_eq!(clock2.compare(&clock1), CausalityRelation::Concurrent);
        
        // Node 2 receives message from node 1
        clock2.update(&clock1);
        assert_eq!(clock1.compare(&clock2), CausalityRelation::HappensBefore);
        assert_eq!(clock2.compare(&clock1), CausalityRelation::HappensAfter);
    }
    
    #[test]
    fn test_vector_clock_merge() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        let node3 = Uuid::new_v4();
        
        let mut clock1 = VectorClock::new(node1);
        let mut clock2 = VectorClock::new(node2);
        
        // Set up different times for each node
        clock1.tick();
        clock1.tick();
        clock1.clocks.insert(node3, 5);
        
        clock2.tick();
        clock2.clocks.insert(node3, 3);
        
        let merged = clock1.merge(&clock2);
        
        // Should take maximum time for each node
        assert_eq!(merged.get_time(&node1), 2);
        assert_eq!(merged.get_time(&node2), 1);
        assert_eq!(merged.get_time(&node3), 5);
    }
    
    #[test]
    fn test_vector_clock_helper_methods() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let mut clock1 = VectorClock::new(node1);
        let mut clock2 = VectorClock::new(node2);
        
        clock1.tick();
        
        assert!(clock1.happens_after(&clock2));
        assert!(clock2.happens_before(&clock1));
        
        clock2.tick();
        
        assert!(clock1.is_concurrent(&clock2));
        assert!(clock2.is_concurrent(&clock1));
        
        clock2.update(&clock1);
        
        assert!(clock2.happens_after(&clock1));
        assert!(clock1.happens_before(&clock2));
    }
    
    #[test]
    fn test_vector_clock_serialization() {
        let node_id = Uuid::new_v4();
        let mut clock = VectorClock::new(node_id);
        clock.tick();
        clock.tick();
        
        let serialized = serde_json::to_string(&clock).unwrap();
        let deserialized: VectorClock = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(clock, deserialized);
        assert_eq!(deserialized.get_own_time(), 2);
    }
    
    #[test]
    fn test_vector_clock_statistics() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let mut clock = VectorClock::new(node1);
        clock.tick();
        clock.tick();
        clock.clocks.insert(node2, 3);
        
        assert_eq!(clock.total_events(), 5); // 2 + 3
        assert_eq!(clock.node_count(), 2);
    }
}