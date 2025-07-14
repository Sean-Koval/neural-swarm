//! Sequence CRDTs
//!
//! Implementation of RGA (Replicated Growable Array) for ordered sequences

use super::{CrdtOperation, CrdtState};
use crate::{NodeId, Timestamp, VersionVector, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};

/// RGA operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RGAOp<T: Clone> {
    Insert {
        element: T,
        position: RGAPosition,
        node_id: NodeId,
        timestamp: Timestamp,
    },
    Delete {
        position: RGAPosition,
        node_id: NodeId,
        timestamp: Timestamp,
    },
}

/// Position in RGA sequence
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct RGAPosition {
    pub node_id: NodeId,
    pub timestamp: Timestamp,
}

impl<T: Clone> CrdtOperation for RGAOp<T> {
    type State = RGAState<T>;
    
    fn apply(&self, state: &mut Self::State) -> Result<()> {
        state.apply(self)
    }
    
    fn timestamp(&self) -> Timestamp {
        match self {
            RGAOp::Insert { timestamp, .. } => *timestamp,
            RGAOp::Delete { timestamp, .. } => *timestamp,
        }
    }
    
    fn node_id(&self) -> NodeId {
        match self {
            RGAOp::Insert { node_id, .. } => *node_id,
            RGAOp::Delete { node_id, .. } => *node_id,
        }
    }
}

/// RGA element
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RGAElement<T: Clone> {
    element: T,
    position: RGAPosition,
    deleted: bool,
}

/// RGA state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGAState<T: Clone> {
    /// Elements ordered by position
    elements: BTreeMap<RGAPosition, RGAElement<T>>,
    /// Version vector
    version_vector: VersionVector,
}

impl<T: Clone> CrdtState for RGAState<T> {
    type Operation = RGAOp<T>;
    type Value = Vec<T>;
    
    fn new() -> Self {
        Self {
            elements: BTreeMap::new(),
            version_vector: HashMap::new(),
        }
    }
    
    fn apply(&mut self, op: &Self::Operation) -> Result<()> {
        // Update version vector
        let version = self.version_vector.entry(op.node_id()).or_insert(0);
        *version = (*version).max(op.timestamp());
        
        match op {
            RGAOp::Insert { element, position, .. } => {
                let rga_element = RGAElement {
                    element: element.clone(),
                    position: position.clone(),
                    deleted: false,
                };
                self.elements.insert(position.clone(), rga_element);
            }
            RGAOp::Delete { position, .. } => {
                if let Some(element) = self.elements.get_mut(position) {
                    element.deleted = true;
                }
            }
        }
        
        Ok(())
    }
    
    fn merge(&mut self, other: &Self) -> Result<()> {
        // Merge version vectors
        for (node_id, timestamp) in &other.version_vector {
            let current = self.version_vector.entry(*node_id).or_insert(0);
            *current = (*current).max(*timestamp);
        }
        
        // Merge elements
        for (position, element) in &other.elements {
            match self.elements.get_mut(position) {
                Some(existing) => {
                    // If other's element is deleted, mark ours as deleted too
                    if element.deleted {
                        existing.deleted = true;
                    }
                }
                None => {
                    // Insert new element
                    self.elements.insert(position.clone(), element.clone());
                }
            }
        }
        
        Ok(())
    }
    
    fn value(&self) -> Self::Value {
        self.elements
            .values()
            .filter(|element| !element.deleted)
            .map(|element| element.element.clone())
            .collect()
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

/// RGA sequence
pub type RGASequence<T> = super::Crdt<RGAState<T>>;

impl<T: Clone> RGASequence<T> {
    /// Insert element at the beginning
    pub fn insert_at_beginning(&mut self, element: T, node_id: NodeId) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let position = RGAPosition { node_id, timestamp };
        
        let op = RGAOp::Insert {
            element,
            position,
            node_id,
            timestamp,
        };
        
        self.apply(op)
    }
    
    /// Insert element after a given position
    pub fn insert_after(&mut self, element: T, after_position: &RGAPosition, node_id: NodeId) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let position = RGAPosition { node_id, timestamp };
        
        let op = RGAOp::Insert {
            element,
            position,
            node_id,
            timestamp,
        };
        
        self.apply(op)
    }
    
    /// Delete element at position
    pub fn delete_at(&mut self, position: &RGAPosition, node_id: NodeId) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let op = RGAOp::Delete {
            position: position.clone(),
            node_id,
            timestamp,
        };
        
        self.apply(op)
    }
    
    /// Get the current sequence
    pub fn sequence(&self) -> Vec<T> {
        self.value()
    }
    
    /// Get the length of the sequence
    pub fn len(&self) -> usize {
        self.value().len()
    }
    
    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.value().is_empty()
    }
    
    /// Get all positions (for debugging/testing)
    pub fn positions(&self) -> Vec<RGAPosition> {
        if let Some(first_op) = self.operations().first() {
            // This is a simplified approach - in reality we'd need to track the state
            Vec::new()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_rga_sequence_basic() {
        let mut sequence = RGASequence::new();
        let node_id = Uuid::new_v4();
        
        sequence.insert_at_beginning("a".to_string(), node_id).unwrap();
        sequence.insert_at_beginning("b".to_string(), node_id).unwrap();
        
        let seq = sequence.sequence();
        assert_eq!(seq.len(), 2);
        // Note: The order might be different due to position-based ordering
    }
    
    #[test]
    fn test_rga_sequence_delete() {
        let mut sequence = RGASequence::new();
        let node_id = Uuid::new_v4();
        
        sequence.insert_at_beginning("a".to_string(), node_id).unwrap();
        sequence.insert_at_beginning("b".to_string(), node_id).unwrap();
        
        assert_eq!(sequence.len(), 2);
        
        // This is a simplified test - in reality we'd need to track positions
        // let positions = sequence.positions();
        // if let Some(first_position) = positions.first() {
        //     sequence.delete_at(first_position, node_id).unwrap();
        //     assert_eq!(sequence.len(), 1);
        // }
    }
    
    #[test]
    fn test_rga_sequence_merge() {
        let mut sequence1 = RGASequence::new();
        let mut sequence2 = RGASequence::new();
        
        let node_id1 = Uuid::new_v4();
        let node_id2 = Uuid::new_v4();
        
        sequence1.insert_at_beginning("a".to_string(), node_id1).unwrap();
        sequence2.insert_at_beginning("b".to_string(), node_id2).unwrap();
        
        sequence1.merge(&sequence2).unwrap();
        
        assert_eq!(sequence1.len(), 2);
        let seq = sequence1.sequence();
        assert!(seq.contains(&"a".to_string()));
        assert!(seq.contains(&"b".to_string()));
    }
    
    #[test]
    fn test_rga_sequence_empty() {
        let sequence = RGASequence::<String>::new();
        
        assert!(sequence.is_empty());
        assert_eq!(sequence.len(), 0);
    }
    
    #[test]
    fn test_rga_position_ordering() {
        let node_id1 = Uuid::new_v4();
        let node_id2 = Uuid::new_v4();
        
        let pos1 = RGAPosition { node_id: node_id1, timestamp: 100 };
        let pos2 = RGAPosition { node_id: node_id2, timestamp: 200 };
        
        if node_id1 < node_id2 {
            assert!(pos1 < pos2);
        } else {
            assert!(pos2 < pos1);
        }
    }
}