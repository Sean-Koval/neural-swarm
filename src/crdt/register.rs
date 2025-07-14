//! Register CRDTs
//!
//! Implementation of LWW-Register (last-writer-wins register)

use super::{CrdtOperation, CrdtState};
use crate::{NodeId, Timestamp, VersionVector, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LWW-Register operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LWWRegisterOp<T: Clone> {
    pub value: T,
    pub node_id: NodeId,
    pub timestamp: Timestamp,
}

impl<T: Clone> CrdtOperation for LWWRegisterOp<T> {
    type State = LWWRegisterState<T>;
    
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

/// LWW-Register state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LWWRegisterState<T: Clone> {
    /// Current value
    value: Option<T>,
    /// Timestamp of the current value
    timestamp: Timestamp,
    /// Node that set the current value
    node_id: Option<NodeId>,
    /// Version vector
    version_vector: VersionVector,
}

impl<T: Clone> CrdtState for LWWRegisterState<T> {
    type Operation = LWWRegisterOp<T>;
    type Value = Option<T>;
    
    fn new() -> Self {
        Self {
            value: None,
            timestamp: 0,
            node_id: None,
            version_vector: HashMap::new(),
        }
    }
    
    fn apply(&mut self, op: &Self::Operation) -> Result<()> {
        // Update version vector
        let version = self.version_vector.entry(op.node_id).or_insert(0);
        *version = (*version).max(op.timestamp);
        
        // Apply operation if it's newer or from a higher node ID (for deterministic ordering)
        if op.timestamp > self.timestamp || 
           (op.timestamp == self.timestamp && op.node_id > self.node_id.unwrap_or(NodeId::nil())) {
            self.value = Some(op.value.clone());
            self.timestamp = op.timestamp;
            self.node_id = Some(op.node_id);
        }
        
        Ok(())
    }
    
    fn merge(&mut self, other: &Self) -> Result<()> {
        // Merge version vectors
        for (node_id, timestamp) in &other.version_vector {
            let current = self.version_vector.entry(*node_id).or_insert(0);
            *current = (*current).max(*timestamp);
        }
        
        // Apply other's value if it's newer
        if let Some(ref other_value) = other.value {
            if other.timestamp > self.timestamp || 
               (other.timestamp == self.timestamp && other.node_id > self.node_id) {
                self.value = Some(other_value.clone());
                self.timestamp = other.timestamp;
                self.node_id = other.node_id;
            }
        }
        
        Ok(())
    }
    
    fn value(&self) -> Self::Value {
        self.value.clone()
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

/// LWW-Register (last-writer-wins register)
pub type LWWRegister<T> = super::Crdt<LWWRegisterState<T>>;

impl<T: Clone> LWWRegister<T> {
    /// Set the register value
    pub fn set(&mut self, value: T, node_id: NodeId) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let op = LWWRegisterOp {
            value,
            node_id,
            timestamp,
        };
        
        self.apply(op)
    }
    
    /// Get the current value
    pub fn get(&self) -> Option<T> {
        self.value()
    }
    
    /// Check if the register has a value
    pub fn is_set(&self) -> bool {
        self.value().is_some()
    }
}

/// Multi-value register that preserves concurrent values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MVRegisterOp<T: Clone> {
    pub value: T,
    pub node_id: NodeId,
    pub timestamp: Timestamp,
}

impl<T: Clone> CrdtOperation for MVRegisterOp<T> {
    type State = MVRegisterState<T>;
    
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

/// Multi-value register state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MVRegisterState<T: Clone> {
    /// Current values with their timestamps
    values: HashMap<NodeId, (T, Timestamp)>,
    /// Version vector
    version_vector: VersionVector,
}

impl<T: Clone> CrdtState for MVRegisterState<T> {
    type Operation = MVRegisterOp<T>;
    type Value = Vec<T>;
    
    fn new() -> Self {
        Self {
            values: HashMap::new(),
            version_vector: HashMap::new(),
        }
    }
    
    fn apply(&mut self, op: &Self::Operation) -> Result<()> {
        // Update version vector
        let version = self.version_vector.entry(op.node_id).or_insert(0);
        *version = (*version).max(op.timestamp);
        
        // Store value for this node
        self.values.insert(op.node_id, (op.value.clone(), op.timestamp));
        
        Ok(())
    }
    
    fn merge(&mut self, other: &Self) -> Result<()> {
        // Merge version vectors
        for (node_id, timestamp) in &other.version_vector {
            let current = self.version_vector.entry(*node_id).or_insert(0);
            *current = (*current).max(*timestamp);
        }
        
        // Merge values
        for (node_id, (value, timestamp)) in &other.values {
            if let Some((_, our_timestamp)) = self.values.get(node_id) {
                if *timestamp > *our_timestamp {
                    self.values.insert(*node_id, (value.clone(), *timestamp));
                }
            } else {
                self.values.insert(*node_id, (value.clone(), *timestamp));
            }
        }
        
        Ok(())
    }
    
    fn value(&self) -> Self::Value {
        if self.values.is_empty() {
            return Vec::new();
        }
        
        // Find the maximum timestamp
        let max_timestamp = self.values.values()
            .map(|(_, ts)| *ts)
            .max()
            .unwrap_or(0);
        
        // Return all values with the maximum timestamp
        self.values.values()
            .filter(|(_, ts)| *ts == max_timestamp)
            .map(|(value, _)| value.clone())
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

/// Multi-value register
pub type MVRegister<T> = super::Crdt<MVRegisterState<T>>;

impl<T: Clone> MVRegister<T> {
    /// Set the register value
    pub fn set(&mut self, value: T, node_id: NodeId) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let op = MVRegisterOp {
            value,
            node_id,
            timestamp,
        };
        
        self.apply(op)
    }
    
    /// Get all current values
    pub fn get(&self) -> Vec<T> {
        self.value()
    }
    
    /// Check if the register has any values
    pub fn is_set(&self) -> bool {
        !self.value().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_lww_register_basic() {
        let mut register = LWWRegister::new();
        let node_id = Uuid::new_v4();
        
        register.set("hello".to_string(), node_id).unwrap();
        assert_eq!(register.get(), Some("hello".to_string()));
        
        register.set("world".to_string(), node_id).unwrap();
        assert_eq!(register.get(), Some("world".to_string()));
    }
    
    #[test]
    fn test_lww_register_merge() {
        let mut register1 = LWWRegister::new();
        let mut register2 = LWWRegister::new();
        
        let node_id1 = Uuid::new_v4();
        let node_id2 = Uuid::new_v4();
        
        register1.set("value1".to_string(), node_id1).unwrap();
        
        // Sleep to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        register2.set("value2".to_string(), node_id2).unwrap();
        
        register1.merge(&register2).unwrap();
        
        // Should have the later value
        assert_eq!(register1.get(), Some("value2".to_string()));
    }
    
    #[test]
    fn test_lww_register_concurrent_updates() {
        let mut register1 = LWWRegister::new();
        let mut register2 = LWWRegister::new();
        
        let node_id1 = Uuid::new_v4();
        let node_id2 = Uuid::new_v4();
        
        // Simulate concurrent updates with same timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let op1 = LWWRegisterOp {
            value: "value1".to_string(),
            node_id: node_id1,
            timestamp,
        };
        
        let op2 = LWWRegisterOp {
            value: "value2".to_string(),
            node_id: node_id2,
            timestamp,
        };
        
        register1.apply(op1).unwrap();
        register2.apply(op2).unwrap();
        
        register1.merge(&register2).unwrap();
        
        // Should resolve deterministically based on node ID
        if node_id1 > node_id2 {
            assert_eq!(register1.get(), Some("value1".to_string()));
        } else {
            assert_eq!(register1.get(), Some("value2".to_string()));
        }
    }
    
    #[test]
    fn test_mv_register_basic() {
        let mut register = MVRegister::new();
        let node_id = Uuid::new_v4();
        
        register.set("hello".to_string(), node_id).unwrap();
        assert_eq!(register.get(), vec!["hello".to_string()]);
        
        register.set("world".to_string(), node_id).unwrap();
        assert_eq!(register.get(), vec!["world".to_string()]);
    }
    
    #[test]
    fn test_mv_register_concurrent_values() {
        let mut register1 = MVRegister::new();
        let mut register2 = MVRegister::new();
        
        let node_id1 = Uuid::new_v4();
        let node_id2 = Uuid::new_v4();
        
        // Simulate concurrent updates with same timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let op1 = MVRegisterOp {
            value: "value1".to_string(),
            node_id: node_id1,
            timestamp,
        };
        
        let op2 = MVRegisterOp {
            value: "value2".to_string(),
            node_id: node_id2,
            timestamp,
        };
        
        register1.apply(op1).unwrap();
        register2.apply(op2).unwrap();
        
        register1.merge(&register2).unwrap();
        
        // Should have both values
        let values = register1.get();
        assert_eq!(values.len(), 2);
        assert!(values.contains(&"value1".to_string()));
        assert!(values.contains(&"value2".to_string()));
    }
}