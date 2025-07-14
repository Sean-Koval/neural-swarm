//! LWW-Register (Last-Write-Wins Register) CRDT implementation
//!
//! An LWW-Register is a register that resolves conflicts by choosing the value
//! with the latest timestamp. In case of timestamp ties, node IDs are used for deterministic resolution.

use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::{Result, NodeId};
use super::{Crdt, CrdtOperation, CrdtError};

/// LWW-Register (Last-Write-Wins Register) implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LWWRegister<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + std::fmt::Debug,
{
    /// The current value
    value: T,
    /// Timestamp of the last write
    timestamp: u64,
    /// Node ID of the last writer
    node_id: NodeId,
}

impl<T> LWWRegister<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + std::fmt::Debug,
{
    /// Create a new LWW-Register with initial value
    pub fn new(value: T, node_id: NodeId) -> Self {
        Self {
            value,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            node_id,
        }
    }
    
    /// Create a new LWW-Register with value and timestamp
    pub fn with_timestamp(value: T, node_id: NodeId, timestamp: u64) -> Self {
        Self {
            value,
            timestamp,
            node_id,
        }
    }
    
    /// Set a new value
    pub fn set(&mut self, value: T, node_id: NodeId, timestamp: u64) {
        if self.should_update(timestamp, node_id) {
            self.value = value;
            self.timestamp = timestamp;
            self.node_id = node_id;
        }
    }
    
    /// Get the current value
    pub fn get(&self) -> &T {
        &self.value
    }
    
    /// Get the timestamp of the last write
    pub fn get_timestamp(&self) -> u64 {
        self.timestamp
    }
    
    /// Get the node ID of the last writer
    pub fn get_node_id(&self) -> NodeId {
        self.node_id
    }
    
    /// Check if an update should be applied based on timestamp and node ID
    fn should_update(&self, timestamp: u64, node_id: NodeId) -> bool {
        if timestamp > self.timestamp {
            true
        } else if timestamp == self.timestamp {
            // In case of timestamp tie, use node ID for deterministic resolution
            node_id > self.node_id
        } else {
            false
        }
    }
    
    /// Compare this register with another for causality
    pub fn compare(&self, other: &LWWRegister<T>) -> CausalityRelation {
        match self.timestamp.cmp(&other.timestamp) {
            std::cmp::Ordering::Greater => CausalityRelation::Greater,
            std::cmp::Ordering::Less => CausalityRelation::Less,
            std::cmp::Ordering::Equal => {
                match self.node_id.cmp(&other.node_id) {
                    std::cmp::Ordering::Greater => CausalityRelation::Greater,
                    std::cmp::Ordering::Less => CausalityRelation::Less,
                    std::cmp::Ordering::Equal => CausalityRelation::Equal,
                }
            }
        }
    }
    
    /// Reset the register with a new value (for testing purposes)
    pub fn reset(&mut self, value: T, node_id: NodeId) {
        self.value = value;
        self.timestamp = chrono::Utc::now().timestamp_millis() as u64;
        self.node_id = node_id;
    }
}

/// Causality relation between two LWW-Registers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CausalityRelation {
    Less,
    Greater,
    Equal,
}

impl<T> Crdt for LWWRegister<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + std::fmt::Debug + Send + Sync,
{
    fn merge(&self, other: &Self) -> Result<Self> {
        if self.should_update(other.timestamp, other.node_id) {
            Ok(other.clone())
        } else {
            Ok(self.clone())
        }
    }
    
    fn apply_operation(&mut self, operation: &CrdtOperation) -> Result<()> {
        match operation {
            CrdtOperation::LWWRegisterSet { value, node_id, timestamp } => {
                // This requires T to be convertible from String
                // For now, we'll assume T is String for simplicity
                if let Ok(typed_value) = serde_json::from_str::<T>(&format!("\"{}\"", value)) {
                    self.set(typed_value, *node_id, *timestamp);
                    Ok(())
                } else {
                    Err(CrdtError::InvalidOperation(format!(
                        "Cannot convert string value to register type: {}",
                        value
                    )).into())
                }
            }
            _ => Err(CrdtError::InvalidOperation(format!(
                "Operation {:?} not applicable to LWW-Register",
                operation
            )).into()),
        }
    }
    
    fn value(&self) -> serde_json::Value {
        serde_json::to_value(&self.value).unwrap_or(serde_json::Value::Null)
    }
    
    fn is_causally_dependent(&self, other: &Self) -> bool {
        matches!(self.compare(other), CausalityRelation::Greater)
    }
}

impl<T> Default for LWWRegister<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + std::fmt::Debug + Default,
{
    fn default() -> Self {
        Self::new(T::default(), Uuid::new_v4())
    }
}

// Specialization for String type to handle the operation conversion
impl LWWRegister<String> {
    /// Set a string value from operation
    pub fn set_from_operation(&mut self, value: String, node_id: NodeId, timestamp: u64) {
        self.set(value, node_id, timestamp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lww_register_creation() {
        let node_id = Uuid::new_v4();
        let register = LWWRegister::new("hello".to_string(), node_id);
        
        assert_eq!(register.get(), &"hello".to_string());
        assert_eq!(register.get_node_id(), node_id);
    }
    
    #[test]
    fn test_lww_register_set() {
        let node_id = Uuid::new_v4();
        let mut register = LWWRegister::new("hello".to_string(), node_id);
        
        let new_timestamp = register.get_timestamp() + 1000;
        register.set("world".to_string(), node_id, new_timestamp);
        
        assert_eq!(register.get(), &"world".to_string());
        assert_eq!(register.get_timestamp(), new_timestamp);
    }
    
    #[test]
    fn test_lww_register_older_timestamp_ignored() {
        let node_id = Uuid::new_v4();
        let mut register = LWWRegister::new("hello".to_string(), node_id);
        
        let old_timestamp = register.get_timestamp() - 1000;
        register.set("world".to_string(), node_id, old_timestamp);
        
        // Should still have the original value
        assert_eq!(register.get(), &"hello".to_string());
    }
    
    #[test]
    fn test_lww_register_timestamp_tie_resolution() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        let timestamp = 1000;
        
        let mut register = LWWRegister::with_timestamp("hello".to_string(), node1, timestamp);
        
        // Determine which node has higher ID
        let (higher_node, lower_node, expected_value) = if node1 > node2 {
            (node1, node2, "hello".to_string())
        } else {
            (node2, node1, "world".to_string())
        };
        
        register.set("world".to_string(), lower_node, timestamp);
        
        // Should keep the value from the higher node ID
        if higher_node == node1 {
            assert_eq!(register.get(), &"hello".to_string());
        } else {
            assert_eq!(register.get(), &"world".to_string());
        }
    }
    
    #[test]
    fn test_lww_register_merge() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let register1 = LWWRegister::with_timestamp("hello".to_string(), node1, 1000);
        let register2 = LWWRegister::with_timestamp("world".to_string(), node2, 2000);
        
        let merged = register1.merge(&register2).unwrap();
        
        // Should have the value with the later timestamp
        assert_eq!(merged.get(), &"world".to_string());
        assert_eq!(merged.get_timestamp(), 2000);
        assert_eq!(merged.get_node_id(), node2);
    }
    
    #[test]
    fn test_lww_register_causality() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        let register1 = LWWRegister::with_timestamp("hello".to_string(), node1, 1000);
        let register2 = LWWRegister::with_timestamp("world".to_string(), node2, 2000);
        
        assert_eq!(register1.compare(&register2), CausalityRelation::Less);
        assert_eq!(register2.compare(&register1), CausalityRelation::Greater);
        
        let register3 = LWWRegister::with_timestamp("test".to_string(), node1, 1000);
        assert_eq!(register1.compare(&register3), CausalityRelation::Equal);
    }
    
    #[test]
    fn test_lww_register_apply_operation() {
        let node_id = Uuid::new_v4();
        let mut register = LWWRegister::new("hello".to_string(), node_id);
        
        let operation = CrdtOperation::LWWRegisterSet {
            value: "world".to_string(),
            node_id,
            timestamp: register.get_timestamp() + 1000,
        };
        
        register.apply_operation(&operation).unwrap();
        assert_eq!(register.get(), &"world".to_string());
    }
    
    #[test]
    fn test_lww_register_serialization() {
        let node_id = Uuid::new_v4();
        let register = LWWRegister::new("hello world".to_string(), node_id);
        
        let serialized = serde_json::to_string(&register).unwrap();
        let deserialized: LWWRegister<String> = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(register, deserialized);
        assert_eq!(deserialized.get(), &"hello world".to_string());
    }
}