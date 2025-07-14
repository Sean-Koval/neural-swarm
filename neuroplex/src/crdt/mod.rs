//! CRDT (Conflict-free Replicated Data Type) implementation
//!
//! This module provides various CRDT types for distributed data structures
//! that can be merged without conflicts.

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use thiserror::Error;

use crate::{Result, NodeId};

pub mod g_counter;
pub mod pn_counter;
pub mod or_set;
pub mod lww_register;
pub mod vector_clock;

pub use g_counter::GCounter;
pub use pn_counter::PNCounter;
pub use or_set::ORSet;
pub use lww_register::LWWRegister;
pub use vector_clock::VectorClock;

/// CRDT operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrdtOperation {
    GCounterIncrement { node_id: NodeId, amount: u64 },
    PNCounterIncrement { node_id: NodeId, amount: i64 },
    PNCounterDecrement { node_id: NodeId, amount: i64 },
    ORSetAdd { element: String, node_id: NodeId, timestamp: u64 },
    ORSetRemove { element: String, node_id: NodeId, timestamp: u64 },
    LWWRegisterSet { value: String, node_id: NodeId, timestamp: u64 },
}

/// CRDT value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrdtValue {
    GCounter(GCounter),
    PNCounter(PNCounter),
    ORSet(ORSet<String>),
    LWWRegister(LWWRegister<String>),
}

/// CRDT error types
#[derive(Error, Debug)]
pub enum CrdtError {
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("Merge conflict: {0}")]
    MergeConflict(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
}

/// Trait for CRDT operations
pub trait Crdt: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync {
    /// Merge this CRDT with another CRDT of the same type
    fn merge(&self, other: &Self) -> Result<Self>;
    
    /// Apply an operation to this CRDT
    fn apply_operation(&mut self, operation: &CrdtOperation) -> Result<()>;
    
    /// Get the current value of this CRDT
    fn value(&self) -> serde_json::Value;
    
    /// Check if this CRDT is causally dependent on another
    fn is_causally_dependent(&self, other: &Self) -> bool;
}

impl CrdtValue {
    /// Create a new G-Counter
    pub fn new_g_counter(node_id: NodeId) -> Self {
        CrdtValue::GCounter(GCounter::new(node_id))
    }
    
    /// Create a new PN-Counter
    pub fn new_pn_counter(node_id: NodeId) -> Self {
        CrdtValue::PNCounter(PNCounter::new(node_id))
    }
    
    /// Create a new OR-Set
    pub fn new_or_set() -> Self {
        CrdtValue::ORSet(ORSet::new())
    }
    
    /// Create a new LWW-Register
    pub fn new_lww_register(value: String, node_id: NodeId) -> Self {
        CrdtValue::LWWRegister(LWWRegister::new(value, node_id))
    }
    
    /// Merge this CRDT value with another
    pub fn merge(&self, other: &CrdtValue) -> Result<CrdtValue> {
        match (self, other) {
            (CrdtValue::GCounter(a), CrdtValue::GCounter(b)) => {
                Ok(CrdtValue::GCounter(a.merge(b)?))
            }
            (CrdtValue::PNCounter(a), CrdtValue::PNCounter(b)) => {
                Ok(CrdtValue::PNCounter(a.merge(b)?))
            }
            (CrdtValue::ORSet(a), CrdtValue::ORSet(b)) => {
                Ok(CrdtValue::ORSet(a.merge(b)?))
            }
            (CrdtValue::LWWRegister(a), CrdtValue::LWWRegister(b)) => {
                Ok(CrdtValue::LWWRegister(a.merge(b)?))
            }
            _ => Err(CrdtError::TypeMismatch {
                expected: self.type_name().to_string(),
                actual: other.type_name().to_string(),
            }.into()),
        }
    }
    
    /// Apply an operation to this CRDT value
    pub fn apply_operation(&mut self, operation: &CrdtOperation) -> Result<()> {
        match (self, operation) {
            (CrdtValue::GCounter(ref mut counter), CrdtOperation::GCounterIncrement { node_id, amount }) => {
                counter.increment(*node_id, *amount);
                Ok(())
            }
            (CrdtValue::PNCounter(ref mut counter), CrdtOperation::PNCounterIncrement { node_id, amount }) => {
                counter.increment(*node_id, *amount);
                Ok(())
            }
            (CrdtValue::PNCounter(ref mut counter), CrdtOperation::PNCounterDecrement { node_id, amount }) => {
                counter.decrement(*node_id, *amount);
                Ok(())
            }
            (CrdtValue::ORSet(ref mut set), CrdtOperation::ORSetAdd { element, node_id, timestamp }) => {
                set.add(element.clone(), *node_id, *timestamp);
                Ok(())
            }
            (CrdtValue::ORSet(ref mut set), CrdtOperation::ORSetRemove { element, node_id, timestamp }) => {
                set.remove(element, *node_id, *timestamp);
                Ok(())
            }
            (CrdtValue::LWWRegister(ref mut register), CrdtOperation::LWWRegisterSet { value, node_id, timestamp }) => {
                register.set(value.clone(), *node_id, *timestamp);
                Ok(())
            }
            _ => Err(CrdtError::InvalidOperation(format!(
                "Operation {:?} not applicable to CRDT type {}",
                operation,
                self.type_name()
            )).into()),
        }
    }
    
    /// Get the current value of this CRDT
    pub fn value(&self) -> serde_json::Value {
        match self {
            CrdtValue::GCounter(counter) => counter.value(),
            CrdtValue::PNCounter(counter) => counter.value(),
            CrdtValue::ORSet(set) => set.value(),
            CrdtValue::LWWRegister(register) => register.value(),
        }
    }
    
    /// Get the type name of this CRDT
    pub fn type_name(&self) -> &'static str {
        match self {
            CrdtValue::GCounter(_) => "GCounter",
            CrdtValue::PNCounter(_) => "PNCounter",
            CrdtValue::ORSet(_) => "ORSet",
            CrdtValue::LWWRegister(_) => "LWWRegister",
        }
    }
    
    /// Check if this CRDT is causally dependent on another
    pub fn is_causally_dependent(&self, other: &CrdtValue) -> bool {
        match (self, other) {
            (CrdtValue::GCounter(a), CrdtValue::GCounter(b)) => a.is_causally_dependent(b),
            (CrdtValue::PNCounter(a), CrdtValue::PNCounter(b)) => a.is_causally_dependent(b),
            (CrdtValue::ORSet(a), CrdtValue::ORSet(b)) => a.is_causally_dependent(b),
            (CrdtValue::LWWRegister(a), CrdtValue::LWWRegister(b)) => a.is_causally_dependent(b),
            _ => false,
        }
    }
}

/// CRDT factory for creating different types of CRDTs
pub struct CrdtFactory;

impl CrdtFactory {
    /// Create a CRDT from a type name and initial value
    pub fn create(crdt_type: &str, node_id: NodeId, initial_value: Option<serde_json::Value>) -> Result<CrdtValue> {
        match crdt_type {
            "GCounter" => Ok(CrdtValue::new_g_counter(node_id)),
            "PNCounter" => Ok(CrdtValue::new_pn_counter(node_id)),
            "ORSet" => Ok(CrdtValue::new_or_set()),
            "LWWRegister" => {
                let value = initial_value
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| String::new());
                Ok(CrdtValue::new_lww_register(value, node_id))
            }
            _ => Err(CrdtError::InvalidOperation(format!("Unknown CRDT type: {}", crdt_type)).into()),
        }
    }
    
    /// Get available CRDT types
    pub fn available_types() -> Vec<&'static str> {
        vec!["GCounter", "PNCounter", "ORSet", "LWWRegister"]
    }
}

/// CRDT merge resolver for handling conflicts
pub struct CrdtMergeResolver;

impl CrdtMergeResolver {
    /// Resolve conflicts between multiple CRDT values
    pub fn resolve_conflicts(values: Vec<CrdtValue>) -> Result<CrdtValue> {
        if values.is_empty() {
            return Err(CrdtError::InvalidOperation("No values to resolve".to_string()).into());
        }
        
        if values.len() == 1 {
            return Ok(values.into_iter().next().unwrap());
        }
        
        // Check that all values are of the same type
        let first_type = values[0].type_name();
        for value in &values {
            if value.type_name() != first_type {
                return Err(CrdtError::TypeMismatch {
                    expected: first_type.to_string(),
                    actual: value.type_name().to_string(),
                }.into());
            }
        }
        
        // Merge all values
        let mut result = values[0].clone();
        for value in values.iter().skip(1) {
            result = result.merge(value)?;
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crdt_factory() {
        let node_id = Uuid::new_v4();
        
        // Test G-Counter creation
        let g_counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
        assert!(matches!(g_counter, CrdtValue::GCounter(_)));
        
        // Test PN-Counter creation
        let pn_counter = CrdtFactory::create("PNCounter", node_id, None).unwrap();
        assert!(matches!(pn_counter, CrdtValue::PNCounter(_)));
        
        // Test OR-Set creation
        let or_set = CrdtFactory::create("ORSet", node_id, None).unwrap();
        assert!(matches!(or_set, CrdtValue::ORSet(_)));
        
        // Test LWW-Register creation
        let lww_register = CrdtFactory::create("LWWRegister", node_id, Some(serde_json::Value::String("test".to_string()))).unwrap();
        assert!(matches!(lww_register, CrdtValue::LWWRegister(_)));
    }
    
    #[test]
    fn test_crdt_merge_resolver() {
        let node_id = Uuid::new_v4();
        
        let mut counter1 = CrdtValue::new_g_counter(node_id);
        let mut counter2 = CrdtValue::new_g_counter(node_id);
        
        // Apply operations
        counter1.apply_operation(&CrdtOperation::GCounterIncrement { node_id, amount: 5 }).unwrap();
        counter2.apply_operation(&CrdtOperation::GCounterIncrement { node_id, amount: 3 }).unwrap();
        
        let resolved = CrdtMergeResolver::resolve_conflicts(vec![counter1, counter2]).unwrap();
        
        // The merged counter should have value 8 (5 + 3)
        assert_eq!(resolved.value(), serde_json::Value::Number(serde_json::Number::from(8)));
    }
    
    #[test]
    fn test_available_types() {
        let types = CrdtFactory::available_types();
        assert!(types.contains(&"GCounter"));
        assert!(types.contains(&"PNCounter"));
        assert!(types.contains(&"ORSet"));
        assert!(types.contains(&"LWWRegister"));
    }
}