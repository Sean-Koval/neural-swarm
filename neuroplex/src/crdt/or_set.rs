//! OR-Set (Observed-Remove Set) CRDT implementation
//!
//! An OR-Set is a set that supports both add and remove operations.
//! Elements are tagged with unique identifiers to handle concurrent adds and removes.

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::{Result, NodeId};
use super::{Crdt, CrdtOperation, CrdtError};

/// Unique identifier for set elements
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ElementId {
    pub node_id: NodeId,
    pub timestamp: u64,
}

impl ElementId {
    pub fn new(node_id: NodeId, timestamp: u64) -> Self {
        Self { node_id, timestamp }
    }
}

/// OR-Set (Observed-Remove Set) implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ORSet<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + std::hash::Hash + Eq + std::fmt::Debug,
{
    /// Elements that have been added with their unique identifiers
    added: HashMap<T, HashSet<ElementId>>,
    /// Elements that have been removed with their unique identifiers
    removed: HashMap<T, HashSet<ElementId>>,
}

impl<T> ORSet<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + std::hash::Hash + Eq + std::fmt::Debug,
{
    /// Create a new OR-Set
    pub fn new() -> Self {
        Self {
            added: HashMap::new(),
            removed: HashMap::new(),
        }
    }
    
    /// Add an element to the set
    pub fn add(&mut self, element: T, node_id: NodeId, timestamp: u64) {
        let element_id = ElementId::new(node_id, timestamp);
        
        self.added
            .entry(element)
            .or_insert_with(HashSet::new)
            .insert(element_id);
    }
    
    /// Remove an element from the set
    pub fn remove(&mut self, element: &T, node_id: NodeId, timestamp: u64) {
        let element_id = ElementId::new(node_id, timestamp);
        
        // Only remove if the element exists in added
        if let Some(added_ids) = self.added.get(element) {
            // Remove all current add tags
            for added_id in added_ids {
                self.removed
                    .entry(element.clone())
                    .or_insert_with(HashSet::new)
                    .insert(added_id.clone());
            }
            
            // Also add the specific remove tag
            self.removed
                .entry(element.clone())
                .or_insert_with(HashSet::new)
                .insert(element_id);
        }
    }
    
    /// Check if an element is in the set
    pub fn contains(&self, element: &T) -> bool {
        if let Some(added_ids) = self.added.get(element) {
            if let Some(removed_ids) = self.removed.get(element) {
                // Element is in the set if there are add tags not in remove tags
                return added_ids.iter().any(|id| !removed_ids.contains(id));
            } else {
                // Element is in the set if it has add tags and no remove tags
                return !added_ids.is_empty();
            }
        }
        false
    }
    
    /// Get all elements in the set
    pub fn elements(&self) -> HashSet<T> {
        let mut result = HashSet::new();
        
        for element in self.added.keys() {
            if self.contains(element) {
                result.insert(element.clone());
            }
        }
        
        result
    }
    
    /// Get the size of the set
    pub fn size(&self) -> usize {
        self.elements().len()
    }
    
    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }
    
    /// Get all add tags for an element
    pub fn get_add_tags(&self, element: &T) -> Option<&HashSet<ElementId>> {
        self.added.get(element)
    }
    
    /// Get all remove tags for an element
    pub fn get_remove_tags(&self, element: &T) -> Option<&HashSet<ElementId>> {
        self.removed.get(element)
    }
    
    /// Clear the set (for testing purposes)
    pub fn clear(&mut self) {
        self.added.clear();
        self.removed.clear();
    }
    
    /// Compare this set with another for causality
    pub fn compare(&self, other: &ORSet<T>) -> CausalityRelation {
        let mut self_subset = true;
        let mut other_subset = true;
        
        // Check if self is a subset of other
        for (element, self_added) in &self.added {
            if let Some(other_added) = other.added.get(element) {
                if !self_added.is_subset(other_added) {
                    self_subset = false;
                    break;
                }
            } else {
                self_subset = false;
                break;
            }
        }
        
        for (element, self_removed) in &self.removed {
            if let Some(other_removed) = other.removed.get(element) {
                if !self_removed.is_subset(other_removed) {
                    self_subset = false;
                    break;
                }
            } else {
                self_subset = false;
                break;
            }
        }
        
        // Check if other is a subset of self
        for (element, other_added) in &other.added {
            if let Some(self_added) = self.added.get(element) {
                if !other_added.is_subset(self_added) {
                    other_subset = false;
                    break;
                }
            } else {
                other_subset = false;
                break;
            }
        }
        
        for (element, other_removed) in &other.removed {
            if let Some(self_removed) = self.removed.get(element) {
                if !other_removed.is_subset(self_removed) {
                    other_subset = false;
                    break;
                }
            } else {
                other_subset = false;
                break;
            }
        }
        
        match (self_subset, other_subset) {
            (true, true) => CausalityRelation::Equal,
            (true, false) => CausalityRelation::Less,
            (false, true) => CausalityRelation::Greater,
            (false, false) => CausalityRelation::Concurrent,
        }
    }
}

/// Causality relation between two OR-Sets
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CausalityRelation {
    Less,
    Greater,
    Equal,
    Concurrent,
}

impl<T> Crdt for ORSet<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + std::hash::Hash + Eq + std::fmt::Debug + Send + Sync,
{
    fn merge(&self, other: &Self) -> Result<Self> {
        let mut merged_added = self.added.clone();
        let mut merged_removed = self.removed.clone();
        
        // Merge added elements
        for (element, other_tags) in &other.added {
            merged_added
                .entry(element.clone())
                .or_insert_with(HashSet::new)
                .extend(other_tags.iter().cloned());
        }
        
        // Merge removed elements
        for (element, other_tags) in &other.removed {
            merged_removed
                .entry(element.clone())
                .or_insert_with(HashSet::new)
                .extend(other_tags.iter().cloned());
        }
        
        Ok(ORSet {
            added: merged_added,
            removed: merged_removed,
        })
    }
    
    fn apply_operation(&mut self, operation: &CrdtOperation) -> Result<()> {
        match operation {
            CrdtOperation::ORSetAdd { element, node_id, timestamp } => {
                self.add(element.clone(), *node_id, *timestamp);
                Ok(())
            }
            CrdtOperation::ORSetRemove { element, node_id, timestamp } => {
                self.remove(element, *node_id, *timestamp);
                Ok(())
            }
            _ => Err(CrdtError::InvalidOperation(format!(
                "Operation {:?} not applicable to OR-Set",
                operation
            )).into()),
        }
    }
    
    fn value(&self) -> serde_json::Value {
        let elements: Vec<serde_json::Value> = self.elements()
            .into_iter()
            .map(|e| serde_json::to_value(e).unwrap_or(serde_json::Value::Null))
            .collect();
        
        serde_json::Value::Array(elements)
    }
    
    fn is_causally_dependent(&self, other: &Self) -> bool {
        matches!(self.compare(other), CausalityRelation::Greater)
    }
}

impl<T> Default for ORSet<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + std::hash::Hash + Eq + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_or_set_creation() {
        let set: ORSet<String> = ORSet::new();
        assert!(set.is_empty());
        assert_eq!(set.size(), 0);
    }
    
    #[test]
    fn test_or_set_add() {
        let mut set = ORSet::new();
        let node_id = Uuid::new_v4();
        
        set.add("hello".to_string(), node_id, 1);
        assert!(set.contains(&"hello".to_string()));
        assert_eq!(set.size(), 1);
        
        set.add("world".to_string(), node_id, 2);
        assert!(set.contains(&"world".to_string()));
        assert_eq!(set.size(), 2);
    }
    
    #[test]
    fn test_or_set_remove() {
        let mut set = ORSet::new();
        let node_id = Uuid::new_v4();
        
        set.add("hello".to_string(), node_id, 1);
        set.add("world".to_string(), node_id, 2);
        assert_eq!(set.size(), 2);
        
        set.remove(&"hello".to_string(), node_id, 3);
        assert!(!set.contains(&"hello".to_string()));
        assert!(set.contains(&"world".to_string()));
        assert_eq!(set.size(), 1);
    }
    
    #[test]
    fn test_or_set_concurrent_add_remove() {
        let mut set1 = ORSet::new();
        let mut set2 = ORSet::new();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        // Node 1 adds "hello"
        set1.add("hello".to_string(), node1, 1);
        
        // Node 2 adds "hello" concurrently
        set2.add("hello".to_string(), node2, 1);
        
        // Node 1 removes "hello"
        set1.remove(&"hello".to_string(), node1, 2);
        
        // Merge the sets
        let merged = set1.merge(&set2).unwrap();
        
        // The element should still be in the merged set because node2's add wasn't observed by node1's remove
        assert!(merged.contains(&"hello".to_string()));
    }
    
    #[test]
    fn test_or_set_merge() {
        let mut set1 = ORSet::new();
        let mut set2 = ORSet::new();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        set1.add("a".to_string(), node1, 1);
        set1.add("b".to_string(), node1, 2);
        
        set2.add("b".to_string(), node2, 1);
        set2.add("c".to_string(), node2, 2);
        
        let merged = set1.merge(&set2).unwrap();
        
        assert!(merged.contains(&"a".to_string()));
        assert!(merged.contains(&"b".to_string()));
        assert!(merged.contains(&"c".to_string()));
        assert_eq!(merged.size(), 3);
    }
    
    #[test]
    fn test_or_set_apply_operation() {
        let mut set = ORSet::new();
        let node_id = Uuid::new_v4();
        
        let add_operation = CrdtOperation::ORSetAdd {
            element: "test".to_string(),
            node_id,
            timestamp: 1,
        };
        set.apply_operation(&add_operation).unwrap();
        assert!(set.contains(&"test".to_string()));
        
        let remove_operation = CrdtOperation::ORSetRemove {
            element: "test".to_string(),
            node_id,
            timestamp: 2,
        };
        set.apply_operation(&remove_operation).unwrap();
        assert!(!set.contains(&"test".to_string()));
    }
    
    #[test]
    fn test_or_set_serialization() {
        let mut set = ORSet::new();
        let node_id = Uuid::new_v4();
        
        set.add("hello".to_string(), node_id, 1);
        set.add("world".to_string(), node_id, 2);
        
        let serialized = serde_json::to_string(&set).unwrap();
        let deserialized: ORSet<String> = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(set, deserialized);
        assert_eq!(deserialized.size(), 2);
    }
}