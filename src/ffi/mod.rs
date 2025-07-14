//! Python FFI Bindings
//!
//! PyO3-based async Python integration for neuroplex distributed memory system

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::exceptions::PyRuntimeError;
use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    NeuroCluster, NeuroNode, NeuroConfig, NeuroError, Result,
    NodeId, Timestamp, VersionVector,
    memory::{DistributedMemory, MemoryConfig, MemoryEvent, MemoryStats},
    crdt::{GCounter, PNCounter, ORSet, LWWRegister},
    consensus::{RaftConsensus, RaftConfig, NodeState, ConsensusStats},
    sync::{GossipSync, SyncConfig, SyncStats},
};

/// Python wrapper for NeuroCluster
#[pyclass]
pub struct PyNeuroCluster {
    inner: Arc<RwLock<NeuroCluster>>,
}

#[pymethods]
impl PyNeuroCluster {
    #[new]
    fn new(node_name: String, address: String) -> PyResult<Self> {
        pyo3_asyncio::tokio::get_runtime()
            .block_on(async {
                let cluster = NeuroCluster::new(&node_name, &address)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                
                Ok(PyNeuroCluster {
                    inner: Arc::new(RwLock::new(cluster)),
                })
            })
    }
    
    fn set<'p>(&self, py: Python<'p>, key: String, value: Vec<u8>) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let cluster = inner.read().await;
            cluster.set(&key, &value)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
    
    fn get<'p>(&self, py: Python<'p>, key: String) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let cluster = inner.read().await;
            let result = cluster.get(&key)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(result)
        })
    }
    
    fn delete<'p>(&self, py: Python<'p>, key: String) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let cluster = inner.read().await;
            let result = cluster.delete(&key)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(result)
        })
    }
    
    fn list_keys<'p>(&self, py: Python<'p>, prefix: Option<String>) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let cluster = inner.read().await;
            let keys = cluster.list(prefix.as_deref())
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(keys)
        })
    }
    
    fn stats<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let cluster = inner.read().await;
            let stats = cluster.stats()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(stats)
        })
    }
    
    fn add_node<'p>(&self, py: Python<'p>, node_id: String, address: String) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let mut cluster = inner.write().await;
            let node_uuid = Uuid::parse_str(&node_id)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            cluster.add_node(node_uuid, &address)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(())
        })
    }
    
    fn remove_node<'p>(&self, py: Python<'p>, node_id: String) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let mut cluster = inner.write().await;
            let node_uuid = Uuid::parse_str(&node_id)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            cluster.remove_node(node_uuid)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(())
        })
    }
    
    fn subscribe<'p>(&self, py: Python<'p>, callback: PyObject) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let cluster = inner.read().await;
            
            // Create a callback wrapper
            let callback_wrapper = Arc::new(move |event: MemoryEvent| {
                Python::with_gil(|py| {
                    // Convert event to Python object
                    let py_event = match event {
                        MemoryEvent::EntryAdded { key, value } => {
                            let dict = PyDict::new(py);
                            dict.set_item("type", "entry_added").unwrap();
                            dict.set_item("key", key).unwrap();
                            dict.set_item("value", PyBytes::new(py, &value)).unwrap();
                            dict.into()
                        }
                        MemoryEvent::EntryUpdated { key, old_value, new_value } => {
                            let dict = PyDict::new(py);
                            dict.set_item("type", "entry_updated").unwrap();
                            dict.set_item("key", key).unwrap();
                            dict.set_item("old_value", PyBytes::new(py, &old_value)).unwrap();
                            dict.set_item("new_value", PyBytes::new(py, &new_value)).unwrap();
                            dict.into()
                        }
                        MemoryEvent::EntryDeleted { key, value } => {
                            let dict = PyDict::new(py);
                            dict.set_item("type", "entry_deleted").unwrap();
                            dict.set_item("key", key).unwrap();
                            dict.set_item("value", PyBytes::new(py, &value)).unwrap();
                            dict.into()
                        }
                        MemoryEvent::CompactionStarted => {
                            let dict = PyDict::new(py);
                            dict.set_item("type", "compaction_started").unwrap();
                            dict.into()
                        }
                        MemoryEvent::CompactionCompleted { freed_bytes } => {
                            let dict = PyDict::new(py);
                            dict.set_item("type", "compaction_completed").unwrap();
                            dict.set_item("freed_bytes", freed_bytes).unwrap();
                            dict.into()
                        }
                    };
                    
                    // Call the Python callback
                    if let Err(e) = callback.call1(py, (py_event,)) {
                        eprintln!("Error in Python callback: {}", e);
                    }
                });
            });
            
            cluster.subscribe(callback_wrapper)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(())
        })
    }
}

/// Python wrapper for G-Counter CRDT
#[pyclass]
pub struct PyGCounter {
    inner: Arc<RwLock<GCounter>>,
    node_id: NodeId,
}

#[pymethods]
impl PyGCounter {
    #[new]
    fn new() -> Self {
        PyGCounter {
            inner: Arc::new(RwLock::new(GCounter::new())),
            node_id: Uuid::new_v4(),
        }
    }
    
    fn increment<'p>(&self, py: Python<'p>, amount: u64) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let node_id = self.node_id;
        
        future_into_py(py, async move {
            let mut counter = inner.write().await;
            counter.increment(node_id, amount)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
    
    fn count<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let counter = inner.read().await;
            Ok(counter.count())
        })
    }
    
    fn merge<'p>(&self, py: Python<'p>, other: &PyGCounter) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let other_inner = other.inner.clone();
        
        future_into_py(py, async move {
            let mut counter = inner.write().await;
            let other_counter = other_inner.read().await;
            
            counter.merge(&other_counter)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(())
        })
    }
}

/// Python wrapper for PN-Counter CRDT
#[pyclass]
pub struct PyPNCounter {
    inner: Arc<RwLock<PNCounter>>,
    node_id: NodeId,
}

#[pymethods]
impl PyPNCounter {
    #[new]
    fn new() -> Self {
        PyPNCounter {
            inner: Arc::new(RwLock::new(PNCounter::new())),
            node_id: Uuid::new_v4(),
        }
    }
    
    fn increment<'p>(&self, py: Python<'p>, amount: i64) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let node_id = self.node_id;
        
        future_into_py(py, async move {
            let mut counter = inner.write().await;
            counter.increment(node_id, amount)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
    
    fn decrement<'p>(&self, py: Python<'p>, amount: i64) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let node_id = self.node_id;
        
        future_into_py(py, async move {
            let mut counter = inner.write().await;
            counter.decrement(node_id, amount)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
    
    fn count<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let counter = inner.read().await;
            Ok(counter.count())
        })
    }
    
    fn merge<'p>(&self, py: Python<'p>, other: &PyPNCounter) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let other_inner = other.inner.clone();
        
        future_into_py(py, async move {
            let mut counter = inner.write().await;
            let other_counter = other_inner.read().await;
            
            counter.merge(&other_counter)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(())
        })
    }
}

/// Python wrapper for OR-Set CRDT
#[pyclass]
pub struct PyORSet {
    inner: Arc<RwLock<ORSet<String>>>,
    node_id: NodeId,
}

#[pymethods]
impl PyORSet {
    #[new]
    fn new() -> Self {
        PyORSet {
            inner: Arc::new(RwLock::new(ORSet::new())),
            node_id: Uuid::new_v4(),
        }
    }
    
    fn add<'p>(&self, py: Python<'p>, element: String) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let node_id = self.node_id;
        
        future_into_py(py, async move {
            let mut set = inner.write().await;
            set.add(element, node_id)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
    
    fn remove<'p>(&self, py: Python<'p>, element: String) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let node_id = self.node_id;
        
        future_into_py(py, async move {
            let mut set = inner.write().await;
            set.remove(element, node_id)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
    
    fn contains<'p>(&self, py: Python<'p>, element: String) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let set = inner.read().await;
            Ok(set.contains(&element))
        })
    }
    
    fn elements<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let set = inner.read().await;
            let elements: Vec<String> = set.elements().into_iter().collect();
            Ok(elements)
        })
    }
    
    fn merge<'p>(&self, py: Python<'p>, other: &PyORSet) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let other_inner = other.inner.clone();
        
        future_into_py(py, async move {
            let mut set = inner.write().await;
            let other_set = other_inner.read().await;
            
            set.merge(&other_set)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(())
        })
    }
}

/// Python wrapper for LWW-Register CRDT
#[pyclass]
pub struct PyLWWRegister {
    inner: Arc<RwLock<LWWRegister<String>>>,
    node_id: NodeId,
}

#[pymethods]
impl PyLWWRegister {
    #[new]
    fn new() -> Self {
        PyLWWRegister {
            inner: Arc::new(RwLock::new(LWWRegister::new())),
            node_id: Uuid::new_v4(),
        }
    }
    
    fn set<'p>(&self, py: Python<'p>, value: String) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let node_id = self.node_id;
        
        future_into_py(py, async move {
            let mut register = inner.write().await;
            register.set(value, node_id)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
    
    fn get<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        
        future_into_py(py, async move {
            let register = inner.read().await;
            Ok(register.get())
        })
    }
    
    fn merge<'p>(&self, py: Python<'p>, other: &PyLWWRegister) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let other_inner = other.inner.clone();
        
        future_into_py(py, async move {
            let mut register = inner.write().await;
            let other_register = other_inner.read().await;
            
            register.merge(&other_register)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(())
        })
    }
}

/// Python helper functions
#[pyfunction]
fn generate_node_id() -> String {
    Uuid::new_v4().to_string()
}

#[pyfunction]
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Python module definition
#[pymodule]
fn neuroplex(_py: Python, m: &PyModule) -> PyResult<()> {
    // Initialize tracing
    crate::init_tracing();
    
    // Add classes
    m.add_class::<PyNeuroCluster>()?;
    m.add_class::<PyGCounter>()?;
    m.add_class::<PyPNCounter>()?;
    m.add_class::<PyORSet>()?;
    m.add_class::<PyLWWRegister>()?;
    
    // Add functions
    m.add_function(wrap_pyfunction!(generate_node_id, m)?)?;
    m.add_function(wrap_pyfunction!(current_timestamp, m)?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    
    #[test]
    fn test_python_module() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "neuroplex").unwrap();
            neuroplex(py, module).unwrap();
            
            // Test that classes are available
            assert!(module.getattr("PyNeuroCluster").is_ok());
            assert!(module.getattr("PyGCounter").is_ok());
            assert!(module.getattr("PyPNCounter").is_ok());
            assert!(module.getattr("PyORSet").is_ok());
            assert!(module.getattr("PyLWWRegister").is_ok());
        });
    }
    
    #[test]
    fn test_generate_node_id() {
        let id1 = generate_node_id();
        let id2 = generate_node_id();
        assert_ne!(id1, id2);
        assert!(Uuid::parse_str(&id1).is_ok());
        assert!(Uuid::parse_str(&id2).is_ok());
    }
    
    #[test]
    fn test_current_timestamp() {
        let ts1 = current_timestamp();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let ts2 = current_timestamp();
        assert!(ts2 > ts1);
    }
}