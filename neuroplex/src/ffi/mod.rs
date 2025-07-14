//! Python FFI bindings for neuroplex
//!
//! This module provides Python bindings using PyO3 for the neuroplex distributed memory system.

use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyBytes};
use pyo3::exceptions::PyRuntimeError;
use pyo3_asyncio::tokio::future_into_py;
use tokio::sync::RwLock;
use uuid::Uuid;
use serde_json;

use crate::{
    NeuroPlexSystem, NeuroPlexConfig, NeuroPlexError,
    NodeId, distributed::StoreEvent, consensus::RaftEvent, sync::SyncEvent,
    crdt::{CrdtValue, CrdtFactory},
    memory::MemoryUsageStats,
};

pub mod python_types;
pub mod async_support;
pub mod error_handling;

use python_types::*;
use async_support::*;
use error_handling::*;

/// Python wrapper for NeuroPlexSystem
#[pyclass]
pub struct PyNeuroPlexSystem {
    system: Arc<NeuroPlexSystem>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyNeuroPlexSystem {
    /// Create a new NeuroPlexSystem
    #[new]
    fn new(config: Option<PyNeuroPlexConfig>) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;
        
        let config = config.unwrap_or_default().into_rust();
        
        let system = runtime.block_on(async {
            NeuroPlexSystem::new(config).await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to create system: {}", e)))?;
        
        Ok(Self {
            system: Arc::new(system),
            runtime,
        })
    }
    
    /// Start the system
    fn start(&self) -> PyResult<()> {
        self.runtime.block_on(async {
            self.system.start().await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to start system: {}", e)))
    }
    
    /// Stop the system
    fn stop(&self) -> PyResult<()> {
        self.runtime.block_on(async {
            self.system.stop().await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to stop system: {}", e)))
    }
    
    /// Get health status
    fn health_status(&self) -> PyResult<HashMap<String, String>> {
        self.runtime.block_on(async {
            Ok(self.system.health_status().await)
        })
    }
    
    /// Get distributed store reference
    fn distributed_store(&self) -> PyResult<PyDistributedStore> {
        Ok(PyDistributedStore {
            store: Arc::clone(self.system.distributed_store()),
        })
    }
    
    /// Get consensus engine reference
    fn consensus_engine(&self) -> PyResult<PyRaftConsensus> {
        Ok(PyRaftConsensus {
            consensus: Arc::clone(self.system.consensus_engine()),
        })
    }
    
    /// Get memory manager reference
    fn memory_manager(&self) -> PyResult<PyMemoryManager> {
        Ok(PyMemoryManager {
            manager: Arc::clone(self.system.memory_manager()),
        })
    }
    
    /// Get sync coordinator reference
    fn sync_coordinator(&self) -> PyResult<PySyncCoordinator> {
        Ok(PySyncCoordinator {
            coordinator: Arc::clone(self.system.sync_coordinator()),
        })
    }
}

/// Python wrapper for DistributedStore
#[pyclass]
pub struct PyDistributedStore {
    store: Arc<crate::distributed::DistributedStore>,
}

#[pymethods]
impl PyDistributedStore {
    /// Get a value from the store
    fn get(&self, py: Python, key: &str) -> PyResult<&PyAny> {
        let store = Arc::clone(&self.store);
        let key = key.to_string();
        
        future_into_py(py, async move {
            match store.get(&key).await {
                Ok(Some(value)) => Ok(PyCrdtValue::from_rust(value)),
                Ok(None) => Ok(PyCrdtValue::none()),
                Err(e) => Err(PyRuntimeError::new_err(format!("Failed to get value: {}", e))),
            }
        })
    }
    
    /// Set a value in the store
    fn set(&self, py: Python, key: &str, value: &PyCrdtValue) -> PyResult<&PyAny> {
        let store = Arc::clone(&self.store);
        let key = key.to_string();
        let value = value.to_rust()?;
        
        future_into_py(py, async move {
            store.set(key, value).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to set value: {}", e)))
        })
    }
    
    /// Delete a value from the store
    fn delete(&self, py: Python, key: &str) -> PyResult<&PyAny> {
        let store = Arc::clone(&self.store);
        let key = key.to_string();
        
        future_into_py(py, async move {
            store.delete(&key).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete value: {}", e)))
        })
    }
    
    /// List all keys in the store
    fn keys(&self, py: Python) -> PyResult<&PyAny> {
        let store = Arc::clone(&self.store);
        
        future_into_py(py, async move {
            store.keys().await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get keys: {}", e)))
        })
    }
    
    /// Get the size of the store
    fn size(&self, py: Python) -> PyResult<&PyAny> {
        let store = Arc::clone(&self.store);
        
        future_into_py(py, async move {
            store.size().await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get size: {}", e)))
        })
    }
    
    /// Subscribe to store events
    fn subscribe(&self) -> PyResult<PyStoreEventReceiver> {
        let receiver = self.store.subscribe();
        Ok(PyStoreEventReceiver { receiver })
    }
}

/// Python wrapper for RaftConsensus
#[pyclass]
pub struct PyRaftConsensus {
    consensus: Arc<crate::consensus::RaftConsensus>,
}

#[pymethods]
impl PyRaftConsensus {
    /// Get the current state
    fn state(&self, py: Python) -> PyResult<&PyAny> {
        let consensus = Arc::clone(&self.consensus);
        
        future_into_py(py, async move {
            let state = consensus.state().await;
            Ok(PyNodeState::from_rust(state))
        })
    }
    
    /// Get the current term
    fn term(&self, py: Python) -> PyResult<&PyAny> {
        let consensus = Arc::clone(&self.consensus);
        
        future_into_py(py, async move {
            Ok(consensus.term().await)
        })
    }
    
    /// Get the current leader
    fn leader(&self, py: Python) -> PyResult<&PyAny> {
        let consensus = Arc::clone(&self.consensus);
        
        future_into_py(py, async move {
            Ok(consensus.leader().await.map(|id| id.to_string()))
        })
    }
    
    /// Propose a new log entry
    fn propose(&self, py: Python, data: &PyBytes) -> PyResult<&PyAny> {
        let consensus = Arc::clone(&self.consensus);
        let data = data.as_bytes().to_vec();
        
        future_into_py(py, async move {
            consensus.propose(data).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to propose: {}", e)))
        })
    }
    
    /// Get cluster status
    fn cluster_status(&self, py: Python) -> PyResult<&PyAny> {
        let consensus = Arc::clone(&self.consensus);
        
        future_into_py(py, async move {
            Ok(consensus.cluster_status().await)
        })
    }
    
    /// Subscribe to consensus events
    fn subscribe(&self) -> PyResult<PyRaftEventReceiver> {
        let receiver = self.consensus.subscribe();
        Ok(PyRaftEventReceiver { receiver })
    }
}

/// Python wrapper for MemoryManager
#[pyclass]
pub struct PyMemoryManager {
    manager: Arc<crate::memory::MemoryManager>,
}

#[pymethods]
impl PyMemoryManager {
    /// Allocate memory
    fn allocate(&self, py: Python, id: &str, data: &PyBytes) -> PyResult<&PyAny> {
        let manager = Arc::clone(&self.manager);
        let id = id.to_string();
        let data = data.as_bytes().to_vec();
        
        future_into_py(py, async move {
            manager.allocate(id, data).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate: {}", e)))
        })
    }
    
    /// Read memory
    fn read(&self, py: Python, id: &str) -> PyResult<&PyAny> {
        let manager = Arc::clone(&self.manager);
        let id = id.to_string();
        
        future_into_py(py, async move {
            manager.read(&id).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read: {}", e)))
        })
    }
    
    /// Update memory
    fn update(&self, py: Python, id: &str, data: &PyBytes) -> PyResult<&PyAny> {
        let manager = Arc::clone(&self.manager);
        let id = id.to_string();
        let data = data.as_bytes().to_vec();
        
        future_into_py(py, async move {
            manager.update(&id, data).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to update: {}", e)))
        })
    }
    
    /// Delete memory
    fn delete(&self, py: Python, id: &str) -> PyResult<&PyAny> {
        let manager = Arc::clone(&self.manager);
        let id = id.to_string();
        
        future_into_py(py, async move {
            manager.delete(&id).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete: {}", e)))
        })
    }
    
    /// Get usage statistics
    fn usage_stats(&self, py: Python) -> PyResult<&PyAny> {
        let manager = Arc::clone(&self.manager);
        
        future_into_py(py, async move {
            let stats = manager.usage_stats().await;
            Ok(PyMemoryUsageStats::from_rust(stats))
        })
    }
    
    /// Force garbage collection
    fn force_gc(&self, py: Python) -> PyResult<&PyAny> {
        let manager = Arc::clone(&self.manager);
        
        future_into_py(py, async move {
            manager.force_gc().await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to run GC: {}", e)))
        })
    }
}

/// Python wrapper for SyncCoordinator
#[pyclass]
pub struct PySyncCoordinator {
    coordinator: Arc<crate::sync::SyncCoordinator>,
}

#[pymethods]
impl PySyncCoordinator {
    /// Sync with peers
    fn sync_with_peers(&self, py: Python) -> PyResult<&PyAny> {
        let coordinator = Arc::clone(&self.coordinator);
        
        future_into_py(py, async move {
            coordinator.sync_with_peers().await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to sync with peers: {}", e)))
        })
    }
    
    /// Sync a specific key
    fn sync_key(&self, py: Python, key: &str, value: Option<&PyCrdtValue>) -> PyResult<&PyAny> {
        let coordinator = Arc::clone(&self.coordinator);
        let key = key.to_string();
        let value = value.map(|v| v.to_rust()).transpose()?;
        
        future_into_py(py, async move {
            coordinator.sync_key(&key, value).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to sync key: {}", e)))
        })
    }
    
    /// Get sync status
    fn status(&self, py: Python) -> PyResult<&PyAny> {
        let coordinator = Arc::clone(&self.coordinator);
        
        future_into_py(py, async move {
            Ok(coordinator.status().await)
        })
    }
    
    /// Subscribe to sync events
    fn subscribe(&self) -> PyResult<PySyncEventReceiver> {
        let receiver = self.coordinator.subscribe();
        Ok(PySyncEventReceiver { receiver })
    }
}

/// Python wrapper for CRDT factory
#[pyclass]
pub struct PyCrdtFactory;

#[pymethods]
impl PyCrdtFactory {
    /// Create a CRDT
    #[staticmethod]
    fn create(crdt_type: &str, node_id: &str, initial_value: Option<&PyAny>) -> PyResult<PyCrdtValue> {
        let node_id = Uuid::parse_str(node_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {}", e)))?;
        
        let initial_value = initial_value.map(|v| {
            // Convert Python value to serde_json::Value
            serde_json::to_value(v).map_err(|e| PyRuntimeError::new_err(format!("Failed to convert value: {}", e)))
        }).transpose()?;
        
        let crdt = CrdtFactory::create(crdt_type, node_id, initial_value)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create CRDT: {}", e)))?;
        
        Ok(PyCrdtValue::from_rust(crdt))
    }
    
    /// Get available CRDT types
    #[staticmethod]
    fn available_types() -> PyResult<Vec<String>> {
        Ok(CrdtFactory::available_types().into_iter().map(|s| s.to_string()).collect())
    }
}

/// Python module initialization
#[pymodule]
fn neuroplex(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeuroPlexSystem>()?;
    m.add_class::<PyDistributedStore>()?;
    m.add_class::<PyRaftConsensus>()?;
    m.add_class::<PyMemoryManager>()?;
    m.add_class::<PySyncCoordinator>()?;
    m.add_class::<PyCrdtFactory>()?;
    
    // Add Python types
    m.add_class::<PyNeuroPlexConfig>()?;
    m.add_class::<PyCrdtValue>()?;
    m.add_class::<PyNodeState>()?;
    m.add_class::<PyMemoryUsageStats>()?;
    
    // Add event receivers
    m.add_class::<PyStoreEventReceiver>()?;
    m.add_class::<PyRaftEventReceiver>()?;
    m.add_class::<PySyncEventReceiver>()?;
    
    Ok(())
}

/// Initialize the Python module
pub fn init_python_module() -> PyResult<()> {
    pyo3_asyncio::tokio::init_multi_thread_once();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyBytes;
    
    #[test]
    fn test_python_module_creation() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let config = PyNeuroPlexConfig::default();
            let system = PyNeuroPlexSystem::new(Some(config)).unwrap();
            
            // Test that system was created successfully
            assert!(system.health_status().is_ok());
        });
    }
    
    #[test]
    fn test_crdt_factory() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let types = PyCrdtFactory::available_types().unwrap();
            assert!(types.contains(&"GCounter".to_string()));
            assert!(types.contains(&"PNCounter".to_string()));
            assert!(types.contains(&"ORSet".to_string()));
            assert!(types.contains(&"LWWRegister".to_string()));
        });
    }
}