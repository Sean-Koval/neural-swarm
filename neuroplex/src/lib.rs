//! # Neuroplex - Distributed Memory Package
//!
//! A high-performance distributed memory system with CRDT implementation and Raft consensus.
//! 
//! ## Features
//! - Distributed in-memory state store with async operations
//! - CRDT implementation (G-Counter, PN-Counter, OR-Set, LWW-Register)
//! - Raft consensus protocol for coordination
//! - Multi-node synchronization with conflict resolution
//! - Python FFI bindings with async support
//! - Lock-free data structures for high concurrency
//! - Memory-efficient storage with compression
//! - SIMD optimizations for performance

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;
use thiserror::Error;
use tracing::{info, warn, error, debug};

// Core modules
pub mod distributed;
pub mod crdt;
pub mod consensus;
pub mod memory;
pub mod sync;

// FFI bindings
#[cfg(feature = "python-bindings")]
pub mod ffi;

// Re-exports for convenience
pub use distributed::*;
pub use crdt::*;
pub use consensus::*;
pub use memory::*;
pub use sync::*;

/// Node identifier for distributed operations
pub type NodeId = Uuid;

/// Result type for neuroplex operations
pub type Result<T> = std::result::Result<T, NeuroPlexError>;

/// Main error type for neuroplex operations
#[derive(Error, Debug)]
pub enum NeuroPlexError {
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Consensus error: {0}")]
    Consensus(String),
    
    #[error("CRDT error: {0}")]
    Crdt(String),
    
    #[error("Memory error: {0}")]
    Memory(String),
    
    #[error("Synchronization error: {0}")]
    Sync(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Generic error: {0}")]
    Generic(String),
}

/// Configuration for neuroplex cluster
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuroPlexConfig {
    pub node_id: NodeId,
    pub cluster_nodes: Vec<String>,
    pub raft_config: consensus::RaftConfig,
    pub memory_config: memory::MemoryConfig,
    pub sync_config: sync::SyncConfig,
}

impl Default for NeuroPlexConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            cluster_nodes: vec!["127.0.0.1:8000".to_string()],
            raft_config: consensus::RaftConfig::default(),
            memory_config: memory::MemoryConfig::default(),
            sync_config: sync::SyncConfig::default(),
        }
    }
}

/// Main neuroplex distributed memory system
pub struct NeuroPlexSystem {
    config: NeuroPlexConfig,
    distributed_store: Arc<distributed::DistributedStore>,
    consensus_engine: Arc<consensus::RaftConsensus>,
    memory_manager: Arc<memory::MemoryManager>,
    sync_coordinator: Arc<sync::SyncCoordinator>,
}

impl NeuroPlexSystem {
    /// Create a new neuroplex system with given configuration
    pub async fn new(config: NeuroPlexConfig) -> Result<Self> {
        info!("Initializing NeuroPlexSystem with config: {:?}", config);
        
        let memory_manager = Arc::new(memory::MemoryManager::new(config.memory_config.clone()));
        let consensus_engine = Arc::new(consensus::RaftConsensus::new(config.raft_config.clone(), config.node_id).await?);
        let sync_coordinator = Arc::new(sync::SyncCoordinator::new(config.sync_config.clone()).await?);
        let distributed_store = Arc::new(distributed::DistributedStore::new(
            config.node_id,
            Arc::clone(&memory_manager),
            Arc::clone(&consensus_engine),
            Arc::clone(&sync_coordinator),
        ).await?);
        
        Ok(Self {
            config,
            distributed_store,
            consensus_engine,
            memory_manager,
            sync_coordinator,
        })
    }
    
    /// Start the neuroplex system
    pub async fn start(&self) -> Result<()> {
        info!("Starting NeuroPlexSystem");
        
        // Start consensus engine
        self.consensus_engine.start().await?;
        
        // Start sync coordinator
        self.sync_coordinator.start().await?;
        
        // Start distributed store
        self.distributed_store.start().await?;
        
        info!("NeuroPlexSystem started successfully");
        Ok(())
    }
    
    /// Stop the neuroplex system
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping NeuroPlexSystem");
        
        // Stop distributed store
        self.distributed_store.stop().await?;
        
        // Stop sync coordinator
        self.sync_coordinator.stop().await?;
        
        // Stop consensus engine
        self.consensus_engine.stop().await?;
        
        info!("NeuroPlexSystem stopped successfully");
        Ok(())
    }
    
    /// Get distributed store reference
    pub fn distributed_store(&self) -> &Arc<distributed::DistributedStore> {
        &self.distributed_store
    }
    
    /// Get consensus engine reference
    pub fn consensus_engine(&self) -> &Arc<consensus::RaftConsensus> {
        &self.consensus_engine
    }
    
    /// Get memory manager reference
    pub fn memory_manager(&self) -> &Arc<memory::MemoryManager> {
        &self.memory_manager
    }
    
    /// Get sync coordinator reference
    pub fn sync_coordinator(&self) -> &Arc<sync::SyncCoordinator> {
        &self.sync_coordinator
    }
    
    /// Get system configuration
    pub fn config(&self) -> &NeuroPlexConfig {
        &self.config
    }
    
    /// Get system health status
    pub async fn health_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        
        status.insert("node_id".to_string(), self.config.node_id.to_string());
        status.insert("consensus_state".to_string(), self.consensus_engine.state().await.to_string());
        status.insert("memory_usage".to_string(), self.memory_manager.usage_stats().await.to_string());
        status.insert("sync_status".to_string(), self.sync_coordinator.status().await.to_string());
        
        status
    }
}

/// Initialize tracing for neuroplex
pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter("neuroplex=debug")
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    
    #[tokio::test]
    async fn test_neuroplex_system_creation() {
        let config = NeuroPlexConfig::default();
        let system = NeuroPlexSystem::new(config).await.unwrap();
        
        assert_eq!(system.config().cluster_nodes.len(), 1);
    }
    
    #[tokio::test]
    async fn test_system_start_stop() {
        let config = NeuroPlexConfig::default();
        let system = NeuroPlexSystem::new(config).await.unwrap();
        
        // Start system
        system.start().await.unwrap();
        
        // Check health status
        let status = system.health_status().await;
        assert!(status.contains_key("node_id"));
        assert!(status.contains_key("consensus_state"));
        
        // Stop system
        system.stop().await.unwrap();
    }
}