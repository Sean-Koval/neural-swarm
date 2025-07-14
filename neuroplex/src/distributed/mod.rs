//! Distributed memory store implementation
//!
//! This module provides a distributed in-memory store with async operations,
//! built on top of CRDTs and consensus protocols.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use dashmap::DashMap;
use tracing::{info, warn, error, debug};

use crate::{Result, NeuroPlexError, NodeId};
use crate::crdt::{CrdtValue, CrdtOperation};
use crate::consensus::RaftConsensus;
use crate::memory::MemoryManager;
use crate::sync::SyncCoordinator;

pub mod network;
pub mod operations;
pub mod replication;

use network::NetworkLayer;
use operations::DistributedOperation;
use replication::ReplicationManager;

/// Key type for distributed store
pub type StoreKey = String;

/// Value type for distributed store
pub type StoreValue = CrdtValue;

/// Event types for distributed store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoreEvent {
    ValueUpdated { key: StoreKey, value: StoreValue, node_id: NodeId },
    ValueDeleted { key: StoreKey, node_id: NodeId },
    NodeJoined { node_id: NodeId },
    NodeLeft { node_id: NodeId },
    SyncCompleted { node_id: NodeId },
}

/// Configuration for distributed store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedStoreConfig {
    pub replication_factor: usize,
    pub consistency_level: ConsistencyLevel,
    pub sync_interval_ms: u64,
    pub max_batch_size: usize,
    pub compression_enabled: bool,
}

impl Default for DistributedStoreConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            consistency_level: ConsistencyLevel::Strong,
            sync_interval_ms: 1000,
            max_batch_size: 1000,
            compression_enabled: true,
        }
    }
}

/// Consistency levels for distributed operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Eventual,
    Strong,
    Causal,
}

/// Distributed store implementation
pub struct DistributedStore {
    node_id: NodeId,
    config: DistributedStoreConfig,
    
    // Core storage
    storage: Arc<DashMap<StoreKey, StoreValue>>,
    
    // Components
    memory_manager: Arc<MemoryManager>,
    consensus_engine: Arc<RaftConsensus>,
    sync_coordinator: Arc<SyncCoordinator>,
    network_layer: Arc<NetworkLayer>,
    replication_manager: Arc<ReplicationManager>,
    
    // Event system
    event_sender: broadcast::Sender<StoreEvent>,
    event_receiver: broadcast::Receiver<StoreEvent>,
    
    // State
    is_running: Arc<RwLock<bool>>,
}

impl DistributedStore {
    /// Create a new distributed store
    pub async fn new(
        node_id: NodeId,
        memory_manager: Arc<MemoryManager>,
        consensus_engine: Arc<RaftConsensus>,
        sync_coordinator: Arc<SyncCoordinator>,
    ) -> Result<Self> {
        let config = DistributedStoreConfig::default();
        let storage = Arc::new(DashMap::new());
        
        let network_layer = Arc::new(NetworkLayer::new(node_id).await?);
        let replication_manager = Arc::new(ReplicationManager::new(
            node_id,
            config.replication_factor,
            Arc::clone(&storage),
            Arc::clone(&network_layer),
        ).await?);
        
        let (event_sender, event_receiver) = broadcast::channel(1000);
        
        Ok(Self {
            node_id,
            config,
            storage,
            memory_manager,
            consensus_engine,
            sync_coordinator,
            network_layer,
            replication_manager,
            event_sender,
            event_receiver,
            is_running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Start the distributed store
    pub async fn start(&self) -> Result<()> {
        info!("Starting distributed store for node {}", self.node_id);
        
        *self.is_running.write().await = true;
        
        // Start network layer
        self.network_layer.start().await?;
        
        // Start replication manager
        self.replication_manager.start().await?;
        
        // Start background tasks
        self.start_sync_task().await;
        self.start_event_processing().await;
        
        info!("Distributed store started successfully");
        Ok(())
    }
    
    /// Stop the distributed store
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping distributed store for node {}", self.node_id);
        
        *self.is_running.write().await = false;
        
        // Stop replication manager
        self.replication_manager.stop().await?;
        
        // Stop network layer
        self.network_layer.stop().await?;
        
        info!("Distributed store stopped successfully");
        Ok(())
    }
    
    /// Get a value from the distributed store
    pub async fn get(&self, key: &StoreKey) -> Result<Option<StoreValue>> {
        debug!("Getting value for key: {}", key);
        
        // Check local storage first
        if let Some(value) = self.storage.get(key) {
            return Ok(Some(value.clone()));
        }
        
        // If not found locally, try to sync from other nodes
        if self.config.consistency_level == ConsistencyLevel::Strong {
            self.sync_key_from_peers(key).await?;
            if let Some(value) = self.storage.get(key) {
                return Ok(Some(value.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Set a value in the distributed store
    pub async fn set(&self, key: StoreKey, value: StoreValue) -> Result<()> {
        debug!("Setting value for key: {}", key);
        
        // Create operation
        let operation = DistributedOperation::Set {
            key: key.clone(),
            value: value.clone(),
            node_id: self.node_id,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        
        // Apply locally first
        self.apply_operation(&operation).await?;
        
        // Replicate to other nodes
        self.replication_manager.replicate_operation(operation).await?;
        
        // Send event
        let event = StoreEvent::ValueUpdated {
            key,
            value,
            node_id: self.node_id,
        };
        let _ = self.event_sender.send(event);
        
        Ok(())
    }
    
    /// Delete a value from the distributed store
    pub async fn delete(&self, key: &StoreKey) -> Result<()> {
        debug!("Deleting value for key: {}", key);
        
        // Create operation
        let operation = DistributedOperation::Delete {
            key: key.clone(),
            node_id: self.node_id,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        
        // Apply locally first
        self.apply_operation(&operation).await?;
        
        // Replicate to other nodes
        self.replication_manager.replicate_operation(operation).await?;
        
        // Send event
        let event = StoreEvent::ValueDeleted {
            key: key.clone(),
            node_id: self.node_id,
        };
        let _ = self.event_sender.send(event);
        
        Ok(())
    }
    
    /// List all keys in the distributed store
    pub async fn keys(&self) -> Result<Vec<StoreKey>> {
        Ok(self.storage.iter().map(|entry| entry.key().clone()).collect())
    }
    
    /// Get the size of the distributed store
    pub async fn size(&self) -> Result<usize> {
        Ok(self.storage.len())
    }
    
    /// Subscribe to store events
    pub fn subscribe(&self) -> broadcast::Receiver<StoreEvent> {
        self.event_sender.subscribe()
    }
    
    /// Merge CRDT values for conflict resolution
    pub async fn merge_crdt(&self, key: &StoreKey, local_value: &StoreValue, remote_value: &StoreValue) -> Result<StoreValue> {
        // Use CRDT merge logic
        let merged = local_value.merge(remote_value)?;
        
        // Update local storage
        self.storage.insert(key.clone(), merged.clone());
        
        Ok(merged)
    }
    
    /// Apply a distributed operation
    async fn apply_operation(&self, operation: &DistributedOperation) -> Result<()> {
        match operation {
            DistributedOperation::Set { key, value, .. } => {
                self.storage.insert(key.clone(), value.clone());
            }
            DistributedOperation::Delete { key, .. } => {
                self.storage.remove(key);
            }
            DistributedOperation::Merge { key, value, .. } => {
                if let Some(existing) = self.storage.get(key) {
                    let merged = existing.merge(value)?;
                    self.storage.insert(key.clone(), merged);
                } else {
                    self.storage.insert(key.clone(), value.clone());
                }
            }
        }
        
        Ok(())
    }
    
    /// Sync a specific key from peer nodes
    async fn sync_key_from_peers(&self, key: &StoreKey) -> Result<()> {
        // Implementation would involve network calls to peer nodes
        // For now, this is a placeholder
        debug!("Syncing key {} from peers", key);
        Ok(())
    }
    
    /// Start background sync task
    async fn start_sync_task(&self) {
        let storage = Arc::clone(&self.storage);
        let sync_coordinator = Arc::clone(&self.sync_coordinator);
        let is_running = Arc::clone(&self.is_running);
        let interval = self.config.sync_interval_ms;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(interval));
            
            while *is_running.read().await {
                interval.tick().await;
                
                // Perform periodic sync
                if let Err(e) = sync_coordinator.sync_with_peers().await {
                    error!("Sync error: {}", e);
                }
            }
        });
    }
    
    /// Start event processing task
    async fn start_event_processing(&self) {
        let mut receiver = self.event_sender.subscribe();
        let is_running = Arc::clone(&self.is_running);
        
        tokio::spawn(async move {
            while *is_running.read().await {
                if let Ok(event) = receiver.recv().await {
                    debug!("Processing event: {:?}", event);
                    // Process event (logging, metrics, etc.)
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::RaftConfig;
    use crate::memory::MemoryConfig;
    use crate::sync::SyncConfig;
    use crate::crdt::GCounter;
    
    #[tokio::test]
    async fn test_distributed_store_creation() {
        let node_id = Uuid::new_v4();
        let memory_manager = Arc::new(MemoryManager::new(MemoryConfig::default()));
        let consensus_engine = Arc::new(RaftConsensus::new(RaftConfig::default(), node_id).await.unwrap());
        let sync_coordinator = Arc::new(SyncCoordinator::new(SyncConfig::default()).await.unwrap());
        
        let store = DistributedStore::new(
            node_id,
            memory_manager,
            consensus_engine,
            sync_coordinator,
        ).await.unwrap();
        
        assert_eq!(store.node_id, node_id);
    }
    
    #[tokio::test]
    async fn test_store_operations() {
        let node_id = Uuid::new_v4();
        let memory_manager = Arc::new(MemoryManager::new(MemoryConfig::default()));
        let consensus_engine = Arc::new(RaftConsensus::new(RaftConfig::default(), node_id).await.unwrap());
        let sync_coordinator = Arc::new(SyncCoordinator::new(SyncConfig::default()).await.unwrap());
        
        let store = DistributedStore::new(
            node_id,
            memory_manager,
            consensus_engine,
            sync_coordinator,
        ).await.unwrap();
        
        // Test set and get
        let key = "test_key".to_string();
        let value = CrdtValue::GCounter(GCounter::new(node_id));
        
        store.set(key.clone(), value.clone()).await.unwrap();
        let retrieved = store.get(&key).await.unwrap();
        
        assert!(retrieved.is_some());
    }
}