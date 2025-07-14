//! Synchronization module
//!
//! This module provides synchronization primitives and conflict resolution
//! for distributed operations.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex, broadcast, Semaphore};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

use crate::{Result, NeuroPlexError, NodeId};
use crate::crdt::{CrdtValue, VectorClock};

pub mod conflict_resolution;
pub mod coordination;
pub mod barriers;
pub mod locks;

use conflict_resolution::ConflictResolver;
use coordination::DistributedCoordinator;
use barriers::DistributedBarrier;
use locks::DistributedLock;

/// Synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Sync interval in milliseconds
    pub sync_interval_ms: u64,
    
    /// Timeout for sync operations
    pub sync_timeout_ms: u64,
    
    /// Maximum number of concurrent sync operations
    pub max_concurrent_syncs: usize,
    
    /// Conflict resolution strategy
    pub conflict_resolution_strategy: ConflictResolutionStrategy,
    
    /// Enable vector clock synchronization
    pub enable_vector_clocks: bool,
    
    /// Batch size for sync operations
    pub batch_size: usize,
    
    /// Retry configuration
    pub max_retries: usize,
    pub retry_delay_ms: u64,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            sync_interval_ms: 1000,
            sync_timeout_ms: 5000,
            max_concurrent_syncs: 10,
            conflict_resolution_strategy: ConflictResolutionStrategy::LastWriterWins,
            enable_vector_clocks: true,
            batch_size: 100,
            max_retries: 3,
            retry_delay_ms: 100,
        }
    }
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Last writer wins based on timestamp
    LastWriterWins,
    /// First writer wins
    FirstWriterWins,
    /// Merge using CRDT semantics
    CrdtMerge,
    /// Manual resolution required
    Manual,
}

/// Synchronization operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncOperation {
    /// Sync a specific key
    SyncKey { key: String, value: Option<CrdtValue> },
    /// Sync all keys
    SyncAll,
    /// Sync a batch of keys
    SyncBatch { keys: Vec<String> },
    /// Heartbeat operation
    Heartbeat { node_id: NodeId, timestamp: u64 },
}

/// Synchronization event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    /// Sync operation started
    SyncStarted { operation: SyncOperation, node_id: NodeId },
    /// Sync operation completed
    SyncCompleted { operation: SyncOperation, node_id: NodeId, duration: Duration },
    /// Sync operation failed
    SyncFailed { operation: SyncOperation, node_id: NodeId, error: String },
    /// Conflict detected
    ConflictDetected { key: String, conflicting_nodes: Vec<NodeId> },
    /// Conflict resolved
    ConflictResolved { key: String, resolution: ConflictResolution },
    /// Node joined
    NodeJoined { node_id: NodeId },
    /// Node left
    NodeLeft { node_id: NodeId },
}

/// Conflict resolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolution {
    /// The key that had the conflict
    pub key: String,
    /// The resolved value
    pub resolved_value: CrdtValue,
    /// The strategy used to resolve the conflict
    pub strategy: ConflictResolutionStrategy,
    /// The nodes that were involved in the conflict
    pub involved_nodes: Vec<NodeId>,
    /// The timestamp when the conflict was resolved
    pub resolved_at: u64,
}

/// Synchronization state for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    /// Node ID
    pub node_id: NodeId,
    /// Vector clock for causality tracking
    pub vector_clock: VectorClock,
    /// Last sync timestamp
    pub last_sync: u64,
    /// Keys that this node has
    pub keys: HashSet<String>,
    /// Status of the node
    pub status: NodeStatus,
}

/// Node status in the cluster
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Inactive,
    Syncing,
    Conflicted,
}

/// Sync coordinator for distributed synchronization
pub struct SyncCoordinator {
    /// Configuration
    config: SyncConfig,
    
    /// Node ID
    node_id: NodeId,
    
    /// Cluster state
    nodes: Arc<RwLock<HashMap<NodeId, SyncState>>>,
    
    /// Vector clock for this node
    vector_clock: Arc<RwLock<VectorClock>>,
    
    /// Components
    conflict_resolver: Arc<ConflictResolver>,
    distributed_coordinator: Arc<DistributedCoordinator>,
    
    /// Synchronization primitives
    sync_semaphore: Arc<Semaphore>,
    barriers: Arc<RwLock<HashMap<String, Arc<DistributedBarrier>>>>,
    locks: Arc<RwLock<HashMap<String, Arc<DistributedLock>>>>,
    
    /// Events
    event_sender: broadcast::Sender<SyncEvent>,
    event_receiver: broadcast::Receiver<SyncEvent>,
    
    /// State
    is_running: Arc<RwLock<bool>>,
    sync_in_progress: Arc<RwLock<HashSet<String>>>,
}

impl SyncCoordinator {
    /// Create a new sync coordinator
    pub async fn new(config: SyncConfig) -> Result<Self> {
        let node_id = Uuid::new_v4();
        let vector_clock = Arc::new(RwLock::new(VectorClock::new(node_id)));
        
        let conflict_resolver = Arc::new(ConflictResolver::new(config.conflict_resolution_strategy));
        let distributed_coordinator = Arc::new(DistributedCoordinator::new(node_id).await?);
        
        let sync_semaphore = Arc::new(Semaphore::new(config.max_concurrent_syncs));
        
        let (event_sender, event_receiver) = broadcast::channel(1000);
        
        Ok(Self {
            config,
            node_id,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            vector_clock,
            conflict_resolver,
            distributed_coordinator,
            sync_semaphore,
            barriers: Arc::new(RwLock::new(HashMap::new())),
            locks: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            event_receiver,
            is_running: Arc::new(RwLock::new(false)),
            sync_in_progress: Arc::new(RwLock::new(HashSet::new())),
        })
    }
    
    /// Start the sync coordinator
    pub async fn start(&self) -> Result<()> {
        info!("Starting sync coordinator for node {}", self.node_id);
        
        *self.is_running.write().await = true;
        
        // Start distributed coordinator
        self.distributed_coordinator.start().await?;
        
        // Start background tasks
        self.start_sync_task().await;
        self.start_heartbeat_task().await;
        self.start_cleanup_task().await;
        
        info!("Sync coordinator started successfully");
        Ok(())
    }
    
    /// Stop the sync coordinator
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping sync coordinator for node {}", self.node_id);
        
        *self.is_running.write().await = false;
        
        // Stop distributed coordinator
        self.distributed_coordinator.stop().await?;
        
        // Clear state
        self.nodes.write().await.clear();
        self.barriers.write().await.clear();
        self.locks.write().await.clear();
        
        info!("Sync coordinator stopped successfully");
        Ok(())
    }
    
    /// Sync with all peers
    pub async fn sync_with_peers(&self) -> Result<()> {
        debug!("Starting sync with peers");
        
        let _permit = self.sync_semaphore.acquire().await
            .map_err(|e| NeuroPlexError::Sync(format!("Failed to acquire sync semaphore: {}", e)))?;
        
        let start_time = Instant::now();
        let operation = SyncOperation::SyncAll;
        
        // Send sync started event
        let _ = self.event_sender.send(SyncEvent::SyncStarted {
            operation: operation.clone(),
            node_id: self.node_id,
        });
        
        // Update vector clock
        self.vector_clock.write().await.tick();
        
        // Perform sync with all known nodes
        let nodes: Vec<NodeId> = self.nodes.read().await.keys().cloned().collect();
        
        for node_id in nodes {
            if let Err(e) = self.sync_with_node(node_id).await {
                warn!("Failed to sync with node {}: {}", node_id, e);
            }
        }
        
        let duration = start_time.elapsed();
        
        // Send sync completed event
        let _ = self.event_sender.send(SyncEvent::SyncCompleted {
            operation,
            node_id: self.node_id,
            duration,
        });
        
        debug!("Completed sync with peers in {:?}", duration);
        Ok(())
    }
    
    /// Sync a specific key
    pub async fn sync_key(&self, key: &str, value: Option<CrdtValue>) -> Result<()> {
        debug!("Syncing key: {}", key);
        
        // Check if sync is already in progress for this key
        {
            let mut sync_in_progress = self.sync_in_progress.write().await;
            if sync_in_progress.contains(key) {
                return Ok(());
            }
            sync_in_progress.insert(key.to_string());
        }
        
        let _permit = self.sync_semaphore.acquire().await
            .map_err(|e| NeuroPlexError::Sync(format!("Failed to acquire sync semaphore: {}", e)))?;
        
        let start_time = Instant::now();
        let operation = SyncOperation::SyncKey {
            key: key.to_string(),
            value: value.clone(),
        };
        
        // Send sync started event
        let _ = self.event_sender.send(SyncEvent::SyncStarted {
            operation: operation.clone(),
            node_id: self.node_id,
        });
        
        // Update vector clock
        self.vector_clock.write().await.tick();
        
        // Perform sync operation
        let result = self.perform_key_sync(key, value).await;
        
        // Remove from sync in progress
        self.sync_in_progress.write().await.remove(key);
        
        let duration = start_time.elapsed();
        
        match result {
            Ok(_) => {
                let _ = self.event_sender.send(SyncEvent::SyncCompleted {
                    operation,
                    node_id: self.node_id,
                    duration,
                });
                debug!("Successfully synced key: {}", key);
            }
            Err(e) => {
                let _ = self.event_sender.send(SyncEvent::SyncFailed {
                    operation,
                    node_id: self.node_id,
                    error: e.to_string(),
                });
                error!("Failed to sync key {}: {}", key, e);
            }
        }
        
        result
    }
    
    /// Resolve conflicts for a key
    pub async fn resolve_conflict(&self, key: &str, conflicting_values: Vec<(NodeId, CrdtValue)>) -> Result<CrdtValue> {
        debug!("Resolving conflict for key: {}", key);
        
        let conflicting_nodes: Vec<NodeId> = conflicting_values.iter().map(|(node_id, _)| *node_id).collect();
        
        // Send conflict detected event
        let _ = self.event_sender.send(SyncEvent::ConflictDetected {
            key: key.to_string(),
            conflicting_nodes: conflicting_nodes.clone(),
        });
        
        // Use conflict resolver
        let resolved_value = self.conflict_resolver.resolve(key, conflicting_values).await?;
        
        // Create conflict resolution record
        let resolution = ConflictResolution {
            key: key.to_string(),
            resolved_value: resolved_value.clone(),
            strategy: self.config.conflict_resolution_strategy,
            involved_nodes: conflicting_nodes,
            resolved_at: chrono::Utc::now().timestamp_millis() as u64,
        };
        
        // Send conflict resolved event
        let _ = self.event_sender.send(SyncEvent::ConflictResolved {
            key: key.to_string(),
            resolution,
        });
        
        debug!("Resolved conflict for key: {}", key);
        Ok(resolved_value)
    }
    
    /// Create a distributed barrier
    pub async fn create_barrier(&self, name: &str, count: usize) -> Result<Arc<DistributedBarrier>> {
        let barrier = Arc::new(DistributedBarrier::new(name.to_string(), count, self.node_id));
        self.barriers.write().await.insert(name.to_string(), Arc::clone(&barrier));
        Ok(barrier)
    }
    
    /// Create a distributed lock
    pub async fn create_lock(&self, name: &str) -> Result<Arc<DistributedLock>> {
        let lock = Arc::new(DistributedLock::new(name.to_string(), self.node_id));
        self.locks.write().await.insert(name.to_string(), Arc::clone(&lock));
        Ok(lock)
    }
    
    /// Get sync status
    pub async fn status(&self) -> HashMap<String, serde_json::Value> {
        let mut status = HashMap::new();
        
        status.insert("node_id".to_string(), serde_json::Value::String(self.node_id.to_string()));
        status.insert("is_running".to_string(), serde_json::Value::Bool(*self.is_running.read().await));
        status.insert("node_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.nodes.read().await.len())));
        status.insert("sync_in_progress".to_string(), serde_json::Value::Number(serde_json::Number::from(self.sync_in_progress.read().await.len())));
        status.insert("barriers".to_string(), serde_json::Value::Number(serde_json::Number::from(self.barriers.read().await.len())));
        status.insert("locks".to_string(), serde_json::Value::Number(serde_json::Number::from(self.locks.read().await.len())));
        
        status
    }
    
    /// Subscribe to sync events
    pub fn subscribe(&self) -> broadcast::Receiver<SyncEvent> {
        self.event_sender.subscribe()
    }
    
    /// Sync with a specific node
    async fn sync_with_node(&self, node_id: NodeId) -> Result<()> {
        debug!("Syncing with node: {}", node_id);
        
        // Get node state
        let node_state = self.nodes.read().await.get(&node_id).cloned();
        
        if let Some(state) = node_state {
            // Update vector clock with node's clock
            self.vector_clock.write().await.update(&state.vector_clock);
        }
        
        // Perform actual sync logic here
        // This would involve network communication with the node
        
        Ok(())
    }
    
    /// Perform key synchronization
    async fn perform_key_sync(&self, key: &str, value: Option<CrdtValue>) -> Result<()> {
        debug!("Performing key sync for: {}", key);
        
        // This would involve:
        // 1. Sending the key/value to other nodes
        // 2. Receiving responses with their values
        // 3. Detecting conflicts
        // 4. Resolving conflicts if any
        
        // For now, this is a placeholder
        Ok(())
    }
    
    /// Start background sync task
    async fn start_sync_task(&self) {
        let is_running = Arc::clone(&self.is_running);
        let sync_coordinator = Arc::new(self as *const SyncCoordinator);
        let interval = Duration::from_millis(self.config.sync_interval_ms);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            
            while *is_running.read().await {
                interval.tick().await;
                
                // Safety: This is safe because the task is spawned from within the SyncCoordinator
                unsafe {
                    if let Err(e) = (*sync_coordinator.as_ref()).sync_with_peers().await {
                        debug!("Periodic sync failed: {}", e);
                    }
                }
            }
        });
    }
    
    /// Start heartbeat task
    async fn start_heartbeat_task(&self) {
        let is_running = Arc::clone(&self.is_running);
        let event_sender = self.event_sender.clone();
        let node_id = self.node_id;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            while *is_running.read().await {
                interval.tick().await;
                
                let operation = SyncOperation::Heartbeat {
                    node_id,
                    timestamp: chrono::Utc::now().timestamp_millis() as u64,
                };
                
                let _ = event_sender.send(SyncEvent::SyncStarted {
                    operation,
                    node_id,
                });
            }
        });
    }
    
    /// Start cleanup task
    async fn start_cleanup_task(&self) {
        let is_running = Arc::clone(&self.is_running);
        let barriers = Arc::clone(&self.barriers);
        let locks = Arc::clone(&self.locks);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
            
            while *is_running.read().await {
                interval.tick().await;
                
                // Clean up expired barriers and locks
                barriers.write().await.retain(|_, barrier| {
                    Arc::strong_count(barrier) > 1 // Keep if there are external references
                });
                
                locks.write().await.retain(|_, lock| {
                    Arc::strong_count(lock) > 1 // Keep if there are external references
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    
    #[tokio::test]
    async fn test_sync_coordinator_creation() {
        let config = SyncConfig::default();
        let coordinator = SyncCoordinator::new(config).await.unwrap();
        
        let status = coordinator.status().await;
        assert!(status.contains_key("node_id"));
        assert!(status.contains_key("is_running"));
    }
    
    #[tokio::test]
    async fn test_sync_coordinator_start_stop() {
        let config = SyncConfig::default();
        let coordinator = SyncCoordinator::new(config).await.unwrap();
        
        coordinator.start().await.unwrap();
        assert!(*coordinator.is_running.read().await);
        
        coordinator.stop().await.unwrap();
        assert!(!*coordinator.is_running.read().await);
    }
    
    #[tokio::test]
    async fn test_barrier_creation() {
        let config = SyncConfig::default();
        let coordinator = SyncCoordinator::new(config).await.unwrap();
        
        let barrier = coordinator.create_barrier("test_barrier", 3).await.unwrap();
        
        assert_eq!(barrier.name(), "test_barrier");
        assert_eq!(barrier.count(), 3);
    }
    
    #[tokio::test]
    async fn test_lock_creation() {
        let config = SyncConfig::default();
        let coordinator = SyncCoordinator::new(config).await.unwrap();
        
        let lock = coordinator.create_lock("test_lock").await.unwrap();
        
        assert_eq!(lock.name(), "test_lock");
    }
}