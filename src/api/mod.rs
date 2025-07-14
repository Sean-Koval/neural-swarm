//! High-level API
//!
//! High-level API for neuroplex distributed memory system

use crate::{
    NodeId, Result, NeuroError, NeuroConfig,
    memory::{DistributedMemory, MemoryManager, MemoryStats, MemoryCallback},
    consensus::{RaftConsensus, Consensus, ConsensusStats},
    sync::{GossipSync, Synchronizer, SyncStats},
    crdt::{GCounter, PNCounter, ORSet, LWWRegister},
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use tracing::{info, warn, error, debug};

/// High-level cluster interface
pub struct NeuroCluster {
    /// Node configuration
    config: NeuroConfig,
    /// Distributed memory
    memory: Arc<dyn MemoryManager>,
    /// Consensus layer
    consensus: Arc<RwLock<dyn Consensus>>,
    /// Synchronization layer
    sync: Arc<RwLock<dyn Synchronizer>>,
    /// CRDT instances
    crdts: Arc<RwLock<HashMap<String, CrdtInstance>>>,
    /// Running state
    running: Arc<RwLock<bool>>,
}

/// CRDT instance wrapper
#[derive(Debug, Clone)]
pub enum CrdtInstance {
    GCounter(Arc<RwLock<GCounter>>),
    PNCounter(Arc<RwLock<PNCounter>>),
    ORSet(Arc<RwLock<ORSet<String>>>),
    LWWRegister(Arc<RwLock<LWWRegister<String>>>),
}

impl NeuroCluster {
    /// Create a new cluster instance
    pub async fn new(node_name: &str, address: &str) -> Result<Self> {
        let config = NeuroConfig {
            node_id: Uuid::new_v4(),
            address: address.to_string(),
            peers: Vec::new(),
            ..Default::default()
        };
        
        info!("Creating new cluster node {} at {}", node_name, address);
        
        // Create memory layer
        let memory = Arc::new(DistributedMemory::new(
            config.node_id,
            config.memory.clone(),
        ));
        
        // Create consensus layer
        let consensus = Arc::new(RwLock::new(RaftConsensus::new(
            config.node_id,
            vec![config.node_id],
            config.consensus.clone(),
        )));
        
        // Create sync layer
        let sync = Arc::new(RwLock::new(GossipSync::new(
            config.node_id,
            address.parse().map_err(|e| NeuroError::config(format!("Invalid address: {}", e)))?,
            config.sync.clone(),
        )));
        
        Ok(Self {
            config,
            memory,
            consensus,
            sync,
            crdts: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Start the cluster
    pub async fn start(&self) -> Result<()> {
        info!("Starting cluster node {}", self.config.node_id);
        
        let mut running = self.running.write().await;
        if *running {
            return Err(NeuroError::invalid_operation("Cluster already running"));
        }
        
        // Start consensus
        let mut consensus = self.consensus.write().await;
        consensus.start().await?;
        drop(consensus);
        
        // Start sync
        let mut sync = self.sync.write().await;
        sync.start().await?;
        drop(sync);
        
        *running = true;
        
        info!("Cluster node {} started successfully", self.config.node_id);
        Ok(())
    }
    
    /// Stop the cluster
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping cluster node {}", self.config.node_id);
        
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }
        
        // Stop sync
        let mut sync = self.sync.write().await;
        sync.stop().await?;
        drop(sync);
        
        // Stop consensus
        let mut consensus = self.consensus.write().await;
        consensus.stop().await?;
        drop(consensus);
        
        *running = false;
        
        info!("Cluster node {} stopped", self.config.node_id);
        Ok(())
    }
    
    /// Set a key-value pair
    pub async fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        self.memory.set(key, value).await
    }
    
    /// Get a value by key
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.memory.get(key).await
    }
    
    /// Delete a key
    pub async fn delete(&self, key: &str) -> Result<bool> {
        self.memory.delete(key).await
    }
    
    /// List keys with optional prefix
    pub async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        self.memory.list(prefix).await
    }
    
    /// Check if key exists
    pub async fn exists(&self, key: &str) -> Result<bool> {
        self.memory.exists(key).await
    }
    
    /// Get memory statistics
    pub async fn stats(&self) -> Result<MemoryStats> {
        self.memory.stats().await
    }
    
    /// Subscribe to memory events
    pub async fn subscribe(&self, callback: MemoryCallback) -> Result<()> {
        self.memory.subscribe(callback).await
    }
    
    /// Compact memory
    pub async fn compact(&self) -> Result<usize> {
        self.memory.compact().await
    }
    
    /// Add a node to the cluster
    pub async fn add_node(&mut self, node_id: NodeId, address: &str) -> Result<()> {
        info!("Adding node {} at {}", node_id, address);
        
        // Add to config
        if !self.config.peers.contains(&address.to_string()) {
            self.config.peers.push(address.to_string());
        }
        
        // Add to consensus
        let mut consensus = self.consensus.write().await;
        consensus.add_node(node_id).await?;
        drop(consensus);
        
        // Add to sync
        let mut sync = self.sync.write().await;
        let socket_addr = address.parse()
            .map_err(|e| NeuroError::config(format!("Invalid address: {}", e)))?;
        sync.add_node(node_id, socket_addr).await?;
        drop(sync);
        
        info!("Node {} added successfully", node_id);
        Ok(())
    }
    
    /// Remove a node from the cluster
    pub async fn remove_node(&mut self, node_id: NodeId) -> Result<()> {
        info!("Removing node {}", node_id);
        
        // Remove from consensus
        let mut consensus = self.consensus.write().await;
        consensus.remove_node(node_id).await?;
        drop(consensus);
        
        // Remove from sync
        let mut sync = self.sync.write().await;
        sync.remove_node(node_id).await?;
        drop(sync);
        
        info!("Node {} removed successfully", node_id);
        Ok(())
    }
    
    /// Get cluster members
    pub async fn get_members(&self) -> Result<Vec<NodeId>> {
        let consensus = self.consensus.read().await;
        consensus.get_cluster_members().await
    }
    
    /// Get consensus statistics
    pub async fn consensus_stats(&self) -> Result<ConsensusStats> {
        let consensus = self.consensus.read().await;
        consensus.get_stats().await
    }
    
    /// Get synchronization statistics
    pub async fn sync_stats(&self) -> Result<SyncStats> {
        let sync = self.sync.read().await;
        sync.get_stats().await
    }
    
    /// Create a G-Counter CRDT
    pub async fn create_g_counter(&self, name: &str) -> Result<()> {
        let mut crdts = self.crdts.write().await;
        
        if crdts.contains_key(name) {
            return Err(NeuroError::invalid_operation("CRDT already exists"));
        }
        
        let counter = Arc::new(RwLock::new(GCounter::new()));
        crdts.insert(name.to_string(), CrdtInstance::GCounter(counter));
        
        info!("Created G-Counter CRDT: {}", name);
        Ok(())
    }
    
    /// Create a PN-Counter CRDT
    pub async fn create_pn_counter(&self, name: &str) -> Result<()> {
        let mut crdts = self.crdts.write().await;
        
        if crdts.contains_key(name) {
            return Err(NeuroError::invalid_operation("CRDT already exists"));
        }
        
        let counter = Arc::new(RwLock::new(PNCounter::new()));
        crdts.insert(name.to_string(), CrdtInstance::PNCounter(counter));
        
        info!("Created PN-Counter CRDT: {}", name);
        Ok(())
    }
    
    /// Create an OR-Set CRDT
    pub async fn create_or_set(&self, name: &str) -> Result<()> {
        let mut crdts = self.crdts.write().await;
        
        if crdts.contains_key(name) {
            return Err(NeuroError::invalid_operation("CRDT already exists"));
        }
        
        let set = Arc::new(RwLock::new(ORSet::new()));
        crdts.insert(name.to_string(), CrdtInstance::ORSet(set));
        
        info!("Created OR-Set CRDT: {}", name);
        Ok(())
    }
    
    /// Create an LWW-Register CRDT
    pub async fn create_lww_register(&self, name: &str) -> Result<()> {
        let mut crdts = self.crdts.write().await;
        
        if crdts.contains_key(name) {
            return Err(NeuroError::invalid_operation("CRDT already exists"));
        }
        
        let register = Arc::new(RwLock::new(LWWRegister::new()));
        crdts.insert(name.to_string(), CrdtInstance::LWWRegister(register));
        
        info!("Created LWW-Register CRDT: {}", name);
        Ok(())
    }
    
    /// Get CRDT instance
    pub async fn get_crdt(&self, name: &str) -> Result<Option<CrdtInstance>> {
        let crdts = self.crdts.read().await;
        Ok(crdts.get(name).cloned())
    }
    
    /// List CRDT names
    pub async fn list_crdts(&self) -> Result<Vec<String>> {
        let crdts = self.crdts.read().await;
        Ok(crdts.keys().cloned().collect())
    }
    
    /// Remove a CRDT
    pub async fn remove_crdt(&self, name: &str) -> Result<bool> {
        let mut crdts = self.crdts.write().await;
        let removed = crdts.remove(name).is_some();
        
        if removed {
            info!("Removed CRDT: {}", name);
        }
        
        Ok(removed)
    }
    
    /// Get node ID
    pub fn node_id(&self) -> NodeId {
        self.config.node_id
    }
    
    /// Get node address
    pub fn address(&self) -> &str {
        &self.config.address
    }
    
    /// Check if cluster is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
    
    /// Get configuration
    pub fn config(&self) -> &NeuroConfig {
        &self.config
    }
}

/// Individual node interface
pub struct NeuroNode {
    /// Node ID
    node_id: NodeId,
    /// Node address
    address: SocketAddr,
    /// Distributed memory
    memory: Arc<dyn MemoryManager>,
    /// Running state
    running: Arc<RwLock<bool>>,
}

impl NeuroNode {
    /// Create a new node
    pub async fn new(address: SocketAddr) -> Result<Self> {
        let node_id = Uuid::new_v4();
        let config = crate::memory::MemoryConfig::default();
        
        let memory = Arc::new(DistributedMemory::new(node_id, config));
        
        Ok(Self {
            node_id,
            address,
            memory,
            running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Start the node
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(NeuroError::invalid_operation("Node already running"));
        }
        
        *running = true;
        info!("Node {} started at {}", self.node_id, self.address);
        
        Ok(())
    }
    
    /// Stop the node
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }
        
        *running = false;
        info!("Node {} stopped", self.node_id);
        
        Ok(())
    }
    
    /// Set a key-value pair
    pub async fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        self.memory.set(key, value).await
    }
    
    /// Get a value by key
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.memory.get(key).await
    }
    
    /// Delete a key
    pub async fn delete(&self, key: &str) -> Result<bool> {
        self.memory.delete(key).await
    }
    
    /// List keys with optional prefix
    pub async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        self.memory.list(prefix).await
    }
    
    /// Get memory statistics
    pub async fn stats(&self) -> Result<MemoryStats> {
        self.memory.stats().await
    }
    
    /// Get node ID
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
    
    /// Get node address
    pub fn address(&self) -> SocketAddr {
        self.address
    }
    
    /// Check if node is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    
    #[tokio::test]
    async fn test_neuro_cluster_creation() {
        let cluster = NeuroCluster::new("test-node", "127.0.0.1:8080").await.unwrap();
        assert_eq!(cluster.address(), "127.0.0.1:8080");
        assert!(!cluster.is_running().await);
    }
    
    #[tokio::test]
    async fn test_neuro_node_creation() {
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = NeuroNode::new(address).await.unwrap();
        assert_eq!(node.address(), address);
        assert!(!node.is_running().await);
    }
    
    #[tokio::test]
    async fn test_cluster_basic_operations() {
        let cluster = NeuroCluster::new("test-node", "127.0.0.1:8080").await.unwrap();
        
        // Test set/get
        cluster.set("key1", b"value1").await.unwrap();
        let value = cluster.get("key1").await.unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));
        
        // Test exists
        assert!(cluster.exists("key1").await.unwrap());
        assert!(!cluster.exists("nonexistent").await.unwrap());
        
        // Test delete
        assert!(cluster.delete("key1").await.unwrap());
        assert!(!cluster.exists("key1").await.unwrap());
    }
    
    #[tokio::test]
    async fn test_cluster_crdts() {
        let cluster = NeuroCluster::new("test-node", "127.0.0.1:8080").await.unwrap();
        
        // Test G-Counter creation
        cluster.create_g_counter("counter1").await.unwrap();
        assert!(cluster.get_crdt("counter1").await.unwrap().is_some());
        
        // Test PN-Counter creation
        cluster.create_pn_counter("counter2").await.unwrap();
        assert!(cluster.get_crdt("counter2").await.unwrap().is_some());
        
        // Test OR-Set creation
        cluster.create_or_set("set1").await.unwrap();
        assert!(cluster.get_crdt("set1").await.unwrap().is_some());
        
        // Test LWW-Register creation
        cluster.create_lww_register("register1").await.unwrap();
        assert!(cluster.get_crdt("register1").await.unwrap().is_some());
        
        // Test list CRDTs
        let crdts = cluster.list_crdts().await.unwrap();
        assert_eq!(crdts.len(), 4);
        
        // Test remove CRDT
        assert!(cluster.remove_crdt("counter1").await.unwrap());
        assert!(!cluster.remove_crdt("nonexistent").await.unwrap());
    }
}