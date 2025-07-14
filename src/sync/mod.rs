//! Multi-Node Synchronization
//!
//! Implementation of gossip protocol and delta synchronization for
//! efficient state propagation across nodes.

pub mod gossip;
pub mod delta_sync;
pub mod membership;
pub mod failure_detection;

pub use gossip::GossipSync;
pub use delta_sync::{DeltaSync, DeltaSyncMessage};
pub use membership::{ClusterMembership, MembershipEvent};
pub use failure_detection::{FailureDetector, FailureDetectorConfig};

use crate::{NodeId, Timestamp, VersionVector, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;

/// Sync configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Gossip interval in milliseconds
    pub gossip_interval: u64,
    /// Maximum gossip fanout
    pub gossip_fanout: usize,
    /// Delta sync batch size
    pub delta_sync_batch_size: usize,
    /// Failure detection timeout
    pub failure_detection_timeout: u64,
    /// Network timeout
    pub network_timeout: u64,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            gossip_interval: 100,
            gossip_fanout: 3,
            delta_sync_batch_size: 1000,
            failure_detection_timeout: 5000,
            network_timeout: 1000,
        }
    }
}

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: NodeId,
    pub address: SocketAddr,
    pub last_seen: Timestamp,
    pub version_vector: VersionVector,
    pub metadata: HashMap<String, String>,
}

/// Sync message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncMessage {
    /// Gossip message containing node state
    Gossip {
        sender: NodeId,
        nodes: Vec<NodeInfo>,
        timestamp: Timestamp,
    },
    /// Delta sync request
    DeltaRequest {
        sender: NodeId,
        version_vector: VersionVector,
        timestamp: Timestamp,
    },
    /// Delta sync response
    DeltaResponse {
        sender: NodeId,
        deltas: Vec<u8>, // Serialized delta data
        timestamp: Timestamp,
    },
    /// Heartbeat message
    Heartbeat {
        sender: NodeId,
        timestamp: Timestamp,
    },
    /// Ping message for failure detection
    Ping {
        sender: NodeId,
        timestamp: Timestamp,
    },
    /// Pong response
    Pong {
        sender: NodeId,
        timestamp: Timestamp,
    },
}

/// Sync event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    /// Node joined the cluster
    NodeJoined { node_id: NodeId, address: SocketAddr },
    /// Node left the cluster
    NodeLeft { node_id: NodeId },
    /// Node failed (suspected)
    NodeFailed { node_id: NodeId },
    /// Node recovered
    NodeRecovered { node_id: NodeId },
    /// Sync completed
    SyncCompleted { node_id: NodeId, entries_synced: usize },
    /// Network partition detected
    NetworkPartition { affected_nodes: Vec<NodeId> },
    /// Network partition healed
    NetworkPartitionHealed { recovered_nodes: Vec<NodeId> },
}

/// Sync statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStats {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub failed_nodes: usize,
    pub gossip_rounds: u64,
    pub delta_syncs: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub average_latency: f64,
}

/// Synchronization trait
#[async_trait::async_trait]
pub trait Synchronizer: Send + Sync {
    /// Start synchronization
    async fn start(&mut self) -> Result<()>;
    
    /// Stop synchronization
    async fn stop(&mut self) -> Result<()>;
    
    /// Add a node to sync with
    async fn add_node(&mut self, node_id: NodeId, address: SocketAddr) -> Result<()>;
    
    /// Remove a node from sync
    async fn remove_node(&mut self, node_id: NodeId) -> Result<()>;
    
    /// Get list of active nodes
    async fn get_active_nodes(&self) -> Result<Vec<NodeId>>;
    
    /// Send sync message
    async fn send_message(&self, target: NodeId, message: SyncMessage) -> Result<()>;
    
    /// Handle incoming sync message
    async fn handle_message(&mut self, message: SyncMessage) -> Result<()>;
    
    /// Trigger delta sync with a node
    async fn trigger_delta_sync(&mut self, node_id: NodeId) -> Result<()>;
    
    /// Get sync statistics
    async fn get_stats(&self) -> Result<SyncStats>;
    
    /// Subscribe to sync events
    async fn subscribe_events(&self, callback: Box<dyn Fn(SyncEvent) + Send + Sync>) -> Result<()>;
}

/// Network transport trait
#[async_trait::async_trait]
pub trait NetworkTransport: Send + Sync {
    /// Send message to a node
    async fn send(&self, target: SocketAddr, message: &[u8]) -> Result<()>;
    
    /// Receive message from network
    async fn receive(&self) -> Result<(SocketAddr, Vec<u8>)>;
    
    /// Bind to an address
    async fn bind(&mut self, address: SocketAddr) -> Result<()>;
    
    /// Close the transport
    async fn close(&mut self) -> Result<()>;
}

/// Anti-entropy session for delta synchronization
#[derive(Debug, Clone)]
pub struct AntiEntropySession {
    pub session_id: String,
    pub node_id: NodeId,
    pub started_at: Timestamp,
    pub last_activity: Timestamp,
    pub state: SessionState,
}

/// Session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionState {
    Initiating,
    Syncing,
    Completed,
    Failed,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Last writer wins
    LastWriterWins,
    /// Use version vectors
    VectorClock,
    /// Custom resolution function
    Custom(String),
}

/// Sync protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncProtocolConfig {
    pub conflict_resolution: ConflictResolution,
    pub enable_compression: bool,
    pub max_message_size: usize,
    pub retry_attempts: usize,
    pub retry_backoff: u64,
}

impl Default for SyncProtocolConfig {
    fn default() -> Self {
        Self {
            conflict_resolution: ConflictResolution::VectorClock,
            enable_compression: true,
            max_message_size: 1024 * 1024, // 1MB
            retry_attempts: 3,
            retry_backoff: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_sync_config_default() {
        let config = SyncConfig::default();
        assert_eq!(config.gossip_interval, 100);
        assert_eq!(config.gossip_fanout, 3);
        assert_eq!(config.delta_sync_batch_size, 1000);
    }
    
    #[test]
    fn test_node_info_serialization() {
        let node_info = NodeInfo {
            id: uuid::Uuid::new_v4(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            last_seen: 123456789,
            version_vector: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        let serialized = serde_json::to_string(&node_info).unwrap();
        let deserialized: NodeInfo = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(node_info.id, deserialized.id);
        assert_eq!(node_info.address, deserialized.address);
        assert_eq!(node_info.last_seen, deserialized.last_seen);
    }
    
    #[test]
    fn test_sync_message_serialization() {
        let message = SyncMessage::Heartbeat {
            sender: uuid::Uuid::new_v4(),
            timestamp: 123456789,
        };
        
        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: SyncMessage = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            SyncMessage::Heartbeat { sender, timestamp } => {
                assert_eq!(timestamp, 123456789);
            }
            _ => panic!("Expected Heartbeat message"),
        }
    }
    
    #[test]
    fn test_session_state_transitions() {
        let state = SessionState::Initiating;
        assert!(matches!(state, SessionState::Initiating));
        
        let state = SessionState::Syncing;
        assert!(matches!(state, SessionState::Syncing));
        
        let state = SessionState::Completed;
        assert!(matches!(state, SessionState::Completed));
        
        let state = SessionState::Failed;
        assert!(matches!(state, SessionState::Failed));
    }
}