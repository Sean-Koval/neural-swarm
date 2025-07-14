//! Neuroplex - Distributed Memory System
//!
//! A high-performance distributed memory system with CRDT support,
//! consensus protocols, and async multi-node synchronization.
//!
//! ## Features
//!
//! - **Distributed Memory**: Async HashMap with distributed consistency
//! - **CRDT Support**: G-Counter, PN-Counter, OR-Set, LWW-Register
//! - **Consensus**: Raft implementation with leader election
//! - **Multi-Node Sync**: Gossip protocol and delta synchronization
//! - **Python FFI**: PyO3-based async Python integration
//! - **Performance**: Lock-free concurrent data structures
//!
//! ## Quick Start
//!
//! ```rust
//! use neuroplex::{NeuroCluster, NeuroNode};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut cluster = NeuroCluster::new("node1", "127.0.0.1:8080").await?;
//!     
//!     // Store distributed data
//!     cluster.set("key1", "value1").await?;
//!     
//!     // Retrieve from any node
//!     let value = cluster.get("key1").await?;
//!     println!("Retrieved: {:?}", value);
//!     
//!     Ok(())
//! }
//! ```

pub mod memory;
pub mod consensus;
pub mod crdt;
pub mod sync;
pub mod api;
pub mod error;
pub mod common_error;
pub mod integration;

#[cfg(feature = "python-ffi")]
pub mod ffi;

// Re-export main types
pub use api::{NeuroCluster, NeuroNode};
pub use error::{NeuroError, Result};
pub use common_error::{NeuralSwarmError, NeuralResult, ErrorCategory};
pub use memory::DistributedMemory;
pub use consensus::RaftConsensus;
pub use crdt::{GCounter, PNCounter, ORSet, LWWRegister};
pub use sync::GossipSync;

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Node identifier
pub type NodeId = Uuid;

/// Logical timestamp for ordering events
pub type Timestamp = u64;

/// Version vector for distributed versioning
pub type VersionVector = HashMap<NodeId, Timestamp>;

/// Common configuration for neuroplex nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroConfig {
    /// Node identifier
    pub node_id: NodeId,
    /// Node address
    pub address: String,
    /// Cluster peers
    pub peers: Vec<String>,
    /// Memory configuration
    pub memory: MemoryConfig,
    /// Consensus configuration
    pub consensus: ConsensusConfig,
    /// Synchronization configuration
    pub sync: SyncConfig,
}

/// Memory subsystem configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory size in bytes
    pub max_size: usize,
    /// Compression algorithm
    pub compression: CompressionAlgorithm,
    /// Replication factor
    pub replication_factor: usize,
}

/// Consensus subsystem configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Election timeout in milliseconds
    pub election_timeout: u64,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval: u64,
    /// Log compaction threshold
    pub log_compaction_threshold: usize,
}

/// Synchronization subsystem configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Gossip interval in milliseconds
    pub gossip_interval: u64,
    /// Maximum gossip fanout
    pub gossip_fanout: usize,
    /// Delta sync batch size
    pub delta_sync_batch_size: usize,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
}

impl Default for NeuroConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            address: "127.0.0.1:8080".to_string(),
            peers: Vec::new(),
            memory: MemoryConfig {
                max_size: 1024 * 1024 * 1024, // 1GB
                compression: CompressionAlgorithm::Lz4,
                replication_factor: 3,
            },
            consensus: ConsensusConfig {
                election_timeout: 5000,
                heartbeat_interval: 1000,
                log_compaction_threshold: 10000,
            },
            sync: SyncConfig {
                gossip_interval: 100,
                gossip_fanout: 3,
                delta_sync_batch_size: 1000,
            },
        }
    }
}

/// Initialize tracing for the neuroplex system
pub fn init_tracing() {
    use tracing_subscriber::prelude::*;
    
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuro_config_default() {
        let config = NeuroConfig::default();
        assert_eq!(config.address, "127.0.0.1:8080");
        assert_eq!(config.peers.len(), 0);
        assert_eq!(config.memory.replication_factor, 3);
    }

    #[test]
    fn test_node_id_generation() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        assert_ne!(id1, id2);
    }
}