//! Raft consensus protocol implementation
//!
//! This module implements the Raft consensus algorithm for distributed coordination.
//! It provides leader election, log replication, and safety guarantees.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex, broadcast};
use tokio::time::{sleep, timeout};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

use crate::{Result, NeuroPlexError, NodeId};

pub mod log;
pub mod state_machine;
pub mod network;
pub mod election;
pub mod replication;

use log::{LogEntry, LogIndex, Term};
use state_machine::StateMachine;
use network::RaftNetwork;
use election::ElectionManager;
use replication::ReplicationManager;

/// Raft node states
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
}

impl std::fmt::Display for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeState::Follower => write!(f, "Follower"),
            NodeState::Candidate => write!(f, "Candidate"),
            NodeState::Leader => write!(f, "Leader"),
        }
    }
}

/// Raft configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Election timeout range in milliseconds
    pub election_timeout_min: u64,
    pub election_timeout_max: u64,
    
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval: u64,
    
    /// Maximum number of log entries per AppendEntries RPC
    pub max_entries_per_append: usize,
    
    /// Cluster configuration
    pub cluster_nodes: Vec<NodeId>,
    
    /// Snapshot configuration
    pub snapshot_interval: u64,
    pub snapshot_threshold: usize,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            election_timeout_min: 150,
            election_timeout_max: 300,
            heartbeat_interval: 50,
            max_entries_per_append: 100,
            cluster_nodes: Vec::new(),
            snapshot_interval: 10000,
            snapshot_threshold: 1000,
        }
    }
}

/// Raft consensus engine
pub struct RaftConsensus {
    /// Node configuration
    node_id: NodeId,
    config: RaftConfig,
    
    /// Persistent state
    current_term: Arc<RwLock<Term>>,
    voted_for: Arc<RwLock<Option<NodeId>>>,
    log: Arc<RwLock<Vec<LogEntry>>>,
    
    /// Volatile state
    commit_index: Arc<RwLock<LogIndex>>,
    last_applied: Arc<RwLock<LogIndex>>,
    
    /// Leader state
    next_index: Arc<RwLock<HashMap<NodeId, LogIndex>>>,
    match_index: Arc<RwLock<HashMap<NodeId, LogIndex>>>,
    
    /// Node state
    state: Arc<RwLock<NodeState>>,
    leader_id: Arc<RwLock<Option<NodeId>>>,
    
    /// Components
    state_machine: Arc<StateMachine>,
    network: Arc<RaftNetwork>,
    election_manager: Arc<ElectionManager>,
    replication_manager: Arc<ReplicationManager>,
    
    /// Events
    event_sender: broadcast::Sender<RaftEvent>,
    event_receiver: broadcast::Receiver<RaftEvent>,
    
    /// Control
    is_running: Arc<RwLock<bool>>,
    shutdown_sender: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

/// Raft events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftEvent {
    StateChanged { old_state: NodeState, new_state: NodeState },
    LeaderElected { leader_id: NodeId, term: Term },
    LogEntryCommitted { index: LogIndex, term: Term },
    SnapshotCreated { index: LogIndex },
    NetworkPartition { disconnected_nodes: Vec<NodeId> },
    ClusterConfigurationChanged { new_nodes: Vec<NodeId> },
}

impl RaftConsensus {
    /// Create a new Raft consensus engine
    pub async fn new(config: RaftConfig, node_id: NodeId) -> Result<Self> {
        let state_machine = Arc::new(StateMachine::new());
        let network = Arc::new(RaftNetwork::new(node_id, config.cluster_nodes.clone()).await?);
        
        let (event_sender, event_receiver) = broadcast::channel(1000);
        
        // Initialize state
        let current_term = Arc::new(RwLock::new(0));
        let voted_for = Arc::new(RwLock::new(None));
        let log = Arc::new(RwLock::new(vec![]));
        let commit_index = Arc::new(RwLock::new(0));
        let last_applied = Arc::new(RwLock::new(0));
        let next_index = Arc::new(RwLock::new(HashMap::new()));
        let match_index = Arc::new(RwLock::new(HashMap::new()));
        let state = Arc::new(RwLock::new(NodeState::Follower));
        let leader_id = Arc::new(RwLock::new(None));
        
        // Initialize leader state for all nodes
        let mut next_idx = HashMap::new();
        let mut match_idx = HashMap::new();
        for node in &config.cluster_nodes {
            next_idx.insert(*node, 1);
            match_idx.insert(*node, 0);
        }
        *next_index.write().await = next_idx;
        *match_index.write().await = match_idx;
        
        let election_manager = Arc::new(ElectionManager::new(
            node_id,
            config.clone(),
            Arc::clone(&current_term),
            Arc::clone(&voted_for),
            Arc::clone(&state),
            Arc::clone(&leader_id),
            Arc::clone(&network),
            event_sender.clone(),
        ));
        
        let replication_manager = Arc::new(ReplicationManager::new(
            node_id,
            config.clone(),
            Arc::clone(&current_term),
            Arc::clone(&log),
            Arc::clone(&commit_index),
            Arc::clone(&next_index),
            Arc::clone(&match_index),
            Arc::clone(&state),
            Arc::clone(&network),
            event_sender.clone(),
        ));
        
        Ok(Self {
            node_id,
            config,
            current_term,
            voted_for,
            log,
            commit_index,
            last_applied,
            next_index,
            match_index,
            state,
            leader_id,
            state_machine,
            network,
            election_manager,
            replication_manager,
            event_sender,
            event_receiver,
            is_running: Arc::new(RwLock::new(false)),
            shutdown_sender: Arc::new(Mutex::new(None)),
        })
    }
    
    /// Start the Raft consensus engine
    pub async fn start(&self) -> Result<()> {
        info!("Starting Raft consensus for node {}", self.node_id);
        
        *self.is_running.write().await = true;
        
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        *self.shutdown_sender.lock().await = Some(shutdown_tx);
        
        // Start network layer
        self.network.start().await?;
        
        // Start election manager
        self.election_manager.start().await?;
        
        // Start replication manager
        self.replication_manager.start().await?;
        
        // Start main consensus loop
        self.start_consensus_loop(shutdown_rx).await;
        
        info!("Raft consensus started successfully");
        Ok(())
    }
    
    /// Stop the Raft consensus engine
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Raft consensus for node {}", self.node_id);
        
        *self.is_running.write().await = false;
        
        // Send shutdown signal
        if let Some(shutdown_tx) = self.shutdown_sender.lock().await.take() {
            let _ = shutdown_tx.send(());
        }
        
        // Stop managers
        self.replication_manager.stop().await?;
        self.election_manager.stop().await?;
        self.network.stop().await?;
        
        info!("Raft consensus stopped successfully");
        Ok(())
    }
    
    /// Get the current state of the node
    pub async fn state(&self) -> NodeState {
        *self.state.read().await
    }
    
    /// Get the current term
    pub async fn term(&self) -> Term {
        *self.current_term.read().await
    }
    
    /// Get the current leader ID
    pub async fn leader(&self) -> Option<NodeId> {
        *self.leader_id.read().await
    }
    
    /// Get the commit index
    pub async fn commit_index(&self) -> LogIndex {
        *self.commit_index.read().await
    }
    
    /// Get the last applied index
    pub async fn last_applied(&self) -> LogIndex {
        *self.last_applied.read().await
    }
    
    /// Propose a new log entry (only works if this node is the leader)
    pub async fn propose(&self, data: Vec<u8>) -> Result<LogIndex> {
        if *self.state.read().await != NodeState::Leader {
            return Err(NeuroPlexError::Consensus("Not the leader".to_string()));
        }
        
        let term = *self.current_term.read().await;
        let mut log = self.log.write().await;
        
        let index = log.len() as LogIndex + 1;
        let entry = LogEntry {
            index,
            term,
            data,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };
        
        log.push(entry);
        
        // Trigger replication
        self.replication_manager.trigger_replication().await?;
        
        Ok(index)
    }
    
    /// Subscribe to Raft events
    pub fn subscribe(&self) -> broadcast::Receiver<RaftEvent> {
        self.event_sender.subscribe()
    }
    
    /// Get cluster status
    pub async fn cluster_status(&self) -> HashMap<String, serde_json::Value> {
        let mut status = HashMap::new();
        
        status.insert("node_id".to_string(), serde_json::Value::String(self.node_id.to_string()));
        status.insert("state".to_string(), serde_json::Value::String(self.state().await.to_string()));
        status.insert("term".to_string(), serde_json::Value::Number(serde_json::Number::from(self.term().await)));
        status.insert("leader".to_string(), 
            self.leader().await.map(|id| serde_json::Value::String(id.to_string()))
                .unwrap_or(serde_json::Value::Null));
        status.insert("commit_index".to_string(), serde_json::Value::Number(serde_json::Number::from(self.commit_index().await)));
        status.insert("last_applied".to_string(), serde_json::Value::Number(serde_json::Number::from(self.last_applied().await)));
        status.insert("log_length".to_string(), serde_json::Value::Number(serde_json::Number::from(self.log.read().await.len())));
        
        status
    }
    
    /// Main consensus loop
    async fn start_consensus_loop(&self, mut shutdown_rx: tokio::sync::oneshot::Receiver<()>) {
        let is_running = Arc::clone(&self.is_running);
        let last_applied = Arc::clone(&self.last_applied);
        let commit_index = Arc::clone(&self.commit_index);
        let state_machine = Arc::clone(&self.state_machine);
        let log = Arc::clone(&self.log);
        let event_sender = self.event_sender.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(10));
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if !*is_running.read().await {
                            break;
                        }
                        
                        // Apply committed entries to state machine
                        let last_applied_idx = *last_applied.read().await;
                        let commit_idx = *commit_index.read().await;
                        
                        if commit_idx > last_applied_idx {
                            let log_entries = log.read().await;
                            
                            for i in (last_applied_idx + 1)..=commit_idx {
                                if let Some(entry) = log_entries.get((i - 1) as usize) {
                                    if let Err(e) = state_machine.apply(&entry.data).await {
                                        error!("Failed to apply entry {} to state machine: {}", i, e);
                                    } else {
                                        debug!("Applied entry {} to state machine", i);
                                        let _ = event_sender.send(RaftEvent::LogEntryCommitted {
                                            index: i,
                                            term: entry.term,
                                        });
                                    }
                                }
                            }
                            
                            *last_applied.write().await = commit_idx;
                        }
                    }
                    _ = &mut shutdown_rx => {
                        info!("Consensus loop shutting down");
                        break;
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    
    #[tokio::test]
    async fn test_raft_consensus_creation() {
        let config = RaftConfig::default();
        let node_id = Uuid::new_v4();
        
        let raft = RaftConsensus::new(config, node_id).await.unwrap();
        
        assert_eq!(raft.node_id, node_id);
        assert_eq!(raft.state().await, NodeState::Follower);
        assert_eq!(raft.term().await, 0);
        assert_eq!(raft.leader().await, None);
    }
    
    #[tokio::test]
    async fn test_raft_consensus_start_stop() {
        let config = RaftConfig::default();
        let node_id = Uuid::new_v4();
        
        let raft = RaftConsensus::new(config, node_id).await.unwrap();
        
        // Start and immediately stop
        raft.start().await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        raft.stop().await.unwrap();
        
        assert!(!*raft.is_running.read().await);
    }
    
    #[tokio::test]
    async fn test_raft_consensus_cluster_status() {
        let config = RaftConfig::default();
        let node_id = Uuid::new_v4();
        
        let raft = RaftConsensus::new(config, node_id).await.unwrap();
        
        let status = raft.cluster_status().await;
        
        assert!(status.contains_key("node_id"));
        assert!(status.contains_key("state"));
        assert!(status.contains_key("term"));
        assert!(status.contains_key("leader"));
    }
}