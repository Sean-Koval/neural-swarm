//! Consensus Protocols
//!
//! Implementation of Raft consensus algorithm for distributed coordination

pub mod raft;
pub mod leader_election;
pub mod log_replication;
pub mod state_machine;

pub use raft::RaftConsensus;
pub use leader_election::{LeaderElection, ElectionState};
pub use log_replication::{LogReplication, LogEntry};
pub use state_machine::{StateMachine, StateMachineCommand};

use crate::{NodeId, Timestamp, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Raft term number
pub type Term = u64;

/// Log index
pub type LogIndex = u64;

/// Node state in Raft
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
}

/// Raft configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Election timeout in milliseconds
    pub election_timeout: u64,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval: u64,
    /// Log compaction threshold
    pub log_compaction_threshold: usize,
    /// Minimum cluster size for quorum
    pub min_cluster_size: usize,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            election_timeout: 150,
            heartbeat_interval: 50,
            log_compaction_threshold: 1000,
            min_cluster_size: 3,
        }
    }
}

/// Vote request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRequest {
    pub term: Term,
    pub candidate_id: NodeId,
    pub last_log_index: LogIndex,
    pub last_log_term: Term,
}

/// Vote response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResponse {
    pub term: Term,
    pub vote_granted: bool,
    pub voter_id: NodeId,
}

/// Append entries request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    pub term: Term,
    pub leader_id: NodeId,
    pub prev_log_index: LogIndex,
    pub prev_log_term: Term,
    pub entries: Vec<LogEntry>,
    pub leader_commit: LogIndex,
}

/// Append entries response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    pub term: Term,
    pub success: bool,
    pub follower_id: NodeId,
    pub match_index: LogIndex,
}

/// Consensus message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    VoteRequest(VoteRequest),
    VoteResponse(VoteResponse),
    AppendEntries(AppendEntriesRequest),
    AppendEntriesResponse(AppendEntriesResponse),
}

/// Consensus event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusEvent {
    ElectionStarted { term: Term, candidate_id: NodeId },
    VoteCast { term: Term, candidate_id: NodeId, granted: bool },
    LeaderElected { term: Term, leader_id: NodeId },
    LogEntryAppended { index: LogIndex, term: Term },
    LogEntryCommitted { index: LogIndex, term: Term },
    TermChanged { old_term: Term, new_term: Term },
}

/// Consensus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusStats {
    pub current_term: Term,
    pub current_leader: Option<NodeId>,
    pub node_state: NodeState,
    pub log_length: usize,
    pub commit_index: LogIndex,
    pub last_applied: LogIndex,
    pub election_count: u64,
    pub heartbeat_count: u64,
}

/// Consensus trait
#[async_trait::async_trait]
pub trait Consensus: Send + Sync {
    /// Start the consensus algorithm
    async fn start(&mut self) -> Result<()>;
    
    /// Stop the consensus algorithm
    async fn stop(&mut self) -> Result<()>;
    
    /// Submit a command to be replicated
    async fn submit_command(&mut self, command: Vec<u8>) -> Result<LogIndex>;
    
    /// Handle incoming consensus message
    async fn handle_message(&mut self, message: ConsensusMessage) -> Result<()>;
    
    /// Get current consensus state
    async fn get_state(&self) -> Result<NodeState>;
    
    /// Get current term
    async fn get_term(&self) -> Result<Term>;
    
    /// Get current leader
    async fn get_leader(&self) -> Result<Option<NodeId>>;
    
    /// Get consensus statistics
    async fn get_stats(&self) -> Result<ConsensusStats>;
    
    /// Check if this node is the leader
    async fn is_leader(&self) -> Result<bool>;
    
    /// Add a new node to the cluster
    async fn add_node(&mut self, node_id: NodeId) -> Result<()>;
    
    /// Remove a node from the cluster
    async fn remove_node(&mut self, node_id: NodeId) -> Result<()>;
    
    /// Get cluster membership
    async fn get_cluster_members(&self) -> Result<Vec<NodeId>>;
    
    /// Trigger log compaction
    async fn compact_log(&mut self) -> Result<()>;
}

/// Quorum calculation
pub fn calculate_quorum(cluster_size: usize) -> usize {
    cluster_size / 2 + 1
}

/// Generate random election timeout
pub fn random_election_timeout(base_timeout: u64) -> u64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    base_timeout + rng.gen_range(0..base_timeout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quorum_calculation() {
        assert_eq!(calculate_quorum(1), 1);
        assert_eq!(calculate_quorum(3), 2);
        assert_eq!(calculate_quorum(5), 3);
        assert_eq!(calculate_quorum(7), 4);
    }
    
    #[test]
    fn test_random_election_timeout() {
        let base_timeout = 150;
        let timeout = random_election_timeout(base_timeout);
        assert!(timeout >= base_timeout);
        assert!(timeout < base_timeout * 2);
    }
    
    #[test]
    fn test_raft_config_default() {
        let config = RaftConfig::default();
        assert_eq!(config.election_timeout, 150);
        assert_eq!(config.heartbeat_interval, 50);
        assert_eq!(config.log_compaction_threshold, 1000);
        assert_eq!(config.min_cluster_size, 3);
    }
    
    #[test]
    fn test_node_state_equality() {
        assert_eq!(NodeState::Follower, NodeState::Follower);
        assert_eq!(NodeState::Candidate, NodeState::Candidate);
        assert_eq!(NodeState::Leader, NodeState::Leader);
        assert_ne!(NodeState::Follower, NodeState::Leader);
    }
    
    #[test]
    fn test_consensus_message_serialization() {
        let vote_request = VoteRequest {
            term: 1,
            candidate_id: uuid::Uuid::new_v4(),
            last_log_index: 0,
            last_log_term: 0,
        };
        
        let message = ConsensusMessage::VoteRequest(vote_request);
        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: ConsensusMessage = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            ConsensusMessage::VoteRequest(req) => {
                assert_eq!(req.term, 1);
                assert_eq!(req.last_log_index, 0);
            }
            _ => panic!("Expected VoteRequest"),
        }
    }
}