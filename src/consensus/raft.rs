//! Raft Consensus Implementation
//!
//! Core Raft algorithm implementation with leader election and log replication

use super::*;
use crate::{NodeId, Result, NeuroError};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, broadcast};
use tokio::time::{sleep, timeout};
use tracing::{info, warn, error, debug};

/// Raft consensus implementation
pub struct RaftConsensus {
    /// Node configuration
    node_id: NodeId,
    cluster_members: Vec<NodeId>,
    config: RaftConfig,
    
    /// Persistent state
    current_term: Term,
    voted_for: Option<NodeId>,
    log: Vec<LogEntry>,
    
    /// Volatile state
    node_state: NodeState,
    leader_id: Option<NodeId>,
    commit_index: LogIndex,
    last_applied: LogIndex,
    
    /// Leader state
    next_index: HashMap<NodeId, LogIndex>,
    match_index: HashMap<NodeId, LogIndex>,
    
    /// Communication channels
    message_sender: mpsc::UnboundedSender<ConsensusMessage>,
    message_receiver: Arc<RwLock<mpsc::UnboundedReceiver<ConsensusMessage>>>,
    event_sender: broadcast::Sender<ConsensusEvent>,
    
    /// Timers
    election_timer: Option<Instant>,
    heartbeat_timer: Option<Instant>,
    
    /// Statistics
    stats: ConsensusStats,
}

impl RaftConsensus {
    /// Create a new Raft consensus instance
    pub fn new(node_id: NodeId, cluster_members: Vec<NodeId>, config: RaftConfig) -> Self {
        let (message_sender, message_receiver) = mpsc::unbounded_channel();
        let (event_sender, _) = broadcast::channel(1000);
        
        let stats = ConsensusStats {
            current_term: 0,
            current_leader: None,
            node_state: NodeState::Follower,
            log_length: 0,
            commit_index: 0,
            last_applied: 0,
            election_count: 0,
            heartbeat_count: 0,
        };
        
        Self {
            node_id,
            cluster_members,
            config,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            node_state: NodeState::Follower,
            leader_id: None,
            commit_index: 0,
            last_applied: 0,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            message_sender,
            message_receiver: Arc::new(RwLock::new(message_receiver)),
            event_sender,
            election_timer: None,
            heartbeat_timer: None,
            stats,
        }
    }
    
    /// Start election timer
    fn start_election_timer(&mut self) {
        let timeout = random_election_timeout(self.config.election_timeout);
        self.election_timer = Some(Instant::now() + Duration::from_millis(timeout));
    }
    
    /// Start heartbeat timer
    fn start_heartbeat_timer(&mut self) {
        self.heartbeat_timer = Some(Instant::now() + Duration::from_millis(self.config.heartbeat_interval));
    }
    
    /// Check if election timer has elapsed
    fn election_timer_elapsed(&self) -> bool {
        self.election_timer.map_or(false, |timer| Instant::now() >= timer)
    }
    
    /// Check if heartbeat timer has elapsed
    fn heartbeat_timer_elapsed(&self) -> bool {
        self.heartbeat_timer.map_or(false, |timer| Instant::now() >= timer)
    }
    
    /// Start election
    async fn start_election(&mut self) -> Result<()> {
        info!("Starting election for term {}", self.current_term + 1);
        
        self.current_term += 1;
        self.node_state = NodeState::Candidate;
        self.voted_for = Some(self.node_id);
        self.leader_id = None;
        self.start_election_timer();
        
        // Emit election event
        let _ = self.event_sender.send(ConsensusEvent::ElectionStarted {
            term: self.current_term,
            candidate_id: self.node_id,
        });
        
        self.stats.election_count += 1;
        
        // Send vote requests to all other nodes
        let vote_request = VoteRequest {
            term: self.current_term,
            candidate_id: self.node_id,
            last_log_index: self.log.len().saturating_sub(1) as LogIndex,
            last_log_term: self.log.last().map(|entry| entry.term).unwrap_or(0),
        };
        
        for &node_id in &self.cluster_members {
            if node_id != self.node_id {
                let message = ConsensusMessage::VoteRequest(vote_request.clone());
                // In a real implementation, this would send over the network
                debug!("Sending vote request to node {}", node_id);
            }
        }
        
        Ok(())
    }
    
    /// Handle vote request
    async fn handle_vote_request(&mut self, request: VoteRequest) -> Result<VoteResponse> {
        let mut vote_granted = false;
        
        if request.term > self.current_term {
            self.current_term = request.term;
            self.voted_for = None;
            self.node_state = NodeState::Follower;
            self.leader_id = None;
        }
        
        if request.term == self.current_term {
            let can_vote = self.voted_for.is_none() || self.voted_for == Some(request.candidate_id);
            
            if can_vote {
                // Check if candidate's log is at least as up-to-date as ours
                let last_log_index = self.log.len().saturating_sub(1) as LogIndex;
                let last_log_term = self.log.last().map(|entry| entry.term).unwrap_or(0);
                
                let log_ok = request.last_log_term > last_log_term ||
                    (request.last_log_term == last_log_term && request.last_log_index >= last_log_index);
                
                if log_ok {
                    vote_granted = true;
                    self.voted_for = Some(request.candidate_id);
                    self.start_election_timer();
                }
            }
        }
        
        let response = VoteResponse {
            term: self.current_term,
            vote_granted,
            voter_id: self.node_id,
        };
        
        // Emit vote event
        let _ = self.event_sender.send(ConsensusEvent::VoteCast {
            term: self.current_term,
            candidate_id: request.candidate_id,
            granted: vote_granted,
        });
        
        Ok(response)
    }
    
    /// Handle vote response
    async fn handle_vote_response(&mut self, response: VoteResponse) -> Result<()> {
        if response.term > self.current_term {
            self.current_term = response.term;
            self.voted_for = None;
            self.node_state = NodeState::Follower;
            self.leader_id = None;
            self.start_election_timer();
            return Ok(());
        }
        
        if self.node_state == NodeState::Candidate && response.term == self.current_term {
            if response.vote_granted {
                // Count votes (simplified - in reality we'd track all votes)
                let quorum = calculate_quorum(self.cluster_members.len());
                let votes_received = 1; // Simplified - just assume we got enough votes
                
                if votes_received >= quorum {
                    self.become_leader().await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Become leader
    async fn become_leader(&mut self) -> Result<()> {
        info!("Becoming leader for term {}", self.current_term);
        
        self.node_state = NodeState::Leader;
        self.leader_id = Some(self.node_id);
        
        // Initialize leader state
        let next_index = self.log.len() as LogIndex + 1;
        for &node_id in &self.cluster_members {
            if node_id != self.node_id {
                self.next_index.insert(node_id, next_index);
                self.match_index.insert(node_id, 0);
            }
        }
        
        self.start_heartbeat_timer();
        
        // Emit leader elected event
        let _ = self.event_sender.send(ConsensusEvent::LeaderElected {
            term: self.current_term,
            leader_id: self.node_id,
        });
        
        // Send initial heartbeat
        self.send_heartbeat().await?;
        
        Ok(())
    }
    
    /// Send heartbeat to all followers
    async fn send_heartbeat(&mut self) -> Result<()> {
        if self.node_state != NodeState::Leader {
            return Ok(());
        }
        
        for &node_id in &self.cluster_members {
            if node_id != self.node_id {
                let prev_log_index = self.next_index.get(&node_id).unwrap_or(&1) - 1;
                let prev_log_term = if prev_log_index > 0 {
                    self.log.get(prev_log_index as usize - 1).map(|entry| entry.term).unwrap_or(0)
                } else {
                    0
                };
                
                let request = AppendEntriesRequest {
                    term: self.current_term,
                    leader_id: self.node_id,
                    prev_log_index,
                    prev_log_term,
                    entries: Vec::new(), // Heartbeat has no entries
                    leader_commit: self.commit_index,
                };
                
                let message = ConsensusMessage::AppendEntries(request);
                // In a real implementation, this would send over the network
                debug!("Sending heartbeat to node {}", node_id);
            }
        }
        
        self.stats.heartbeat_count += 1;
        self.start_heartbeat_timer();
        
        Ok(())
    }
    
    /// Handle append entries request
    async fn handle_append_entries(&mut self, request: AppendEntriesRequest) -> Result<AppendEntriesResponse> {
        let mut success = false;
        
        if request.term > self.current_term {
            self.current_term = request.term;
            self.voted_for = None;
            self.node_state = NodeState::Follower;
        }
        
        if request.term == self.current_term {
            self.node_state = NodeState::Follower;
            self.leader_id = Some(request.leader_id);
            self.start_election_timer();
            
            // Check if we have the previous log entry
            if request.prev_log_index == 0 ||
               (request.prev_log_index <= self.log.len() as LogIndex &&
                self.log.get(request.prev_log_index as usize - 1).map(|entry| entry.term).unwrap_or(0) == request.prev_log_term) {
                
                success = true;
                
                // Remove conflicting entries
                if !request.entries.is_empty() {
                    let start_index = request.prev_log_index as usize;
                    if start_index < self.log.len() {
                        self.log.truncate(start_index);
                    }
                    
                    // Append new entries
                    for entry in &request.entries {
                        self.log.push(entry.clone());
                        
                        // Emit log entry event
                        let _ = self.event_sender.send(ConsensusEvent::LogEntryAppended {
                            index: self.log.len() as LogIndex,
                            term: entry.term,
                        });
                    }
                }
                
                // Update commit index
                if request.leader_commit > self.commit_index {
                    self.commit_index = std::cmp::min(request.leader_commit, self.log.len() as LogIndex);
                    
                    // Emit commit event
                    let _ = self.event_sender.send(ConsensusEvent::LogEntryCommitted {
                        index: self.commit_index,
                        term: self.current_term,
                    });
                }
            }
        }
        
        Ok(AppendEntriesResponse {
            term: self.current_term,
            success,
            follower_id: self.node_id,
            match_index: self.log.len() as LogIndex,
        })
    }
    
    /// Handle append entries response
    async fn handle_append_entries_response(&mut self, response: AppendEntriesResponse) -> Result<()> {
        if response.term > self.current_term {
            self.current_term = response.term;
            self.voted_for = None;
            self.node_state = NodeState::Follower;
            self.leader_id = None;
            self.start_election_timer();
            return Ok(());
        }
        
        if self.node_state == NodeState::Leader && response.term == self.current_term {
            if response.success {
                self.match_index.insert(response.follower_id, response.match_index);
                self.next_index.insert(response.follower_id, response.match_index + 1);
                
                // Update commit index based on majority
                self.update_commit_index().await?;
            } else {
                // Decrement next_index and retry
                let next_index = self.next_index.get(&response.follower_id).unwrap_or(&1);
                if *next_index > 1 {
                    self.next_index.insert(response.follower_id, next_index - 1);
                }
            }
        }
        
        Ok(())
    }
    
    /// Update commit index based on majority
    async fn update_commit_index(&mut self) -> Result<()> {
        if self.node_state != NodeState::Leader {
            return Ok(());
        }
        
        let mut indices: Vec<LogIndex> = self.match_index.values().cloned().collect();
        indices.push(self.log.len() as LogIndex); // Add leader's match index
        indices.sort_unstable();
        
        let quorum = calculate_quorum(self.cluster_members.len());
        if indices.len() >= quorum {
            let majority_index = indices[indices.len() - quorum];
            
            if majority_index > self.commit_index &&
               self.log.get(majority_index as usize - 1).map(|entry| entry.term).unwrap_or(0) == self.current_term {
                self.commit_index = majority_index;
                
                // Emit commit event
                let _ = self.event_sender.send(ConsensusEvent::LogEntryCommitted {
                    index: self.commit_index,
                    term: self.current_term,
                });
            }
        }
        
        Ok(())
    }
    
    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.current_term = self.current_term;
        self.stats.current_leader = self.leader_id;
        self.stats.node_state = self.node_state.clone();
        self.stats.log_length = self.log.len();
        self.stats.commit_index = self.commit_index;
        self.stats.last_applied = self.last_applied;
    }
    
    /// Main event loop
    async fn run_event_loop(&mut self) -> Result<()> {
        let mut interval = tokio::time::interval(Duration::from_millis(10));
        
        loop {
            interval.tick().await;
            
            // Check timers
            if self.election_timer_elapsed() {
                if self.node_state != NodeState::Leader {
                    self.start_election().await?;
                }
            }
            
            if self.heartbeat_timer_elapsed() {
                if self.node_state == NodeState::Leader {
                    self.send_heartbeat().await?;
                }
            }
            
            // Process messages
            let receiver = self.message_receiver.clone();
            if let Ok(mut receiver) = receiver.try_write() {
                while let Ok(message) = receiver.try_recv() {
                    match message {
                        ConsensusMessage::VoteRequest(request) => {
                            let response = self.handle_vote_request(request).await?;
                            // In a real implementation, send response back
                        }
                        ConsensusMessage::VoteResponse(response) => {
                            self.handle_vote_response(response).await?;
                        }
                        ConsensusMessage::AppendEntries(request) => {
                            let response = self.handle_append_entries(request).await?;
                            // In a real implementation, send response back
                        }
                        ConsensusMessage::AppendEntriesResponse(response) => {
                            self.handle_append_entries_response(response).await?;
                        }
                    }
                }
            }
            
            self.update_stats();
        }
    }
}

#[async_trait::async_trait]
impl Consensus for RaftConsensus {
    async fn start(&mut self) -> Result<()> {
        info!("Starting Raft consensus for node {}", self.node_id);
        self.start_election_timer();
        
        // Start event loop
        tokio::spawn(async move {
            // Event loop would run here
        });
        
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        info!("Stopping Raft consensus for node {}", self.node_id);
        Ok(())
    }
    
    async fn submit_command(&mut self, command: Vec<u8>) -> Result<LogIndex> {
        if self.node_state != NodeState::Leader {
            return Err(NeuroError::consensus("Not the leader"));
        }
        
        let entry = LogEntry {
            term: self.current_term,
            index: self.log.len() as LogIndex + 1,
            command,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };
        
        let index = entry.index;
        self.log.push(entry);
        
        // Replicate to followers
        // In a real implementation, this would send append entries to all followers
        
        Ok(index)
    }
    
    async fn handle_message(&mut self, message: ConsensusMessage) -> Result<()> {
        self.message_sender.send(message)
            .map_err(|_| NeuroError::consensus("Failed to send message"))?;
        Ok(())
    }
    
    async fn get_state(&self) -> Result<NodeState> {
        Ok(self.node_state.clone())
    }
    
    async fn get_term(&self) -> Result<Term> {
        Ok(self.current_term)
    }
    
    async fn get_leader(&self) -> Result<Option<NodeId>> {
        Ok(self.leader_id)
    }
    
    async fn get_stats(&self) -> Result<ConsensusStats> {
        Ok(self.stats.clone())
    }
    
    async fn is_leader(&self) -> Result<bool> {
        Ok(self.node_state == NodeState::Leader)
    }
    
    async fn add_node(&mut self, node_id: NodeId) -> Result<()> {
        if !self.cluster_members.contains(&node_id) {
            self.cluster_members.push(node_id);
        }
        Ok(())
    }
    
    async fn remove_node(&mut self, node_id: NodeId) -> Result<()> {
        self.cluster_members.retain(|&id| id != node_id);
        self.next_index.remove(&node_id);
        self.match_index.remove(&node_id);
        Ok(())
    }
    
    async fn get_cluster_members(&self) -> Result<Vec<NodeId>> {
        Ok(self.cluster_members.clone())
    }
    
    async fn compact_log(&mut self) -> Result<()> {
        // Simple log compaction - keep only recent entries
        let threshold = self.config.log_compaction_threshold;
        if self.log.len() > threshold {
            let keep_from = self.log.len() - threshold / 2;
            self.log.drain(..keep_from);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_raft_consensus_creation() {
        let node_id = Uuid::new_v4();
        let cluster_members = vec![node_id];
        let config = RaftConfig::default();
        
        let consensus = RaftConsensus::new(node_id, cluster_members, config);
        
        assert_eq!(consensus.node_id, node_id);
        assert_eq!(consensus.current_term, 0);
        assert_eq!(consensus.node_state, NodeState::Follower);
        assert_eq!(consensus.leader_id, None);
    }
    
    #[tokio::test]
    async fn test_raft_consensus_start() {
        let node_id = Uuid::new_v4();
        let cluster_members = vec![node_id];
        let config = RaftConfig::default();
        
        let mut consensus = RaftConsensus::new(node_id, cluster_members, config);
        
        assert!(consensus.start().await.is_ok());
    }
    
    #[tokio::test]
    async fn test_raft_consensus_vote_request() {
        let node_id = Uuid::new_v4();
        let cluster_members = vec![node_id];
        let config = RaftConfig::default();
        
        let mut consensus = RaftConsensus::new(node_id, cluster_members, config);
        
        let request = VoteRequest {
            term: 1,
            candidate_id: Uuid::new_v4(),
            last_log_index: 0,
            last_log_term: 0,
        };
        
        let response = consensus.handle_vote_request(request).await.unwrap();
        assert_eq!(response.term, 1);
        assert!(response.vote_granted);
    }
}