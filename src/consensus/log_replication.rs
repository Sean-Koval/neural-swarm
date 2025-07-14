//! Log Replication
//!
//! Implementation of Raft log replication mechanism

use super::*;
use crate::{NodeId, Result, NeuroError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

/// Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: Term,
    pub index: LogIndex,
    pub command: Vec<u8>,
    pub timestamp: u64,
}

/// Log replication state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogReplicationState {
    pub next_index: HashMap<NodeId, LogIndex>,
    pub match_index: HashMap<NodeId, LogIndex>,
    pub commit_index: LogIndex,
    pub last_applied: LogIndex,
}

/// Log replication implementation
pub struct LogReplication {
    /// Log entries
    log: Arc<RwLock<Vec<LogEntry>>>,
    /// Replication state
    state: Arc<RwLock<LogReplicationState>>,
    /// Node ID
    node_id: NodeId,
    /// Cluster members
    cluster_members: Vec<NodeId>,
    /// Configuration
    config: RaftConfig,
}

impl LogReplication {
    /// Create a new log replication instance
    pub fn new(node_id: NodeId, cluster_members: Vec<NodeId>, config: RaftConfig) -> Self {
        let state = LogReplicationState {
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            commit_index: 0,
            last_applied: 0,
        };
        
        Self {
            log: Arc::new(RwLock::new(Vec::new())),
            state: Arc::new(RwLock::new(state)),
            node_id,
            cluster_members,
            config,
        }
    }
    
    /// Append a new log entry
    pub async fn append_entry(&self, term: Term, command: Vec<u8>) -> Result<LogIndex> {
        let mut log = self.log.write().await;
        let index = log.len() as LogIndex + 1;
        
        let entry = LogEntry {
            term,
            index,
            command,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };
        
        log.push(entry);
        
        info!("Appended log entry {} for term {}", index, term);
        
        Ok(index)
    }
    
    /// Get log entry at index
    pub async fn get_entry(&self, index: LogIndex) -> Result<Option<LogEntry>> {
        let log = self.log.read().await;
        
        if index == 0 || index > log.len() as LogIndex {
            return Ok(None);
        }
        
        Ok(Some(log[index as usize - 1].clone()))
    }
    
    /// Get log entries from start_index to end_index
    pub async fn get_entries(&self, start_index: LogIndex, end_index: LogIndex) -> Result<Vec<LogEntry>> {
        let log = self.log.read().await;
        
        if start_index == 0 || start_index > log.len() as LogIndex {
            return Ok(Vec::new());
        }
        
        let start = (start_index - 1) as usize;
        let end = std::cmp::min(end_index as usize, log.len());
        
        Ok(log[start..end].to_vec())
    }
    
    /// Get the last log index
    pub async fn last_log_index(&self) -> LogIndex {
        let log = self.log.read().await;
        log.len() as LogIndex
    }
    
    /// Get the last log term
    pub async fn last_log_term(&self) -> Term {
        let log = self.log.read().await;
        log.last().map(|entry| entry.term).unwrap_or(0)
    }
    
    /// Get the term of entry at index
    pub async fn get_term(&self, index: LogIndex) -> Result<Term> {
        if index == 0 {
            return Ok(0);
        }
        
        let log = self.log.read().await;
        if index > log.len() as LogIndex {
            return Err(NeuroError::consensus("Index out of bounds"));
        }
        
        Ok(log[index as usize - 1].term)
    }
    
    /// Truncate log from index
    pub async fn truncate_from(&self, index: LogIndex) -> Result<()> {
        let mut log = self.log.write().await;
        
        if index == 0 {
            log.clear();
        } else if index <= log.len() as LogIndex {
            log.truncate(index as usize - 1);
        }
        
        info!("Truncated log from index {}", index);
        
        Ok(())
    }
    
    /// Initialize replication state for a node
    pub async fn initialize_node(&self, node_id: NodeId) -> Result<()> {
        let mut state = self.state.write().await;
        let last_index = self.last_log_index().await;
        
        state.next_index.insert(node_id, last_index + 1);
        state.match_index.insert(node_id, 0);
        
        debug!("Initialized replication state for node {}", node_id);
        
        Ok(())
    }
    
    /// Update next index for a node
    pub async fn update_next_index(&self, node_id: NodeId, index: LogIndex) -> Result<()> {
        let mut state = self.state.write().await;
        state.next_index.insert(node_id, index);
        
        debug!("Updated next index for node {} to {}", node_id, index);
        
        Ok(())
    }
    
    /// Update match index for a node
    pub async fn update_match_index(&self, node_id: NodeId, index: LogIndex) -> Result<()> {
        let mut state = self.state.write().await;
        state.match_index.insert(node_id, index);
        
        // Update next index as well
        state.next_index.insert(node_id, index + 1);
        
        debug!("Updated match index for node {} to {}", node_id, index);
        
        Ok(())
    }
    
    /// Get next index for a node
    pub async fn get_next_index(&self, node_id: NodeId) -> LogIndex {
        let state = self.state.read().await;
        state.next_index.get(&node_id).copied().unwrap_or(1)
    }
    
    /// Get match index for a node
    pub async fn get_match_index(&self, node_id: NodeId) -> LogIndex {
        let state = self.state.read().await;
        state.match_index.get(&node_id).copied().unwrap_or(0)
    }
    
    /// Calculate commit index based on majority
    pub async fn calculate_commit_index(&self, current_term: Term) -> Result<LogIndex> {
        let state = self.state.read().await;
        let log = self.log.read().await;
        
        let mut match_indices: Vec<LogIndex> = state.match_index.values().copied().collect();
        match_indices.push(log.len() as LogIndex); // Add leader's match index
        match_indices.sort_unstable();
        
        let quorum = calculate_quorum(self.cluster_members.len());
        
        if match_indices.len() >= quorum {
            let majority_index = match_indices[match_indices.len() - quorum];
            
            // Only commit entries from current term
            if majority_index > 0 && majority_index <= log.len() as LogIndex {
                let entry_term = log[majority_index as usize - 1].term;
                if entry_term == current_term {
                    return Ok(majority_index);
                }
            }
        }
        
        Ok(state.commit_index)
    }
    
    /// Update commit index
    pub async fn update_commit_index(&self, index: LogIndex) -> Result<()> {
        let mut state = self.state.write().await;
        
        if index > state.commit_index {
            state.commit_index = index;
            info!("Updated commit index to {}", index);
        }
        
        Ok(())
    }
    
    /// Get commit index
    pub async fn get_commit_index(&self) -> LogIndex {
        let state = self.state.read().await;
        state.commit_index
    }
    
    /// Get last applied index
    pub async fn get_last_applied(&self) -> LogIndex {
        let state = self.state.read().await;
        state.last_applied
    }
    
    /// Update last applied index
    pub async fn update_last_applied(&self, index: LogIndex) -> Result<()> {
        let mut state = self.state.write().await;
        state.last_applied = index;
        
        debug!("Updated last applied index to {}", index);
        
        Ok(())
    }
    
    /// Get entries to replicate to a node
    pub async fn get_entries_for_node(&self, node_id: NodeId) -> Result<Vec<LogEntry>> {
        let state = self.state.read().await;
        let next_index = state.next_index.get(&node_id).copied().unwrap_or(1);
        
        drop(state); // Release lock before calling get_entries
        
        let log = self.log.read().await;
        let max_entries = 100; // Limit batch size
        let end_index = std::cmp::min(next_index + max_entries, log.len() as LogIndex + 1);
        
        if next_index > log.len() as LogIndex {
            return Ok(Vec::new());
        }
        
        let start = (next_index - 1) as usize;
        let end = std::cmp::min(end_index as usize, log.len());
        
        Ok(log[start..end].to_vec())
    }
    
    /// Get previous log index and term for a node
    pub async fn get_prev_log_info(&self, node_id: NodeId) -> Result<(LogIndex, Term)> {
        let state = self.state.read().await;
        let next_index = state.next_index.get(&node_id).copied().unwrap_or(1);
        
        drop(state); // Release lock
        
        if next_index <= 1 {
            return Ok((0, 0));
        }
        
        let prev_index = next_index - 1;
        let prev_term = self.get_term(prev_index).await?;
        
        Ok((prev_index, prev_term))
    }
    
    /// Handle successful append entries response
    pub async fn handle_append_success(&self, node_id: NodeId, match_index: LogIndex) -> Result<()> {
        self.update_match_index(node_id, match_index).await?;
        
        debug!("Successfully replicated to node {} up to index {}", node_id, match_index);
        
        Ok(())
    }
    
    /// Handle failed append entries response
    pub async fn handle_append_failure(&self, node_id: NodeId) -> Result<()> {
        let current_next = self.get_next_index(node_id).await;
        
        if current_next > 1 {
            self.update_next_index(node_id, current_next - 1).await?;
            debug!("Decremented next index for node {} to {}", node_id, current_next - 1);
        }
        
        Ok(())
    }
    
    /// Compact log (remove old entries)
    pub async fn compact(&self, up_to_index: LogIndex) -> Result<usize> {
        let mut log = self.log.write().await;
        
        if up_to_index >= log.len() as LogIndex {
            return Ok(0);
        }
        
        let entries_before = log.len();
        log.drain(..up_to_index as usize);
        let entries_after = log.len();
        
        let removed = entries_before - entries_after;
        
        info!("Compacted log: removed {} entries up to index {}", removed, up_to_index);
        
        Ok(removed)
    }
    
    /// Get log statistics
    pub async fn get_stats(&self) -> Result<LogStats> {
        let log = self.log.read().await;
        let state = self.state.read().await;
        
        let stats = LogStats {
            total_entries: log.len(),
            commit_index: state.commit_index,
            last_applied: state.last_applied,
            last_log_index: log.len() as LogIndex,
            last_log_term: log.last().map(|e| e.term).unwrap_or(0),
            replication_progress: state.match_index.clone(),
        };
        
        Ok(stats)
    }
}

/// Log statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStats {
    pub total_entries: usize,
    pub commit_index: LogIndex,
    pub last_applied: LogIndex,
    pub last_log_index: LogIndex,
    pub last_log_term: Term,
    pub replication_progress: HashMap<NodeId, LogIndex>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_log_replication_append_entry() {
        let node_id = Uuid::new_v4();
        let cluster = vec![node_id];
        let config = RaftConfig::default();
        
        let log_replication = LogReplication::new(node_id, cluster, config);
        
        let index = log_replication.append_entry(1, b"test command".to_vec()).await.unwrap();
        assert_eq!(index, 1);
        
        let entry = log_replication.get_entry(1).await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().term, 1);
    }
    
    #[tokio::test]
    async fn test_log_replication_get_entries() {
        let node_id = Uuid::new_v4();
        let cluster = vec![node_id];
        let config = RaftConfig::default();
        
        let log_replication = LogReplication::new(node_id, cluster, config);
        
        log_replication.append_entry(1, b"command1".to_vec()).await.unwrap();
        log_replication.append_entry(1, b"command2".to_vec()).await.unwrap();
        log_replication.append_entry(2, b"command3".to_vec()).await.unwrap();
        
        let entries = log_replication.get_entries(1, 3).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].term, 1);
        assert_eq!(entries[1].term, 1);
    }
    
    #[tokio::test]
    async fn test_log_replication_truncate() {
        let node_id = Uuid::new_v4();
        let cluster = vec![node_id];
        let config = RaftConfig::default();
        
        let log_replication = LogReplication::new(node_id, cluster, config);
        
        log_replication.append_entry(1, b"command1".to_vec()).await.unwrap();
        log_replication.append_entry(1, b"command2".to_vec()).await.unwrap();
        log_replication.append_entry(2, b"command3".to_vec()).await.unwrap();
        
        assert_eq!(log_replication.last_log_index().await, 3);
        
        log_replication.truncate_from(2).await.unwrap();
        
        assert_eq!(log_replication.last_log_index().await, 1);
    }
    
    #[tokio::test]
    async fn test_log_replication_commit_index() {
        let node_id = Uuid::new_v4();
        let peer_id = Uuid::new_v4();
        let cluster = vec![node_id, peer_id];
        let config = RaftConfig::default();
        
        let log_replication = LogReplication::new(node_id, cluster, config);
        
        log_replication.append_entry(1, b"command1".to_vec()).await.unwrap();
        log_replication.append_entry(1, b"command2".to_vec()).await.unwrap();
        
        // Initialize peer
        log_replication.initialize_node(peer_id).await.unwrap();
        
        // Simulate successful replication
        log_replication.handle_append_success(peer_id, 2).await.unwrap();
        
        let commit_index = log_replication.calculate_commit_index(1).await.unwrap();
        assert_eq!(commit_index, 2);
    }
}