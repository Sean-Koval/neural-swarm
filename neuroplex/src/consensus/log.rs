//! Raft log implementation
//!
//! This module implements the replicated log that maintains a sequence of state machine commands.

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Log index type
pub type LogIndex = u64;

/// Term type for Raft
pub type Term = u64;

/// Log entry in the Raft log
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogEntry {
    /// Index of the entry in the log
    pub index: LogIndex,
    /// Term when the entry was received by the leader
    pub term: Term,
    /// State machine command data
    pub data: Vec<u8>,
    /// Timestamp when the entry was created
    pub timestamp: u64,
}

impl LogEntry {
    /// Create a new log entry
    pub fn new(index: LogIndex, term: Term, data: Vec<u8>) -> Self {
        Self {
            index,
            term,
            data,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }
    
    /// Get the size of the entry in bytes
    pub fn size(&self) -> usize {
        std::mem::size_of::<LogIndex>() + 
        std::mem::size_of::<Term>() + 
        self.data.len() + 
        std::mem::size_of::<u64>()
    }
}

/// Raft log implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftLog {
    /// Log entries
    entries: Vec<LogEntry>,
    /// Commit index
    commit_index: LogIndex,
    /// Last applied index
    last_applied: LogIndex,
}

impl RaftLog {
    /// Create a new Raft log
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            commit_index: 0,
            last_applied: 0,
        }
    }
    
    /// Append a new entry to the log
    pub fn append(&mut self, entry: LogEntry) {
        self.entries.push(entry);
    }
    
    /// Append multiple entries to the log
    pub fn append_entries(&mut self, entries: Vec<LogEntry>) {
        self.entries.extend(entries);
    }
    
    /// Get an entry at a specific index
    pub fn get(&self, index: LogIndex) -> Option<&LogEntry> {
        if index == 0 || index > self.entries.len() as LogIndex {
            return None;
        }
        self.entries.get((index - 1) as usize)
    }
    
    /// Get multiple entries starting from an index
    pub fn get_entries(&self, start_index: LogIndex, max_entries: usize) -> Vec<LogEntry> {
        if start_index == 0 || start_index > self.entries.len() as LogIndex {
            return Vec::new();
        }
        
        let start = (start_index - 1) as usize;
        let end = std::cmp::min(start + max_entries, self.entries.len());
        
        self.entries[start..end].to_vec()
    }
    
    /// Get the last entry in the log
    pub fn last_entry(&self) -> Option<&LogEntry> {
        self.entries.last()
    }
    
    /// Get the index of the last entry
    pub fn last_index(&self) -> LogIndex {
        self.entries.len() as LogIndex
    }
    
    /// Get the term of the last entry
    pub fn last_term(&self) -> Term {
        self.last_entry().map(|e| e.term).unwrap_or(0)
    }
    
    /// Get the term of an entry at a specific index
    pub fn term_at(&self, index: LogIndex) -> Option<Term> {
        self.get(index).map(|e| e.term)
    }
    
    /// Truncate the log from a specific index
    pub fn truncate(&mut self, index: LogIndex) {
        if index == 0 {
            self.entries.clear();
        } else if index <= self.entries.len() as LogIndex {
            self.entries.truncate((index - 1) as usize);
        }
    }
    
    /// Get the commit index
    pub fn commit_index(&self) -> LogIndex {
        self.commit_index
    }
    
    /// Set the commit index
    pub fn set_commit_index(&mut self, index: LogIndex) {
        self.commit_index = std::cmp::min(index, self.last_index());
    }
    
    /// Get the last applied index
    pub fn last_applied(&self) -> LogIndex {
        self.last_applied
    }
    
    /// Set the last applied index
    pub fn set_last_applied(&mut self, index: LogIndex) {
        self.last_applied = std::cmp::min(index, self.commit_index);
    }
    
    /// Get the length of the log
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    
    /// Check if the log is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    
    /// Get entries that need to be applied to the state machine
    pub fn unapplied_entries(&self) -> Vec<&LogEntry> {
        if self.last_applied >= self.commit_index {
            return Vec::new();
        }
        
        let start = self.last_applied as usize;
        let end = std::cmp::min(self.commit_index as usize, self.entries.len());
        
        self.entries[start..end].iter().collect()
    }
    
    /// Check if the log contains an entry at a specific index and term
    pub fn contains_entry(&self, index: LogIndex, term: Term) -> bool {
        if let Some(entry) = self.get(index) {
            entry.term == term
        } else {
            false
        }
    }
    
    /// Get the total size of the log in bytes
    pub fn total_size(&self) -> usize {
        self.entries.iter().map(|e| e.size()).sum()
    }
    
    /// Clear the log
    pub fn clear(&mut self) {
        self.entries.clear();
        self.commit_index = 0;
        self.last_applied = 0;
    }
    
    /// Create a snapshot of the log up to a specific index
    pub fn snapshot(&self, index: LogIndex) -> LogSnapshot {
        LogSnapshot {
            last_included_index: index,
            last_included_term: self.term_at(index).unwrap_or(0),
            data: self.entries.iter()
                .take(index as usize)
                .map(|e| e.data.clone())
                .collect(),
        }
    }
    
    /// Apply a snapshot to the log
    pub fn apply_snapshot(&mut self, snapshot: LogSnapshot) {
        // Remove entries included in the snapshot
        if snapshot.last_included_index > 0 {
            self.entries.drain(0..std::cmp::min(snapshot.last_included_index as usize, self.entries.len()));
        }
        
        // Update indices
        self.last_applied = std::cmp::max(self.last_applied, snapshot.last_included_index);
        self.commit_index = std::cmp::max(self.commit_index, snapshot.last_included_index);
    }
}

/// Log snapshot for compaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogSnapshot {
    /// Index of the last entry included in the snapshot
    pub last_included_index: LogIndex,
    /// Term of the last entry included in the snapshot
    pub last_included_term: Term,
    /// Snapshot data
    pub data: Vec<Vec<u8>>,
}

impl LogSnapshot {
    /// Create a new log snapshot
    pub fn new(last_included_index: LogIndex, last_included_term: Term, data: Vec<Vec<u8>>) -> Self {
        Self {
            last_included_index,
            last_included_term,
            data,
        }
    }
    
    /// Get the size of the snapshot in bytes
    pub fn size(&self) -> usize {
        std::mem::size_of::<LogIndex>() + 
        std::mem::size_of::<Term>() + 
        self.data.iter().map(|d| d.len()).sum::<usize>()
    }
}

impl Default for RaftLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_raft_log_creation() {
        let log = RaftLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), 0);
    }
    
    #[test]
    fn test_raft_log_append() {
        let mut log = RaftLog::new();
        
        let entry1 = LogEntry::new(1, 1, b"command1".to_vec());
        let entry2 = LogEntry::new(2, 1, b"command2".to_vec());
        
        log.append(entry1.clone());
        log.append(entry2.clone());
        
        assert_eq!(log.len(), 2);
        assert_eq!(log.last_index(), 2);
        assert_eq!(log.last_term(), 1);
        
        assert_eq!(log.get(1), Some(&entry1));
        assert_eq!(log.get(2), Some(&entry2));
    }
    
    #[test]
    fn test_raft_log_truncate() {
        let mut log = RaftLog::new();
        
        for i in 1..=5 {
            log.append(LogEntry::new(i, 1, format!("command{}", i).into_bytes()));
        }
        
        assert_eq!(log.len(), 5);
        
        log.truncate(3);
        assert_eq!(log.len(), 2);
        assert_eq!(log.last_index(), 2);
        
        log.truncate(0);
        assert_eq!(log.len(), 0);
        assert_eq!(log.last_index(), 0);
    }
    
    #[test]
    fn test_raft_log_commit_index() {
        let mut log = RaftLog::new();
        
        for i in 1..=3 {
            log.append(LogEntry::new(i, 1, format!("command{}", i).into_bytes()));
        }
        
        assert_eq!(log.commit_index(), 0);
        
        log.set_commit_index(2);
        assert_eq!(log.commit_index(), 2);
        
        // Cannot commit beyond last index
        log.set_commit_index(10);
        assert_eq!(log.commit_index(), 3);
    }
    
    #[test]
    fn test_raft_log_unapplied_entries() {
        let mut log = RaftLog::new();
        
        for i in 1..=3 {
            log.append(LogEntry::new(i, 1, format!("command{}", i).into_bytes()));
        }
        
        log.set_commit_index(3);
        
        let unapplied = log.unapplied_entries();
        assert_eq!(unapplied.len(), 3);
        
        log.set_last_applied(2);
        let unapplied = log.unapplied_entries();
        assert_eq!(unapplied.len(), 1);
        assert_eq!(unapplied[0].index, 3);
    }
    
    #[test]
    fn test_raft_log_get_entries() {
        let mut log = RaftLog::new();
        
        for i in 1..=5 {
            log.append(LogEntry::new(i, 1, format!("command{}", i).into_bytes()));
        }
        
        let entries = log.get_entries(2, 3);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].index, 2);
        assert_eq!(entries[1].index, 3);
        assert_eq!(entries[2].index, 4);
        
        // Test boundary conditions
        let entries = log.get_entries(0, 3);
        assert_eq!(entries.len(), 0);
        
        let entries = log.get_entries(6, 3);
        assert_eq!(entries.len(), 0);
        
        let entries = log.get_entries(4, 10);
        assert_eq!(entries.len(), 2);
    }
    
    #[test]
    fn test_raft_log_snapshot() {
        let mut log = RaftLog::new();
        
        for i in 1..=5 {
            log.append(LogEntry::new(i, 1, format!("command{}", i).into_bytes()));
        }
        
        let snapshot = log.snapshot(3);
        assert_eq!(snapshot.last_included_index, 3);
        assert_eq!(snapshot.last_included_term, 1);
        assert_eq!(snapshot.data.len(), 3);
        
        log.apply_snapshot(snapshot);
        assert_eq!(log.len(), 2);
        assert_eq!(log.last_applied, 3);
        assert_eq!(log.commit_index, 3);
    }
    
    #[test]
    fn test_log_entry_size() {
        let entry = LogEntry::new(1, 1, b"test command".to_vec());
        let size = entry.size();
        
        // Should be size of index + term + data length + timestamp
        assert!(size > 0);
        assert!(size >= 12); // At least the size of the data
    }
}