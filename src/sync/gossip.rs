//! Gossip Protocol Implementation
//!
//! Implementation of gossip-based state propagation for distributed synchronization

use super::*;
use crate::{NodeId, Result, NeuroError};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast};
use tokio::time::interval;
use tracing::{info, warn, error, debug};

/// Gossip synchronization implementation
pub struct GossipSync {
    /// Node ID
    node_id: NodeId,
    /// Node address
    address: SocketAddr,
    /// Configuration
    config: SyncConfig,
    /// Known nodes
    nodes: Arc<RwLock<HashMap<NodeId, NodeInfo>>>,
    /// Event broadcaster
    event_sender: broadcast::Sender<SyncEvent>,
    /// Statistics
    stats: Arc<RwLock<SyncStats>>,
    /// Running state
    running: Arc<RwLock<bool>>,
}

impl GossipSync {
    /// Create a new gossip sync instance
    pub fn new(node_id: NodeId, address: SocketAddr, config: SyncConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        
        let stats = SyncStats {
            total_nodes: 0,
            active_nodes: 0,
            failed_nodes: 0,
            gossip_rounds: 0,
            delta_syncs: 0,
            bytes_sent: 0,
            bytes_received: 0,
            average_latency: 0.0,
        };
        
        let mut nodes = HashMap::new();
        
        // Add self to nodes
        let self_info = NodeInfo {
            id: node_id,
            address,
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            version_vector: HashMap::new(),
            metadata: HashMap::new(),
        };
        nodes.insert(node_id, self_info);
        
        Self {
            node_id,
            address,
            config,
            nodes: Arc::new(RwLock::new(nodes)),
            event_sender,
            stats: Arc::new(RwLock::new(stats)),
            running: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Start gossip protocol
    async fn start_gossip_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_millis(self.config.gossip_interval));
        
        loop {
            let running = *self.running.read().await;
            if !running {
                break;
            }
            
            interval.tick().await;
            
            if let Err(e) = self.gossip_round().await {
                warn!("Gossip round failed: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Perform one gossip round
    async fn gossip_round(&self) -> Result<()> {
        let nodes = self.nodes.read().await;
        let node_list: Vec<NodeInfo> = nodes.values().cloned().collect();
        drop(nodes);
        
        if node_list.len() <= 1 {
            return Ok(());
        }
        
        // Select random nodes to gossip with
        let targets = self.select_gossip_targets(&node_list).await?;
        
        for target in targets {
            if let Err(e) = self.send_gossip_message(target).await {
                debug!("Failed to send gossip to {}: {}", target.id, e);
            }
        }
        
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.gossip_rounds += 1;
        
        Ok(())
    }
    
    /// Select nodes to gossip with
    async fn select_gossip_targets(&self, node_list: &[NodeInfo]) -> Result<Vec<NodeInfo>> {
        use rand::seq::SliceRandom;
        
        let mut targets = Vec::new();
        let mut rng = rand::thread_rng();
        
        // Filter out self and select random nodes
        let other_nodes: Vec<_> = node_list.iter()
            .filter(|node| node.id != self.node_id)
            .collect();
        
        let fanout = std::cmp::min(self.config.gossip_fanout, other_nodes.len());
        
        if fanout > 0 {
            let selected = other_nodes.choose_multiple(&mut rng, fanout);
            targets.extend(selected.into_iter().cloned());
        }
        
        Ok(targets)
    }
    
    /// Send gossip message to a node
    async fn send_gossip_message(&self, target: NodeInfo) -> Result<()> {
        let nodes = self.nodes.read().await;
        let node_list: Vec<NodeInfo> = nodes.values().cloned().collect();
        drop(nodes);
        
        let message = SyncMessage::Gossip {
            sender: self.node_id,
            nodes: node_list,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };
        
        // In a real implementation, this would send over the network
        debug!("Sending gossip message to {}", target.id);
        
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.bytes_sent += 1000; // Estimate
        
        Ok(())
    }
    
    /// Handle incoming gossip message
    async fn handle_gossip_message(&self, sender: NodeId, nodes: Vec<NodeInfo>, timestamp: u64) -> Result<()> {
        debug!("Received gossip message from {} with {} nodes", sender, nodes.len());
        
        let mut our_nodes = self.nodes.write().await;
        let mut updates = Vec::new();
        
        for node_info in nodes {
            match our_nodes.get(&node_info.id) {
                Some(existing) => {
                    // Update if newer
                    if node_info.last_seen > existing.last_seen {
                        our_nodes.insert(node_info.id, node_info.clone());
                        updates.push(node_info);
                    }
                }
                None => {
                    // New node
                    our_nodes.insert(node_info.id, node_info.clone());
                    updates.push(node_info.clone());
                    
                    // Emit node joined event
                    let _ = self.event_sender.send(SyncEvent::NodeJoined {
                        node_id: node_info.id,
                        address: node_info.address,
                    });
                }
            }
        }
        
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.bytes_received += 1000; // Estimate
        stats.total_nodes = our_nodes.len();
        stats.active_nodes = our_nodes.values()
            .filter(|node| self.is_node_active(node))
            .count();
        
        debug!("Updated {} nodes from gossip", updates.len());
        
        Ok(())
    }
    
    /// Check if a node is considered active
    fn is_node_active(&self, node: &NodeInfo) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let timeout = self.config.failure_detection_timeout * 1_000_000; // Convert to nanoseconds
        
        now - node.last_seen < timeout
    }
    
    /// Update node information
    async fn update_node_info(&self, node_id: NodeId, address: SocketAddr) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        
        match nodes.get_mut(&node_id) {
            Some(node) => {
                node.address = address;
                node.last_seen = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
            }
            None => {
                let node_info = NodeInfo {
                    id: node_id,
                    address,
                    last_seen: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64,
                    version_vector: HashMap::new(),
                    metadata: HashMap::new(),
                };
                nodes.insert(node_id, node_info);
            }
        }
        
        Ok(())
    }
    
    /// Detect failed nodes
    async fn detect_failed_nodes(&self) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        let mut failed_nodes = Vec::new();
        
        for node in nodes.values() {
            if node.id != self.node_id && !self.is_node_active(node) {
                failed_nodes.push(node.id);
            }
        }
        
        // Remove failed nodes
        for node_id in &failed_nodes {
            if let Some(node) = nodes.remove(node_id) {
                info!("Detected failed node: {}", node_id);
                
                // Emit node failed event
                let _ = self.event_sender.send(SyncEvent::NodeFailed {
                    node_id: *node_id,
                });
            }
        }
        
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.failed_nodes = failed_nodes.len();
        stats.total_nodes = nodes.len();
        stats.active_nodes = nodes.values()
            .filter(|node| self.is_node_active(node))
            .count();
        
        Ok(())
    }
    
    /// Start failure detection loop
    async fn start_failure_detection(&self) -> Result<()> {
        let mut interval = interval(Duration::from_millis(self.config.failure_detection_timeout));
        
        loop {
            let running = *self.running.read().await;
            if !running {
                break;
            }
            
            interval.tick().await;
            
            if let Err(e) = self.detect_failed_nodes().await {
                warn!("Failure detection failed: {}", e);
            }
        }
        
        Ok(())
    }
}

#[async_trait::async_trait]
impl Synchronizer for GossipSync {
    async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Ok(());
        }
        
        *running = true;
        drop(running);
        
        info!("Starting gossip synchronization for node {}", self.node_id);
        
        // Start gossip loop
        let gossip_sync = self.clone();
        tokio::spawn(async move {
            if let Err(e) = gossip_sync.start_gossip_loop().await {
                error!("Gossip loop failed: {}", e);
            }
        });
        
        // Start failure detection
        let failure_detector = self.clone();
        tokio::spawn(async move {
            if let Err(e) = failure_detector.start_failure_detection().await {
                error!("Failure detection failed: {}", e);
            }
        });
        
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }
        
        *running = false;
        info!("Stopping gossip synchronization for node {}", self.node_id);
        
        Ok(())
    }
    
    async fn add_node(&mut self, node_id: NodeId, address: SocketAddr) -> Result<()> {
        info!("Adding node {} at {} to gossip", node_id, address);
        self.update_node_info(node_id, address).await?;
        Ok(())
    }
    
    async fn remove_node(&mut self, node_id: NodeId) -> Result<()> {
        info!("Removing node {} from gossip", node_id);
        
        let mut nodes = self.nodes.write().await;
        if let Some(_) = nodes.remove(&node_id) {
            // Emit node left event
            let _ = self.event_sender.send(SyncEvent::NodeLeft { node_id });
        }
        
        Ok(())
    }
    
    async fn get_active_nodes(&self) -> Result<Vec<NodeId>> {
        let nodes = self.nodes.read().await;
        let active_nodes: Vec<NodeId> = nodes.values()
            .filter(|node| self.is_node_active(node))
            .map(|node| node.id)
            .collect();
        
        Ok(active_nodes)
    }
    
    async fn send_message(&self, target: NodeId, message: SyncMessage) -> Result<()> {
        // In a real implementation, this would send over the network
        debug!("Sending message to {}: {:?}", target, message);
        Ok(())
    }
    
    async fn handle_message(&mut self, message: SyncMessage) -> Result<()> {
        match message {
            SyncMessage::Gossip { sender, nodes, timestamp } => {
                self.handle_gossip_message(sender, nodes, timestamp).await?;
            }
            SyncMessage::Heartbeat { sender, timestamp } => {
                self.update_node_info(sender, self.address).await?;
                debug!("Received heartbeat from {}", sender);
            }
            SyncMessage::Ping { sender, timestamp } => {
                // Send pong response
                let pong = SyncMessage::Pong {
                    sender: self.node_id,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64,
                };
                self.send_message(sender, pong).await?;
            }
            SyncMessage::Pong { sender, timestamp } => {
                self.update_node_info(sender, self.address).await?;
                debug!("Received pong from {}", sender);
            }
            _ => {
                debug!("Unhandled message type");
            }
        }
        
        Ok(())
    }
    
    async fn trigger_delta_sync(&mut self, node_id: NodeId) -> Result<()> {
        debug!("Triggering delta sync with {}", node_id);
        
        let message = SyncMessage::DeltaRequest {
            sender: self.node_id,
            version_vector: HashMap::new(), // Would use actual version vector
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };
        
        self.send_message(node_id, message).await?;
        
        Ok(())
    }
    
    async fn get_stats(&self) -> Result<SyncStats> {
        Ok(self.stats.read().await.clone())
    }
    
    async fn subscribe_events(&self, callback: Box<dyn Fn(SyncEvent) + Send + Sync>) -> Result<()> {
        let mut receiver = self.event_sender.subscribe();
        
        tokio::spawn(async move {
            while let Ok(event) = receiver.recv().await {
                callback(event);
            }
        });
        
        Ok(())
    }
}

// Add Clone trait for GossipSync
impl Clone for GossipSync {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id,
            address: self.address,
            config: self.config.clone(),
            nodes: self.nodes.clone(),
            event_sender: self.event_sender.clone(),
            stats: self.stats.clone(),
            running: self.running.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use uuid::Uuid;
    
    #[tokio::test]
    async fn test_gossip_sync_creation() {
        let node_id = Uuid::new_v4();
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = SyncConfig::default();
        
        let gossip = GossipSync::new(node_id, address, config);
        
        assert_eq!(gossip.node_id, node_id);
        assert_eq!(gossip.address, address);
    }
    
    #[tokio::test]
    async fn test_gossip_sync_node_management() {
        let node_id = Uuid::new_v4();
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = SyncConfig::default();
        
        let mut gossip = GossipSync::new(node_id, address, config);
        
        let peer_id = Uuid::new_v4();
        let peer_address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081);
        
        gossip.add_node(peer_id, peer_address).await.unwrap();
        
        let active_nodes = gossip.get_active_nodes().await.unwrap();
        assert!(active_nodes.contains(&peer_id));
        
        gossip.remove_node(peer_id).await.unwrap();
        
        let active_nodes = gossip.get_active_nodes().await.unwrap();
        assert!(!active_nodes.contains(&peer_id));
    }
    
    #[tokio::test]
    async fn test_gossip_message_handling() {
        let node_id = Uuid::new_v4();
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = SyncConfig::default();
        
        let mut gossip = GossipSync::new(node_id, address, config);
        
        let peer_id = Uuid::new_v4();
        let peer_address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081);
        
        let peer_info = NodeInfo {
            id: peer_id,
            address: peer_address,
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            version_vector: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        let message = SyncMessage::Gossip {
            sender: peer_id,
            nodes: vec![peer_info],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };
        
        gossip.handle_message(message).await.unwrap();
        
        let active_nodes = gossip.get_active_nodes().await.unwrap();
        assert!(active_nodes.contains(&peer_id));
    }
}