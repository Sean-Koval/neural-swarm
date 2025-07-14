//! Integration tests for neuroplex distributed memory system

use neuroplex::{NeuroCluster, NeuroNode, init_tracing};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

#[tokio::test]
async fn test_cluster_basic_operations() {
    init_tracing();
    
    let cluster = NeuroCluster::new("test-node", "127.0.0.1:8080").await.unwrap();
    
    // Test basic operations
    cluster.set("key1", b"value1").await.unwrap();
    
    let value = cluster.get("key1").await.unwrap();
    assert_eq!(value, Some(b"value1".to_vec()));
    
    assert!(cluster.exists("key1").await.unwrap());
    assert!(!cluster.exists("nonexistent").await.unwrap());
    
    assert!(cluster.delete("key1").await.unwrap());
    assert!(!cluster.exists("key1").await.unwrap());
}

#[tokio::test]
async fn test_cluster_list_operations() {
    init_tracing();
    
    let cluster = NeuroCluster::new("test-node", "127.0.0.1:8081").await.unwrap();
    
    // Add some keys
    cluster.set("user:alice", b"Alice").await.unwrap();
    cluster.set("user:bob", b"Bob").await.unwrap();
    cluster.set("post:1", b"Post 1").await.unwrap();
    cluster.set("post:2", b"Post 2").await.unwrap();
    
    // Test list all
    let all_keys = cluster.list(None).await.unwrap();
    assert_eq!(all_keys.len(), 4);
    
    // Test list with prefix
    let user_keys = cluster.list(Some("user:")).await.unwrap();
    assert_eq!(user_keys.len(), 2);
    assert!(user_keys.contains(&"user:alice".to_string()));
    assert!(user_keys.contains(&"user:bob".to_string()));
    
    let post_keys = cluster.list(Some("post:")).await.unwrap();
    assert_eq!(post_keys.len(), 2);
    assert!(post_keys.contains(&"post:1".to_string()));
    assert!(post_keys.contains(&"post:2".to_string()));
}

#[tokio::test]
async fn test_cluster_lifecycle() {
    init_tracing();
    
    let cluster = NeuroCluster::new("test-node", "127.0.0.1:8082").await.unwrap();
    
    // Test initial state
    assert!(!cluster.is_running().await);
    
    // Start cluster
    cluster.start().await.unwrap();
    assert!(cluster.is_running().await);
    
    // Test operations while running
    cluster.set("test", b"value").await.unwrap();
    let value = cluster.get("test").await.unwrap();
    assert_eq!(value, Some(b"value".to_vec()));
    
    // Stop cluster
    cluster.stop().await.unwrap();
    assert!(!cluster.is_running().await);
}

#[tokio::test]
async fn test_cluster_crdts() {
    init_tracing();
    
    let cluster = NeuroCluster::new("test-node", "127.0.0.1:8083").await.unwrap();
    
    // Test CRDT creation
    cluster.create_g_counter("counter1").await.unwrap();
    cluster.create_pn_counter("counter2").await.unwrap();
    cluster.create_or_set("set1").await.unwrap();
    cluster.create_lww_register("register1").await.unwrap();
    
    // Test CRDT listing
    let crdts = cluster.list_crdts().await.unwrap();
    assert_eq!(crdts.len(), 4);
    assert!(crdts.contains(&"counter1".to_string()));
    assert!(crdts.contains(&"counter2".to_string()));
    assert!(crdts.contains(&"set1".to_string()));
    assert!(crdts.contains(&"register1".to_string()));
    
    // Test CRDT retrieval
    let counter1 = cluster.get_crdt("counter1").await.unwrap();
    assert!(counter1.is_some());
    
    let nonexistent = cluster.get_crdt("nonexistent").await.unwrap();
    assert!(nonexistent.is_none());
    
    // Test CRDT removal
    assert!(cluster.remove_crdt("counter1").await.unwrap());
    assert!(!cluster.remove_crdt("nonexistent").await.unwrap());
    
    let crdts_after = cluster.list_crdts().await.unwrap();
    assert_eq!(crdts_after.len(), 3);
}

#[tokio::test]
async fn test_cluster_node_management() {
    init_tracing();
    
    let mut cluster = NeuroCluster::new("test-node", "127.0.0.1:8084").await.unwrap();
    
    let node2_id = Uuid::new_v4();
    let node3_id = Uuid::new_v4();
    
    // Test adding nodes
    cluster.add_node(node2_id, "127.0.0.1:8085").await.unwrap();
    cluster.add_node(node3_id, "127.0.0.1:8086").await.unwrap();
    
    let members = cluster.get_members().await.unwrap();
    assert!(members.contains(&node2_id));
    assert!(members.contains(&node3_id));
    
    // Test removing nodes
    cluster.remove_node(node2_id).await.unwrap();
    
    let members_after = cluster.get_members().await.unwrap();
    assert!(!members_after.contains(&node2_id));
    assert!(members_after.contains(&node3_id));
}

#[tokio::test]
async fn test_cluster_statistics() {
    init_tracing();
    
    let cluster = NeuroCluster::new("test-node", "127.0.0.1:8087").await.unwrap();
    
    // Initial stats
    let initial_stats = cluster.stats().await.unwrap();
    assert_eq!(initial_stats.total_entries, 0);
    
    // Add some data
    for i in 0..10 {
        cluster.set(&format!("key{}", i), &format!("value{}", i).into_bytes()).await.unwrap();
    }
    
    let stats_after = cluster.stats().await.unwrap();
    assert_eq!(stats_after.total_entries, 10);
    assert!(stats_after.total_size > 0);
    
    // Test consensus stats
    let consensus_stats = cluster.consensus_stats().await.unwrap();
    assert_eq!(consensus_stats.current_term, 0);
    
    // Test sync stats
    let sync_stats = cluster.sync_stats().await.unwrap();
    assert!(sync_stats.total_nodes > 0);
}

#[tokio::test]
async fn test_cluster_compaction() {
    init_tracing();
    
    let cluster = NeuroCluster::new("test-node", "127.0.0.1:8088").await.unwrap();
    
    // Add data
    for i in 0..100 {
        cluster.set(&format!("key{}", i), &format!("value{}", i).into_bytes()).await.unwrap();
    }
    
    let stats_before = cluster.stats().await.unwrap();
    assert_eq!(stats_before.total_entries, 100);
    
    // Compact
    let freed = cluster.compact().await.unwrap();
    // Note: In our implementation, compaction might not free anything
    // but it should complete without error
    
    let stats_after = cluster.stats().await.unwrap();
    assert_eq!(stats_after.total_entries, 100);
}

#[tokio::test]
async fn test_node_basic_operations() {
    init_tracing();
    
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8089);
    let node = NeuroNode::new(address).await.unwrap();
    
    // Test basic operations
    node.set("key1", b"value1").await.unwrap();
    
    let value = node.get("key1").await.unwrap();
    assert_eq!(value, Some(b"value1".to_vec()));
    
    assert!(node.delete("key1").await.unwrap());
    
    let value_after = node.get("key1").await.unwrap();
    assert_eq!(value_after, None);
}

#[tokio::test]
async fn test_node_lifecycle() {
    init_tracing();
    
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8090);
    let node = NeuroNode::new(address).await.unwrap();
    
    // Test initial state
    assert!(!node.is_running().await);
    
    // Start node
    node.start().await.unwrap();
    assert!(node.is_running().await);
    
    // Stop node
    node.stop().await.unwrap();
    assert!(!node.is_running().await);
}

#[tokio::test]
async fn test_multiple_nodes() {
    init_tracing();
    
    // Create multiple nodes
    let node1 = NeuroNode::new(
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8091)
    ).await.unwrap();
    
    let node2 = NeuroNode::new(
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8092)
    ).await.unwrap();
    
    // Start both nodes
    node1.start().await.unwrap();
    node2.start().await.unwrap();
    
    // Test independent operations
    node1.set("node1_key", b"node1_value").await.unwrap();
    node2.set("node2_key", b"node2_value").await.unwrap();
    
    let value1 = node1.get("node1_key").await.unwrap();
    let value2 = node2.get("node2_key").await.unwrap();
    
    assert_eq!(value1, Some(b"node1_value".to_vec()));
    assert_eq!(value2, Some(b"node2_value".to_vec()));
    
    // Cross-node reads should return None (no sync in this test)
    let cross_value1 = node1.get("node2_key").await.unwrap();
    let cross_value2 = node2.get("node1_key").await.unwrap();
    
    assert_eq!(cross_value1, None);
    assert_eq!(cross_value2, None);
    
    // Stop nodes
    node1.stop().await.unwrap();
    node2.stop().await.unwrap();
}

#[tokio::test]
async fn test_concurrent_operations() {
    init_tracing();
    
    let cluster = NeuroCluster::new("test-node", "127.0.0.1:8093").await.unwrap();
    
    // Spawn concurrent operations
    let handles: Vec<_> = (0..10).map(|i| {
        let cluster = &cluster;
        tokio::spawn(async move {
            cluster.set(&format!("concurrent_key_{}", i), &format!("value_{}", i).into_bytes()).await.unwrap();
        })
    }).collect();
    
    // Wait for all operations to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all keys were set
    for i in 0..10 {
        let value = cluster.get(&format!("concurrent_key_{}", i)).await.unwrap();
        assert_eq!(value, Some(format!("value_{}", i).into_bytes()));
    }
    
    let stats = cluster.stats().await.unwrap();
    assert_eq!(stats.total_entries, 10);
}

#[tokio::test]
async fn test_large_data_operations() {
    init_tracing();
    
    let cluster = NeuroCluster::new("test-node", "127.0.0.1:8094").await.unwrap();
    
    // Create large data
    let large_data = vec![0u8; 1024 * 1024]; // 1MB
    
    cluster.set("large_key", &large_data).await.unwrap();
    
    let retrieved = cluster.get("large_key").await.unwrap();
    assert_eq!(retrieved, Some(large_data));
    
    let stats = cluster.stats().await.unwrap();
    assert!(stats.total_size >= 1024 * 1024);
}