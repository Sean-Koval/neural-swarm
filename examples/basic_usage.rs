//! Basic usage example for neuroplex distributed memory system

use neuroplex::{NeuroCluster, init_tracing};
use std::error::Error;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize tracing
    init_tracing();
    
    println!("=== Neuroplex Distributed Memory System Example ===");
    
    // Create a cluster
    let cluster = NeuroCluster::new("node1", "127.0.0.1:8080").await?;
    
    println!("Created cluster node: {}", cluster.node_id());
    println!("Node address: {}", cluster.address());
    
    // Start the cluster
    cluster.start().await?;
    println!("Cluster started successfully");
    
    // Basic key-value operations
    println!("\n--- Basic Key-Value Operations ---");
    
    // Set some values
    cluster.set("greeting", b"Hello, World!").await?;
    cluster.set("number", &42u64.to_be_bytes()).await?;
    cluster.set("binary", &[0x01, 0x02, 0x03, 0x04]).await?;
    
    println!("Stored 3 key-value pairs");
    
    // Get values
    if let Some(value) = cluster.get("greeting").await? {
        println!("greeting: {}", String::from_utf8_lossy(&value));
    }
    
    if let Some(value) = cluster.get("number").await? {
        if value.len() == 8 {
            let number = u64::from_be_bytes(value.try_into().unwrap());
            println!("number: {}", number);
        }
    }
    
    if let Some(value) = cluster.get("binary").await? {
        println!("binary: {:?}", value);
    }
    
    // List all keys
    let keys = cluster.list(None).await?;
    println!("All keys: {:?}", keys);
    
    // List keys with prefix
    cluster.set("user:alice", b"Alice Smith").await?;
    cluster.set("user:bob", b"Bob Jones").await?;
    cluster.set("user:charlie", b"Charlie Brown").await?;
    
    let user_keys = cluster.list(Some("user:")).await?;
    println!("User keys: {:?}", user_keys);
    
    // Memory statistics
    println!("\n--- Memory Statistics ---");
    let stats = cluster.stats().await?;
    println!("Total entries: {}", stats.total_entries);
    println!("Total size: {} bytes", stats.total_size);
    println!("Compression ratio: {:.2}", stats.compression_ratio);
    
    // CRDT operations
    println!("\n--- CRDT Operations ---");
    
    // Create a G-Counter
    cluster.create_g_counter("page_views").await?;
    println!("Created G-Counter: page_views");
    
    // Create a PN-Counter
    cluster.create_pn_counter("likes").await?;
    println!("Created PN-Counter: likes");
    
    // Create an OR-Set
    cluster.create_or_set("tags").await?;
    println!("Created OR-Set: tags");
    
    // Create an LWW-Register
    cluster.create_lww_register("status").await?;
    println!("Created LWW-Register: status");
    
    // List all CRDTs
    let crdts = cluster.list_crdts().await?;
    println!("Available CRDTs: {:?}", crdts);
    
    // Memory compaction
    println!("\n--- Memory Compaction ---");
    let freed = cluster.compact().await?;
    println!("Freed {} bytes during compaction", freed);
    
    // Consensus statistics
    println!("\n--- Consensus Statistics ---");
    let consensus_stats = cluster.consensus_stats().await?;
    println!("Current term: {}", consensus_stats.current_term);
    println!("Node state: {:?}", consensus_stats.node_state);
    println!("Log length: {}", consensus_stats.log_length);
    
    // Synchronization statistics
    println!("\n--- Synchronization Statistics ---");
    let sync_stats = cluster.sync_stats().await?;
    println!("Total nodes: {}", sync_stats.total_nodes);
    println!("Active nodes: {}", sync_stats.active_nodes);
    println!("Gossip rounds: {}", sync_stats.gossip_rounds);
    
    // Simulate some work
    println!("\n--- Simulating distributed operations ---");
    for i in 1..=5 {
        cluster.set(&format!("task_{}", i), format!("Task {} data", i).as_bytes()).await?;
        sleep(Duration::from_millis(100)).await;
    }
    
    // Final statistics
    println!("\n--- Final Statistics ---");
    let final_stats = cluster.stats().await?;
    println!("Final entry count: {}", final_stats.total_entries);
    println!("Final size: {} bytes", final_stats.total_size);
    
    // Clean up
    cluster.stop().await?;
    println!("\nCluster stopped successfully");
    
    Ok(())
}