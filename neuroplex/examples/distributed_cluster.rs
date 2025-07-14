//! Distributed cluster example for neuroplex
//!
//! This example demonstrates how to set up a multi-node neuroplex cluster
//! with distributed consensus and synchronization.

use neuroplex::{
    NeuroPlexSystem, NeuroPlexConfig,
    crdt::{CrdtValue, CrdtFactory, CrdtOperation},
    consensus::RaftConfig,
    memory::MemoryConfig,
    sync::SyncConfig,
};
use std::time::Duration;
use tokio::{time::sleep, task::JoinHandle};
use uuid::Uuid;

async fn create_node(node_id: Uuid, port: u16, cluster_nodes: Vec<String>) -> Result<NeuroPlexSystem, Box<dyn std::error::Error>> {
    let config = NeuroPlexConfig {
        node_id,
        cluster_nodes,
        raft_config: RaftConfig {
            election_timeout_min: 150,
            election_timeout_max: 300,
            heartbeat_interval: 50,
            max_entries_per_append: 100,
            cluster_nodes: vec![],
            snapshot_interval: 10000,
            snapshot_threshold: 1000,
        },
        memory_config: MemoryConfig::default(),
        sync_config: SyncConfig::default(),
    };
    
    let system = NeuroPlexSystem::new(config).await?;
    system.start().await?;
    
    println!("✅ Node {} started on port {}", node_id, port);
    
    Ok(system)
}

async fn run_node_operations(system: &NeuroPlexSystem, node_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let store = system.distributed_store();
    let node_id = system.config().node_id;
    
    println!("🔄 {} performing operations...", node_name);
    
    // Create a counter for this node
    let counter_key = format!("counter_{}", node_name);
    let counter = CrdtFactory::create("GCounter", node_id, None)?;
    store.set(counter_key.clone(), counter).await?;
    
    // Increment counter multiple times
    for i in 1..=5 {
        if let Some(mut counter) = store.get(&counter_key).await? {
            counter.apply_operation(&CrdtOperation::GCounterIncrement { 
                node_id, 
                amount: i 
            })?;
            store.set(counter_key.clone(), counter.clone()).await?;
            println!("  {} incremented counter to: {}", node_name, counter.value());
        }
        
        sleep(Duration::from_millis(500)).await;
    }
    
    // Create a shared set
    let set_key = "shared_set".to_string();
    if store.get(&set_key).await?.is_none() {
        let or_set = CrdtFactory::create("ORSet", node_id, None)?;
        store.set(set_key.clone(), or_set).await?;
    }
    
    // Add items to shared set
    if let Some(mut or_set) = store.get(&set_key).await? {
        for i in 1..=3 {
            let element = format!("{}_{}", node_name, i);
            or_set.apply_operation(&CrdtOperation::ORSetAdd { 
                element, 
                node_id, 
                timestamp: chrono::Utc::now().timestamp_millis() as u64 
            })?;
        }
        store.set(set_key.clone(), or_set.clone()).await?;
        println!("  {} added items to shared set: {}", node_name, or_set.value());
    }
    
    // Sync with other nodes
    system.sync_coordinator().sync_with_peers().await?;
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    neuroplex::init_tracing();
    
    println!("🚀 Starting neuroplex distributed cluster example");
    
    // Define cluster configuration
    let cluster_nodes = vec![
        "127.0.0.1:8000".to_string(),
        "127.0.0.1:8001".to_string(),
        "127.0.0.1:8002".to_string(),
    ];
    
    // Create three nodes
    let node1_id = Uuid::new_v4();
    let node2_id = Uuid::new_v4();
    let node3_id = Uuid::new_v4();
    
    println!("🔧 Creating cluster nodes...");
    
    // Start nodes
    let node1 = create_node(node1_id, 8000, cluster_nodes.clone()).await?;
    let node2 = create_node(node2_id, 8001, cluster_nodes.clone()).await?;
    let node3 = create_node(node3_id, 8002, cluster_nodes.clone()).await?;
    
    println!("✅ All nodes started successfully");
    
    // Let nodes discover each other
    println!("⏳ Waiting for cluster formation...");
    sleep(Duration::from_secs(3)).await;
    
    // Run operations on all nodes concurrently
    println!("🔄 Running concurrent operations on all nodes...");
    
    let handles: Vec<JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>>> = vec![
        tokio::spawn(async move {
            run_node_operations(&node1, "Node1").await.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
        }),
        tokio::spawn(async move {
            run_node_operations(&node2, "Node2").await.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
        }),
        tokio::spawn(async move {
            run_node_operations(&node3, "Node3").await.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
        }),
    ];
    
    // Wait for all operations to complete
    for handle in handles {
        handle.await??;
    }
    
    println!("✅ All operations completed");
    
    // Let synchronization happen
    println!("⏳ Allowing synchronization...");
    sleep(Duration::from_secs(5)).await;
    
    // Check final state on all nodes
    println!("📊 Final cluster state:");
    
    for (i, system) in [&node1, &node2, &node3].iter().enumerate() {
        let node_name = format!("Node{}", i + 1);
        let store = system.distributed_store();
        
        println!("\n{} final state:", node_name);
        
        let keys = store.keys().await?;
        for key in keys {
            if let Some(value) = store.get(&key).await? {
                println!("  {}: {}", key, value.value());
            }
        }
        
        let size = store.size().await?;
        println!("  Total items: {}", size);
        
        // Show consensus state
        let consensus_status = system.consensus_engine().cluster_status().await;
        println!("  Consensus state: {:?}", consensus_status.get("state"));
        
        // Show memory usage
        let memory_stats = system.memory_manager().usage_stats().await;
        println!("  Memory usage: {} bytes, {} blocks", memory_stats.total_used, memory_stats.block_count);
    }
    
    // Demonstrate conflict resolution
    println!("\n🔧 Demonstrating conflict resolution...");
    
    // Create conflicting updates
    let conflict_key = "conflict_test".to_string();
    let register1 = CrdtFactory::create("LWWRegister", node1_id, Some(serde_json::Value::String("Value from Node1".to_string())))?;
    let register2 = CrdtFactory::create("LWWRegister", node2_id, Some(serde_json::Value::String("Value from Node2".to_string())))?;
    
    // Set conflicting values
    node1.distributed_store().set(conflict_key.clone(), register1).await?;
    node2.distributed_store().set(conflict_key.clone(), register2).await?;
    
    // Force synchronization
    node1.sync_coordinator().sync_with_peers().await?;
    node2.sync_coordinator().sync_with_peers().await?;
    node3.sync_coordinator().sync_with_peers().await?;
    
    sleep(Duration::from_secs(2)).await;
    
    // Check resolved value
    println!("Conflict resolution results:");
    for (i, system) in [&node1, &node2, &node3].iter().enumerate() {
        let node_name = format!("Node{}", i + 1);
        if let Some(value) = system.distributed_store().get(&conflict_key).await? {
            println!("  {}: {}", node_name, value.value());
        }
    }
    
    // Show final cluster health
    println!("\n🏥 Final cluster health:");
    for (i, system) in [&node1, &node2, &node3].iter().enumerate() {
        let node_name = format!("Node{}", i + 1);
        let health = system.health_status().await;
        println!("{}:", node_name);
        for (key, value) in health {
            println!("  {}: {}", key, value);
        }
    }
    
    // Stop all nodes
    println!("\n🛑 Stopping all nodes...");
    node1.stop().await?;
    node2.stop().await?;
    node3.stop().await?;
    
    println!("✅ All nodes stopped gracefully");
    println!("🎉 Distributed cluster example completed!");
    
    Ok(())
}