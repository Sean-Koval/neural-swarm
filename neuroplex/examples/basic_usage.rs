//! Basic usage example for neuroplex
//!
//! This example demonstrates how to use the neuroplex distributed memory system
//! with CRDT operations and consensus.

use neuroplex::{
    NeuroPlexSystem, NeuroPlexConfig,
    crdt::{CrdtValue, CrdtFactory, CrdtOperation},
    consensus::RaftConfig,
    memory::MemoryConfig,
    sync::SyncConfig,
};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    neuroplex::init_tracing();
    
    println!("🚀 Starting neuroplex basic usage example");
    
    // Create configuration
    let node_id = Uuid::new_v4();
    let config = NeuroPlexConfig {
        node_id,
        cluster_nodes: vec!["127.0.0.1:8000".to_string()],
        raft_config: RaftConfig::default(),
        memory_config: MemoryConfig::default(),
        sync_config: SyncConfig::default(),
    };
    
    // Create neuroplex system
    let system = NeuroPlexSystem::new(config).await?;
    
    // Start the system
    system.start().await?;
    
    println!("✅ System started successfully");
    
    // Get distributed store reference
    let store = system.distributed_store();
    
    // Create different CRDT types
    let g_counter = CrdtFactory::create("GCounter", node_id, None)?;
    let pn_counter = CrdtFactory::create("PNCounter", node_id, None)?;
    let or_set = CrdtFactory::create("ORSet", node_id, None)?;
    let lww_register = CrdtFactory::create("LWWRegister", node_id, 
        Some(serde_json::Value::String("Hello, World!".to_string())))?;
    
    // Store CRDTs in the distributed store
    store.set("counter".to_string(), g_counter).await?;
    store.set("pn_counter".to_string(), pn_counter).await?;
    store.set("my_set".to_string(), or_set).await?;
    store.set("register".to_string(), lww_register).await?;
    
    println!("✅ Stored CRDTs in distributed store");
    
    // Demonstrate CRDT operations
    println!("\n📊 Demonstrating CRDT operations:");
    
    // G-Counter operations
    if let Some(mut counter) = store.get("counter").await? {
        println!("Current G-Counter value: {}", counter.value());
        
        // Increment counter
        counter.apply_operation(&CrdtOperation::GCounterIncrement { 
            node_id, 
            amount: 5 
        })?;
        
        store.set("counter".to_string(), counter.clone()).await?;
        println!("After incrementing by 5: {}", counter.value());
    }
    
    // PN-Counter operations
    if let Some(mut pn_counter) = store.get("pn_counter").await? {
        println!("Current PN-Counter value: {}", pn_counter.value());
        
        // Increment and decrement
        pn_counter.apply_operation(&CrdtOperation::PNCounterIncrement { 
            node_id, 
            amount: 10 
        })?;
        pn_counter.apply_operation(&CrdtOperation::PNCounterDecrement { 
            node_id, 
            amount: 3 
        })?;
        
        store.set("pn_counter".to_string(), pn_counter.clone()).await?;
        println!("After increment(10) and decrement(3): {}", pn_counter.value());
    }
    
    // OR-Set operations
    if let Some(mut or_set) = store.get("my_set").await? {
        println!("Current OR-Set value: {}", or_set.value());
        
        // Add elements
        or_set.apply_operation(&CrdtOperation::ORSetAdd { 
            element: "apple".to_string(), 
            node_id, 
            timestamp: chrono::Utc::now().timestamp_millis() as u64 
        })?;
        or_set.apply_operation(&CrdtOperation::ORSetAdd { 
            element: "banana".to_string(), 
            node_id, 
            timestamp: chrono::Utc::now().timestamp_millis() as u64 
        })?;
        
        store.set("my_set".to_string(), or_set.clone()).await?;
        println!("After adding apple and banana: {}", or_set.value());
    }
    
    // LWW-Register operations
    if let Some(mut register) = store.get("register").await? {
        println!("Current LWW-Register value: {}", register.value());
        
        // Update register
        register.apply_operation(&CrdtOperation::LWWRegisterSet { 
            value: "Updated value!".to_string(), 
            node_id, 
            timestamp: chrono::Utc::now().timestamp_millis() as u64 
        })?;
        
        store.set("register".to_string(), register.clone()).await?;
        println!("After update: {}", register.value());
    }
    
    // List all keys in the store
    println!("\n📋 All keys in the store:");
    let keys = store.keys().await?;
    for key in keys {
        println!("  - {}", key);
    }
    
    // Get store size
    let size = store.size().await?;
    println!("Store contains {} items", size);
    
    // Demonstrate memory management
    println!("\n💾 Memory management:");
    let memory_manager = system.memory_manager();
    
    // Allocate some memory
    memory_manager.allocate("test_data".to_string(), b"This is test data".to_vec()).await?;
    
    // Read it back
    let data = memory_manager.read("test_data").await?;
    println!("Read data: {}", String::from_utf8_lossy(&data));
    
    // Get memory usage stats
    let stats = memory_manager.usage_stats().await;
    println!("Memory usage: {} bytes allocated, {} blocks", stats.total_allocated, stats.block_count);
    
    // Demonstrate consensus
    println!("\n🗳️ Consensus status:");
    let consensus = system.consensus_engine();
    let cluster_status = consensus.cluster_status().await;
    
    for (key, value) in cluster_status {
        println!("  {}: {}", key, value);
    }
    
    // Demonstrate synchronization
    println!("\n🔄 Synchronization:");
    let sync_coordinator = system.sync_coordinator();
    
    // Sync with peers
    sync_coordinator.sync_with_peers().await?;
    
    let sync_status = sync_coordinator.status().await;
    for (key, value) in sync_status {
        println!("  {}: {}", key, value);
    }
    
    // Get overall system health
    println!("\n🏥 System health:");
    let health = system.health_status().await;
    for (key, value) in health {
        println!("  {}: {}", key, value);
    }
    
    // Let the system run for a bit
    println!("\n⏳ Letting system run for 5 seconds...");
    sleep(Duration::from_secs(5)).await;
    
    // Stop the system
    system.stop().await?;
    
    println!("✅ System stopped gracefully");
    println!("🎉 Basic usage example completed!");
    
    Ok(())
}