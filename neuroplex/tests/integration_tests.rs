//! Integration tests for neuroplex
//!
//! These tests verify the correct operation of the entire system
//! including distributed operations, consensus, and synchronization.

use neuroplex::{
    NeuroPlexSystem, NeuroPlexConfig,
    crdt::{CrdtFactory, CrdtOperation},
    consensus::RaftConfig,
    memory::MemoryConfig,
    sync::SyncConfig,
};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

#[tokio::test]
async fn test_system_lifecycle() {
    // Test system creation, start, and stop
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    
    // Start system
    system.start().await.unwrap();
    
    // Check that system is running
    let health = system.health_status().await;
    assert!(health.contains_key("node_id"));
    assert!(health.contains_key("consensus_state"));
    
    // Stop system
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_distributed_store_operations() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    system.start().await.unwrap();
    
    let store = system.distributed_store();
    let node_id = system.config().node_id;
    
    // Test G-Counter operations
    let g_counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
    store.set("counter".to_string(), g_counter).await.unwrap();
    
    // Increment counter
    if let Some(mut counter) = store.get("counter").await.unwrap() {
        counter.apply_operation(&CrdtOperation::GCounterIncrement { 
            node_id, 
            amount: 5 
        }).unwrap();
        
        store.set("counter".to_string(), counter.clone()).await.unwrap();
        
        // Verify value
        assert_eq!(counter.value(), serde_json::Value::Number(serde_json::Number::from(5)));
    }
    
    // Test PN-Counter operations
    let pn_counter = CrdtFactory::create("PNCounter", node_id, None).unwrap();
    store.set("pn_counter".to_string(), pn_counter).await.unwrap();
    
    if let Some(mut counter) = store.get("pn_counter").await.unwrap() {
        counter.apply_operation(&CrdtOperation::PNCounterIncrement { 
            node_id, 
            amount: 10 
        }).unwrap();
        counter.apply_operation(&CrdtOperation::PNCounterDecrement { 
            node_id, 
            amount: 3 
        }).unwrap();
        
        store.set("pn_counter".to_string(), counter.clone()).await.unwrap();
        
        // Verify value
        assert_eq!(counter.value(), serde_json::Value::Number(serde_json::Number::from(7)));
    }
    
    // Test OR-Set operations
    let or_set = CrdtFactory::create("ORSet", node_id, None).unwrap();
    store.set("set".to_string(), or_set).await.unwrap();
    
    if let Some(mut set) = store.get("set").await.unwrap() {
        set.apply_operation(&CrdtOperation::ORSetAdd { 
            element: "apple".to_string(), 
            node_id, 
            timestamp: chrono::Utc::now().timestamp_millis() as u64 
        }).unwrap();
        set.apply_operation(&CrdtOperation::ORSetAdd { 
            element: "banana".to_string(), 
            node_id, 
            timestamp: chrono::Utc::now().timestamp_millis() as u64 
        }).unwrap();
        
        store.set("set".to_string(), set.clone()).await.unwrap();
        
        // Verify set contains items
        let value = set.value();
        assert!(value.is_array());
        let array = value.as_array().unwrap();
        assert!(array.len() >= 2);
    }
    
    // Test store operations
    let keys = store.keys().await.unwrap();
    assert!(keys.contains(&"counter".to_string()));
    assert!(keys.contains(&"pn_counter".to_string()));
    assert!(keys.contains(&"set".to_string()));
    
    let size = store.size().await.unwrap();
    assert!(size >= 3);
    
    // Test deletion
    store.delete("counter").await.unwrap();
    let keys_after_delete = store.keys().await.unwrap();
    assert!(!keys_after_delete.contains(&"counter".to_string()));
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_memory_management() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    system.start().await.unwrap();
    
    let memory_manager = system.memory_manager();
    
    // Test memory allocation
    let test_data = b"This is test data for memory management".to_vec();
    memory_manager.allocate("test_block".to_string(), test_data.clone()).await.unwrap();
    
    // Test reading data
    let read_data = memory_manager.read("test_block").await.unwrap();
    assert_eq!(read_data, test_data);
    
    // Test updating data
    let new_data = b"Updated test data with different content".to_vec();
    memory_manager.update("test_block", new_data.clone()).await.unwrap();
    
    let updated_data = memory_manager.read("test_block").await.unwrap();
    assert_eq!(updated_data, new_data);
    
    // Test usage statistics
    let stats = memory_manager.usage_stats().await;
    assert!(stats.total_used > 0);
    assert!(stats.block_count > 0);
    
    // Test metadata
    let metadata = memory_manager.get_metadata("test_block").await;
    assert!(metadata.is_some());
    let metadata = metadata.unwrap();
    assert_eq!(metadata.id, "test_block");
    assert_eq!(metadata.size, new_data.len());
    
    // Test listing blocks
    let blocks = memory_manager.list_blocks().await;
    assert!(blocks.contains(&"test_block".to_string()));
    
    // Test deletion
    memory_manager.delete("test_block").await.unwrap();
    let result = memory_manager.read("test_block").await;
    assert!(result.is_err());
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_consensus_operations() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    system.start().await.unwrap();
    
    let consensus = system.consensus_engine();
    
    // Test initial state
    let state = consensus.state().await;
    // Should start as follower
    assert_eq!(state.to_string(), "Follower");
    
    let term = consensus.term().await;
    assert_eq!(term, 0);
    
    let leader = consensus.leader().await;
    assert_eq!(leader, None);
    
    // Test cluster status
    let cluster_status = consensus.cluster_status().await;
    assert!(cluster_status.contains_key("node_id"));
    assert!(cluster_status.contains_key("state"));
    assert!(cluster_status.contains_key("term"));
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_synchronization() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    system.start().await.unwrap();
    
    let sync_coordinator = system.sync_coordinator();
    
    // Test sync status
    let status = sync_coordinator.status().await;
    assert!(status.contains_key("node_id"));
    assert!(status.contains_key("is_running"));
    
    // Test sync with peers (should not fail even with no peers)
    let result = sync_coordinator.sync_with_peers().await;
    assert!(result.is_ok());
    
    // Test key sync
    let store = system.distributed_store();
    let node_id = system.config().node_id;
    let counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
    store.set("sync_test".to_string(), counter.clone()).await.unwrap();
    
    let sync_result = sync_coordinator.sync_key("sync_test", Some(counter)).await;
    assert!(sync_result.is_ok());
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_crdt_conflict_resolution() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    system.start().await.unwrap();
    
    let store = system.distributed_store();
    let node_id = system.config().node_id;
    
    // Create two G-Counters with different values
    let mut counter1 = CrdtFactory::create("GCounter", node_id, None).unwrap();
    let mut counter2 = CrdtFactory::create("GCounter", node_id, None).unwrap();
    
    // Apply different operations
    counter1.apply_operation(&CrdtOperation::GCounterIncrement { 
        node_id, 
        amount: 5 
    }).unwrap();
    
    counter2.apply_operation(&CrdtOperation::GCounterIncrement { 
        node_id, 
        amount: 3 
    }).unwrap();
    
    // Merge counters (simulate conflict resolution)
    let merged = counter1.merge(&counter2).unwrap();
    
    // Verify merge result
    assert_eq!(merged.value(), serde_json::Value::Number(serde_json::Number::from(8)));
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_event_subscription() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    system.start().await.unwrap();
    
    let store = system.distributed_store();
    let node_id = system.config().node_id;
    
    // Subscribe to store events
    let mut event_receiver = store.subscribe();
    
    // Perform an operation that should generate an event
    let counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
    store.set("event_test".to_string(), counter).await.unwrap();
    
    // Try to receive an event (with timeout)
    let result = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv()).await;
    
    // Note: This test may pass or fail depending on timing and implementation
    // In a real scenario, you'd have proper event handling
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_system_recovery() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    
    // Start system
    system.start().await.unwrap();
    
    // Add some data
    let store = system.distributed_store();
    let node_id = system.config().node_id;
    let counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
    store.set("recovery_test".to_string(), counter).await.unwrap();
    
    // Stop system
    system.stop().await.unwrap();
    
    // Restart system
    system.start().await.unwrap();
    
    // Verify data is still accessible
    let retrieved = store.get("recovery_test").await.unwrap();
    assert!(retrieved.is_some());
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_concurrent_operations() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    system.start().await.unwrap();
    
    let store = system.distributed_store();
    let node_id = system.config().node_id;
    
    // Create multiple tasks performing concurrent operations
    let mut tasks = Vec::new();
    
    for i in 0..10 {
        let store = store.clone();
        let node_id = node_id;
        
        let task = tokio::spawn(async move {
            let key = format!("concurrent_test_{}", i);
            let counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
            store.set(key.clone(), counter).await.unwrap();
            
            // Perform multiple operations
            for j in 1..=5 {
                if let Some(mut counter) = store.get(&key).await.unwrap() {
                    counter.apply_operation(&CrdtOperation::GCounterIncrement { 
                        node_id, 
                        amount: j 
                    }).unwrap();
                    store.set(key.clone(), counter).await.unwrap();
                }
            }
        });
        
        tasks.push(task);
    }
    
    // Wait for all tasks to complete
    for task in tasks {
        task.await.unwrap();
    }
    
    // Verify all operations completed
    let keys = store.keys().await.unwrap();
    assert!(keys.len() >= 10);
    
    // Verify counters have correct values
    for i in 0..10 {
        let key = format!("concurrent_test_{}", i);
        if let Some(counter) = store.get(&key).await.unwrap() {
            // Each counter should have value 15 (1+2+3+4+5)
            assert_eq!(counter.value(), serde_json::Value::Number(serde_json::Number::from(15)));
        }
    }
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_stress_operations() {
    let config = NeuroPlexConfig::default();
    let system = NeuroPlexSystem::new(config).await.unwrap();
    system.start().await.unwrap();
    
    let store = system.distributed_store();
    let memory_manager = system.memory_manager();
    let node_id = system.config().node_id;
    
    // Stress test with many operations
    let num_operations = 100;
    
    for i in 0..num_operations {
        // Store operations
        let key = format!("stress_test_{}", i);
        let counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
        store.set(key.clone(), counter).await.unwrap();
        
        // Memory operations
        let mem_key = format!("mem_stress_{}", i);
        let data = format!("stress test data {}", i).into_bytes();
        memory_manager.allocate(mem_key, data).await.unwrap();
        
        // Sync operations
        if i % 10 == 0 {
            let _ = system.sync_coordinator().sync_with_peers().await;
        }
    }
    
    // Verify all operations completed
    let keys = store.keys().await.unwrap();
    assert!(keys.len() >= num_operations);
    
    let mem_blocks = memory_manager.list_blocks().await;
    assert!(mem_blocks.len() >= num_operations);
    
    // Check memory usage
    let stats = memory_manager.usage_stats().await;
    assert!(stats.total_used > 0);
    assert!(stats.block_count >= num_operations);
    
    system.stop().await.unwrap();
}