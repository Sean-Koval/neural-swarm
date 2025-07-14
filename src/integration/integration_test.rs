//! Comprehensive Integration Test Suite
//!
//! Tests all neural swarm integration components working together.

use super::*;
use crate::neural::NeuralNetwork;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Comprehensive integration test
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_full_neural_swarm_integration() {
        // Initialize integration registry
        let mut registry = IntegrationRegistry::new();
        
        // Test 1: Neural Communication Integration
        println!("Testing Neural Communication Integration...");
        let neural_comm_info = IntegrationInfo {
            name: "neural_comm".to_string(),
            version: "1.0.0".to_string(),
            description: "Neural communication integration".to_string(),
            capabilities: vec!["secure_messaging".to_string(), "task_distribution".to_string()],
            dependencies: vec![],
            config_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "encryption_key": {"type": "string"},
                    "port": {"type": "number"}
                }
            }),
        };
        
        registry.register(neural_comm_info).expect("Failed to register neural_comm");
        
        let neural_comm_config = serde_json::json!({
            "encryption_key": "test_key_123",
            "port": 8080
        });
        
        let neural_comm_id = registry.create_instance("neural_comm", &neural_comm_config)
            .expect("Failed to create neural_comm instance");
        
        registry.start_instance(neural_comm_id).expect("Failed to start neural_comm");
        
        // Test 2: Neuroplex Integration
        println!("Testing Neuroplex Integration...");
        let neuroplex_info = IntegrationInfo {
            name: "neuroplex".to_string(),
            version: "1.0.0".to_string(),
            description: "Neuroplex distributed memory integration".to_string(),
            capabilities: vec!["distributed_memory".to_string(), "consensus".to_string()],
            dependencies: vec![],
            config_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "cluster_size": {"type": "number"},
                    "replication_factor": {"type": "number"}
                }
            }),
        };
        
        registry.register(neuroplex_info).expect("Failed to register neuroplex");
        
        let neuroplex_config = serde_json::json!({
            "cluster_size": 3,
            "replication_factor": 2
        });
        
        let neuroplex_id = registry.create_instance("neuroplex", &neuroplex_config)
            .expect("Failed to create neuroplex instance");
        
        registry.start_instance(neuroplex_id).expect("Failed to start neuroplex");
        
        // Test 3: FANN Core Integration
        println!("Testing FANN Core Integration...");
        let fann_core_info = IntegrationInfo {
            name: "fann_core".to_string(),
            version: "1.0.0".to_string(),
            description: "FANN neural network integration".to_string(),
            capabilities: vec!["neural_acceleration".to_string(), "gpu_compute".to_string()],
            dependencies: vec![],
            config_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "gpu_enabled": {"type": "boolean"},
                    "model_cache_size": {"type": "number"}
                }
            }),
        };
        
        registry.register(fann_core_info).expect("Failed to register fann_core");
        
        let fann_core_config = serde_json::json!({
            "gpu_enabled": false,
            "model_cache_size": 100
        });
        
        let fann_core_id = registry.create_instance("fann_core", &fann_core_config)
            .expect("Failed to create fann_core instance");
        
        registry.start_instance(fann_core_id).expect("Failed to start fann_core");
        
        // Test 4: Agent Orchestration Integration
        println!("Testing Agent Orchestration Integration...");
        let agent_orchestration_info = IntegrationInfo {
            name: "agent_orchestration".to_string(),
            version: "1.0.0".to_string(),
            description: "Agent orchestration integration".to_string(),
            capabilities: vec!["task_assignment".to_string(), "load_balancing".to_string()],
            dependencies: vec!["neural_comm".to_string()],
            config_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "max_agents": {"type": "number"},
                    "assignment_strategy": {"type": "string"}
                }
            }),
        };
        
        registry.register(agent_orchestration_info).expect("Failed to register agent_orchestration");
        
        let agent_orchestration_config = serde_json::json!({
            "max_agents": 10,
            "assignment_strategy": "capability_based"
        });
        
        let agent_orchestration_id = registry.create_instance("agent_orchestration", &agent_orchestration_config)
            .expect("Failed to create agent_orchestration instance");
        
        registry.start_instance(agent_orchestration_id).expect("Failed to start agent_orchestration");
        
        // Test 5: Swarm Coordination Integration
        println!("Testing Swarm Coordination Integration...");
        let swarm_coordination_info = IntegrationInfo {
            name: "swarm_coordination".to_string(),
            version: "1.0.0".to_string(),
            description: "Swarm coordination integration".to_string(),
            capabilities: vec!["discovery".to_string(), "consensus".to_string()],
            dependencies: vec!["neural_comm".to_string(), "neuroplex".to_string()],
            config_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "topology": {"type": "string"},
                    "discovery_interval": {"type": "number"}
                }
            }),
        };
        
        registry.register(swarm_coordination_info).expect("Failed to register swarm_coordination");
        
        let swarm_coordination_config = serde_json::json!({
            "topology": "hierarchical",
            "discovery_interval": 30
        });
        
        let swarm_coordination_id = registry.create_instance("swarm_coordination", &swarm_coordination_config)
            .expect("Failed to create swarm_coordination instance");
        
        registry.start_instance(swarm_coordination_id).expect("Failed to start swarm_coordination");
        
        // Test 6: Integration Event Flow
        println!("Testing Integration Event Flow...");
        let agent_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();
        
        // Agent registration event
        let agent_event = IntegrationEvent::AgentRegistered {
            agent_id,
            capabilities: vec!["neural_processing".to_string(), "task_execution".to_string()],
        };
        
        registry.send_event(neural_comm_id, agent_event.clone()).expect("Failed to send agent event to neural_comm");
        registry.send_event(agent_orchestration_id, agent_event.clone()).expect("Failed to send agent event to agent_orchestration");
        registry.send_event(swarm_coordination_id, agent_event).expect("Failed to send agent event to swarm_coordination");
        
        // Task assignment event
        let task_event = IntegrationEvent::TaskAssigned {
            task_id,
            agent_id,
            task_data: vec![1, 2, 3, 4, 5],
        };
        
        registry.send_event(neural_comm_id, task_event.clone()).expect("Failed to send task event to neural_comm");
        registry.send_event(neuroplex_id, task_event.clone()).expect("Failed to send task event to neuroplex");
        registry.send_event(fann_core_id, task_event.clone()).expect("Failed to send task event to fann_core");
        registry.send_event(swarm_coordination_id, task_event).expect("Failed to send task event to swarm_coordination");
        
        // Task completion event
        let completion_event = IntegrationEvent::TaskCompleted {
            task_id,
            agent_id,
            result: vec![10, 20, 30],
        };
        
        registry.send_event(neuroplex_id, completion_event.clone()).expect("Failed to send completion event to neuroplex");
        registry.send_event(agent_orchestration_id, completion_event.clone()).expect("Failed to send completion event to agent_orchestration");
        registry.send_event(swarm_coordination_id, completion_event).expect("Failed to send completion event to swarm_coordination");
        
        // Test 7: Integration Status Verification
        println!("Verifying Integration Status...");
        let neural_comm_status = registry.get_status(neural_comm_id).expect("Failed to get neural_comm status");
        let neuroplex_status = registry.get_status(neuroplex_id).expect("Failed to get neuroplex status");
        let fann_core_status = registry.get_status(fann_core_id).expect("Failed to get fann_core status");
        let agent_orchestration_status = registry.get_status(agent_orchestration_id).expect("Failed to get agent_orchestration status");
        let swarm_coordination_status = registry.get_status(swarm_coordination_id).expect("Failed to get swarm_coordination status");
        
        assert!(matches!(neural_comm_status, IntegrationStatus::Running));
        assert!(matches!(neuroplex_status, IntegrationStatus::Running));
        assert!(matches!(fann_core_status, IntegrationStatus::Running));
        assert!(matches!(agent_orchestration_status, IntegrationStatus::Running));
        assert!(matches!(swarm_coordination_status, IntegrationStatus::Running));
        
        // Test 8: Integration Shutdown
        println!("Testing Integration Shutdown...");
        registry.stop_instance(neural_comm_id).expect("Failed to stop neural_comm");
        registry.stop_instance(neuroplex_id).expect("Failed to stop neuroplex");
        registry.stop_instance(fann_core_id).expect("Failed to stop fann_core");
        registry.stop_instance(agent_orchestration_id).expect("Failed to stop agent_orchestration");
        registry.stop_instance(swarm_coordination_id).expect("Failed to stop swarm_coordination");
        
        println!("All integration tests passed successfully!");
    }
    
    #[tokio::test]
    async fn test_integration_builder_workflow() {
        println!("Testing Integration Builder Workflow...");
        
        // Build complete integration using builder pattern
        let builder = IntegrationBuilder::new()
            .with_neural_comm(serde_json::json!({
                "encryption_key": "builder_test_key",
                "port": 8081
            }))
            .with_neuroplex(serde_json::json!({
                "cluster_size": 5,
                "replication_factor": 3
            }))
            .with_fann_core(serde_json::json!({
                "gpu_enabled": true,
                "model_cache_size": 200
            }))
            .with_agent_orchestration(serde_json::json!({
                "max_agents": 20,
                "assignment_strategy": "optimal_matching"
            }))
            .with_swarm_coordination(serde_json::json!({
                "topology": "mesh",
                "discovery_interval": 60
            }));
        
        let registry = builder.build().expect("Failed to build integration registry");
        
        // Verify all integrations are registered
        let integrations = registry.list_integrations();
        assert_eq!(integrations.len(), 5);
        
        let integration_names: Vec<&str> = integrations.iter().map(|info| info.name.as_str()).collect();
        assert!(integration_names.contains(&"neural_comm"));
        assert!(integration_names.contains(&"neuroplex"));
        assert!(integration_names.contains(&"fann_core"));
        assert!(integration_names.contains(&"agent_orchestration"));
        assert!(integration_names.contains(&"swarm_coordination"));
        
        println!("Integration builder workflow test passed!");
    }
    
    #[tokio::test]
    async fn test_integration_error_handling() {
        println!("Testing Integration Error Handling...");
        
        let mut registry = IntegrationRegistry::new();
        
        // Test 1: Missing dependency error
        let invalid_integration = IntegrationInfo {
            name: "invalid_integration".to_string(),
            version: "1.0.0".to_string(),
            description: "Invalid integration with missing dependency".to_string(),
            capabilities: vec![],
            dependencies: vec!["non_existent_dependency".to_string()],
            config_schema: serde_json::json!({}),
        };
        
        let result = registry.register(invalid_integration);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing dependency"));
        
        // Test 2: Invalid configuration error
        let valid_integration = IntegrationInfo {
            name: "test_integration".to_string(),
            version: "1.0.0".to_string(),
            description: "Test integration".to_string(),
            capabilities: vec![],
            dependencies: vec![],
            config_schema: serde_json::json!({}),
        };
        
        registry.register(valid_integration).expect("Failed to register valid integration");
        
        // Test 3: Non-existent integration error
        let result = registry.create_instance("non_existent", &serde_json::json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Integration not found"));
        
        // Test 4: Non-existent instance error
        let result = registry.get_status(Uuid::new_v4());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Integration instance not found"));
        
        println!("Integration error handling test passed!");
    }
    
    #[tokio::test]
    async fn test_integration_performance() {
        println!("Testing Integration Performance...");
        
        let mut registry = IntegrationRegistry::new();
        
        // Register test integration
        let test_integration = IntegrationInfo {
            name: "performance_test".to_string(),
            version: "1.0.0".to_string(),
            description: "Performance test integration".to_string(),
            capabilities: vec![],
            dependencies: vec![],
            config_schema: serde_json::json!({}),
        };
        
        registry.register(test_integration).expect("Failed to register test integration");
        
        // Performance test: Create and start multiple instances
        let start_time = std::time::Instant::now();
        let mut instance_ids = Vec::new();
        
        for i in 0..100 {
            let config = serde_json::json!({
                "instance_id": i,
                "test_data": format!("test_{}", i)
            });
            
            let instance_id = registry.create_instance("performance_test", &config)
                .expect("Failed to create instance");
            
            registry.start_instance(instance_id).expect("Failed to start instance");
            instance_ids.push(instance_id);
        }
        
        let creation_time = start_time.elapsed();
        println!("Created and started 100 instances in {:?}", creation_time);
        
        // Performance test: Send events to all instances
        let event_start = std::time::Instant::now();
        let test_event = IntegrationEvent::ConfigurationChanged {
            component: "performance_test".to_string(),
            changes: {
                let mut changes = std::collections::HashMap::new();
                changes.insert("test_key".to_string(), serde_json::json!("test_value"));
                changes
            },
        };
        
        for &instance_id in &instance_ids {
            registry.send_event(instance_id, test_event.clone()).expect("Failed to send event");
        }
        
        let event_time = event_start.elapsed();
        println!("Sent events to 100 instances in {:?}", event_time);
        
        // Performance test: Stop all instances
        let stop_start = std::time::Instant::now();
        
        for &instance_id in &instance_ids {
            registry.stop_instance(instance_id).expect("Failed to stop instance");
        }
        
        let stop_time = stop_start.elapsed();
        println!("Stopped 100 instances in {:?}", stop_time);
        
        let total_time = start_time.elapsed();
        println!("Total performance test time: {:?}", total_time);
        
        // Verify performance thresholds
        assert!(creation_time.as_millis() < 5000, "Instance creation took too long");
        assert!(event_time.as_millis() < 1000, "Event processing took too long");
        assert!(stop_time.as_millis() < 1000, "Instance stopping took too long");
        
        println!("Integration performance test passed!");
    }
    
    #[tokio::test]
    async fn test_integration_memory_management() {
        println!("Testing Integration Memory Management...");
        
        let mut registry = IntegrationRegistry::new();
        
        // Register memory test integration
        let memory_integration = IntegrationInfo {
            name: "memory_test".to_string(),
            version: "1.0.0".to_string(),
            description: "Memory test integration".to_string(),
            capabilities: vec![],
            dependencies: vec![],
            config_schema: serde_json::json!({}),
        };
        
        registry.register(memory_integration).expect("Failed to register memory integration");
        
        // Create and destroy instances multiple times
        for cycle in 0..10 {
            let mut instance_ids = Vec::new();
            
            // Create instances
            for i in 0..20 {
                let config = serde_json::json!({
                    "cycle": cycle,
                    "instance": i
                });
                
                let instance_id = registry.create_instance("memory_test", &config)
                    .expect("Failed to create instance");
                
                registry.start_instance(instance_id).expect("Failed to start instance");
                instance_ids.push(instance_id);
            }
            
            // Send events
            let test_event = IntegrationEvent::TaskAssigned {
                task_id: Uuid::new_v4(),
                agent_id: Uuid::new_v4(),
                task_data: vec![1, 2, 3, 4, 5],
            };
            
            for &instance_id in &instance_ids {
                registry.send_event(instance_id, test_event.clone()).expect("Failed to send event");
            }
            
            // Stop instances
            for &instance_id in &instance_ids {
                registry.stop_instance(instance_id).expect("Failed to stop instance");
            }
            
            // Verify instances are cleaned up
            let active_instances = registry.list_instances();
            assert_eq!(active_instances.len(), 0, "Memory leak detected in cycle {}", cycle);
        }
        
        println!("Integration memory management test passed!");
    }
    
    #[tokio::test]
    async fn test_integration_concurrent_access() {
        println!("Testing Integration Concurrent Access...");
        
        let registry = Arc::new(RwLock::new(IntegrationRegistry::new()));
        
        // Register test integration
        {
            let mut reg = registry.write().await;
            let concurrent_integration = IntegrationInfo {
                name: "concurrent_test".to_string(),
                version: "1.0.0".to_string(),
                description: "Concurrent test integration".to_string(),
                capabilities: vec![],
                dependencies: vec![],
                config_schema: serde_json::json!({}),
            };
            
            reg.register(concurrent_integration).expect("Failed to register concurrent integration");
        }
        
        // Spawn multiple tasks for concurrent access
        let mut handles = Vec::new();
        
        for i in 0..10 {
            let registry_clone = registry.clone();
            let handle = tokio::spawn(async move {
                let config = serde_json::json!({
                    "thread_id": i,
                    "timestamp": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                });
                
                let instance_id = {
                    let mut reg = registry_clone.write().await;
                    reg.create_instance("concurrent_test", &config)
                        .expect("Failed to create instance")
                };
                
                {
                    let mut reg = registry_clone.write().await;
                    reg.start_instance(instance_id).expect("Failed to start instance");
                }
                
                // Send some events
                for j in 0..5 {
                    let event = IntegrationEvent::TaskAssigned {
                        task_id: Uuid::new_v4(),
                        agent_id: Uuid::new_v4(),
                        task_data: vec![i as u8, j as u8],
                    };
                    
                    let mut reg = registry_clone.write().await;
                    reg.send_event(instance_id, event).expect("Failed to send event");
                }
                
                // Check status
                {
                    let reg = registry_clone.read().await;
                    let status = reg.get_status(instance_id).expect("Failed to get status");
                    assert!(matches!(status, IntegrationStatus::Running));
                }
                
                // Stop instance
                {
                    let mut reg = registry_clone.write().await;
                    reg.stop_instance(instance_id).expect("Failed to stop instance");
                }
                
                instance_id
            });
            
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        let mut completed_instances = Vec::new();
        for handle in handles {
            let instance_id = handle.await.expect("Task failed");
            completed_instances.push(instance_id);
        }
        
        // Verify all instances completed successfully
        assert_eq!(completed_instances.len(), 10);
        
        // Verify registry state is clean
        let reg = registry.read().await;
        let active_instances = reg.list_instances();
        assert_eq!(active_instances.len(), 0);
        
        println!("Integration concurrent access test passed!");
    }
}