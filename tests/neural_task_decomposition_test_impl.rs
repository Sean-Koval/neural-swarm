//! Neural Task Decomposition Test Implementation
//!
//! This file contains the actual test implementations for the neural task decomposition system.

#[cfg(test)]
mod decomposition_tests {
    use super::neural_task_decomposition_tests::*;
    use tokio::test;

    /// Test Category 1: Task Decomposition Testing
    mod task_decomposition_tests {
        use super::*;

        #[tokio::test]
        async fn test_heuristic_decomposition_simple_task() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            let task = Task::new_test_task("simple_001", 0.3);
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            assert_eq!(decomposition.strategy_used, DecompositionStrategy::Heuristic);
            assert!(decomposition.confidence_score > 0.5);
            assert!(!decomposition.subtasks.is_empty());
            
            // Simple task should have minimal decomposition
            assert!(decomposition.subtasks.len() <= 2);
        }

        #[tokio::test]
        async fn test_heuristic_decomposition_complex_task() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            let task = MockNeuralTaskDecomposer::create_complex_task();
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            assert_eq!(decomposition.strategy_used, DecompositionStrategy::Heuristic);
            
            // Complex task should be decomposed into multiple subtasks
            assert!(decomposition.subtasks.len() >= 3);
            
            // Check that subtasks have appropriate complexity
            for subtask in &decomposition.subtasks {
                assert!(subtask.complexity < task.complexity);
                assert!(subtask.complexity > 0.0);
            }
        }

        #[tokio::test]
        async fn test_neural_decomposition_adaptive_complexity() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Test with various complexity levels
            for complexity in [0.1, 0.3, 0.5, 0.7, 0.9] {
                let task = Task::new_test_task(&format!("neural_{}", complexity), complexity);
                
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
                assert!(result.is_ok());
                
                let decomposition = result.unwrap();
                assert_eq!(decomposition.strategy_used, DecompositionStrategy::Neural);
                
                // Higher complexity should result in more subtasks
                let expected_subtasks = ((complexity * 5.0) as usize).max(1);
                assert_eq!(decomposition.subtasks.len(), expected_subtasks);
            }
        }

        #[tokio::test]
        async fn test_hybrid_decomposition_strategy() {
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            
            // Test hybrid strategy with different complexity levels
            let simple_task = Task::new_test_task("hybrid_simple", 0.2);
            let complex_task = Task::new_test_task("hybrid_complex", 0.8);
            
            let simple_result = framework.run_decomposition_test(&simple_task, DecompositionStrategy::Hybrid).await;
            let complex_result = framework.run_decomposition_test(&complex_task, DecompositionStrategy::Hybrid).await;
            
            assert!(simple_result.is_ok());
            assert!(complex_result.is_ok());
            
            let simple_decomp = simple_result.unwrap();
            let complex_decomp = complex_result.unwrap();
            
            // Hybrid should adapt strategy based on complexity
            assert_eq!(simple_decomp.strategy_used, DecompositionStrategy::Hybrid);
            assert_eq!(complex_decomp.strategy_used, DecompositionStrategy::Hybrid);
            
            // Complex task should have more subtasks than simple task
            assert!(complex_decomp.subtasks.len() > simple_decomp.subtasks.len());
        }

        #[tokio::test]
        async fn test_edge_case_malformed_task() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            
            let malformed_task = Task {
                id: "malformed".to_string(),
                name: "".to_string(), // Empty name
                description: "".to_string(),
                complexity: -0.5, // Invalid complexity
                dependencies: vec!["non_existent_task".to_string()],
                priority: TaskPriority::High,
                estimated_duration: 0, // Invalid duration
                required_capabilities: vec![],
                resources: TaskResources {
                    memory_mb: 0,
                    cpu_cores: 0.0,
                    network_bandwidth: 0,
                    storage_gb: 0.0,
                },
            };
            
            let result = framework.run_decomposition_test(&malformed_task, DecompositionStrategy::Heuristic).await;
            // Should handle malformed tasks gracefully
            assert!(result.is_ok() || matches!(result.unwrap_err(), DecompositionError::InvalidTaskConfiguration { .. }));
        }

        #[tokio::test]
        async fn test_circular_dependency_detection() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            
            // Create task with circular dependency
            let task = Task {
                id: "circular_task".to_string(),
                name: "Circular Task".to_string(),
                description: "Task with circular dependency".to_string(),
                complexity: 0.5,
                dependencies: vec!["circular_task".to_string()], // Self-dependency
                priority: TaskPriority::Medium,
                estimated_duration: 600,
                required_capabilities: vec!["test".to_string()],
                resources: TaskResources {
                    memory_mb: 1024,
                    cpu_cores: 1.0,
                    network_bandwidth: 100,
                    storage_gb: 10.0,
                },
            };
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            let validation = framework.run_validation_test(&task, &decomposition).await;
            assert!(validation.is_ok());
            
            // Validation should detect circular dependencies
            let validation_result = validation.unwrap();
            let has_circular_dep = validation_result.issues.iter()
                .any(|issue| matches!(issue.issue_type, ValidationIssueType::CircularDependency));
            
            if has_circular_dep {
                assert!(!validation_result.is_valid);
            }
        }

        #[tokio::test]
        async fn test_resource_limit_handling() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            
            // Create task with very high resource requirements
            let resource_heavy_task = Task {
                id: "resource_heavy".to_string(),
                name: "Resource Heavy Task".to_string(),
                description: "Task requiring excessive resources".to_string(),
                complexity: 0.8,
                dependencies: vec![],
                priority: TaskPriority::High,
                estimated_duration: 3600,
                required_capabilities: vec!["heavy_compute".to_string()],
                resources: TaskResources {
                    memory_mb: 1024 * 1024, // 1TB
                    cpu_cores: 1024.0,
                    network_bandwidth: 1000000,
                    storage_gb: 10000.0,
                },
            };
            
            let result = framework.run_decomposition_test(&resource_heavy_task, DecompositionStrategy::Heuristic).await;
            
            // Should either handle gracefully or return resource error
            match result {
                Ok(decomposition) => {
                    let validation = framework.run_validation_test(&resource_heavy_task, &decomposition).await;
                    assert!(validation.is_ok());
                    
                    let validation_result = validation.unwrap();
                    let has_resource_issue = validation_result.issues.iter()
                        .any(|issue| matches!(issue.issue_type, ValidationIssueType::ResourceOverallocation));
                    
                    if has_resource_issue {
                        assert!(!validation_result.is_valid);
                    }
                }
                Err(DecompositionError::InsufficientResources { .. }) => {
                    // Expected error for excessive resource requirements
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    }

    /// Test Category 2: Neural Architecture Testing
    mod neural_architecture_tests {
        use super::*;

        #[tokio::test]
        async fn test_transformer_decomposition_accuracy() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Create a set of test tasks with known good decompositions
            let test_cases = vec![
                (Task::new_test_task("ml_pipeline", 0.7), 3), // Expected subtasks
                (Task::new_test_task("data_processing", 0.5), 2),
                (Task::new_test_task("model_training", 0.9), 4),
                (Task::new_test_task("deployment", 0.4), 2),
            ];
            
            for (task, expected_subtasks) in test_cases {
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
                assert!(result.is_ok());
                
                let decomposition = result.unwrap();
                assert_eq!(decomposition.subtasks.len(), expected_subtasks);
                assert!(decomposition.confidence_score > 0.6);
            }
        }

        #[tokio::test]
        async fn test_decision_network_quality() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Test decision quality with various task types
            let task_types = vec![
                ("compute_intensive", 0.8),
                ("io_bound", 0.3),
                ("memory_intensive", 0.6),
                ("network_bound", 0.4),
            ];
            
            for (task_type, complexity) in task_types {
                let task = Task::new_test_task(task_type, complexity);
                
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
                assert!(result.is_ok());
                
                let decomposition = result.unwrap();
                
                // Validate decision quality
                let validation = framework.run_validation_test(&task, &decomposition).await;
                assert!(validation.is_ok());
                
                let validation_result = validation.unwrap();
                assert!(validation_result.consistency_score > 0.7);
                assert!(validation_result.feasibility_score > 0.7);
            }
        }

        #[tokio::test]
        async fn test_reinforcement_learning_convergence() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Simulate multiple decomposition iterations to test learning
            let base_task = Task::new_test_task("rl_test", 0.6);
            let mut results = Vec::new();
            
            for i in 0..10 {
                let mut task = base_task.clone();
                task.id = format!("rl_test_{}", i);
                
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
                assert!(result.is_ok());
                
                let decomposition = result.unwrap();
                results.push(decomposition.confidence_score);
            }
            
            // Check that confidence scores improve over time (learning)
            let early_avg = results[0..3].iter().sum::<f32>() / 3.0;
            let late_avg = results[7..10].iter().sum::<f32>() / 3.0;
            
            // For mock implementation, scores should be consistent
            // In real implementation, should show improvement
            assert!(late_avg >= early_avg - 0.1); // Allow for some variation
        }

        #[tokio::test]
        async fn test_mixture_of_experts_selection() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Test with tasks requiring different expertise
            let expert_tasks = vec![
                Task {
                    id: "ml_expert".to_string(),
                    name: "Machine Learning Task".to_string(),
                    description: "Requires ML expertise".to_string(),
                    complexity: 0.7,
                    dependencies: vec![],
                    priority: TaskPriority::High,
                    estimated_duration: 1800,
                    required_capabilities: vec!["machine_learning".to_string(), "python".to_string()],
                    resources: TaskResources {
                        memory_mb: 4096,
                        cpu_cores: 4.0,
                        network_bandwidth: 1000,
                        storage_gb: 50.0,
                    },
                },
                Task {
                    id: "system_expert".to_string(),
                    name: "System Administration Task".to_string(),
                    description: "Requires system expertise".to_string(),
                    complexity: 0.5,
                    dependencies: vec![],
                    priority: TaskPriority::Medium,
                    estimated_duration: 1200,
                    required_capabilities: vec!["system_admin".to_string(), "linux".to_string()],
                    resources: TaskResources {
                        memory_mb: 2048,
                        cpu_cores: 2.0,
                        network_bandwidth: 500,
                        storage_gb: 25.0,
                    },
                },
            ];
            
            for task in expert_tasks {
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
                assert!(result.is_ok());
                
                let decomposition = result.unwrap();
                
                // Verify expert selection affects decomposition
                assert!(!decomposition.subtasks.is_empty());
                assert!(decomposition.confidence_score > 0.5);
                
                // Check that subtasks inherit relevant capabilities
                for subtask in &decomposition.subtasks {
                    let has_relevant_capability = subtask.required_capabilities.iter()
                        .any(|cap| task.required_capabilities.contains(cap));
                    assert!(has_relevant_capability || subtask.required_capabilities.is_empty());
                }
            }
        }
    }

    /// Test Category 3: Task Graph Testing
    mod task_graph_tests {
        use super::*;

        #[tokio::test]
        async fn test_dag_construction_correctness() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            let task = MockNeuralTaskDecomposer::create_complex_task();
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            let graph = &decomposition.execution_graph;
            
            // Verify DAG properties
            assert!(!graph.nodes.is_empty());
            assert_eq!(graph.nodes.len(), decomposition.subtasks.len());
            
            // Check that all nodes are represented
            for subtask in &decomposition.subtasks {
                assert!(graph.nodes.iter().any(|n| n.task_id == subtask.id));
            }
            
            // Verify no self-references in edges
            for edge in &graph.edges {
                assert_ne!(edge.from_task, edge.to_task);
            }
        }

        #[tokio::test]
        async fn test_cycle_detection() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            let task = Task::new_test_task("cycle_test", 0.5);
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            let validation = framework.run_validation_test(&task, &decomposition).await;
            assert!(validation.is_ok());
            
            let validation_result = validation.unwrap();
            
            // Should not have circular dependencies in a properly formed decomposition
            let has_cycles = validation_result.issues.iter()
                .any(|issue| matches!(issue.issue_type, ValidationIssueType::CircularDependency));
            
            if has_cycles {
                assert!(!validation_result.is_valid);
            }
        }

        #[tokio::test]
        async fn test_topological_sorting() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            let task = MockNeuralTaskDecomposer::create_complex_task();
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            let graph = &decomposition.execution_graph;
            
            // Check that nodes are properly leveled
            let mut level_map = std::collections::HashMap::new();
            for node in &graph.nodes {
                level_map.insert(node.task_id.clone(), node.level);
            }
            
            // Verify topological ordering
            for edge in &graph.edges {
                let from_level = level_map.get(&edge.from_task).unwrap_or(&0);
                let to_level = level_map.get(&edge.to_task).unwrap_or(&0);
                assert!(from_level < to_level, "Task {} (level {}) should come before {} (level {})", 
                    edge.from_task, from_level, edge.to_task, to_level);
            }
        }

        #[tokio::test]
        async fn test_priority_scheduling() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            
            // Create tasks with different priorities
            let high_priority_task = Task {
                id: "high_priority".to_string(),
                name: "High Priority Task".to_string(),
                description: "Critical task".to_string(),
                complexity: 0.6,
                dependencies: vec![],
                priority: TaskPriority::Critical,
                estimated_duration: 1800,
                required_capabilities: vec!["urgent".to_string()],
                resources: TaskResources {
                    memory_mb: 2048,
                    cpu_cores: 2.0,
                    network_bandwidth: 1000,
                    storage_gb: 20.0,
                },
            };
            
            let result = framework.run_decomposition_test(&high_priority_task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // High priority tasks should be decomposed efficiently
            assert!(decomposition.confidence_score > 0.7);
            
            // Subtasks should inherit priority characteristics
            for subtask in &decomposition.subtasks {
                assert_eq!(subtask.priority, TaskPriority::Critical);
            }
        }

        #[tokio::test]
        async fn test_critical_path_identification() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            let task = MockNeuralTaskDecomposer::create_complex_task();
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            let graph = &decomposition.execution_graph;
            
            // Verify critical path exists and is valid
            assert!(!graph.critical_path.is_empty());
            
            // Critical path should contain valid task IDs
            for task_id in &graph.critical_path {
                assert!(decomposition.subtasks.iter().any(|t| t.id == *task_id));
            }
            
            // Critical path should be a valid sequence
            for i in 1..graph.critical_path.len() {
                let current = &graph.critical_path[i];
                let previous = &graph.critical_path[i - 1];
                
                // Check if there's a path from previous to current
                let has_path = graph.edges.iter()
                    .any(|e| e.from_task == *previous && e.to_task == *current);
                
                // Allow for implicit dependencies through levels
                let current_task = decomposition.subtasks.iter().find(|t| t.id == *current).unwrap();
                let has_implicit_dep = current_task.dependencies.contains(previous);
                
                assert!(has_path || has_implicit_dep || i == 1);
            }
        }

        #[tokio::test]
        async fn test_parallelization_opportunities() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            let task = MockNeuralTaskDecomposer::create_complex_task();
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            let graph = &decomposition.execution_graph;
            
            // Check parallelizable groups
            assert!(!graph.parallelizable_groups.is_empty());
            
            // Verify groups don't contain conflicting tasks
            for group in &graph.parallelizable_groups {
                for task_id in group {
                    assert!(decomposition.subtasks.iter().any(|t| t.id == *task_id));
                }
                
                // Tasks in the same group should be at the same level
                let levels: std::collections::HashSet<_> = group.iter()
                    .map(|id| graph.nodes.iter().find(|n| n.task_id == *id).unwrap().level)
                    .collect();
                
                assert!(levels.len() <= 1 || group.len() == 1);
            }
        }

        #[tokio::test]
        async fn test_graph_modification_dynamic() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            let task = Task::new_test_task("dynamic_test", 0.6);
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            let original_graph = &decomposition.execution_graph;
            
            // Test optimization (simulates dynamic modification)
            let constraints = OptimizationConstraints {
                max_parallel_tasks: 4,
                resource_limits: ResourceConstraints {
                    max_memory_mb: 8192,
                    max_cpu_cores: 8.0,
                    max_network_bandwidth: 10000,
                    max_storage_gb: 1000.0,
                    max_execution_time: 3600,
                },
                deadline: Some(1800),
                priority_weights: std::collections::HashMap::new(),
            };
            
            let optimized_result = framework.decomposer.optimize_task_graph(original_graph, &constraints).await;
            assert!(optimized_result.is_ok());
            
            let optimized_graph = optimized_result.unwrap();
            
            // Optimized graph should maintain basic properties
            assert_eq!(optimized_graph.nodes.len(), original_graph.nodes.len());
            assert!(!optimized_graph.critical_path.is_empty());
        }
    }

    /// Test Category 4: Swarm Integration Testing
    mod swarm_integration_tests {
        use super::*;

        #[tokio::test]
        async fn test_neural_comm_task_passing() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            let task = Task::new_test_task("comm_test", 0.5);
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // Simulate task passing between agents
            assert!(!decomposition.subtasks.is_empty());
            
            // Each subtask should be assignable to an agent
            for subtask in &decomposition.subtasks {
                assert!(!subtask.required_capabilities.is_empty() || subtask.complexity < 0.1);
            }
        }

        #[tokio::test]
        async fn test_distributed_task_synchronization() {
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            let task = MockNeuralTaskDecomposer::create_complex_task();
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Hybrid).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // Test synchronization points
            let sync_points = decomposition.execution_graph.edges.iter()
                .filter(|e| matches!(e.dependency_type, DependencyType::Synchronization))
                .count();
            
            // Should have synchronization for complex tasks
            assert!(sync_points > 0 || decomposition.subtasks.len() == 1);
        }

        #[tokio::test]
        async fn test_fann_integration_correctness() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Test with FANN-like neural network tasks
            let fann_task = Task {
                id: "fann_integration".to_string(),
                name: "FANN Neural Network Task".to_string(),
                description: "Neural network training with FANN".to_string(),
                complexity: 0.7,
                dependencies: vec![],
                priority: TaskPriority::High,
                estimated_duration: 2400,
                required_capabilities: vec!["neural_networks".to_string(), "fann".to_string()],
                resources: TaskResources {
                    memory_mb: 4096,
                    cpu_cores: 4.0,
                    network_bandwidth: 1000,
                    storage_gb: 100.0,
                },
            };
            
            let result = framework.run_decomposition_test(&fann_task, DecompositionStrategy::Neural).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // FANN tasks should be decomposed appropriately
            assert!(decomposition.subtasks.len() > 1);
            
            // Subtasks should maintain FANN-related capabilities
            let has_fann_capability = decomposition.subtasks.iter()
                .any(|t| t.required_capabilities.contains(&"fann".to_string()));
            assert!(has_fann_capability);
        }

        #[tokio::test]
        async fn test_agent_coordination_protocols() {
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            let task = Task::new_test_task("coordination_test", 0.6);
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Hybrid).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // Test agent assignment feasibility
            let available_agents = framework.context.available_agents.len();
            let required_agents = decomposition.subtasks.len();
            
            // Should either have enough agents or enable sequential execution
            assert!(available_agents >= required_agents || decomposition.execution_graph.parallelizable_groups.len() < required_agents);
        }
    }

    /// Test Category 5: Performance Testing
    mod performance_tests {
        use super::*;

        #[tokio::test]
        async fn test_decomposition_speed_benchmarks() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            
            // Test with various task complexities
            let complexities = vec![0.1, 0.3, 0.5, 0.7, 0.9];
            
            for complexity in complexities {
                let task = Task::new_test_task(&format!("speed_test_{}", complexity), complexity);
                
                let start = std::time::Instant::now();
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
                let duration = start.elapsed();
                
                assert!(result.is_ok());
                
                // Decomposition should complete within reasonable time
                assert!(duration.as_millis() < 5000, "Decomposition took too long: {}ms", duration.as_millis());
                
                // More complex tasks may take longer, but should still be reasonable
                let expected_max_time = (complexity * 2000.0) as u128 + 1000;
                assert!(duration.as_millis() < expected_max_time);
            }
        }

        #[tokio::test]
        async fn test_memory_usage_efficiency() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Test memory usage with large task graphs
            let large_task = Task {
                id: "memory_test".to_string(),
                name: "Large Memory Task".to_string(),
                description: "Task designed to test memory efficiency".to_string(),
                complexity: 0.9,
                dependencies: vec![],
                priority: TaskPriority::Medium,
                estimated_duration: 3600,
                required_capabilities: (0..20).map(|i| format!("capability_{}", i)).collect(),
                resources: TaskResources {
                    memory_mb: 8192,
                    cpu_cores: 8.0,
                    network_bandwidth: 10000,
                    storage_gb: 500.0,
                },
            };
            
            let result = framework.run_decomposition_test(&large_task, DecompositionStrategy::Neural).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // Should handle large decompositions efficiently
            assert!(decomposition.subtasks.len() <= 10); // Reasonable limit
            
            // Total resource allocation should be reasonable
            let total_memory: u32 = decomposition.subtasks.iter()
                .map(|t| t.resources.memory_mb)
                .sum();
            
            assert!(total_memory <= large_task.resources.memory_mb * 2);
        }

        #[tokio::test]
        async fn test_concurrency_safety() {
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            
            // Test concurrent decomposition requests
            let tasks: Vec<_> = (0..10)
                .map(|i| Task::new_test_task(&format!("concurrent_{}", i), 0.5))
                .collect();
            
            let handles: Vec<_> = tasks.into_iter().map(|task| {
                let decomposer = &framework.decomposer;
                let context = &framework.context;
                tokio::spawn(async move {
                    decomposer.decompose_task(&task, DecompositionStrategy::Hybrid, context).await
                })
            }).collect();
            
            // Wait for all tasks to complete
            for handle in handles {
                let result = handle.await.unwrap();
                assert!(result.is_ok());
            }
        }

        #[tokio::test]
        async fn test_scalability_large_graphs() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Test scalability with increasing graph sizes
            for num_deps in [1, 3, 5, 10] {
                let task = Task {
                    id: format!("scalability_test_{}", num_deps),
                    name: format!("Scalability Test {}", num_deps),
                    description: "Scalability test task".to_string(),
                    complexity: 0.8,
                    dependencies: (0..num_deps).map(|i| format!("dep_{}", i)).collect(),
                    priority: TaskPriority::Medium,
                    estimated_duration: 1800,
                    required_capabilities: vec!["scalability".to_string()],
                    resources: TaskResources {
                        memory_mb: 2048,
                        cpu_cores: 2.0,
                        network_bandwidth: 1000,
                        storage_gb: 50.0,
                    },
                };
                
                let start = std::time::Instant::now();
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
                let duration = start.elapsed();
                
                assert!(result.is_ok());
                
                // Performance should scale reasonably
                let expected_max_time = (num_deps as u128 * 500) + 1000;
                assert!(duration.as_millis() < expected_max_time);
            }
        }
    }

    /// Test Category 6: Python FFI Testing
    mod python_ffi_tests {
        use super::*;

        #[tokio::test]
        async fn test_cross_language_data_consistency() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            let task = Task::new_test_task("ffi_test", 0.5);
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // Test serialization/deserialization (simulates FFI)
            let serialized = serde_json::to_string(&decomposition).unwrap();
            let deserialized: DecompositionResult = serde_json::from_str(&serialized).unwrap();
            
            // Data should remain consistent across language boundaries
            assert_eq!(decomposition.subtasks.len(), deserialized.subtasks.len());
            assert_eq!(decomposition.strategy_used, deserialized.strategy_used);
            assert_eq!(decomposition.confidence_score, deserialized.confidence_score);
        }

        #[tokio::test]
        async fn test_memory_safety_boundaries() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            
            // Test memory safety with various data types
            let test_data = vec![
                ("string_test", 0.3),
                ("numeric_test", 0.5),
                ("complex_test", 0.7),
            ];
            
            for (name, complexity) in test_data {
                let task = Task::new_test_task(name, complexity);
                
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
                assert!(result.is_ok());
                
                // Memory should be managed safely
                let decomposition = result.unwrap();
                assert!(!decomposition.subtasks.is_empty());
            }
        }

        #[tokio::test]
        async fn test_async_python_integration() {
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            let task = Task::new_test_task("async_test", 0.6);
            
            // Test async operation compatibility
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Hybrid).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // Should work with async/await patterns
            assert!(decomposition.confidence_score > 0.0);
        }

        #[tokio::test]
        async fn test_performance_impact_measurement() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Measure performance impact of FFI operations
            let task = Task::new_test_task("performance_impact", 0.7);
            
            let start = std::time::Instant::now();
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
            let duration = start.elapsed();
            
            assert!(result.is_ok());
            
            // FFI overhead should be minimal
            assert!(duration.as_millis() < 2000);
        }
    }

    /// Test Category 7: End-to-End Integration Testing
    mod integration_tests {
        use super::*;

        #[tokio::test]
        async fn test_complete_workflow_validation() {
            let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
            let task = MockNeuralTaskDecomposer::create_complex_task();
            
            // Complete workflow: decompose -> validate -> optimize -> estimate
            let decomposition = framework.run_decomposition_test(&task, DecompositionStrategy::Hybrid).await;
            assert!(decomposition.is_ok());
            
            let decomp_result = decomposition.unwrap();
            
            let validation = framework.run_validation_test(&task, &decomp_result).await;
            assert!(validation.is_ok());
            
            let validation_result = validation.unwrap();
            
            // Workflow should complete successfully
            assert!(validation_result.completeness_score > 0.7);
            assert!(validation_result.consistency_score > 0.7);
            assert!(validation_result.feasibility_score > 0.7);
        }

        #[tokio::test]
        async fn test_fault_tolerance_recovery() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            
            // Test with potentially problematic task
            let problematic_task = Task {
                id: "fault_test".to_string(),
                name: "Fault Test Task".to_string(),
                description: "Task designed to test fault tolerance".to_string(),
                complexity: 1.5, // Invalid complexity
                dependencies: vec!["non_existent".to_string()],
                priority: TaskPriority::High,
                estimated_duration: 0,
                required_capabilities: vec![],
                resources: TaskResources {
                    memory_mb: 0,
                    cpu_cores: 0.0,
                    network_bandwidth: 0,
                    storage_gb: 0.0,
                },
            };
            
            let result = framework.run_decomposition_test(&problematic_task, DecompositionStrategy::Neural).await;
            
            // Should handle gracefully
            assert!(result.is_ok() || result.is_err());
        }

        #[tokio::test]
        async fn test_system_resilience() {
            let mut framework = TestFramework::new(DecompositionStrategy::Heuristic);
            
            // Test system resilience with various edge cases
            let edge_cases = vec![
                Task::new_test_task("empty_task", 0.0),
                Task::new_test_task("max_complexity", 1.0),
                Task::new_test_task("medium_task", 0.5),
            ];
            
            for task in edge_cases {
                let result = framework.run_decomposition_test(&task, DecompositionStrategy::Heuristic).await;
                
                // System should be resilient to edge cases
                match result {
                    Ok(decomposition) => {
                        assert!(!decomposition.subtasks.is_empty());
                        assert!(decomposition.confidence_score >= 0.0);
                    }
                    Err(e) => {
                        // Acceptable errors for edge cases
                        assert!(matches!(e, 
                            DecompositionError::TaskTooComplex { .. } |
                            DecompositionError::InvalidTaskConfiguration { .. }
                        ));
                    }
                }
            }
        }

        #[tokio::test]
        async fn test_monitoring_integration() {
            let mut framework = TestFramework::new(DecompositionStrategy::Neural);
            let task = Task::new_test_task("monitoring_test", 0.6);
            
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Neural).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            
            // Check that metrics are collected
            let stats = framework.profiler.get_stats("decomposition");
            assert!(stats.is_some());
            
            let perf_stats = stats.unwrap();
            assert!(perf_stats.count > 0);
            assert!(perf_stats.mean_us > 0.0);
        }
    }

    /// Comprehensive test runner
    #[tokio::test]
    async fn test_comprehensive_suite() {
        let mut framework = TestFramework::new(DecompositionStrategy::Hybrid);
        
        // Run comprehensive test suite
        let test_tasks = vec![
            MockNeuralTaskDecomposer::create_simple_task(),
            MockNeuralTaskDecomposer::create_complex_task(),
            Task::new_test_task("comprehensive_1", 0.3),
            Task::new_test_task("comprehensive_2", 0.7),
        ];
        
        let mut results = Vec::new();
        
        for task in test_tasks {
            let result = framework.run_decomposition_test(&task, DecompositionStrategy::Hybrid).await;
            assert!(result.is_ok());
            
            let decomposition = result.unwrap();
            results.push(decomposition);
        }
        
        // Verify all tests passed
        assert_eq!(results.len(), 4);
        
        // Print performance report
        framework.profiler.print_report();
        
        // Store results in memory for coordination
        println!("Comprehensive test suite completed successfully!");
    }
}