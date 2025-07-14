//! Comprehensive tests for neural coordination protocols
//!
//! Tests all aspects of the neural coordination protocol suite including
//! neural consensus, real-time messaging, fault tolerance, load balancing,
//! and swarm intelligence.

use neuroplex::coordination::{
    SwarmCoordinator, CoordinationConfig, CoordinationStrategy,
    NeuralConsensusEngine, NeuralConsensusConfig, NeuralProposal, NeuralVote,
    RealTimeMessagingEngine, RealTimeMessagingConfig, CoordinationMessage,
    FaultToleranceEngine, ByzantineFaultToleranceConfig, RecoveryConfig,
    DynamicLoadBalancer, LoadBalancingConfig, TaskAssignmentRequest,
    SwarmIntelligenceEngine, SwarmIntelligenceConfig, SwarmAgent,
    DecisionType, Priority, MessageType, MessagePriority, AgentType,
    LoadBalancingStrategy, TaskPriority, ResourceType
};
use neuroplex::{NodeId, NeuroConfig};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

/// Test neural consensus engine functionality
#[tokio::test]
async fn test_neural_consensus_engine() {
    let config = NeuralConsensusConfig::default();
    let node_id = Uuid::new_v4();
    
    // Create and start neural consensus engine
    let engine = NeuralConsensusEngine::new(config.clone(), node_id).unwrap();
    engine.start().await.unwrap();
    
    // Test proposal submission
    let proposal = NeuralProposal {
        id: Uuid::new_v4(),
        proposer: node_id,
        timestamp: 1234567890,
        decision_type: DecisionType::TaskAssignment,
        data: b"test proposal".to_vec(),
        confidence: 0.8,
        neural_weights: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        deadline: Some(1234567900),
        priority: Priority::High,
    };
    
    engine.submit_proposal(proposal.clone()).await.unwrap();
    
    // Test vote submission
    let vote = NeuralVote {
        proposal_id: proposal.id,
        voter: node_id,
        timestamp: 1234567891,
        decision: true,
        confidence: 0.9,
        reasoning: vec![0.8, 0.7, 0.6],
        learning_feedback: None,
    };
    
    engine.submit_vote(vote).await.unwrap();
    
    // Wait for processing
    sleep(Duration::from_millis(100)).await;
    
    // Check consensus state
    let state = engine.get_state().await.unwrap();
    assert_eq!(state.pending_proposals.len(), 1);
    assert_eq!(state.active_votes.len(), 1);
    
    engine.stop().await.unwrap();
}

/// Test real-time messaging engine functionality
#[tokio::test]
async fn test_real_time_messaging_engine() {
    let config = RealTimeMessagingConfig::default();
    let node_id = Uuid::new_v4();
    
    // Create and start messaging engine
    let engine = RealTimeMessagingEngine::new(config.clone(), node_id).unwrap();
    engine.start().await.unwrap();
    
    // Test message enqueueing
    let message = CoordinationMessage {
        id: Uuid::new_v4(),
        sender: node_id,
        recipient: None,
        message_type: MessageType::Ping(neuroplex::coordination::real_time_messaging::PingMessage {
            sequence: 1,
            timestamp: 1234567890,
        }),
        priority: MessagePriority::High,
        timestamp: 1234567890,
        deadline: Some(1234567900),
        retry_count: 0,
        correlation_id: None,
        metadata: HashMap::new(),
    };
    
    engine.enqueue_message(message).await.unwrap();
    
    // Wait for processing
    sleep(Duration::from_millis(100)).await;
    
    // Check statistics
    let stats = engine.get_stats().await;
    assert_eq!(stats.total_messages, 1);
    assert_eq!(stats.messages_by_priority.get(&MessagePriority::High), Some(&1));
    
    engine.stop().await.unwrap();
}

/// Test fault tolerance engine functionality
#[tokio::test]
async fn test_fault_tolerance_engine() {
    let bft_config = ByzantineFaultToleranceConfig::default();
    let recovery_config = RecoveryConfig::default();
    let node_id = Uuid::new_v4();
    
    // Create and start fault tolerance engine
    let engine = FaultToleranceEngine::new(bft_config.clone(), recovery_config.clone(), node_id).unwrap();
    engine.start().await.unwrap();
    
    // Test circuit breaker creation
    let circuit_config = neuroplex::coordination::fault_tolerance::CircuitBreakerConfig::default();
    engine.create_circuit_breaker("test_circuit".to_string(), circuit_config).await.unwrap();
    
    // Test circuit breaker state
    let state = engine.get_circuit_breaker_state("test_circuit").await;
    assert!(state.is_some());
    
    // Test checkpoint creation
    let checkpoint_id = engine.create_checkpoint(b"test state".to_vec()).await.unwrap();
    assert_ne!(checkpoint_id, Uuid::nil());
    
    engine.stop().await.unwrap();
}

/// Test dynamic load balancer functionality
#[tokio::test]
async fn test_dynamic_load_balancer() {
    let config = LoadBalancingConfig {
        strategy: LoadBalancingStrategy::NeuralAdaptive,
        rebalancing_threshold: 0.2,
        rebalancing_interval: 5000,
        prediction_window_size: 100,
        learning_rate: 0.01,
        max_task_queue_size: 1000,
        monitoring_interval: 1000,
        load_smoothing_factor: 0.9,
    };
    
    // Create and start load balancer
    let balancer = DynamicLoadBalancer::new(config.clone()).unwrap();
    balancer.start().await.unwrap();
    
    // Add a node with load information
    let node_id = Uuid::new_v4();
    let mut resource_metrics = HashMap::new();
    resource_metrics.insert(ResourceType::CPU, neuroplex::coordination::load_balancing::ResourceMetrics {
        resource_type: ResourceType::CPU,
        current_usage: 50.0,
        available_capacity: 100.0,
        utilization_percentage: 50.0,
        predicted_usage: 55.0,
        historical_usage: std::collections::VecDeque::new(),
        last_updated: 1234567890,
    });
    
    let node_load = neuroplex::coordination::load_balancing::NodeLoad {
        node_id,
        current_tasks: 5,
        max_tasks: 10,
        cpu_usage: 0.5,
        memory_usage: 0.4,
        network_usage: 0.3,
        response_time: 100.0,
        throughput: 50.0,
        error_rate: 0.01,
        quality_score: 0.8,
        resource_metrics,
        predicted_load: 0.6,
        health_status: 0.9,
        last_updated: 1234567890,
    };
    
    balancer.update_node_load(node_id, node_load).await.unwrap();
    
    // Test task assignment
    let mut resource_requirements = HashMap::new();
    resource_requirements.insert(ResourceType::CPU, 10.0);
    resource_requirements.insert(ResourceType::Memory, 512.0);
    
    let task = TaskAssignmentRequest {
        task_id: Uuid::new_v4(),
        task_type: "computation".to_string(),
        priority: TaskPriority::High,
        resource_requirements,
        estimated_duration: 5000,
        deadline: Some(1234567900),
        dependencies: Vec::new(),
        constraints: Vec::new(),
        preferences: neuroplex::coordination::load_balancing::AssignmentPreferences {
            preferred_nodes: Vec::new(),
            avoid_nodes: Vec::new(),
            load_balancing_weight: 1.0,
            locality_preference: 0.5,
            performance_preference: 0.8,
            cost_preference: 0.3,
        },
    };
    
    let assignment_result = balancer.submit_task(task).await.unwrap();
    assert_eq!(assignment_result.assigned_node, node_id);
    
    // Check statistics
    let stats = balancer.get_stats().await;
    assert_eq!(stats.total_assignments, 1);
    assert_eq!(stats.successful_assignments, 1);
    
    balancer.stop().await.unwrap();
}

/// Test swarm intelligence engine functionality
#[tokio::test]
async fn test_swarm_intelligence_engine() {
    let config = SwarmIntelligenceConfig::default();
    
    // Create and start swarm intelligence engine
    let engine = SwarmIntelligenceEngine::new(config.clone()).unwrap();
    engine.start().await.unwrap();
    
    // Test agent addition
    let agent = SwarmAgent {
        agent_id: Uuid::new_v4(),
        node_id: Uuid::new_v4(),
        agent_type: AgentType::Worker,
        position: neuroplex::coordination::swarm_intelligence::Position {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            dimension: 3,
        },
        velocity: neuroplex::coordination::swarm_intelligence::Velocity {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        },
        state: neuroplex::coordination::swarm_intelligence::AgentState::Idle,
        capabilities: vec!["coordination".to_string()],
        fitness: 0.5,
        energy: 100.0,
        age: 0,
        experience: neuroplex::coordination::swarm_intelligence::Experience {
            total_tasks: 0,
            successful_tasks: 0,
            failed_tasks: 0,
            average_performance: 0.0,
            specialization_level: 0.0,
            collaboration_score: 0.0,
            learning_efficiency: 0.0,
            adaptation_rate: 0.0,
        },
        social_connections: Vec::new(),
        pheromone_trail: neuroplex::coordination::swarm_intelligence::PheromoneTrail {
            trail_id: Uuid::new_v4(),
            pheromone_type: neuroplex::coordination::swarm_intelligence::PheromoneType::Attraction,
            intensity: 1.0,
            decay_rate: 0.1,
            last_updated: 0,
            trail_points: Vec::new(),
        },
        memory: neuroplex::coordination::swarm_intelligence::AgentMemory {
            short_term: HashMap::new(),
            long_term: HashMap::new(),
            episodic: std::collections::VecDeque::new(),
            semantic: HashMap::new(),
            procedural: HashMap::new(),
            working_memory_capacity: 10,
            memory_decay_rate: 0.05,
        },
        behavior_rules: Vec::new(),
    };
    
    engine.add_agent(agent).await.unwrap();
    
    // Test collective decision making
    let proposal = neuroplex::coordination::swarm_intelligence::Proposal {
        proposal_id: Uuid::new_v4(),
        proposer: Uuid::new_v4(),
        proposal_type: neuroplex::coordination::swarm_intelligence::DecisionType::ResourceAllocation,
        content: b"test proposal".to_vec(),
        confidence: 0.8,
        expected_outcome: neuroplex::coordination::swarm_intelligence::Outcome {
            outcome_id: Uuid::new_v4(),
            outcome_type: neuroplex::coordination::swarm_intelligence::OutcomeType::ResourceGain,
            value: 10.0,
            timestamp: 1234567890,
            success: true,
            side_effects: Vec::new(),
        },
        resource_requirements: HashMap::new(),
        risk_assessment: neuroplex::coordination::swarm_intelligence::RiskAssessment {
            overall_risk: 0.2,
            risk_factors: Vec::new(),
            mitigation_strategies: Vec::new(),
            uncertainty_level: 0.1,
        },
    };
    
    engine.submit_proposal(proposal.clone()).await.unwrap();
    
    // Test voting
    let vote = neuroplex::coordination::swarm_intelligence::Vote {
        voter: Uuid::new_v4(),
        proposal_id: proposal.proposal_id,
        vote_value: 0.8,
        confidence: 0.9,
        reasoning: b"positive vote".to_vec(),
        timestamp: 1234567891,
    };
    
    engine.cast_vote(vote).await.unwrap();
    
    // Wait for processing
    sleep(Duration::from_millis(100)).await;
    
    // Check swarm statistics
    let stats = engine.get_swarm_stats().await;
    assert_eq!(stats.total_agents, 1);
    assert_eq!(stats.active_agents, 0); // Agent is idle
    
    engine.stop().await.unwrap();
}

/// Test integrated swarm coordinator functionality
#[tokio::test]
async fn test_integrated_swarm_coordinator() {
    let config = CoordinationConfig::default();
    let node_id = Uuid::new_v4();
    
    // Create and initialize swarm coordinator
    let mut coordinator = SwarmCoordinator::new(node_id, config).unwrap();
    coordinator.initialize().await.unwrap();
    coordinator.start().await.unwrap();
    
    // Test agent addition
    let agent_id = Uuid::new_v4();
    coordinator.add_agent(agent_id).await.unwrap();
    
    // Check agent list
    let agents = coordinator.get_agents();
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0], agent_id);
    
    // Test strategy change
    coordinator.set_strategy(CoordinationStrategy::NeuralConsensus);
    assert_eq!(coordinator.get_strategy(), &CoordinationStrategy::NeuralConsensus);
    
    // Test statistics
    let stats = coordinator.get_stats().await.unwrap();
    assert_eq!(stats.active_agents, 1);
    assert_eq!(stats.coordination_id, coordinator.get_coordination_id());
    
    // Test agent removal
    coordinator.remove_agent(agent_id).await.unwrap();
    let agents = coordinator.get_agents();
    assert_eq!(agents.len(), 0);
    
    coordinator.stop().await.unwrap();
}

/// Test fault tolerance under stress conditions
#[tokio::test]
async fn test_fault_tolerance_stress() {
    let bft_config = ByzantineFaultToleranceConfig {
        max_byzantine_nodes: 2,
        min_confirmations: 3,
        detection_threshold: 0.7,
        quarantine_duration: 10000,
        evidence_retention: 60000,
        reputation_decay: 0.95,
        recovery_interval: 5000,
    };
    let recovery_config = RecoveryConfig::default();
    let node_id = Uuid::new_v4();
    
    let engine = FaultToleranceEngine::new(bft_config.clone(), recovery_config.clone(), node_id).unwrap();
    engine.start().await.unwrap();
    
    // Test multiple Byzantine evidence reports
    for i in 0..5 {
        let evidence = neuroplex::coordination::fault_tolerance::ByzantineEvidence {
            evidence_id: Uuid::new_v4(),
            suspect_node: Uuid::new_v4(),
            reporter: node_id,
            evidence_type: neuroplex::coordination::fault_tolerance::EvidenceType::DoubleVoting,
            timestamp: 1234567890 + i,
            severity: neuroplex::coordination::fault_tolerance::Severity::High,
            proof: vec![1, 2, 3, i as u8],
            witnesses: Vec::new(),
            confidence: 0.9,
        };
        
        engine.report_byzantine_evidence(evidence).await.unwrap();
    }
    
    // Test circuit breaker under load
    let circuit_config = neuroplex::coordination::fault_tolerance::CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout_duration: 1000,
        half_open_retry_interval: 500,
        max_request_rate: 10.0,
        sliding_window_size: 20,
    };
    
    engine.create_circuit_breaker("stress_test".to_string(), circuit_config).await.unwrap();
    
    // Test multiple checkpoints
    for i in 0..3 {
        let checkpoint_data = format!("checkpoint_{}", i).into_bytes();
        let checkpoint_id = engine.create_checkpoint(checkpoint_data).await.unwrap();
        assert_ne!(checkpoint_id, Uuid::nil());
    }
    
    engine.stop().await.unwrap();
}

/// Test load balancer with multiple nodes and tasks
#[tokio::test]
async fn test_load_balancer_multiple_nodes() {
    let config = LoadBalancingConfig {
        strategy: LoadBalancingStrategy::PredictiveML,
        rebalancing_threshold: 0.3,
        rebalancing_interval: 1000,
        prediction_window_size: 50,
        learning_rate: 0.02,
        max_task_queue_size: 500,
        monitoring_interval: 500,
        load_smoothing_factor: 0.8,
    };
    
    let balancer = DynamicLoadBalancer::new(config.clone()).unwrap();
    balancer.start().await.unwrap();
    
    // Add multiple nodes
    let node_ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
    
    for (i, &node_id) in node_ids.iter().enumerate() {
        let mut resource_metrics = HashMap::new();
        resource_metrics.insert(ResourceType::CPU, neuroplex::coordination::load_balancing::ResourceMetrics {
            resource_type: ResourceType::CPU,
            current_usage: 20.0 + i as f64 * 10.0,
            available_capacity: 100.0,
            utilization_percentage: 20.0 + i as f64 * 10.0,
            predicted_usage: 25.0 + i as f64 * 10.0,
            historical_usage: std::collections::VecDeque::new(),
            last_updated: 1234567890,
        });
        
        let node_load = neuroplex::coordination::load_balancing::NodeLoad {
            node_id,
            current_tasks: i,
            max_tasks: 20,
            cpu_usage: 0.2 + i as f64 * 0.1,
            memory_usage: 0.3 + i as f64 * 0.1,
            network_usage: 0.1 + i as f64 * 0.05,
            response_time: 50.0 + i as f64 * 20.0,
            throughput: 100.0 - i as f64 * 10.0,
            error_rate: 0.01 + i as f64 * 0.005,
            quality_score: 0.9 - i as f64 * 0.1,
            resource_metrics,
            predicted_load: 0.3 + i as f64 * 0.1,
            health_status: 0.95 - i as f64 * 0.05,
            last_updated: 1234567890,
        };
        
        balancer.update_node_load(node_id, node_load).await.unwrap();
    }
    
    // Submit multiple tasks
    for i in 0..10 {
        let mut resource_requirements = HashMap::new();
        resource_requirements.insert(ResourceType::CPU, 5.0 + i as f64);
        resource_requirements.insert(ResourceType::Memory, 256.0 + i as f64 * 64.0);
        
        let task = TaskAssignmentRequest {
            task_id: Uuid::new_v4(),
            task_type: format!("task_{}", i),
            priority: if i % 2 == 0 { TaskPriority::High } else { TaskPriority::Medium },
            resource_requirements,
            estimated_duration: 1000 + i as u64 * 500,
            deadline: Some(1234567890 + i as u64 * 1000),
            dependencies: Vec::new(),
            constraints: Vec::new(),
            preferences: neuroplex::coordination::load_balancing::AssignmentPreferences {
                preferred_nodes: Vec::new(),
                avoid_nodes: Vec::new(),
                load_balancing_weight: 1.0,
                locality_preference: 0.5,
                performance_preference: 0.8,
                cost_preference: 0.3,
            },
        };
        
        let assignment_result = balancer.submit_task(task).await.unwrap();
        assert!(node_ids.contains(&assignment_result.assigned_node));
    }
    
    // Check final statistics
    let stats = balancer.get_stats().await;
    assert_eq!(stats.total_assignments, 10);
    assert_eq!(stats.successful_assignments, 10);
    
    balancer.stop().await.unwrap();
}

/// Test neural consensus with multiple agents
#[tokio::test]
async fn test_neural_consensus_multiple_agents() {
    let config = NeuralConsensusConfig {
        learning_rate: 0.02,
        confidence_threshold: 0.7,
        decision_window: 2000,
        neural_depth: 4,
        adaptive_timeout_base: 200,
        max_retries: 5,
        consensus_group_size: 3,
    };
    
    let node_id = Uuid::new_v4();
    let engine = NeuralConsensusEngine::new(config.clone(), node_id).unwrap();
    engine.start().await.unwrap();
    
    // Create a complex proposal
    let proposal = NeuralProposal {
        id: Uuid::new_v4(),
        proposer: node_id,
        timestamp: 1234567890,
        decision_type: DecisionType::ResourceAllocation,
        data: b"complex resource allocation proposal".to_vec(),
        confidence: 0.85,
        neural_weights: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        deadline: Some(1234567890 + 5000),
        priority: Priority::High,
    };
    
    engine.submit_proposal(proposal.clone()).await.unwrap();
    
    // Submit multiple votes from different agents
    let voter_ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
    
    for (i, &voter_id) in voter_ids.iter().enumerate() {
        let vote = NeuralVote {
            proposal_id: proposal.id,
            voter: voter_id,
            timestamp: 1234567890 + i as u64,
            decision: i % 3 != 0, // 2/3 positive votes
            confidence: 0.8 + i as f64 * 0.02,
            reasoning: vec![0.7 + i as f64 * 0.05, 0.6 + i as f64 * 0.04, 0.5 + i as f64 * 0.03],
            learning_feedback: Some(neuroplex::coordination::neural_consensus::LearningFeedback {
                outcome_quality: 0.8 + i as f64 * 0.02,
                execution_time: 1000 + i as u64 * 100,
                resource_usage: 0.5 + i as f64 * 0.1,
                success_rate: 0.9 + i as f64 * 0.01,
                adaptation_suggestions: Vec::new(),
            }),
        };
        
        engine.submit_vote(vote).await.unwrap();
    }
    
    // Wait for consensus processing
    sleep(Duration::from_millis(200)).await;
    
    // Check final state
    let state = engine.get_state().await.unwrap();
    assert!(state.active_votes.len() > 0);
    
    engine.stop().await.unwrap();
}

/// Test swarm emergent behavior patterns
#[tokio::test]
async fn test_swarm_emergent_behavior() {
    let config = SwarmIntelligenceConfig {
        topology: neuroplex::coordination::swarm_intelligence::SwarmTopology::SmallWorld,
        max_swarm_size: 20,
        min_consensus_size: 5,
        pheromone_decay_rate: 0.05,
        exploration_factor: 0.3,
        learning_rate: 0.03,
        convergence_threshold: 0.9,
        communication_range: 15.0,
        stigmergy_strength: 1.2,
    };
    
    let engine = SwarmIntelligenceEngine::new(config.clone()).unwrap();
    engine.start().await.unwrap();
    
    // Add multiple agents with different types
    let agent_types = vec![
        AgentType::Worker,
        AgentType::Scout,
        AgentType::Forager,
        AgentType::Coordinator,
        AgentType::Specialist("optimizer".to_string()),
    ];
    
    for (i, agent_type) in agent_types.iter().enumerate() {
        let agent = SwarmAgent {
            agent_id: Uuid::new_v4(),
            node_id: Uuid::new_v4(),
            agent_type: agent_type.clone(),
            position: neuroplex::coordination::swarm_intelligence::Position {
                x: i as f64 * 2.0,
                y: i as f64 * 1.5,
                z: 0.0,
                dimension: 3,
            },
            velocity: neuroplex::coordination::swarm_intelligence::Velocity {
                vx: 0.1 + i as f64 * 0.05,
                vy: 0.2 + i as f64 * 0.03,
                vz: 0.0,
            },
            state: match i % 3 {
                0 => neuroplex::coordination::swarm_intelligence::AgentState::Exploring,
                1 => neuroplex::coordination::swarm_intelligence::AgentState::Foraging,
                _ => neuroplex::coordination::swarm_intelligence::AgentState::Collaborating,
            },
            capabilities: vec![
                format!("capability_{}", i),
                "coordination".to_string(),
                "communication".to_string(),
            ],
            fitness: 0.6 + i as f64 * 0.08,
            energy: 80.0 + i as f64 * 4.0,
            age: i as u64 * 10,
            experience: neuroplex::coordination::swarm_intelligence::Experience {
                total_tasks: i * 2,
                successful_tasks: i * 2 - (i / 3),
                failed_tasks: i / 3,
                average_performance: 0.7 + i as f64 * 0.05,
                specialization_level: 0.5 + i as f64 * 0.1,
                collaboration_score: 0.6 + i as f64 * 0.08,
                learning_efficiency: 0.8 + i as f64 * 0.03,
                adaptation_rate: 0.4 + i as f64 * 0.1,
            },
            social_connections: Vec::new(),
            pheromone_trail: neuroplex::coordination::swarm_intelligence::PheromoneTrail {
                trail_id: Uuid::new_v4(),
                pheromone_type: match i % 3 {
                    0 => neuroplex::coordination::swarm_intelligence::PheromoneType::Attraction,
                    1 => neuroplex::coordination::swarm_intelligence::PheromoneType::Information,
                    _ => neuroplex::coordination::swarm_intelligence::PheromoneType::Path,
                },
                intensity: 0.8 + i as f64 * 0.04,
                decay_rate: 0.05 + i as f64 * 0.01,
                last_updated: 1234567890 + i as u64,
                trail_points: Vec::new(),
            },
            memory: neuroplex::coordination::swarm_intelligence::AgentMemory {
                short_term: HashMap::new(),
                long_term: HashMap::new(),
                episodic: std::collections::VecDeque::new(),
                semantic: HashMap::new(),
                procedural: HashMap::new(),
                working_memory_capacity: 8 + i * 2,
                memory_decay_rate: 0.03 + i as f64 * 0.01,
            },
            behavior_rules: Vec::new(),
        };
        
        engine.add_agent(agent).await.unwrap();
    }
    
    // Wait for agents to interact
    sleep(Duration::from_millis(300)).await;
    
    // Check swarm statistics
    let stats = engine.get_swarm_stats().await;
    assert_eq!(stats.total_agents, 5);
    assert!(stats.average_fitness > 0.5);
    assert!(stats.swarm_cohesion > 0.0);
    
    engine.stop().await.unwrap();
}

/// Performance benchmark test
#[tokio::test]
async fn test_coordination_performance_benchmark() {
    let config = CoordinationConfig {
        strategy: CoordinationStrategy::HybridOptimized,
        enable_neural_consensus: true,
        enable_real_time_messaging: true,
        enable_fault_tolerance: true,
        enable_load_balancing: true,
        enable_swarm_intelligence: true,
        neural_consensus_config: Some(NeuralConsensusConfig {
            learning_rate: 0.01,
            confidence_threshold: 0.8,
            decision_window: 1000,
            neural_depth: 3,
            adaptive_timeout_base: 150,
            max_retries: 3,
            consensus_group_size: 5,
        }),
        messaging_config: Some(RealTimeMessagingConfig {
            websocket_port: 8080,
            max_message_size: 1024 * 1024,
            connection_timeout: 30000,
            heartbeat_interval: 5000,
            max_retries: 3,
            queue_capacity: 10000,
            priority_levels: 5,
            deadline_enforcement: true,
        }),
        fault_tolerance_config: Some((
            ByzantineFaultToleranceConfig::default(),
            RecoveryConfig::default(),
        )),
        load_balancing_config: Some(LoadBalancingConfig {
            strategy: LoadBalancingStrategy::HybridOptimized,
            rebalancing_threshold: 0.2,
            rebalancing_interval: 5000,
            prediction_window_size: 100,
            learning_rate: 0.01,
            max_task_queue_size: 1000,
            monitoring_interval: 1000,
            load_smoothing_factor: 0.9,
        }),
        swarm_intelligence_config: Some(SwarmIntelligenceConfig::default()),
    };
    
    let node_id = Uuid::new_v4();
    let mut coordinator = SwarmCoordinator::new(node_id, config).unwrap();
    
    // Measure initialization time
    let start_time = std::time::Instant::now();
    coordinator.initialize().await.unwrap();
    let init_duration = start_time.elapsed();
    
    // Measure startup time
    let start_time = std::time::Instant::now();
    coordinator.start().await.unwrap();
    let startup_duration = start_time.elapsed();
    
    // Add multiple agents and measure scaling
    let agent_count = 10;
    let start_time = std::time::Instant::now();
    
    for i in 0..agent_count {
        let agent_id = Uuid::new_v4();
        coordinator.add_agent(agent_id).await.unwrap();
    }
    
    let agent_addition_duration = start_time.elapsed();
    
    // Measure statistics collection time
    let start_time = std::time::Instant::now();
    let stats = coordinator.get_stats().await.unwrap();
    let stats_duration = start_time.elapsed();
    
    // Measure shutdown time
    let start_time = std::time::Instant::now();
    coordinator.stop().await.unwrap();
    let shutdown_duration = start_time.elapsed();
    
    // Performance assertions
    assert!(init_duration.as_millis() < 1000, "Initialization took too long: {:?}", init_duration);
    assert!(startup_duration.as_millis() < 2000, "Startup took too long: {:?}", startup_duration);
    assert!(agent_addition_duration.as_millis() < 1000, "Agent addition took too long: {:?}", agent_addition_duration);
    assert!(stats_duration.as_millis() < 100, "Statistics collection took too long: {:?}", stats_duration);
    assert!(shutdown_duration.as_millis() < 1000, "Shutdown took too long: {:?}", shutdown_duration);
    
    // Verify final state
    assert_eq!(stats.active_agents, agent_count);
    assert_eq!(stats.coordination_id, coordinator.get_coordination_id());
    
    println!("Performance Benchmark Results:");
    println!("  Initialization: {:?}", init_duration);
    println!("  Startup: {:?}", startup_duration);
    println!("  Agent Addition ({}): {:?}", agent_count, agent_addition_duration);
    println!("  Statistics Collection: {:?}", stats_duration);
    println!("  Shutdown: {:?}", shutdown_duration);
}