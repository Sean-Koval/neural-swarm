//! Agent Orchestration Integration
//!
//! Capability-based agent selection and load balancing for neural swarm.

use super::{Integration, IntegrationInfo, IntegrationEvent, IntegrationStatus};
use crate::{Result, NeuroError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Agent orchestration integration implementation
pub struct AgentOrchestrationIntegration {
    info: IntegrationInfo,
    status: IntegrationStatus,
    config: OrchestrationConfig,
    agents: HashMap<Uuid, AgentInfo>,
    task_queue: Vec<TaskAssignment>,
    load_balancer: LoadBalancer,
    failure_detector: FailureDetector,
}

/// Orchestration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    /// Task assignment strategy
    pub assignment_strategy: AssignmentStrategy,
    /// Load balancing algorithm
    pub load_balancing: LoadBalancingAlgorithm,
    /// Health check interval in seconds
    pub health_check_interval: u64,
    /// Task timeout in seconds
    pub task_timeout: u64,
    /// Retry attempts for failed tasks
    pub retry_attempts: u32,
}

/// Task assignment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignmentStrategy {
    Random,
    RoundRobin,
    CapabilityBased,
    LoadBased,
    OptimalMatching,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    LeastConnections,
    WeightedRoundRobin,
    LeastResponseTime,
    ResourceBased,
}

/// Agent information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent ID
    pub id: Uuid,
    /// Agent name
    pub name: String,
    /// Agent capabilities
    pub capabilities: Vec<String>,
    /// Resource capacity
    pub capacity: ResourceCapacity,
    /// Current load
    pub current_load: ResourceLoad,
    /// Performance metrics
    pub metrics: AgentMetrics,
    /// Status
    pub status: AgentStatus,
    /// Last heartbeat
    pub last_heartbeat: u64,
}

/// Resource capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapacity {
    /// CPU cores
    pub cpu_cores: f32,
    /// Memory in MB
    pub memory_mb: u32,
    /// Network bandwidth in Mbps
    pub network_mbps: u32,
    /// GPU memory in MB
    pub gpu_memory_mb: u32,
    /// Storage in GB
    pub storage_gb: u32,
}

/// Current resource load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLoad {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f32,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f32,
    /// Network utilization (0.0 to 1.0)
    pub network_utilization: f32,
    /// GPU utilization (0.0 to 1.0)
    pub gpu_utilization: f32,
    /// Active tasks
    pub active_tasks: usize,
    /// Queue length
    pub queue_length: usize,
}

/// Agent performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total tasks failed
    pub tasks_failed: u64,
    /// Average response time in ms
    pub avg_response_time: f32,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Uptime in seconds
    pub uptime: u64,
    /// Last updated
    pub last_updated: u64,
}

/// Agent status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Available,
    Busy,
    Offline,
    Maintenance,
    Failed,
}

/// Task assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAssignment {
    /// Task ID
    pub task_id: Uuid,
    /// Agent ID
    pub agent_id: Option<Uuid>,
    /// Task data
    pub task_data: Vec<u8>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Priority
    pub priority: TaskPriority,
    /// Creation time
    pub created_at: u64,
    /// Assignment time
    pub assigned_at: Option<u64>,
    /// Deadline
    pub deadline: Option<u64>,
    /// Retry count
    pub retry_count: u32,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores needed
    pub cpu_cores: f32,
    /// Memory needed in MB
    pub memory_mb: u32,
    /// Network bandwidth needed in Mbps
    pub network_mbps: u32,
    /// GPU memory needed in MB
    pub gpu_memory_mb: u32,
    /// Estimated execution time in seconds
    pub execution_time_s: u32,
}

/// Task priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Load balancer
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    algorithm: LoadBalancingAlgorithm,
    agent_weights: HashMap<Uuid, f32>,
    last_assigned: Option<Uuid>,
}

/// Failure detector
#[derive(Debug, Clone)]
pub struct FailureDetector {
    failure_threshold: u32,
    failure_counts: HashMap<Uuid, u32>,
    last_check: u64,
}

impl AgentOrchestrationIntegration {
    /// Create a new agent orchestration integration
    pub fn new(info: IntegrationInfo) -> Self {
        Self {
            info,
            status: IntegrationStatus::Initializing,
            config: OrchestrationConfig {
                max_agents: 100,
                assignment_strategy: AssignmentStrategy::CapabilityBased,
                load_balancing: LoadBalancingAlgorithm::LeastConnections,
                health_check_interval: 30,
                task_timeout: 300,
                retry_attempts: 3,
            },
            agents: HashMap::new(),
            task_queue: Vec::new(),
            load_balancer: LoadBalancer {
                algorithm: LoadBalancingAlgorithm::LeastConnections,
                agent_weights: HashMap::new(),
                last_assigned: None,
            },
            failure_detector: FailureDetector {
                failure_threshold: 3,
                failure_counts: HashMap::new(),
                last_check: 0,
            },
        }
    }
    
    /// Register an agent
    pub fn register_agent(&mut self, agent: AgentInfo) -> Result<()> {
        if self.agents.len() >= self.config.max_agents {
            return Err(NeuroError::integration("Maximum agents reached"));
        }
        
        self.agents.insert(agent.id, agent.clone());
        self.load_balancer.agent_weights.insert(agent.id, 1.0);
        
        Ok(())
    }
    
    /// Unregister an agent
    pub fn unregister_agent(&mut self, agent_id: Uuid) -> Result<()> {
        self.agents.remove(&agent_id);
        self.load_balancer.agent_weights.remove(&agent_id);
        self.failure_detector.failure_counts.remove(&agent_id);
        
        // Reassign tasks from this agent
        self.reassign_tasks_from_agent(agent_id)?;
        
        Ok(())
    }
    
    /// Assign task to agent
    pub fn assign_task(&mut self, mut task: TaskAssignment) -> Result<Uuid> {
        // Select agent based on strategy
        let agent_id = self.select_agent(&task)?;
        
        task.agent_id = Some(agent_id);
        task.assigned_at = Some(self.current_timestamp());
        
        // Update agent load
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.current_load.active_tasks += 1;
            agent.current_load.queue_length += 1;
        }
        
        self.task_queue.push(task.clone());
        
        Ok(agent_id)
    }
    
    /// Complete task
    pub fn complete_task(&mut self, task_id: Uuid, agent_id: Uuid, success: bool) -> Result<()> {
        // Remove task from queue
        self.task_queue.retain(|task| task.task_id != task_id);
        
        // Update agent metrics
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.current_load.active_tasks = agent.current_load.active_tasks.saturating_sub(1);
            agent.current_load.queue_length = agent.current_load.queue_length.saturating_sub(1);
            
            if success {
                agent.metrics.tasks_completed += 1;
            } else {
                agent.metrics.tasks_failed += 1;
            }
            
            // Update success rate
            let total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed;
            if total_tasks > 0 {
                agent.metrics.success_rate = agent.metrics.tasks_completed as f32 / total_tasks as f32;
            }
        }
        
        Ok(())
    }
    
    /// Update agent heartbeat
    pub fn update_heartbeat(&mut self, agent_id: Uuid) -> Result<()> {
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.last_heartbeat = self.current_timestamp();
            agent.status = AgentStatus::Available;
        }
        
        Ok(())
    }
    
    /// Get agent statistics
    pub fn get_agent_stats(&self, agent_id: Uuid) -> Result<AgentInfo> {
        self.agents.get(&agent_id)
            .cloned()
            .ok_or_else(|| NeuroError::integration("Agent not found"))
    }
    
    /// Get orchestration statistics
    pub fn get_orchestration_stats(&self) -> OrchestrationStats {
        let mut stats = OrchestrationStats {
            total_agents: self.agents.len(),
            available_agents: 0,
            busy_agents: 0,
            offline_agents: 0,
            total_tasks: self.task_queue.len(),
            pending_tasks: 0,
            assigned_tasks: 0,
            average_load: 0.0,
            success_rate: 0.0,
        };
        
        let mut total_load = 0.0;
        let mut total_completed = 0u64;
        let mut total_failed = 0u64;
        
        for agent in self.agents.values() {
            match agent.status {
                AgentStatus::Available => stats.available_agents += 1,
                AgentStatus::Busy => stats.busy_agents += 1,
                AgentStatus::Offline => stats.offline_agents += 1,
                _ => {}
            }
            
            total_load += agent.current_load.cpu_utilization;
            total_completed += agent.metrics.tasks_completed;
            total_failed += agent.metrics.tasks_failed;
        }
        
        for task in &self.task_queue {
            if task.agent_id.is_some() {
                stats.assigned_tasks += 1;
            } else {
                stats.pending_tasks += 1;
            }
        }
        
        if !self.agents.is_empty() {
            stats.average_load = total_load / self.agents.len() as f32;
        }
        
        let total_tasks = total_completed + total_failed;
        if total_tasks > 0 {
            stats.success_rate = total_completed as f32 / total_tasks as f32;
        }
        
        stats
    }
    
    /// Select agent for task assignment
    fn select_agent(&self, task: &TaskAssignment) -> Result<Uuid> {
        let eligible_agents = self.find_eligible_agents(task)?;
        
        if eligible_agents.is_empty() {
            return Err(NeuroError::integration("No eligible agents available"));
        }
        
        let selected_agent = match self.config.assignment_strategy {
            AssignmentStrategy::Random => {
                let index = fastrand::usize(..eligible_agents.len());
                eligible_agents[index]
            }
            AssignmentStrategy::RoundRobin => {
                self.select_round_robin(&eligible_agents)
            }
            AssignmentStrategy::CapabilityBased => {
                self.select_capability_based(task, &eligible_agents)
            }
            AssignmentStrategy::LoadBased => {
                self.select_load_based(&eligible_agents)
            }
            AssignmentStrategy::OptimalMatching => {
                self.select_optimal_matching(task, &eligible_agents)
            }
        };
        
        Ok(selected_agent)
    }
    
    /// Find eligible agents for task
    fn find_eligible_agents(&self, task: &TaskAssignment) -> Result<Vec<Uuid>> {
        let mut eligible = Vec::new();
        
        for (agent_id, agent) in &self.agents {
            // Check if agent is available
            if agent.status != AgentStatus::Available {
                continue;
            }
            
            // Check capabilities
            if !task.required_capabilities.iter().all(|cap| agent.capabilities.contains(cap)) {
                continue;
            }
            
            // Check resource requirements
            if !self.check_resource_requirements(agent, &task.resource_requirements) {
                continue;
            }
            
            eligible.push(*agent_id);
        }
        
        Ok(eligible)
    }
    
    /// Check if agent meets resource requirements
    fn check_resource_requirements(&self, agent: &AgentInfo, requirements: &ResourceRequirements) -> bool {
        // Check CPU
        let available_cpu = agent.capacity.cpu_cores * (1.0 - agent.current_load.cpu_utilization);
        if available_cpu < requirements.cpu_cores {
            return false;
        }
        
        // Check memory
        let available_memory = (agent.capacity.memory_mb as f32 * (1.0 - agent.current_load.memory_utilization)) as u32;
        if available_memory < requirements.memory_mb {
            return false;
        }
        
        // Check network
        let available_network = (agent.capacity.network_mbps as f32 * (1.0 - agent.current_load.network_utilization)) as u32;
        if available_network < requirements.network_mbps {
            return false;
        }
        
        // Check GPU memory
        let available_gpu = (agent.capacity.gpu_memory_mb as f32 * (1.0 - agent.current_load.gpu_utilization)) as u32;
        if available_gpu < requirements.gpu_memory_mb {
            return false;
        }
        
        true
    }
    
    /// Select agent using round-robin
    fn select_round_robin(&self, eligible: &[Uuid]) -> Uuid {
        if let Some(last_assigned) = self.load_balancer.last_assigned {
            if let Some(pos) = eligible.iter().position(|&id| id == last_assigned) {
                let next_pos = (pos + 1) % eligible.len();
                return eligible[next_pos];
            }
        }
        
        eligible[0]
    }
    
    /// Select agent based on capabilities
    fn select_capability_based(&self, task: &TaskAssignment, eligible: &[Uuid]) -> Uuid {
        let mut best_agent = eligible[0];
        let mut best_score = 0.0;
        
        for &agent_id in eligible {
            if let Some(agent) = self.agents.get(&agent_id) {
                // Calculate capability match score
                let capability_score = task.required_capabilities.iter()
                    .map(|cap| if agent.capabilities.contains(cap) { 1.0 } else { 0.0 })
                    .sum::<f32>() / task.required_capabilities.len() as f32;
                
                // Factor in agent performance
                let performance_score = agent.metrics.success_rate;
                
                let total_score = capability_score * 0.7 + performance_score * 0.3;
                
                if total_score > best_score {
                    best_score = total_score;
                    best_agent = agent_id;
                }
            }
        }
        
        best_agent
    }
    
    /// Select agent based on load
    fn select_load_based(&self, eligible: &[Uuid]) -> Uuid {
        let mut best_agent = eligible[0];
        let mut lowest_load = f32::MAX;
        
        for &agent_id in eligible {
            if let Some(agent) = self.agents.get(&agent_id) {
                let load = agent.current_load.cpu_utilization + 
                          agent.current_load.memory_utilization + 
                          agent.current_load.network_utilization;
                
                if load < lowest_load {
                    lowest_load = load;
                    best_agent = agent_id;
                }
            }
        }
        
        best_agent
    }
    
    /// Select agent using optimal matching
    fn select_optimal_matching(&self, task: &TaskAssignment, eligible: &[Uuid]) -> Uuid {
        // Simplified optimal matching - in practice would use Hungarian algorithm
        let mut best_agent = eligible[0];
        let mut best_score = 0.0;
        
        for &agent_id in eligible {
            if let Some(agent) = self.agents.get(&agent_id) {
                // Calculate composite score
                let capability_score = task.required_capabilities.iter()
                    .map(|cap| if agent.capabilities.contains(cap) { 1.0 } else { 0.0 })
                    .sum::<f32>() / task.required_capabilities.len().max(1) as f32;
                
                let load_score = 1.0 - (agent.current_load.cpu_utilization + 
                                       agent.current_load.memory_utilization) / 2.0;
                
                let performance_score = agent.metrics.success_rate;
                
                let total_score = capability_score * 0.4 + load_score * 0.3 + performance_score * 0.3;
                
                if total_score > best_score {
                    best_score = total_score;
                    best_agent = agent_id;
                }
            }
        }
        
        best_agent
    }
    
    /// Reassign tasks from failed agent
    fn reassign_tasks_from_agent(&mut self, failed_agent_id: Uuid) -> Result<()> {
        let tasks_to_reassign: Vec<TaskAssignment> = self.task_queue.iter()
            .filter(|task| task.agent_id == Some(failed_agent_id))
            .cloned()
            .collect();
        
        for mut task in tasks_to_reassign {
            task.agent_id = None;
            task.assigned_at = None;
            task.retry_count += 1;
            
            if task.retry_count <= self.config.retry_attempts {
                // Try to reassign
                if let Ok(new_agent_id) = self.select_agent(&task) {
                    task.agent_id = Some(new_agent_id);
                    task.assigned_at = Some(self.current_timestamp());
                }
            }
        }
        
        Ok(())
    }
    
    /// Get current timestamp
    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

/// Orchestration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationStats {
    pub total_agents: usize,
    pub available_agents: usize,
    pub busy_agents: usize,
    pub offline_agents: usize,
    pub total_tasks: usize,
    pub pending_tasks: usize,
    pub assigned_tasks: usize,
    pub average_load: f32,
    pub success_rate: f32,
}

impl Integration for AgentOrchestrationIntegration {
    fn initialize(&mut self, config: &serde_json::Value) -> Result<()> {
        // Parse configuration
        if let Ok(orchestration_config) = serde_json::from_value::<OrchestrationConfig>(config.clone()) {
            self.config = orchestration_config;
            self.load_balancer.algorithm = self.config.load_balancing.clone();
        }
        
        self.status = IntegrationStatus::Initializing;
        Ok(())
    }
    
    fn start(&mut self) -> Result<()> {
        self.status = IntegrationStatus::Running;
        Ok(())
    }
    
    fn stop(&mut self) -> Result<()> {
        self.agents.clear();
        self.task_queue.clear();
        
        self.status = IntegrationStatus::Stopped;
        Ok(())
    }
    
    fn info(&self) -> &IntegrationInfo {
        &self.info
    }
    
    fn handle_event(&mut self, event: IntegrationEvent) -> Result<()> {
        match event {
            IntegrationEvent::TaskAssigned { task_id, agent_id, task_data } => {
                self.update_heartbeat(agent_id)?;
            }
            IntegrationEvent::TaskCompleted { task_id, agent_id, result } => {
                self.complete_task(task_id, agent_id, true)?;
            }
            IntegrationEvent::AgentRegistered { agent_id, capabilities } => {
                let agent = AgentInfo {
                    id: agent_id,
                    name: format!("Agent-{}", agent_id),
                    capabilities,
                    capacity: ResourceCapacity {
                        cpu_cores: 4.0,
                        memory_mb: 8192,
                        network_mbps: 1000,
                        gpu_memory_mb: 2048,
                        storage_gb: 100,
                    },
                    current_load: ResourceLoad {
                        cpu_utilization: 0.0,
                        memory_utilization: 0.0,
                        network_utilization: 0.0,
                        gpu_utilization: 0.0,
                        active_tasks: 0,
                        queue_length: 0,
                    },
                    metrics: AgentMetrics {
                        tasks_completed: 0,
                        tasks_failed: 0,
                        avg_response_time: 0.0,
                        success_rate: 1.0,
                        uptime: 0,
                        last_updated: self.current_timestamp(),
                    },
                    status: AgentStatus::Available,
                    last_heartbeat: self.current_timestamp(),
                };
                
                self.register_agent(agent)?;
            }
            IntegrationEvent::AgentDeregistered { agent_id } => {
                self.unregister_agent(agent_id)?;
            }
            _ => {
                // Handle other events
            }
        }
        
        Ok(())
    }
    
    fn status(&self) -> IntegrationStatus {
        self.status.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_agent_orchestration_creation() {
        let info = IntegrationInfo {
            name: "agent_orchestration".to_string(),
            version: "1.0.0".to_string(),
            description: "Test integration".to_string(),
            capabilities: vec![],
            dependencies: vec![],
            config_schema: serde_json::json!({}),
        };
        
        let integration = AgentOrchestrationIntegration::new(info);
        assert_eq!(integration.status(), IntegrationStatus::Initializing);
    }
    
    #[test]
    fn test_agent_registration() {
        let info = IntegrationInfo {
            name: "agent_orchestration".to_string(),
            version: "1.0.0".to_string(),
            description: "Test integration".to_string(),
            capabilities: vec![],
            dependencies: vec![],
            config_schema: serde_json::json!({}),
        };
        
        let mut integration = AgentOrchestrationIntegration::new(info);
        
        let agent = AgentInfo {
            id: Uuid::new_v4(),
            name: "test_agent".to_string(),
            capabilities: vec!["task_processing".to_string()],
            capacity: ResourceCapacity {
                cpu_cores: 4.0,
                memory_mb: 8192,
                network_mbps: 1000,
                gpu_memory_mb: 0,
                storage_gb: 100,
            },
            current_load: ResourceLoad {
                cpu_utilization: 0.0,
                memory_utilization: 0.0,
                network_utilization: 0.0,
                gpu_utilization: 0.0,
                active_tasks: 0,
                queue_length: 0,
            },
            metrics: AgentMetrics {
                tasks_completed: 0,
                tasks_failed: 0,
                avg_response_time: 0.0,
                success_rate: 1.0,
                uptime: 0,
                last_updated: 0,
            },
            status: AgentStatus::Available,
            last_heartbeat: 0,
        };
        
        assert!(integration.register_agent(agent).is_ok());
        assert_eq!(integration.agents.len(), 1);
    }
}