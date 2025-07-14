//! Neural Swarm Integration Module
//!
//! Comprehensive integration patterns for the neural swarm ecosystem.
//! This module provides seamless integration between neural-comm, neuroplex,
//! FANN-rust-core, and agent orchestration components.

pub mod neural_comm;
pub mod neuroplex;
pub mod fann_core;
pub mod agent_orchestration;
pub mod swarm_coordination;
pub mod apis;

#[cfg(test)]
pub mod integration_test;

use crate::{Result, NeuroError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Integration registry for managing component integrations
#[derive(Debug, Clone)]
pub struct IntegrationRegistry {
    /// Registered integrations
    integrations: HashMap<String, IntegrationInfo>,
    /// Active integration instances
    active_integrations: HashMap<Uuid, Box<dyn Integration>>,
}

/// Integration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationInfo {
    /// Integration name
    pub name: String,
    /// Integration version
    pub version: String,
    /// Integration description
    pub description: String,
    /// Integration capabilities
    pub capabilities: Vec<String>,
    /// Integration dependencies
    pub dependencies: Vec<String>,
    /// Integration configuration schema
    pub config_schema: serde_json::Value,
}

/// Integration trait for all neural swarm integrations
pub trait Integration: Send + Sync {
    /// Initialize the integration
    fn initialize(&mut self, config: &serde_json::Value) -> Result<()>;
    
    /// Start the integration
    fn start(&mut self) -> Result<()>;
    
    /// Stop the integration
    fn stop(&mut self) -> Result<()>;
    
    /// Get integration info
    fn info(&self) -> &IntegrationInfo;
    
    /// Handle integration events
    fn handle_event(&mut self, event: IntegrationEvent) -> Result<()>;
    
    /// Get integration status
    fn status(&self) -> IntegrationStatus;
}

/// Integration event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationEvent {
    /// Task assignment event
    TaskAssigned {
        task_id: Uuid,
        agent_id: Uuid,
        task_data: Vec<u8>,
    },
    /// Task completion event
    TaskCompleted {
        task_id: Uuid,
        agent_id: Uuid,
        result: Vec<u8>,
    },
    /// Agent registration event
    AgentRegistered {
        agent_id: Uuid,
        capabilities: Vec<String>,
    },
    /// Agent deregistration event
    AgentDeregistered {
        agent_id: Uuid,
    },
    /// System configuration change
    ConfigurationChanged {
        component: String,
        changes: HashMap<String, serde_json::Value>,
    },
}

/// Integration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationStatus {
    /// Integration is initializing
    Initializing,
    /// Integration is running
    Running,
    /// Integration is stopped
    Stopped,
    /// Integration has failed
    Failed { error: String },
}

impl IntegrationRegistry {
    /// Create a new integration registry
    pub fn new() -> Self {
        Self {
            integrations: HashMap::new(),
            active_integrations: HashMap::new(),
        }
    }
    
    /// Register an integration
    pub fn register(&mut self, info: IntegrationInfo) -> Result<()> {
        // Validate dependencies
        for dep in &info.dependencies {
            if !self.integrations.contains_key(dep) {
                return Err(NeuroError::integration(format!(
                    "Missing dependency: {}", dep
                )));
            }
        }
        
        self.integrations.insert(info.name.clone(), info);
        Ok(())
    }
    
    /// Create an integration instance
    pub fn create_instance(&mut self, name: &str, config: &serde_json::Value) -> Result<Uuid> {
        let info = self.integrations.get(name)
            .ok_or_else(|| NeuroError::integration(format!("Integration not found: {}", name)))?;
        
        // Create integration instance based on type
        let mut integration: Box<dyn Integration> = match name {
            "neural_comm" => Box::new(neural_comm::NeuralCommIntegration::new(info.clone())),
            "neuroplex" => Box::new(neuroplex::NeuroplexIntegration::new(info.clone())),
            "fann_core" => Box::new(fann_core::FannCoreIntegration::new(info.clone())),
            "agent_orchestration" => Box::new(agent_orchestration::AgentOrchestrationIntegration::new(info.clone())),
            "swarm_coordination" => Box::new(swarm_coordination::SwarmCoordinationIntegration::new(info.clone())),
            _ => return Err(NeuroError::integration(format!("Unknown integration: {}", name))),
        };
        
        // Initialize integration
        integration.initialize(config)?;
        
        let instance_id = Uuid::new_v4();
        self.active_integrations.insert(instance_id, integration);
        
        Ok(instance_id)
    }
    
    /// Start an integration instance
    pub fn start_instance(&mut self, instance_id: Uuid) -> Result<()> {
        let integration = self.active_integrations.get_mut(&instance_id)
            .ok_or_else(|| NeuroError::integration("Integration instance not found"))?;
        
        integration.start()
    }
    
    /// Stop an integration instance
    pub fn stop_instance(&mut self, instance_id: Uuid) -> Result<()> {
        let integration = self.active_integrations.get_mut(&instance_id)
            .ok_or_else(|| NeuroError::integration("Integration instance not found"))?;
        
        integration.stop()
    }
    
    /// Send event to integration
    pub fn send_event(&mut self, instance_id: Uuid, event: IntegrationEvent) -> Result<()> {
        let integration = self.active_integrations.get_mut(&instance_id)
            .ok_or_else(|| NeuroError::integration("Integration instance not found"))?;
        
        integration.handle_event(event)
    }
    
    /// Get integration status
    pub fn get_status(&self, instance_id: Uuid) -> Result<IntegrationStatus> {
        let integration = self.active_integrations.get(&instance_id)
            .ok_or_else(|| NeuroError::integration("Integration instance not found"))?;
        
        Ok(integration.status())
    }
    
    /// List all registered integrations
    pub fn list_integrations(&self) -> Vec<&IntegrationInfo> {
        self.integrations.values().collect()
    }
    
    /// List all active instances
    pub fn list_instances(&self) -> Vec<(Uuid, IntegrationStatus)> {
        self.active_integrations.iter()
            .map(|(id, integration)| (*id, integration.status()))
            .collect()
    }
}

/// Integration builder for creating complex integration configurations
pub struct IntegrationBuilder {
    registry: IntegrationRegistry,
    config: HashMap<String, serde_json::Value>,
}

impl IntegrationBuilder {
    /// Create a new integration builder
    pub fn new() -> Self {
        Self {
            registry: IntegrationRegistry::new(),
            config: HashMap::new(),
        }
    }
    
    /// Add neural communication integration
    pub fn with_neural_comm(mut self, config: serde_json::Value) -> Self {
        self.config.insert("neural_comm".to_string(), config);
        self
    }
    
    /// Add neuroplex integration
    pub fn with_neuroplex(mut self, config: serde_json::Value) -> Self {
        self.config.insert("neuroplex".to_string(), config);
        self
    }
    
    /// Add FANN core integration
    pub fn with_fann_core(mut self, config: serde_json::Value) -> Self {
        self.config.insert("fann_core".to_string(), config);
        self
    }
    
    /// Add agent orchestration integration
    pub fn with_agent_orchestration(mut self, config: serde_json::Value) -> Self {
        self.config.insert("agent_orchestration".to_string(), config);
        self
    }
    
    /// Add swarm coordination integration
    pub fn with_swarm_coordination(mut self, config: serde_json::Value) -> Self {
        self.config.insert("swarm_coordination".to_string(), config);
        self
    }
    
    /// Build the integration configuration
    pub fn build(mut self) -> Result<IntegrationRegistry> {
        // Register all integrations
        for (name, config) in &self.config {
            let info = self.create_integration_info(name, config)?;
            self.registry.register(info)?;
        }
        
        Ok(self.registry)
    }
    
    /// Create integration info from configuration
    fn create_integration_info(&self, name: &str, config: &serde_json::Value) -> Result<IntegrationInfo> {
        let info = match name {
            "neural_comm" => IntegrationInfo {
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
            },
            "neuroplex" => IntegrationInfo {
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
            },
            "fann_core" => IntegrationInfo {
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
            },
            "agent_orchestration" => IntegrationInfo {
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
            },
            "swarm_coordination" => IntegrationInfo {
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
            },
            _ => return Err(NeuroError::integration(format!("Unknown integration: {}", name))),
        };
        
        Ok(info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_integration_registry_creation() {
        let registry = IntegrationRegistry::new();
        assert_eq!(registry.integrations.len(), 0);
        assert_eq!(registry.active_integrations.len(), 0);
    }
    
    #[test]
    fn test_integration_builder() {
        let builder = IntegrationBuilder::new()
            .with_neural_comm(serde_json::json!({"port": 8080}))
            .with_neuroplex(serde_json::json!({"cluster_size": 3}));
        
        let registry = builder.build().unwrap();
        assert_eq!(registry.integrations.len(), 2);
    }
}