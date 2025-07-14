// Agents module - Collaborative neural agents

use uuid::Uuid;
use serde::{Deserialize, Serialize};

pub type AgentId = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    pub id: AgentId,
    pub name: String,
    pub capabilities: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub memory_mb: u32,
    pub cpu_cores: f32,
    pub max_connections: u32,
}

pub trait Agent: Send + Sync {
    fn id(&self) -> AgentId;
    fn metadata(&self) -> &AgentMetadata;
}