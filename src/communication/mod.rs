// Communication module - Inter-agent communication protocols

use serde::{Deserialize, Serialize};
use crate::agents::AgentId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: uuid::Uuid,
    pub from: AgentId,
    pub to: Option<AgentId>, // None for broadcast
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    TaskAssignment,
    StatusUpdate,
    NeuralUpdate,
    Coordination,
    Heartbeat,
}

pub trait CommunicationProtocol: Send + Sync {
    // Communication methods
}