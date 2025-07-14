//! Message protocols for neural communication

use crate::{
    error::{ProtocolError, Result},
    security,
    AgentId, MessageId,
};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use bytes::{Bytes, BytesMut, Buf, BufMut};
use zeroize::{Zeroize, ZeroizeOnDrop};

pub mod framing;
pub mod compression;
pub mod validation;

pub use framing::{FrameCodec, Frame, FrameType};
pub use compression::{Compressor, CompressionType};
pub use validation::{MessageValidator, ReplayProtection};

/// Protocol version
pub const PROTOCOL_VERSION: u32 = 1;

/// Message types for neural communication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MessageType {
    /// Handshake initiation
    HandshakeInit = 0x01,
    /// Handshake response
    HandshakeResponse = 0x02,
    /// Handshake completion
    HandshakeComplete = 0x03,
    /// Task assignment
    TaskAssignment = 0x10,
    /// Task status update
    TaskStatus = 0x11,
    /// Task result
    TaskResult = 0x12,
    /// Neural network update
    NeuralUpdate = 0x20,
    /// Neural weights
    NeuralWeights = 0x21,
    /// Neural gradient
    NeuralGradient = 0x22,
    /// Coordination message
    Coordination = 0x30,
    /// Consensus proposal
    Consensus = 0x31,
    /// Consensus vote
    ConsensusVote = 0x32,
    /// Heartbeat/keep-alive
    Heartbeat = 0x40,
    /// Ping request
    Ping = 0x41,
    /// Pong response
    Pong = 0x42,
    /// Error message
    Error = 0x50,
    /// Application-specific data
    ApplicationData = 0x60,
    /// File transfer
    FileTransfer = 0x70,
    /// Stream data
    StreamData = 0x80,
}

impl MessageType {
    /// Check if message type requires encryption
    pub fn requires_encryption(&self) -> bool {
        match self {
            MessageType::Ping | MessageType::Pong | MessageType::Heartbeat => false,
            _ => true,
        }
    }

    /// Check if message type requires authentication
    pub fn requires_authentication(&self) -> bool {
        !matches!(self, MessageType::HandshakeInit)
    }

    /// Get message priority (0 = highest, 255 = lowest)
    pub fn priority(&self) -> u8 {
        match self {
            MessageType::Error => 0,
            MessageType::HandshakeInit | MessageType::HandshakeResponse | MessageType::HandshakeComplete => 1,
            MessageType::Heartbeat | MessageType::Ping | MessageType::Pong => 2,
            MessageType::Consensus | MessageType::ConsensusVote => 3,
            MessageType::TaskAssignment | MessageType::TaskStatus | MessageType::TaskResult => 4,
            MessageType::Coordination => 5,
            MessageType::NeuralUpdate | MessageType::NeuralWeights | MessageType::NeuralGradient => 6,
            MessageType::ApplicationData => 7,
            MessageType::FileTransfer | MessageType::StreamData => 8,
        }
    }
}

impl TryFrom<u8> for MessageType {
    type Error = ProtocolError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(MessageType::HandshakeInit),
            0x02 => Ok(MessageType::HandshakeResponse),
            0x03 => Ok(MessageType::HandshakeComplete),
            0x10 => Ok(MessageType::TaskAssignment),
            0x11 => Ok(MessageType::TaskStatus),
            0x12 => Ok(MessageType::TaskResult),
            0x20 => Ok(MessageType::NeuralUpdate),
            0x21 => Ok(MessageType::NeuralWeights),
            0x22 => Ok(MessageType::NeuralGradient),
            0x30 => Ok(MessageType::Coordination),
            0x31 => Ok(MessageType::Consensus),
            0x32 => Ok(MessageType::ConsensusVote),
            0x40 => Ok(MessageType::Heartbeat),
            0x41 => Ok(MessageType::Ping),
            0x42 => Ok(MessageType::Pong),
            0x50 => Ok(MessageType::Error),
            0x60 => Ok(MessageType::ApplicationData),
            0x70 => Ok(MessageType::FileTransfer),
            0x80 => Ok(MessageType::StreamData),
            _ => Err(ProtocolError::UnknownMessageType(value)),
        }
    }
}

/// Neural communication message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message header
    pub header: MessageHeader,
    /// Message payload
    pub payload: MessagePayload,
}

impl Message {
    /// Create a new message
    pub fn new(msg_type: MessageType, data: Vec<u8>) -> Self {
        Self {
            header: MessageHeader::new(msg_type),
            payload: MessagePayload::Data(data),
        }
    }

    /// Create a new message with specific recipient
    pub fn new_to(msg_type: MessageType, recipient: AgentId, data: Vec<u8>) -> Self {
        Self {
            header: MessageHeader::new_to(msg_type, recipient),
            payload: MessagePayload::Data(data),
        }
    }

    /// Create a heartbeat message
    pub fn heartbeat() -> Self {
        Self {
            header: MessageHeader::new(MessageType::Heartbeat),
            payload: MessagePayload::Heartbeat,
        }
    }

    /// Create a ping message
    pub fn ping(data: Vec<u8>) -> Self {
        Self {
            header: MessageHeader::new(MessageType::Ping),
            payload: MessagePayload::Ping(data),
        }
    }

    /// Create a pong response
    pub fn pong(data: Vec<u8>) -> Self {
        Self {
            header: MessageHeader::new(MessageType::Pong),
            payload: MessagePayload::Pong(data),
        }
    }

    /// Create an error message
    pub fn error(error_code: u32, description: String) -> Self {
        Self {
            header: MessageHeader::new(MessageType::Error),
            payload: MessagePayload::Error { error_code, description },
        }
    }

    /// Get message size in bytes
    pub fn size(&self) -> usize {
        self.header.size() + self.payload.size()
    }

    /// Check if message is valid
    pub fn validate(&self) -> Result<()> {
        self.header.validate()?;
        self.payload.validate()?;
        
        // Check size limits
        if self.size() > security::MAX_MESSAGE_SIZE {
            return Err(ProtocolError::MessageTooLarge {
                size: self.size(),
                max: security::MAX_MESSAGE_SIZE,
            });
        }

        Ok(())
    }

    /// Serialize message to bytes
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| ProtocolError::InvalidFormat(format!("Serialization failed: {}", e)))
    }

    /// Deserialize message from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let message: Self = bincode::deserialize(data)
            .map_err(|e| ProtocolError::InvalidFormat(format!("Deserialization failed: {}", e)))?;
        
        message.validate()?;
        Ok(message)
    }
}

/// Message header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    /// Protocol version
    pub version: u32,
    /// Message type
    pub msg_type: MessageType,
    /// Message ID for tracking
    pub message_id: MessageId,
    /// Sender agent ID
    pub sender: Option<AgentId>,
    /// Recipient agent ID (None for broadcast)
    pub recipient: Option<AgentId>,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Time-to-live in seconds
    pub ttl: u32,
    /// Message flags
    pub flags: MessageFlags,
}

impl MessageHeader {
    /// Create a new message header
    pub fn new(msg_type: MessageType) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            msg_type,
            message_id: Self::generate_message_id(),
            sender: None,
            recipient: None,
            sequence: 0,
            timestamp: Self::current_timestamp(),
            ttl: 300, // 5 minutes default
            flags: MessageFlags::default(),
        }
    }

    /// Create a new message header with recipient
    pub fn new_to(msg_type: MessageType, recipient: AgentId) -> Self {
        let mut header = Self::new(msg_type);
        header.recipient = Some(recipient);
        header
    }

    /// Generate a random message ID
    fn generate_message_id() -> MessageId {
        use crate::crypto::random::SystemRng;
        let mut rng = SystemRng::new().expect("System RNG should be available");
        let mut id = [0u8; 16];
        rng.fill_bytes(&mut id).expect("Random generation should succeed");
        id
    }

    /// Get current timestamp
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Validate header
    pub fn validate(&self) -> Result<()> {
        if self.version != PROTOCOL_VERSION {
            return Err(ProtocolError::VersionMismatch {
                expected: PROTOCOL_VERSION,
                actual: self.version,
            });
        }

        let now = Self::current_timestamp();
        if self.timestamp > now + security::MAX_CLOCK_SKEW {
            return Err(ProtocolError::InvalidTimestamp(
                "Timestamp too far in future".to_string()
            ));
        }

        if self.ttl == 0 {
            return Err(ProtocolError::InvalidTimestamp(
                "Message has expired".to_string()
            ));
        }

        Ok(())
    }

    /// Get header size in bytes
    pub fn size(&self) -> usize {
        // Approximate size calculation
        32 + // Fixed fields
        self.sender.map_or(0, |_| 32) +
        self.recipient.map_or(0, |_| 32)
    }
}

/// Message flags
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageFlags {
    /// Message is compressed
    pub compressed: bool,
    /// Message is encrypted
    pub encrypted: bool,
    /// Message requires acknowledgment
    pub ack_required: bool,
    /// Message is a retransmission
    pub retransmission: bool,
    /// Message is urgent
    pub urgent: bool,
}

/// Message payload variants
#[derive(Debug, Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub enum MessagePayload {
    /// Raw data payload
    Data(Vec<u8>),
    /// Handshake message
    Handshake(HandshakeMessage),
    /// Task assignment
    TaskAssignment {
        task_id: String,
        task_type: String,
        parameters: Vec<u8>,
        deadline: Option<u64>,
    },
    /// Task status update
    TaskStatus {
        task_id: String,
        status: TaskStatus,
        progress: f32,
        message: Option<String>,
    },
    /// Task result
    TaskResult {
        task_id: String,
        success: bool,
        result: Vec<u8>,
        metrics: Option<Vec<u8>>,
    },
    /// Neural network update
    NeuralUpdate {
        layer_id: u32,
        weights: Vec<f32>,
        biases: Vec<f32>,
        activation: String,
    },
    /// Coordination message
    Coordination {
        action: String,
        parameters: Vec<u8>,
        priority: u8,
    },
    /// Consensus proposal
    Consensus {
        proposal_id: String,
        proposal_data: Vec<u8>,
        timeout: u64,
    },
    /// Consensus vote
    ConsensusVote {
        proposal_id: String,
        vote: bool,
        signature: Vec<u8>,
    },
    /// Heartbeat
    Heartbeat,
    /// Ping with optional data
    Ping(Vec<u8>),
    /// Pong response
    Pong(Vec<u8>),
    /// Error message
    Error {
        error_code: u32,
        description: String,
    },
    /// File transfer chunk
    FileTransfer {
        file_id: String,
        chunk_index: u32,
        total_chunks: u32,
        data: Vec<u8>,
        checksum: Vec<u8>,
    },
}

impl MessagePayload {
    /// Get payload size in bytes
    pub fn size(&self) -> usize {
        match self {
            MessagePayload::Data(data) => data.len(),
            MessagePayload::Ping(data) => data.len(),
            MessagePayload::Pong(data) => data.len(),
            MessagePayload::Heartbeat => 0,
            MessagePayload::FileTransfer { data, .. } => data.len() + 64, // Approximate
            _ => 256, // Approximate for complex types
        }
    }

    /// Validate payload
    pub fn validate(&self) -> Result<()> {
        match self {
            MessagePayload::TaskStatus { progress, .. } => {
                if !(0.0..=1.0).contains(progress) {
                    return Err(ProtocolError::InvalidFormat(
                        "Progress must be between 0.0 and 1.0".to_string()
                    ));
                }
            }
            MessagePayload::Error { error_code, .. } => {
                if *error_code == 0 {
                    return Err(ProtocolError::InvalidFormat(
                        "Error code cannot be zero".to_string()
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Task status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is pending
    Pending,
    /// Task is running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
}

/// Handshake message for secure channel establishment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeMessage {
    /// Handshake step
    pub step: HandshakeStep,
    /// Public key
    pub public_key: Vec<u8>,
    /// Signature
    pub signature: Option<Vec<u8>>,
    /// Nonce for replay protection
    pub nonce: Vec<u8>,
    /// Supported cipher suites
    pub cipher_suites: Option<Vec<u8>>,
    /// Additional data
    pub data: Option<Vec<u8>>,
}

/// Handshake steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandshakeStep {
    /// Initial client hello
    ClientHello,
    /// Server hello response
    ServerHello,
    /// Client key exchange
    ClientKeyExchange,
    /// Server key exchange
    ServerKeyExchange,
    /// Handshake completion
    Finished,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_conversion() {
        let msg_type = MessageType::TaskAssignment;
        let byte_val = msg_type as u8;
        let converted = MessageType::try_from(byte_val).unwrap();
        assert_eq!(msg_type, converted);
    }

    #[test]
    fn test_message_type_properties() {
        assert!(MessageType::TaskAssignment.requires_encryption());
        assert!(!MessageType::Ping.requires_encryption());
        assert!(MessageType::TaskAssignment.requires_authentication());
        assert!(!MessageType::HandshakeInit.requires_authentication());
    }

    #[test]
    fn test_message_creation() {
        let msg = Message::new(MessageType::Heartbeat, vec![]);
        assert_eq!(msg.header.msg_type, MessageType::Heartbeat);
        assert_eq!(msg.header.version, PROTOCOL_VERSION);
    }

    #[test]
    fn test_message_validation() {
        let msg = Message::heartbeat();
        assert!(msg.validate().is_ok());
        
        let mut invalid_msg = Message::new(MessageType::TaskAssignment, vec![0; security::MAX_MESSAGE_SIZE + 1]);
        assert!(invalid_msg.validate().is_err());
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::ping(b"test data".to_vec());
        let serialized = msg.serialize().unwrap();
        let deserialized = Message::deserialize(&serialized).unwrap();
        
        assert_eq!(msg.header.msg_type, deserialized.header.msg_type);
    }

    #[test]
    fn test_header_validation() {
        let header = MessageHeader::new(MessageType::Heartbeat);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_payload_validation() {
        let payload = MessagePayload::TaskStatus {
            task_id: "test".to_string(),
            status: TaskStatus::Running,
            progress: 0.5,
            message: None,
        };
        assert!(payload.validate().is_ok());
        
        let invalid_payload = MessagePayload::TaskStatus {
            task_id: "test".to_string(),
            status: TaskStatus::Running,
            progress: 1.5, // Invalid progress
            message: None,
        };
        assert!(invalid_payload.validate().is_err());
    }

    #[test]
    fn test_message_priorities() {
        assert!(MessageType::Error.priority() < MessageType::Heartbeat.priority());
        assert!(MessageType::Heartbeat.priority() < MessageType::TaskAssignment.priority());
    }
}