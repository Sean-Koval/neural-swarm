# Neural-Comm Technical Specifications

## API Specifications

### Core Traits and Interfaces

#### Transport Trait
```rust
use async_trait::async_trait;
use std::net::SocketAddr;

#[async_trait]
pub trait Transport: Send + Sync + Clone {
    type Connection: Connection;
    type Listener: Listener<Connection = Self::Connection>;
    type Config: TransportConfig;

    async fn connect(&self, address: &Address, config: &Self::Config) -> Result<Self::Connection, TransportError>;
    async fn listen(&self, address: &Address, config: &Self::Config) -> Result<Self::Listener, TransportError>;
    fn protocol_name(&self) -> &'static str;
    fn default_port(&self) -> u16;
}

#[async_trait]
pub trait Connection: Send + Sync {
    async fn send(&mut self, data: &[u8]) -> Result<(), ConnectionError>;
    async fn receive(&mut self) -> Result<Vec<u8>, ConnectionError>;
    async fn close(&mut self) -> Result<(), ConnectionError>;
    fn peer_address(&self) -> &Address;
    fn is_connected(&self) -> bool;
}

#[async_trait]
pub trait Listener: Send + Sync {
    type Connection: Connection;
    
    async fn accept(&mut self) -> Result<Self::Connection, ListenerError>;
    async fn close(&mut self) -> Result<(), ListenerError>;
    fn local_address(&self) -> &Address;
}
```

#### Encryption Trait
```rust
#[async_trait]
pub trait Encryption: Send + Sync + Clone {
    type Key: Key;
    type Nonce: Nonce;
    type Config: EncryptionConfig;

    fn generate_key(&self) -> Result<Self::Key, CryptoError>;
    fn encrypt(&self, plaintext: &[u8], key: &Self::Key, nonce: &Self::Nonce) -> Result<Vec<u8>, CryptoError>;
    fn decrypt(&self, ciphertext: &[u8], key: &Self::Key, nonce: &Self::Nonce) -> Result<Vec<u8>, CryptoError>;
    fn algorithm_name(&self) -> &'static str;
    fn key_size(&self) -> usize;
    fn nonce_size(&self) -> usize;
}

pub trait Key: Send + Sync + Clone + Zeroize {
    fn as_bytes(&self) -> &[u8];
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> where Self: Sized;
}

pub trait Nonce: Send + Sync + Clone {
    fn generate() -> Self;
    fn as_bytes(&self) -> &[u8];
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> where Self: Sized;
}
```

#### Protocol Trait
```rust
#[async_trait]
pub trait Protocol: Send + Sync {
    type Message: Message;
    type Config: ProtocolConfig;

    async fn encode_message(&self, message: &Self::Message) -> Result<Vec<u8>, ProtocolError>;
    async fn decode_message(&self, data: &[u8]) -> Result<Self::Message, ProtocolError>;
    fn version(&self) -> ProtocolVersion;
    fn is_compatible(&self, version: ProtocolVersion) -> bool;
}

pub trait Message: Send + Sync + Clone + Serialize + DeserializeOwned {
    fn message_type(&self) -> MessageType;
    fn message_id(&self) -> MessageId;
    fn from_agent(&self) -> AgentId;
    fn to_agent(&self) -> Option<AgentId>; // None for broadcast
    fn timestamp(&self) -> Timestamp;
    fn payload(&self) -> &[u8];
}
```

### Core Data Types

#### Message Types
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MessageId(Uuid);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AgentId(Uuid);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Timestamp(SystemTime);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    // Core coordination messages
    Handshake,
    Heartbeat,
    Goodbye,
    
    // Task management
    TaskAssignment {
        task_id: TaskId,
        priority: Priority,
        deadline: Option<Timestamp>,
    },
    TaskUpdate {
        task_id: TaskId,
        progress: f32,
        status: TaskStatus,
    },
    TaskResult {
        task_id: TaskId,
        result: TaskResult,
    },
    
    // Neural network operations
    ModelUpdate {
        model_id: ModelId,
        layer_updates: Vec<LayerUpdate>,
        gradient_info: GradientInfo,
    },
    WeightSync {
        model_id: ModelId,
        weights: CompressedWeights,
        checksum: Checksum,
    },
    TrainingMetrics {
        model_id: ModelId,
        epoch: u64,
        metrics: TrainingMetrics,
    },
    
    // Swarm coordination
    TopologyUpdate {
        topology: NetworkTopology,
        membership_changes: Vec<MembershipChange>,
    },
    ConsensusProposal {
        proposal_id: ProposalId,
        proposal_type: ProposalType,
        proposal_data: Vec<u8>,
    },
    ConsensusVote {
        proposal_id: ProposalId,
        vote: Vote,
        signature: Signature,
    },
    
    // Discovery and routing
    PeerAdvertisement {
        capabilities: Vec<Capability>,
        load_metrics: LoadMetrics,
        availability: AvailabilityWindow,
    },
    RouteRequest {
        target: AgentId,
        hop_limit: u8,
        request_id: RequestId,
    },
    RouteResponse {
        request_id: RequestId,
        path: Vec<AgentId>,
        cost: RoutingCost,
    },
    
    // Custom extensible messages
    Custom {
        message_name: String,
        version: u32,
        data: Vec<u8>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMessage {
    pub version: ProtocolVersion,
    pub message_id: MessageId,
    pub from: AgentId,
    pub to: Option<AgentId>, // None for broadcast
    pub message_type: MessageType,
    pub timestamp: Timestamp,
    pub ttl: Option<Duration>,
    pub priority: Priority,
    pub payload: Bytes,
    pub signature: Option<Signature>,
    pub encryption_info: Option<EncryptionInfo>,
}
```

#### Configuration Types
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCommConfig {
    pub transport: TransportConfig,
    pub encryption: EncryptionConfig,
    pub protocol: ProtocolConfig,
    pub discovery: DiscoveryConfig,
    pub coordination: CoordinationConfig,
    pub performance: PerformanceConfig,
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    pub transport_type: TransportType,
    pub address: Address,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub keep_alive_interval: Duration,
    pub tcp_config: Option<TcpConfig>,
    pub udp_config: Option<UdpConfig>,
    pub quic_config: Option<QuicConfig>,
    pub ipc_config: Option<IpcConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub algorithm: EncryptionAlgorithm,
    pub key_exchange: KeyExchangeMethod,
    pub signature_algorithm: SignatureAlgorithm,
    pub key_rotation_interval: Duration,
    pub forward_secrecy: bool,
    pub compression: Option<CompressionAlgorithm>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_message_size: usize,
    pub message_buffer_size: usize,
    pub connection_pool_size: usize,
    pub worker_threads: Option<usize>,
    pub batch_size: usize,
    pub zero_copy_threshold: usize,
    pub simd_optimizations: bool,
}
```

## Protocol Specifications

### Handshake Protocol

#### Initial Connection Establishment
```
Client                                    Server
  |                                         |
  |--- ClientHello ----------------------->|
  |    {version, algorithms, nonce}        |
  |                                        |
  |<-- ServerHello -------------------------|
  |    {version, selected_algs, nonce}     |
  |                                        |
  |--- KeyExchange --------------------->  |
  |    {public_key, signature}             |
  |                                        |
  |<-- KeyExchange -------------------------|
  |    {public_key, signature}             |
  |                                        |
  |--- Finished ----------------------->   |
  |    {encrypted_verification}            |
  |                                        |
  |<-- Finished -------------------------|  |
  |    {encrypted_verification}            |
  |                                        |
  |<-- ApplicationData ------------------>  |
```

#### Message Format Details
```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct ClientHello {
    pub protocol_version: ProtocolVersion,
    pub supported_transports: Vec<TransportType>,
    pub supported_encryptions: Vec<EncryptionAlgorithm>,
    pub supported_compressions: Vec<CompressionAlgorithm>,
    pub client_nonce: [u8; 32],
    pub agent_id: AgentId,
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerHello {
    pub protocol_version: ProtocolVersion,
    pub selected_transport: TransportType,
    pub selected_encryption: EncryptionAlgorithm,
    pub selected_compression: Option<CompressionAlgorithm>,
    pub server_nonce: [u8; 32],
    pub agent_id: AgentId,
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<Extension>,
}
```

### Message Serialization

#### MessagePack-based Serialization
```rust
impl NeuralMessage {
    pub fn serialize(&self) -> Result<Vec<u8>, SerializationError> {
        // Custom MessagePack serialization with optimization
        let mut buffer = Vec::with_capacity(self.estimate_size());
        
        // Write version and flags
        buffer.extend_from_slice(&self.version.to_be_bytes());
        
        // Write message header
        let header = MessageHeader {
            message_id: self.message_id,
            from: self.from,
            to: self.to,
            timestamp: self.timestamp,
            ttl: self.ttl,
            priority: self.priority,
        };
        rmp_serde::encode::write(&mut buffer, &header)?;
        
        // Write message type and payload
        rmp_serde::encode::write(&mut buffer, &self.message_type)?;
        buffer.extend_from_slice(&self.payload);
        
        // Add signature if present
        if let Some(ref signature) = self.signature {
            buffer.push(1); // Signature present flag
            buffer.extend_from_slice(signature.as_bytes());
        } else {
            buffer.push(0); // No signature flag
        }
        
        Ok(buffer)
    }
    
    pub fn deserialize(data: &[u8]) -> Result<Self, SerializationError> {
        let mut cursor = std::io::Cursor::new(data);
        
        // Read version
        let mut version_bytes = [0u8; 4];
        cursor.read_exact(&mut version_bytes)?;
        let version = ProtocolVersion::from_be_bytes(version_bytes);
        
        // Read header
        let header: MessageHeader = rmp_serde::decode::from_read(&mut cursor)?;
        
        // Read message type
        let message_type: MessageType = rmp_serde::decode::from_read(&mut cursor)?;
        
        // Read payload
        let payload_start = cursor.position() as usize;
        let signature_flag_pos = data.len() - 1;
        let payload_end = if data[signature_flag_pos] == 1 {
            // Signature present, exclude signature from payload
            signature_flag_pos - 64 // Assuming 64-byte signature
        } else {
            signature_flag_pos
        };
        
        let payload = data[payload_start..payload_end].to_vec().into();
        
        // Read signature if present
        let signature = if data[signature_flag_pos] == 1 {
            Some(Signature::from_bytes(&data[payload_end..signature_flag_pos])?)
        } else {
            None
        };
        
        Ok(NeuralMessage {
            version,
            message_id: header.message_id,
            from: header.from,
            to: header.to,
            message_type,
            timestamp: header.timestamp,
            ttl: header.ttl,
            priority: header.priority,
            payload,
            signature,
            encryption_info: None, // Populated by encryption layer
        })
    }
}
```

### Encryption Protocol

#### ChaCha20-Poly1305 Implementation
```rust
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce, aead::Aead};

pub struct ChaCha20Poly1305Encryption {
    cipher: ChaCha20Poly1305,
}

impl ChaCha20Poly1305Encryption {
    pub fn new(key: &[u8; 32]) -> Self {
        let key = Key::from_slice(key);
        let cipher = ChaCha20Poly1305::new(key);
        Self { cipher }
    }
    
    pub fn encrypt_message(&self, message: &NeuralMessage, session_key: &[u8; 32]) -> Result<Vec<u8>, CryptoError> {
        // Serialize message
        let plaintext = message.serialize()?;
        
        // Generate nonce from message ID and timestamp
        let nonce = self.generate_nonce(&message.message_id, message.timestamp)?;
        
        // Encrypt with AEAD
        let ciphertext = self.cipher
            .encrypt(&nonce, plaintext.as_ref())
            .map_err(|_| CryptoError::EncryptionFailed)?;
        
        // Prepend nonce to ciphertext
        let mut result = Vec::with_capacity(12 + ciphertext.len());
        result.extend_from_slice(&nonce);
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
    pub fn decrypt_message(&self, encrypted_data: &[u8], session_key: &[u8; 32]) -> Result<NeuralMessage, CryptoError> {
        if encrypted_data.len() < 12 {
            return Err(CryptoError::InvalidCiphertext);
        }
        
        // Extract nonce and ciphertext
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        // Decrypt with AEAD
        let plaintext = self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| CryptoError::DecryptionFailed)?;
        
        // Deserialize message
        NeuralMessage::deserialize(&plaintext)
            .map_err(|_| CryptoError::InvalidPlaintext)
    }
    
    fn generate_nonce(&self, message_id: &MessageId, timestamp: Timestamp) -> Result<Nonce, CryptoError> {
        let mut nonce_data = [0u8; 12];
        
        // Use first 8 bytes of message ID
        nonce_data[0..8].copy_from_slice(&message_id.as_bytes()[0..8]);
        
        // Use timestamp for last 4 bytes
        let timestamp_bytes = timestamp.duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_le_bytes();
        nonce_data[8..12].copy_from_slice(&timestamp_bytes[0..4]);
        
        Ok(*Nonce::from_slice(&nonce_data))
    }
}
```

## Performance Optimizations

### SIMD Acceleration
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct SIMDCrypto {
    #[cfg(target_feature = "avx2")]
    avx2_supported: bool,
    #[cfg(target_feature = "aes")]
    aes_ni_supported: bool,
}

impl SIMDCrypto {
    pub fn new() -> Self {
        Self {
            #[cfg(target_feature = "avx2")]
            avx2_supported: is_x86_feature_detected!("avx2"),
            #[cfg(target_feature = "aes")]
            aes_ni_supported: is_x86_feature_detected!("aes"),
        }
    }
    
    #[cfg(target_feature = "avx2")]
    pub unsafe fn parallel_encrypt_avx2(&self, blocks: &[Block], key: &Key) -> Vec<Block> {
        // AVX2-optimized parallel encryption of multiple blocks
        let mut result = Vec::with_capacity(blocks.len());
        
        for chunk in blocks.chunks(8) { // Process 8 blocks at once with AVX2
            let encrypted_chunk = self.encrypt_chunk_avx2(chunk, key);
            result.extend_from_slice(&encrypted_chunk);
        }
        
        result
    }
    
    #[cfg(target_feature = "aes")]
    pub unsafe fn aes_encrypt_ni(&self, plaintext: &[u8], key: &[u8; 16]) -> Vec<u8> {
        // Hardware-accelerated AES encryption using AES-NI
        // Implementation would use _mm_aesenc_si128 and related intrinsics
        todo!("Implement AES-NI acceleration")
    }
}
```

### Zero-Copy Optimizations
```rust
use bytes::{Bytes, BytesMut, BufMut};

pub struct ZeroCopyBuffer {
    inner: BytesMut,
    write_offset: usize,
    read_offset: usize,
}

impl ZeroCopyBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: BytesMut::with_capacity(capacity),
            write_offset: 0,
            read_offset: 0,
        }
    }
    
    pub fn write_message_header(&mut self, header: &MessageHeader) -> Result<(), BufferError> {
        let required_space = std::mem::size_of::<MessageHeader>();
        if self.inner.remaining_mut() < required_space {
            return Err(BufferError::InsufficientSpace);
        }
        
        // Write header directly into buffer without copying
        unsafe {
            let header_ptr = header as *const MessageHeader as *const u8;
            let buffer_ptr = self.inner.as_mut_ptr().add(self.write_offset);
            std::ptr::copy_nonoverlapping(header_ptr, buffer_ptr, required_space);
        }
        
        self.write_offset += required_space;
        unsafe { self.inner.set_len(self.write_offset); }
        
        Ok(())
    }
    
    pub fn get_message_slice(&self, start: usize, len: usize) -> Option<&[u8]> {
        if start + len <= self.inner.len() {
            Some(&self.inner[start..start + len])
        } else {
            None
        }
    }
}
```

### Memory Pool Management
```rust
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

pub struct MessagePool {
    pool: Arc<Mutex<VecDeque<Box<[u8]>>>>,
    buffer_size: usize,
    max_pool_size: usize,
}

impl MessagePool {
    pub fn new(buffer_size: usize, max_pool_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::with_capacity(max_pool_size))),
            buffer_size,
            max_pool_size,
        }
    }
    
    pub fn acquire(&self) -> PooledBuffer {
        let mut pool = self.pool.lock().unwrap();
        let buffer = pool.pop_front()
            .unwrap_or_else(|| vec![0u8; self.buffer_size].into_boxed_slice());
        
        PooledBuffer {
            buffer,
            pool: Arc::clone(&self.pool),
            max_pool_size: self.max_pool_size,
        }
    }
}

pub struct PooledBuffer {
    buffer: Box<[u8]>,
    pool: Arc<Mutex<VecDeque<Box<[u8]>>>>,
    max_pool_size: usize,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.max_pool_size {
            // Zero the buffer before returning to pool for security
            self.buffer.fill(0);
            let buffer = std::mem::replace(&mut self.buffer, Box::new([]));
            pool.push_back(buffer);
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [u8];
    
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}
```

## Integration Guidelines

### Agent Integration Pattern
```rust
use neural_comm::{NeuralComm, NeuralCommBuilder, MessageType, AgentId};

pub struct NeuralAgent {
    id: AgentId,
    comm: NeuralComm,
    capabilities: Vec<Capability>,
}

impl NeuralAgent {
    pub async fn new(config: AgentConfig) -> Result<Self, AgentError> {
        let comm = NeuralCommBuilder::new()
            .transport(config.transport)
            .encryption(config.encryption)
            .agent_id(config.agent_id)
            .build()
            .await?;
        
        Ok(Self {
            id: config.agent_id,
            comm,
            capabilities: config.capabilities,
        })
    }
    
    pub async fn join_swarm(&mut self, discovery_address: &str) -> Result<(), SwarmError> {
        // Connect to swarm coordinator
        self.comm.connect_to_coordinator(discovery_address).await?;
        
        // Register capabilities
        self.comm.send_message(
            None, // Broadcast to coordinator
            MessageType::PeerAdvertisement {
                capabilities: self.capabilities.clone(),
                load_metrics: self.get_load_metrics(),
                availability: self.get_availability_window(),
            },
            &[]
        ).await?;
        
        // Start message processing loop
        self.start_message_processing().await?;
        
        Ok(())
    }
    
    async fn start_message_processing(&mut self) -> Result<(), CommError> {
        let mut message_stream = self.comm.message_stream();
        
        while let Some(message) = message_stream.next().await {
            match message.message_type {
                MessageType::TaskAssignment { task_id, .. } => {
                    self.handle_task_assignment(task_id, &message.payload).await?;
                }
                MessageType::ModelUpdate { model_id, .. } => {
                    self.handle_model_update(model_id, &message.payload).await?;
                }
                MessageType::ConsensusProposal { proposal_id, .. } => {
                    self.handle_consensus_proposal(proposal_id, &message.payload).await?;
                }
                _ => {
                    log::debug!("Received unhandled message type: {:?}", message.message_type);
                }
            }
        }
        
        Ok(())
    }
    
    async fn handle_task_assignment(&mut self, task_id: TaskId, payload: &[u8]) -> Result<(), TaskError> {
        // Deserialize task data
        let task: Task = bincode::deserialize(payload)?;
        
        // Process task asynchronously
        let result = self.process_task(task).await?;
        
        // Send result back
        let result_payload = bincode::serialize(&result)?;
        self.comm.send_message(
            Some(task.coordinator_id),
            MessageType::TaskResult { task_id, result },
            &result_payload
        ).await?;
        
        Ok(())
    }
}
```

### Swarm Coordinator Integration
```rust
use neural_comm::{SwarmCoordinator, ConsensusProtocol, TopologyManager};

pub struct SwarmManager {
    coordinator: SwarmCoordinator,
    consensus: Box<dyn ConsensusProtocol>,
    topology: TopologyManager,
    active_agents: HashMap<AgentId, AgentMetadata>,
}

impl SwarmManager {
    pub async fn new(config: SwarmConfig) -> Result<Self, SwarmError> {
        let coordinator = SwarmCoordinator::new(config.coordination_config).await?;
        let consensus = Box::new(PBFTConsensus::new(config.consensus_config));
        let topology = TopologyManager::new(config.topology_config);
        
        Ok(Self {
            coordinator,
            consensus,
            topology,
            active_agents: HashMap::new(),
        })
    }
    
    pub async fn start_coordination(&mut self) -> Result<(), SwarmError> {
        // Start listening for agent connections
        self.coordinator.start_listening().await?;
        
        // Initialize consensus protocol
        self.consensus.initialize().await?;
        
        // Start topology management
        self.topology.start_discovery().await?;
        
        // Start main coordination loop
        self.coordination_loop().await
    }
    
    async fn coordination_loop(&mut self) -> Result<(), SwarmError> {
        let mut events = self.coordinator.event_stream();
        
        while let Some(event) = events.next().await {
            match event {
                CoordinationEvent::AgentJoined(agent_id, metadata) => {
                    self.handle_agent_joined(agent_id, metadata).await?;
                }
                CoordinationEvent::AgentLeft(agent_id) => {
                    self.handle_agent_left(agent_id).await?;
                }
                CoordinationEvent::ConsensusRequired(proposal) => {
                    self.handle_consensus_proposal(proposal).await?;
                }
                CoordinationEvent::TopologyChange(change) => {
                    self.handle_topology_change(change).await?;
                }
            }
        }
        
        Ok(())
    }
    
    async fn handle_agent_joined(&mut self, agent_id: AgentId, metadata: AgentMetadata) -> Result<(), SwarmError> {
        // Add agent to active list
        self.active_agents.insert(agent_id, metadata.clone());
        
        // Update topology
        self.topology.add_agent(agent_id, &metadata).await?;
        
        // Notify other agents of topology change
        let topology_update = MessageType::TopologyUpdate {
            topology: self.topology.current_topology(),
            membership_changes: vec![MembershipChange::Added(agent_id, metadata)],
        };
        
        self.coordinator.broadcast_message(topology_update, &[]).await?;
        
        Ok(())
    }
}
```

## Testing Infrastructure

### Unit Test Framework
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    
    #[tokio::test]
    async fn test_message_encryption_roundtrip() {
        let key = [42u8; 32];
        let encryptor = ChaCha20Poly1305Encryption::new(&key);
        
        let original_message = NeuralMessage {
            version: ProtocolVersion::V1_0,
            message_id: MessageId::new(),
            from: AgentId::new(),
            to: Some(AgentId::new()),
            message_type: MessageType::Heartbeat,
            timestamp: Timestamp::now(),
            ttl: Some(Duration::from_secs(60)),
            priority: Priority::Normal,
            payload: b"test payload".to_vec().into(),
            signature: None,
            encryption_info: None,
        };
        
        let encrypted = encryptor.encrypt_message(&original_message, &key).unwrap();
        let decrypted = encryptor.decrypt_message(&encrypted, &key).unwrap();
        
        assert_eq!(original_message.message_id, decrypted.message_id);
        assert_eq!(original_message.payload, decrypted.payload);
    }
    
    #[tokio::test]
    async fn test_transport_connection() {
        let transport = TcpTransport::new();
        let config = TcpConfig::default();
        
        // Start listener
        let listener_addr = "127.0.0.1:0".parse().unwrap();
        let mut listener = transport.listen(&listener_addr, &config).await.unwrap();
        let actual_addr = listener.local_address().clone();
        
        // Connect to listener
        let mut connection = transport.connect(&actual_addr, &config).await.unwrap();
        
        // Accept connection
        let mut server_conn = listener.accept().await.unwrap();
        
        // Test bidirectional communication
        let test_data = b"Hello, world!";
        connection.send(test_data).await.unwrap();
        
        let received_data = server_conn.receive().await.unwrap();
        assert_eq!(test_data, received_data.as_slice());
    }
}
```

### Integration Test Suite
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_swarm_formation() {
        // Create coordinator
        let coord_config = SwarmConfig::default();
        let mut coordinator = SwarmManager::new(coord_config).await.unwrap();
        
        // Start coordinator in background
        tokio::spawn(async move {
            coordinator.start_coordination().await.unwrap();
        });
        
        sleep(Duration::from_millis(100)).await;
        
        // Create multiple agents
        let mut agents = Vec::new();
        for i in 0..5 {
            let config = AgentConfig {
                agent_id: AgentId::new(),
                transport: TransportType::Tcp,
                encryption: EncryptionAlgorithm::ChaCha20Poly1305,
                capabilities: vec![format!("capability_{}", i)],
            };
            
            let mut agent = NeuralAgent::new(config).await.unwrap();
            agent.join_swarm("127.0.0.1:8000").await.unwrap();
            agents.push(agent);
        }
        
        // Wait for swarm formation
        sleep(Duration::from_secs(2)).await;
        
        // Verify all agents are connected
        for agent in &agents {
            assert!(agent.comm.is_connected_to_swarm());
        }
    }
    
    #[tokio::test]
    async fn test_message_broadcast() {
        // Set up swarm with coordinator and 3 agents
        let (mut coordinator, mut agents) = setup_test_swarm(3).await;
        
        // Send broadcast message from first agent
        let broadcast_data = b"Broadcast test message";
        agents[0].comm.send_message(
            None, // Broadcast
            MessageType::Custom {
                message_name: "test_broadcast".to_string(),
                version: 1,
                data: broadcast_data.to_vec(),
            },
            broadcast_data
        ).await.unwrap();
        
        // Verify all other agents receive the message
        for i in 1..agents.len() {
            let received = agents[i].comm.wait_for_message(Duration::from_secs(5)).await.unwrap();
            if let MessageType::Custom { message_name, data, .. } = received.message_type {
                assert_eq!(message_name, "test_broadcast");
                assert_eq!(data, broadcast_data);
            } else {
                panic!("Expected Custom message type");
            }
        }
    }
    
    #[tokio::test]
    async fn test_consensus_protocol() {
        let (mut coordinator, mut agents) = setup_test_swarm(5).await;
        
        // Create consensus proposal
        let proposal = ConsensusProposal {
            proposal_id: ProposalId::new(),
            proposal_type: ProposalType::ParameterUpdate,
            proposal_data: b"new_learning_rate=0.001".to_vec(),
            proposer: agents[0].id,
        };
        
        // Submit proposal through first agent
        agents[0].comm.submit_consensus_proposal(proposal.clone()).await.unwrap();
        
        // Wait for consensus to complete
        sleep(Duration::from_secs(3)).await;
        
        // Verify all agents received the decision
        for agent in &agents {
            let decision = agent.comm.get_consensus_decision(proposal.proposal_id).await.unwrap();
            assert!(decision.is_accepted());
        }
    }
    
    async fn setup_test_swarm(num_agents: usize) -> (SwarmManager, Vec<NeuralAgent>) {
        // Implementation helper for test setup
        todo!("Implement test swarm setup")
    }
}
```

### Performance Benchmarks
```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
    
    fn bench_message_serialization(c: &mut Criterion) {
        let message = create_test_message();
        
        c.bench_function("message_serialize", |b| {
            b.iter(|| {
                black_box(message.serialize().unwrap())
            })
        });
        
        let serialized = message.serialize().unwrap();
        c.bench_function("message_deserialize", |b| {
            b.iter(|| {
                black_box(NeuralMessage::deserialize(&serialized).unwrap())
            })
        });
    }
    
    fn bench_encryption_performance(c: &mut Criterion) {
        let key = [42u8; 32];
        let encryptor = ChaCha20Poly1305Encryption::new(&key);
        let message = create_test_message();
        
        let mut group = c.benchmark_group("encryption");
        
        for size in [1024, 4096, 16384, 65536].iter() {
            let payload = vec![0u8; *size];
            let mut test_message = message.clone();
            test_message.payload = payload.into();
            
            group.bench_with_input(
                BenchmarkId::new("encrypt", size),
                size,
                |b, _| {
                    b.iter(|| {
                        black_box(encryptor.encrypt_message(&test_message, &key).unwrap())
                    })
                }
            );
        }
        
        group.finish();
    }
    
    fn bench_transport_throughput(c: &mut Criterion) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        c.bench_function("tcp_throughput", |b| {
            b.to_async(&rt).iter(|| async {
                let transport = TcpTransport::new();
                let config = TcpConfig::default();
                
                // Set up connection
                let (mut client, mut server) = create_test_connection(&transport, &config).await;
                
                // Benchmark message sending
                let test_data = vec![0u8; 1024];
                for _ in 0..1000 {
                    client.send(&test_data).await.unwrap();
                    let _ = server.receive().await.unwrap();
                }
            })
        });
    }
    
    criterion_group!(benches, bench_message_serialization, bench_encryption_performance, bench_transport_throughput);
    criterion_main!(benches);
    
    fn create_test_message() -> NeuralMessage {
        NeuralMessage {
            version: ProtocolVersion::V1_0,
            message_id: MessageId::new(),
            from: AgentId::new(),
            to: Some(AgentId::new()),
            message_type: MessageType::Heartbeat,
            timestamp: Timestamp::now(),
            ttl: Some(Duration::from_secs(60)),
            priority: Priority::Normal,
            payload: b"test payload".to_vec().into(),
            signature: None,
            encryption_info: None,
        }
    }
}
```

This technical specification provides the detailed implementation guidance needed for the neural-comm crate development, with comprehensive API definitions, protocol specifications, performance optimizations, and testing infrastructure.