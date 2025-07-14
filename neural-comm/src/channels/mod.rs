//! Secure communication channels

use crate::{
    crypto::{CipherSuite, KeyPair, SymmetricKey, SharedSecret},
    error::{ChannelError, Result},
    protocols::{Message, MessageType, HandshakeMessage},
    security,
    AgentId, MessageId,
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{mpsc, RwLock},
    time::{timeout, Duration, Instant},
};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{atomic::{AtomicU64, Ordering}, Arc},
};
use serde::{Deserialize, Serialize};
use bytes::{Bytes, BytesMut};

pub mod transport;
pub mod handshake;
pub mod session;

pub use transport::{Transport, TcpTransport, QuicTransport};
pub use handshake::{HandshakeManager, HandshakeState};
pub use session::{Session, SessionManager, SessionConfig};

/// Configuration for secure channels
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Cipher suite to use
    pub cipher_suite: CipherSuite,
    /// Enable forward secrecy
    pub forward_secrecy: bool,
    /// Message timeout in seconds
    pub message_timeout: u64,
    /// Maximum message size
    pub max_message_size: usize,
    /// Keep-alive interval in seconds
    pub keep_alive_interval: u64,
    /// Maximum concurrent sessions
    pub max_sessions: usize,
    /// Session key rotation interval in minutes
    pub key_rotation_interval: u64,
    /// Enable compression
    pub enable_compression: bool,
    /// Heartbeat interval in seconds
    pub heartbeat_interval: u64,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            cipher_suite: CipherSuite::default(),
            forward_secrecy: true,
            message_timeout: 30,
            max_message_size: security::MAX_MESSAGE_SIZE,
            keep_alive_interval: 60,
            max_sessions: 1000,
            key_rotation_interval: security::DEFAULT_KEY_ROTATION_INTERVAL,
            enable_compression: false,
            heartbeat_interval: 30,
        }
    }
}

impl ChannelConfig {
    /// Create new channel configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set cipher suite
    pub fn cipher_suite(mut self, suite: CipherSuite) -> Self {
        self.cipher_suite = suite;
        self
    }

    /// Enable or disable forward secrecy
    pub fn enable_forward_secrecy(mut self, enable: bool) -> Self {
        self.forward_secrecy = enable;
        self
    }

    /// Set message timeout
    pub fn message_timeout(mut self, timeout: u64) -> Self {
        self.message_timeout = timeout;
        self
    }

    /// Set maximum message size
    pub fn max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }

    /// Set keep-alive interval
    pub fn keep_alive_interval(mut self, interval: u64) -> Self {
        self.keep_alive_interval = interval;
        self
    }

    /// Set maximum concurrent sessions
    pub fn max_sessions(mut self, max: usize) -> Self {
        self.max_sessions = max;
        self
    }

    /// Enable or disable compression
    pub fn enable_compression(mut self, enable: bool) -> Self {
        self.enable_compression = enable;
        self
    }
}

/// Secure communication channel
pub struct SecureChannel {
    /// Channel configuration
    config: ChannelConfig,
    /// Local keypair
    keypair: KeyPair,
    /// Agent ID
    agent_id: AgentId,
    /// Session manager
    session_manager: Arc<SessionManager>,
    /// Message sender
    message_tx: mpsc::UnboundedSender<(AgentId, Message)>,
    /// Message receiver
    message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<(AgentId, Message)>>>>,
    /// Active sessions
    sessions: Arc<RwLock<HashMap<AgentId, Arc<Session>>>>,
    /// Message counter for replay protection
    message_counter: AtomicU64,
    /// Channel state
    state: Arc<RwLock<ChannelState>>,
}

/// Channel state
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelState {
    /// Channel is disconnected
    Disconnected,
    /// Channel is connecting
    Connecting,
    /// Channel is connected and ready
    Connected,
    /// Channel is closing
    Closing,
    /// Channel has encountered an error
    Error(String),
}

impl SecureChannel {
    /// Create a new secure channel
    pub async fn new(config: ChannelConfig, keypair: KeyPair) -> Result<Self> {
        let agent_id = Self::derive_agent_id(&keypair);
        let session_manager = Arc::new(SessionManager::new(config.clone()));
        let (message_tx, message_rx) = mpsc::unbounded_channel();

        Ok(Self {
            config,
            keypair,
            agent_id,
            session_manager,
            message_tx,
            message_rx: Arc::new(RwLock::new(Some(message_rx))),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            message_counter: AtomicU64::new(1),
            state: Arc::new(RwLock::new(ChannelState::Disconnected)),
        })
    }

    /// Start listening for incoming connections
    pub async fn listen(&self, addr: SocketAddr) -> Result<()> {
        self.set_state(ChannelState::Connecting).await;
        
        let listener = TcpListener::bind(addr).await
            .map_err(|e| ChannelError::Network(format!("Failed to bind to {}: {}", addr, e)))?;

        tracing::info!("SecureChannel listening on {}", addr);
        self.set_state(ChannelState::Connected).await;

        let session_manager = self.session_manager.clone();
        let config = self.config.clone();
        let keypair = self.keypair.clone();
        let sessions = self.sessions.clone();
        let message_tx = self.message_tx.clone();

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        let session_manager = session_manager.clone();
                        let config = config.clone();
                        let keypair = keypair.clone();
                        let sessions = sessions.clone();
                        let message_tx = message_tx.clone();

                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_incoming_connection(
                                stream,
                                peer_addr,
                                session_manager,
                                config,
                                keypair,
                                sessions,
                                message_tx,
                            ).await {
                                tracing::error!("Error handling connection from {}: {}", peer_addr, e);
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("Failed to accept connection: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Connect to a remote peer
    pub async fn connect(&self, addr: SocketAddr) -> Result<AgentId> {
        self.set_state(ChannelState::Connecting).await;

        let stream = timeout(
            Duration::from_secs(self.config.message_timeout),
            TcpStream::connect(addr)
        ).await
            .map_err(|_| ChannelError::Network("Connection timeout".to_string()))?
            .map_err(|e| ChannelError::Network(format!("Failed to connect to {}: {}", addr, e)))?;

        let transport = TcpTransport::new(stream);
        let handshake_manager = HandshakeManager::new(self.config.clone(), self.keypair.clone());
        
        let (peer_agent_id, session) = handshake_manager.perform_handshake_client(transport).await?;
        
        // Store the session
        self.sessions.write().await.insert(peer_agent_id, Arc::new(session));
        
        self.set_state(ChannelState::Connected).await;
        
        Ok(peer_agent_id)
    }

    /// Send a message to a specific peer
    pub async fn send(&self, peer_id: AgentId, message: Message) -> Result<()> {
        let sessions = self.sessions.read().await;
        let session = sessions.get(&peer_id)
            .ok_or_else(|| ChannelError::NotConnected)?;

        session.send_message(message).await
    }

    /// Send a broadcast message to all connected peers
    pub async fn broadcast(&self, message: Message) -> Result<()> {
        let sessions = self.sessions.read().await;
        
        for session in sessions.values() {
            if let Err(e) = session.send_message(message.clone()).await {
                tracing::warn!("Failed to send broadcast message: {}", e);
            }
        }

        Ok(())
    }

    /// Receive the next message
    pub async fn receive(&self) -> Result<(AgentId, Message)> {
        let mut rx_guard = self.message_rx.write().await;
        let rx = rx_guard.as_mut().ok_or_else(|| ChannelError::Closed)?;

        timeout(
            Duration::from_secs(self.config.message_timeout),
            rx.recv()
        ).await
            .map_err(|_| ChannelError::ReceiveFailed("Receive timeout".to_string()))?
            .ok_or_else(|| ChannelError::Closed)
    }

    /// Get connected peer IDs
    pub async fn connected_peers(&self) -> Vec<AgentId> {
        self.sessions.read().await.keys().copied().collect()
    }

    /// Get channel statistics
    pub async fn stats(&self) -> ChannelStats {
        let sessions = self.sessions.read().await;
        let active_sessions = sessions.len();
        let total_messages = self.message_counter.load(Ordering::Relaxed);
        
        ChannelStats {
            active_sessions,
            total_messages,
            agent_id: self.agent_id,
            state: self.state.read().await.clone(),
            uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Close the channel
    pub async fn close(&self) -> Result<()> {
        self.set_state(ChannelState::Closing).await;
        
        // Close all sessions
        let mut sessions = self.sessions.write().await;
        for session in sessions.values() {
            session.close().await?;
        }
        sessions.clear();
        
        self.set_state(ChannelState::Disconnected).await;
        Ok(())
    }

    /// Derive agent ID from keypair
    fn derive_agent_id(keypair: &KeyPair) -> AgentId {
        use crate::crypto::hash::{Blake3Hash, HashFunction};
        
        let hash_fn = Blake3Hash::new();
        let public_key_bytes = keypair.public_key().as_bytes();
        let hash = hash_fn.hash(&public_key_bytes);
        
        let mut agent_id = [0u8; 32];
        agent_id.copy_from_slice(&hash[..32]);
        agent_id
    }

    /// Set channel state
    async fn set_state(&self, state: ChannelState) {
        *self.state.write().await = state;
    }

    /// Handle incoming connection
    async fn handle_incoming_connection(
        stream: TcpStream,
        peer_addr: SocketAddr,
        session_manager: Arc<SessionManager>,
        config: ChannelConfig,
        keypair: KeyPair,
        sessions: Arc<RwLock<HashMap<AgentId, Arc<Session>>>>,
        message_tx: mpsc::UnboundedSender<(AgentId, Message)>,
    ) -> Result<()> {
        tracing::info!("Handling incoming connection from {}", peer_addr);
        
        let transport = TcpTransport::new(stream);
        let handshake_manager = HandshakeManager::new(config, keypair);
        
        let (peer_agent_id, session) = handshake_manager.perform_handshake_server(transport).await?;
        
        // Store the session
        sessions.write().await.insert(peer_agent_id, Arc::new(session.clone()));
        
        // Start message handling loop
        loop {
            match session.receive_message().await {
                Ok(message) => {
                    if let Err(e) = message_tx.send((peer_agent_id, message)) {
                        tracing::error!("Failed to forward message: {}", e);
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!("Session ended with {}: {}", peer_agent_id.iter().map(|b| format!("{:02x}", b)).collect::<String>(), e);
                    break;
                }
            }
        }

        // Clean up session
        sessions.write().await.remove(&peer_agent_id);
        
        Ok(())
    }
}

/// Channel statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStats {
    /// Number of active sessions
    pub active_sessions: usize,
    /// Total messages processed
    pub total_messages: u64,
    /// Local agent ID
    pub agent_id: AgentId,
    /// Current channel state
    pub state: ChannelState,
    /// Uptime in seconds
    pub uptime: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::KeyPair;

    #[tokio::test]
    async fn test_channel_creation() {
        let config = ChannelConfig::new();
        let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305).unwrap();
        
        let channel = SecureChannel::new(config, keypair).await.unwrap();
        let stats = channel.stats().await;
        
        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.state, ChannelState::Disconnected);
    }

    #[tokio::test]
    async fn test_agent_id_derivation() {
        let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305).unwrap();
        let agent_id1 = SecureChannel::derive_agent_id(&keypair);
        let agent_id2 = SecureChannel::derive_agent_id(&keypair);
        
        assert_eq!(agent_id1, agent_id2); // Should be deterministic
        assert_eq!(agent_id1.len(), 32);
    }

    #[test]
    fn test_channel_config() {
        let config = ChannelConfig::new()
            .cipher_suite(CipherSuite::AesGcm256)
            .message_timeout(60)
            .enable_compression(true);
        
        assert_eq!(config.cipher_suite, CipherSuite::AesGcm256);
        assert_eq!(config.message_timeout, 60);
        assert!(config.enable_compression);
    }

    #[tokio::test]
    async fn test_channel_state_management() {
        let config = ChannelConfig::new();
        let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305).unwrap();
        
        let channel = SecureChannel::new(config, keypair).await.unwrap();
        assert_eq!(*channel.state.read().await, ChannelState::Disconnected);
        
        channel.set_state(ChannelState::Connected).await;
        assert_eq!(*channel.state.read().await, ChannelState::Connected);
    }
}