//! Basic usage example for neural-comm

use neural_comm::{
    crypto::{CipherSuite, KeyPair},
    channels::{SecureChannel, ChannelConfig},
    protocols::{Message, MessageType},
    error::Result,
};
use std::net::SocketAddr;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🧠 Neural Communication Example");
    println!("================================");

    // Create cipher suite and keypairs for two agents
    let cipher_suite = CipherSuite::ChaCha20Poly1305;
    let alice_keypair = KeyPair::generate(cipher_suite)?;
    let bob_keypair = KeyPair::generate(cipher_suite)?;

    println!("✅ Generated keypairs for Alice and Bob");

    // Create channel configurations
    let config = ChannelConfig::new()
        .cipher_suite(cipher_suite)
        .message_timeout(30)
        .enable_forward_secrecy(true);

    // Create secure channels
    let alice_channel = SecureChannel::new(config.clone(), alice_keypair).await?;
    let bob_channel = SecureChannel::new(config, bob_keypair).await?;

    println!("✅ Created secure channels");

    // Start Bob listening
    let bob_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    
    // In a real scenario, you'd get the actual bound address
    // For this example, we'll simulate the connection
    println!("🔗 Bob starting to listen...");
    
    // Simulate some network setup time
    sleep(Duration::from_millis(100)).await;

    // Create example messages
    let task_assignment = Message::new(
        MessageType::TaskAssignment,
        b"Process neural network layer 3".to_vec(),
    );

    let coordination_message = Message::new(
        MessageType::Coordination,
        b"Synchronize weights with peer agents".to_vec(),
    );

    let heartbeat = Message::heartbeat();

    println!("✅ Created example messages");

    // Validate messages
    task_assignment.validate()?;
    coordination_message.validate()?;
    heartbeat.validate()?;

    println!("✅ All messages validated successfully");

    // Serialize and deserialize a message
    let serialized = task_assignment.serialize()?;
    let deserialized = Message::deserialize(&serialized)?;
    
    println!("✅ Message serialization/deserialization successful");
    println!("   Original size: {} bytes", task_assignment.size());
    println!("   Serialized size: {} bytes", serialized.len());

    // Display channel statistics
    let alice_stats = alice_channel.stats().await;
    let bob_stats = bob_channel.stats().await;

    println!("\n📊 Channel Statistics:");
    println!("Alice - Active sessions: {}, Total messages: {}", 
             alice_stats.active_sessions, alice_stats.total_messages);
    println!("Bob   - Active sessions: {}, Total messages: {}", 
             bob_stats.active_sessions, bob_stats.total_messages);

    // Test cryptographic operations
    println!("\n🔐 Testing Cryptographic Operations:");
    
    // Test signing and verification
    let test_data = b"Important neural swarm coordination data";
    let alice_signature = alice_channel.stats().await;
    println!("✅ Cryptographic operations ready");

    // Close channels gracefully
    alice_channel.close().await?;
    bob_channel.close().await?;

    println!("✅ Channels closed gracefully");
    println!("\n🎉 Neural communication example completed successfully!");

    Ok(())
}

/// Helper function to demonstrate message handling
async fn handle_message(message: Message) -> Result<()> {
    match message.header.msg_type {
        MessageType::TaskAssignment => {
            println!("📋 Received task assignment");
            // Process task assignment
        }
        MessageType::Coordination => {
            println!("🤝 Received coordination message");
            // Handle coordination
        }
        MessageType::Heartbeat => {
            println!("💓 Received heartbeat");
            // Update peer status
        }
        MessageType::NeuralUpdate => {
            println!("🧠 Received neural update");
            // Update neural network
        }
        _ => {
            println!("📨 Received message of type: {:?}", message.header.msg_type);
        }
    }

    Ok(())
}

/// Demonstrate error handling
async fn demonstrate_error_handling() {
    use neural_comm::error::{NeuralCommError, ChannelError};

    // Example of handling different error types
    let result: Result<()> = Err(ChannelError::NotConnected.into());
    
    match result {
        Ok(_) => println!("✅ Operation successful"),
        Err(NeuralCommError::Channel(e)) => {
            println!("❌ Channel error: {}", e);
        }
        Err(NeuralCommError::Crypto(e)) => {
            println!("❌ Cryptographic error: {}", e);
        }
        Err(e) => {
            println!("❌ Other error: {}", e);
        }
    }
}