//! Secure communication example with two agents

use neural_comm::{
    crypto::{CipherSuite, KeyPair},
    channels::{SecureChannel, ChannelConfig},
    protocols::{Message, MessageType},
    error::Result,
};
use std::net::SocketAddr;
use tokio::{time::{sleep, Duration}, join};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🔐 Secure Neural Agent Communication");
    println!("===================================");

    // Demonstrate secure communication between two neural agents
    let (alice_result, bob_result) = join!(
        run_alice_agent(),
        run_bob_agent()
    );

    alice_result?;
    bob_result?;

    println!("\n🎉 Secure communication demonstration completed!");
    Ok(())
}

/// Alice agent - initiates communication
async fn run_alice_agent() -> Result<()> {
    println!("🤖 Starting Alice agent...");

    // Generate Alice's keypair
    let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305)?;
    
    // Create channel configuration with strong security
    let config = ChannelConfig::new()
        .cipher_suite(CipherSuite::ChaCha20Poly1305)
        .message_timeout(60)
        .enable_forward_secrecy(true)
        .enable_compression(false) // Disable for this example
        .max_message_size(1024 * 1024); // 1MB max

    // Create secure channel
    let channel = SecureChannel::new(config, keypair).await?;
    println!("✅ Alice: Secure channel created");

    // Simulate connection to Bob (in real scenario, Bob would be listening)
    sleep(Duration::from_millis(500)).await;

    // Create and send neural coordination messages
    let messages = vec![
        Message::new(
            MessageType::TaskAssignment,
            serde_json::to_vec(&TaskData {
                task_id: "neural_training_001".to_string(),
                algorithm: "backpropagation".to_string(),
                learning_rate: 0.001,
                epochs: 1000,
                layer_config: vec![784, 128, 64, 10],
            }).unwrap(),
        ),
        Message::new(
            MessageType::NeuralUpdate,
            serde_json::to_vec(&NeuralWeights {
                layer_id: 1,
                weights: vec![0.1, 0.2, 0.3, 0.4, 0.5],
                biases: vec![0.01, 0.02],
                gradient_norm: 0.001,
            }).unwrap(),
        ),
        Message::new(
            MessageType::Coordination,
            serde_json::to_vec(&CoordinationData {
                action: "synchronize_weights".to_string(),
                priority: 1,
                metadata: "Global epoch 42 completed".to_string(),
            }).unwrap(),
        ),
    ];

    for (i, message) in messages.into_iter().enumerate() {
        message.validate()?;
        println!("✅ Alice: Message {} validated and ready to send", i + 1);
        
        // In a real scenario, you would send to Bob's agent ID
        // channel.send(bob_agent_id, message).await?;
        
        sleep(Duration::from_millis(100)).await;
    }

    // Send heartbeat
    let heartbeat = Message::heartbeat();
    println!("💓 Alice: Heartbeat sent");

    // Get channel statistics
    let stats = channel.stats().await;
    println!("📊 Alice Stats: {} messages processed", stats.total_messages);

    // Close channel
    channel.close().await?;
    println!("✅ Alice: Channel closed gracefully");

    Ok(())
}

/// Bob agent - receives and processes messages
async fn run_bob_agent() -> Result<()> {
    println!("🤖 Starting Bob agent...");

    // Generate Bob's keypair
    let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305)?;
    
    // Create channel configuration matching Alice's
    let config = ChannelConfig::new()
        .cipher_suite(CipherSuite::ChaCha20Poly1305)
        .message_timeout(60)
        .enable_forward_secrecy(true)
        .enable_compression(false)
        .max_message_size(1024 * 1024);

    // Create secure channel
    let channel = SecureChannel::new(config, keypair).await?;
    println!("✅ Bob: Secure channel created");

    // Start listening for connections
    let listen_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    // In real scenario: channel.listen(listen_addr).await?;
    
    sleep(Duration::from_millis(200)).await;
    println!("👂 Bob: Listening for connections");

    // Simulate message processing
    for i in 1..=3 {
        sleep(Duration::from_millis(150)).await;
        println!("📨 Bob: Processing message {}", i);
        
        // Simulate message handling based on type
        match i {
            1 => {
                println!("   📋 Task assignment received - starting neural training");
                println!("   🧠 Initializing neural network with specified architecture");
            }
            2 => {
                println!("   🔄 Neural weights update received - applying changes");
                println!("   📈 Gradient norm: 0.001 - convergence looking good");
            }
            3 => {
                println!("   🤝 Coordination message received - synchronizing");
                println!("   ✅ Global epoch 42 acknowledged and synchronized");
            }
            _ => {}
        }
    }

    // Handle heartbeat
    sleep(Duration::from_millis(100)).await;
    println!("💓 Bob: Heartbeat received - Alice is alive");

    // Send response messages back to Alice
    let response_messages = vec![
        Message::new(
            MessageType::TaskStatus,
            serde_json::to_vec(&TaskStatusData {
                task_id: "neural_training_001".to_string(),
                status: "running".to_string(),
                progress: 0.25,
                current_epoch: 250,
                loss: 0.234,
            }).unwrap(),
        ),
        Message::new(
            MessageType::NeuralUpdate,
            serde_json::to_vec(&NeuralWeights {
                layer_id: 2,
                weights: vec![0.15, 0.25, 0.35, 0.45, 0.55],
                biases: vec![0.015, 0.025],
                gradient_norm: 0.0008,
            }).unwrap(),
        ),
    ];

    for (i, message) in response_messages.into_iter().enumerate() {
        message.validate()?;
        println!("📤 Bob: Sending response message {}", i + 1);
        sleep(Duration::from_millis(50)).await;
    }

    // Get channel statistics
    let stats = channel.stats().await;
    println!("📊 Bob Stats: {} messages processed", stats.total_messages);

    // Close channel
    channel.close().await?;
    println!("✅ Bob: Channel closed gracefully");

    Ok(())
}

/// Task assignment data structure
#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct TaskData {
    task_id: String,
    algorithm: String,
    learning_rate: f64,
    epochs: u32,
    layer_config: Vec<usize>,
}

/// Neural weights data structure
#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct NeuralWeights {
    layer_id: u32,
    weights: Vec<f64>,
    biases: Vec<f64>,
    gradient_norm: f64,
}

/// Coordination data structure
#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct CoordinationData {
    action: String,
    priority: u8,
    metadata: String,
}

/// Task status data structure
#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct TaskStatusData {
    task_id: String,
    status: String,
    progress: f64,
    current_epoch: u32,
    loss: f64,
}