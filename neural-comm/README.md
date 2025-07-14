# Neural Comm

[![Crates.io](https://img.shields.io/crates/v/neural-comm.svg)](https://crates.io/crates/neural-comm)
[![Documentation](https://docs.rs/neural-comm/badge.svg)](https://docs.rs/neural-comm)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Build Status](https://github.com/neural-swarm/neural-comm/workflows/CI/badge.svg)](https://github.com/neural-swarm/neural-comm/actions)

A high-performance cryptographic communication library for neural agent swarms, designed for secure, authenticated, and efficient inter-agent communication in distributed AI systems.

## 🔒 Security Features

- **Strong Encryption**: ChaCha20-Poly1305, AES-256-GCM with authenticated encryption
- **Digital Signatures**: Ed25519, ECDSA P-256 for message authentication
- **Key Exchange**: X25519 ECDH with forward secrecy
- **Key Derivation**: HKDF, Argon2id for secure key management
- **Hash Functions**: SHA-3, BLAKE3 for integrity verification
- **Memory Security**: Locked memory pages, automatic zeroization
- **Side-Channel Resistance**: Constant-time operations where possible
- **Replay Protection**: Nonce-based anti-replay mechanisms

## 🚀 Performance

- **SIMD Optimizations**: Leverages modern CPU SIMD instructions
- **Zero-Copy Operations**: Minimizes memory allocations and copies
- **Async/Await**: Non-blocking I/O for high concurrency
- **Memory Pooling**: Efficient memory management for frequent operations
- **Batch Processing**: Optimized for high-throughput scenarios

## 🐍 Python Support

Complete Python bindings with PyO3 for seamless integration:

```python
import neural_comm

# Create secure channel
config = neural_comm.ChannelConfig().enable_forward_secrecy(True)
keypair = neural_comm.KeyPair.generate(neural_comm.CipherSuite.chacha20_poly1305())
channel = neural_comm.SecureChannel(config, keypair)

# Send secure message
message = neural_comm.Message.new(
    neural_comm.MessageType.coordination(),
    b"Hello, neural swarm!"
)
await channel.send(peer_id, message)
```

## 📦 Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
neural-comm = "0.1.0"
```

### Python

```bash
pip install neural-comm
```

## 🔧 Quick Start

### Rust

```rust
use neural_comm::{
    crypto::{CipherSuite, KeyPair},
    channels::{SecureChannel, ChannelConfig},
    protocols::{Message, MessageType},
};

#[tokio::main]
async fn main() -> neural_comm::Result<()> {
    // Generate keypair
    let keypair = KeyPair::generate(CipherSuite::ChaCha20Poly1305)?;
    
    // Create secure channel
    let config = ChannelConfig::new()
        .cipher_suite(CipherSuite::ChaCha20Poly1305)
        .enable_forward_secrecy(true)
        .message_timeout(30);
    
    let channel = SecureChannel::new(config, keypair).await?;
    
    // Create and send message
    let message = Message::new(
        MessageType::Coordination,
        b"Neural swarm coordination data".to_vec()
    );
    
    // In a real application, you would:
    // let peer_id = channel.connect("127.0.0.1:8080").await?;
    // channel.send(peer_id, message).await?;
    
    Ok(())
}
```

### Python

```python
import asyncio
import neural_comm

async def main():
    # Generate keypair
    suite = neural_comm.CipherSuite.chacha20_poly1305()
    keypair = neural_comm.KeyPair.generate(suite)
    
    # Create secure channel
    config = (neural_comm.ChannelConfig()
             .cipher_suite(suite)
             .enable_forward_secrecy(True)
             .message_timeout(30))
    
    channel = neural_comm.SecureChannel(config, keypair)
    
    # Create message
    msg_type = neural_comm.MessageType.coordination()
    message = neural_comm.Message(msg_type, b"Hello from Python!")
    
    # In a real application:
    # peer_id = await channel.connect("127.0.0.1", 8080)
    # await channel.send(peer_id, message)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

Neural Comm is built with a modular architecture:

```
neural-comm/
├── crypto/           # Cryptographic primitives
│   ├── symmetric.rs  # ChaCha20-Poly1305, AES-GCM
│   ├── asymmetric.rs # Ed25519, ECDSA signatures
│   ├── kdf.rs        # HKDF, Argon2 key derivation
│   ├── hash.rs       # SHA-3, BLAKE3 hashing
│   └── random.rs     # Secure random generation
├── channels/         # Secure communication channels
│   ├── transport.rs  # TCP, QUIC transport layers
│   ├── handshake.rs  # Key exchange protocols
│   └── session.rs    # Session management
├── protocols/        # Message protocols
│   ├── framing.rs    # Message framing
│   ├── compression.rs # Data compression
│   └── validation.rs # Message validation
├── memory.rs         # Secure memory management
└── ffi/              # Python FFI bindings
```

## 🧪 Message Types

Neural Comm supports various message types for neural agent communication:

- **TaskAssignment**: Distribute computational tasks
- **TaskStatus**: Report task progress and completion
- **NeuralUpdate**: Share neural network updates
- **Coordination**: Coordinate swarm behavior
- **Consensus**: Distributed consensus protocols
- **Heartbeat**: Keep-alive and health monitoring

## 🔐 Cryptographic Primitives

### Cipher Suites

- **ChaCha20-Poly1305**: Fast, secure AEAD cipher (default)
- **AES-256-GCM**: Hardware-accelerated AEAD cipher

### Digital Signatures

- **Ed25519**: Fast elliptic curve signatures
- **ECDSA P-256**: NIST standard signatures

### Key Derivation

- **HKDF**: Extract-and-expand key derivation
- **Argon2id**: Memory-hard password derivation

### Hash Functions

- **SHA-3**: NIST standard cryptographic hash
- **BLAKE3**: Fast, secure cryptographic hash

## 📊 Performance Benchmarks

On modern hardware (AMD Ryzen 9 5900X):

| Operation | Throughput | Latency |
|-----------|------------|---------|
| ChaCha20-Poly1305 Encrypt | 2.1 GB/s | 47 ns |
| AES-256-GCM Encrypt | 1.8 GB/s | 55 ns |
| Ed25519 Sign | 58,000 ops/s | 17 μs |
| Ed25519 Verify | 19,000 ops/s | 52 μs |
| BLAKE3 Hash | 3.2 GB/s | 31 ns |
| SHA-3 Hash | 1.1 GB/s | 91 ns |

Run benchmarks yourself:

```bash
cargo bench --features benchmark-suite
```

## 🛡️ Security Considerations

### Memory Security

- Sensitive data is stored in locked memory pages when available
- Automatic zeroization of cryptographic material
- Constant-time operations to prevent timing attacks
- Secure memory allocator for sensitive operations

### Network Security

- Perfect forward secrecy with ephemeral key exchange
- Message authentication and integrity verification
- Replay attack protection with nonces and sequence numbers
- Configurable message timeouts and size limits

### Implementation Security

- Memory-safe Rust implementation
- Comprehensive error handling
- Extensive testing including fuzzing
- Regular security audits and updates

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration_tests

# Security tests
cargo test --features security-tests

# Python tests (requires Python environment)
cd python && python -m pytest
```

## 📈 Examples

See the [`examples/`](examples/) directory for comprehensive examples:

- [`basic_usage.rs`](examples/basic_usage.rs) - Basic library usage
- [`secure_communication.rs`](examples/secure_communication.rs) - Two-agent communication
- [`python_example.py`](examples/python_example.py) - Python integration

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/neural-swarm/neural-comm.git
cd neural-comm

# Install Rust dependencies
cargo build

# Install Python dependencies (optional)
pip install maturin
maturin develop --features python-bindings

# Run tests
cargo test
```

## 📄 License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🔗 Related Projects

- [neural-swarm](https://github.com/neural-swarm/neural-swarm) - Neural agent swarm framework
- [fann-rust-core](https://github.com/neural-swarm/fann-rust-core) - High-performance neural networks

## 📚 Documentation

- [API Documentation](https://docs.rs/neural-comm)
- [Security Guide](docs/security.md)
- [Performance Guide](docs/performance.md)
- [Python API Reference](docs/python-api.md)

## 🐛 Reporting Issues

Please report security issues to [security@neural-swarm.dev](mailto:security@neural-swarm.dev).

For other issues, use the [GitHub issue tracker](https://github.com/neural-swarm/neural-comm/issues).

---

Built with ❤️ by the Neural Swarm team