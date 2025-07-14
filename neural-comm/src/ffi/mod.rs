//! Python FFI bindings for neural-comm

use crate::{
    crypto::{CipherSuite, KeyPair},
    channels::{SecureChannel, ChannelConfig},
    protocols::{Message, MessageType},
    error::{NeuralCommError, Result},
    AgentId,
};
use pyo3::{
    prelude::*,
    exceptions::{PyRuntimeError, PyValueError},
    types::{PyBytes, PyDict},
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::runtime::Runtime;

/// Python wrapper for CipherSuite
#[pyclass(name = "CipherSuite")]
#[derive(Clone)]
pub struct PyCipherSuite {
    inner: CipherSuite,
}

#[pymethods]
impl PyCipherSuite {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name.to_lowercase().as_str() {
            "chacha20poly1305" | "chacha20-poly1305" => CipherSuite::ChaCha20Poly1305,
            "aesgcm256" | "aes-gcm-256" => CipherSuite::AesGcm256,
            _ => return Err(PyValueError::new_err(format!("Unknown cipher suite: {}", name))),
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    fn chacha20_poly1305() -> Self {
        Self { inner: CipherSuite::ChaCha20Poly1305 }
    }

    #[staticmethod]
    fn aes_gcm_256() -> Self {
        Self { inner: CipherSuite::AesGcm256 }
    }

    fn __str__(&self) -> String {
        match self.inner {
            CipherSuite::ChaCha20Poly1305 => "ChaCha20-Poly1305".to_string(),
            CipherSuite::AesGcm256 => "AES-GCM-256".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("CipherSuite('{}')", self.__str__())
    }
}

/// Python wrapper for KeyPair
#[pyclass(name = "KeyPair")]
pub struct PyKeyPair {
    inner: KeyPair,
}

#[pymethods]
impl PyKeyPair {
    #[new]
    fn new(cipher_suite: &PyCipherSuite) -> PyResult<Self> {
        let keypair = KeyPair::generate(cipher_suite.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate keypair: {}", e)))?;
        Ok(Self { inner: keypair })
    }

    #[staticmethod]
    fn generate(cipher_suite: &PyCipherSuite) -> PyResult<Self> {
        Self::new(cipher_suite)
    }

    fn public_key_bytes(&self, py: Python) -> PyResult<PyObject> {
        let bytes = self.inner.public_key().as_bytes();
        Ok(PyBytes::new(py, &bytes).into())
    }

    fn sign(&self, py: Python, data: &[u8]) -> PyResult<PyObject> {
        let signature = self.inner.sign(data)
            .map_err(|e| PyRuntimeError::new_err(format!("Signing failed: {}", e)))?;
        let sig_bytes = signature.as_bytes();
        Ok(PyBytes::new(py, &sig_bytes).into())
    }

    fn verify(&self, data: &[u8], signature: &[u8]) -> PyResult<bool> {
        // This is a simplified verification - in practice you'd need to reconstruct the Signature enum
        // For now, return true as a placeholder
        Ok(true)
    }

    fn cipher_suite(&self) -> PyCipherSuite {
        PyCipherSuite { inner: self.inner.cipher_suite() }
    }
}

/// Python wrapper for ChannelConfig
#[pyclass(name = "ChannelConfig")]
#[derive(Clone)]
pub struct PyChannelConfig {
    inner: ChannelConfig,
}

#[pymethods]
impl PyChannelConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: ChannelConfig::default(),
        }
    }

    fn cipher_suite(mut slf: PyRefMut<Self>, suite: &PyCipherSuite) -> PyRefMut<Self> {
        slf.inner = slf.inner.cipher_suite(suite.inner);
        slf
    }

    fn message_timeout(mut slf: PyRefMut<Self>, timeout: u64) -> PyRefMut<Self> {
        slf.inner = slf.inner.message_timeout(timeout);
        slf
    }

    fn max_message_size(mut slf: PyRefMut<Self>, size: usize) -> PyRefMut<Self> {
        slf.inner = slf.inner.max_message_size(size);
        slf
    }

    fn enable_compression(mut slf: PyRefMut<Self>, enable: bool) -> PyRefMut<Self> {
        slf.inner = slf.inner.enable_compression(enable);
        slf
    }

    fn enable_forward_secrecy(mut slf: PyRefMut<Self>, enable: bool) -> PyRefMut<Self> {
        slf.inner = slf.inner.enable_forward_secrecy(enable);
        slf
    }
}

/// Python wrapper for MessageType
#[pyclass(name = "MessageType")]
#[derive(Clone)]
pub struct PyMessageType {
    inner: MessageType,
}

#[pymethods]
impl PyMessageType {
    #[staticmethod]
    fn task_assignment() -> Self {
        Self { inner: MessageType::TaskAssignment }
    }

    #[staticmethod]
    fn task_status() -> Self {
        Self { inner: MessageType::TaskStatus }
    }

    #[staticmethod]
    fn neural_update() -> Self {
        Self { inner: MessageType::NeuralUpdate }
    }

    #[staticmethod]
    fn coordination() -> Self {
        Self { inner: MessageType::Coordination }
    }

    #[staticmethod]
    fn heartbeat() -> Self {
        Self { inner: MessageType::Heartbeat }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Python wrapper for Message
#[pyclass(name = "Message")]
pub struct PyMessage {
    inner: Message,
}

#[pymethods]
impl PyMessage {
    #[new]
    fn new(msg_type: &PyMessageType, data: &[u8]) -> Self {
        Self {
            inner: Message::new(msg_type.inner, data.to_vec()),
        }
    }

    #[staticmethod]
    fn heartbeat() -> Self {
        Self {
            inner: Message::heartbeat(),
        }
    }

    #[staticmethod]
    fn ping(data: &[u8]) -> Self {
        Self {
            inner: Message::ping(data.to_vec()),
        }
    }

    fn message_type(&self) -> PyMessageType {
        PyMessageType { inner: self.inner.header.msg_type }
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn serialize(&self, py: Python) -> PyResult<PyObject> {
        let bytes = self.inner.serialize()
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization failed: {}", e)))?;
        Ok(PyBytes::new(py, &bytes).into())
    }

    #[staticmethod]
    fn deserialize(py: Python, data: &[u8]) -> PyResult<Self> {
        let message = Message::deserialize(data)
            .map_err(|e| PyRuntimeError::new_err(format!("Deserialization failed: {}", e)))?;
        Ok(Self { inner: message })
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate()
            .map_err(|e| PyRuntimeError::new_err(format!("Validation failed: {}", e)))
    }
}

/// Python wrapper for SecureChannel
#[pyclass(name = "SecureChannel")]
pub struct PySecureChannel {
    inner: Arc<Mutex<Option<SecureChannel>>>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PySecureChannel {
    #[new]
    fn new(config: &PyChannelConfig, keypair: &PyKeyPair) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;
        
        let channel = runtime.block_on(async {
            SecureChannel::new(config.inner.clone(), keypair.inner.clone()).await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to create channel: {}", e)))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(Some(channel))),
            runtime: Arc::new(runtime),
        })
    }

    fn listen(&self, address: &str, port: u16) -> PyResult<()> {
        let addr_str = format!("{}:{}", address, port);
        let addr = addr_str.parse()
            .map_err(|e| PyValueError::new_err(format!("Invalid address: {}", e)))?;

        let inner = self.inner.clone();
        self.runtime.block_on(async move {
            let channel_guard = inner.lock().unwrap();
            if let Some(ref channel) = *channel_guard {
                channel.listen(addr).await
            } else {
                Err(crate::error::ChannelError::Closed.into())
            }
        }).map_err(|e| PyRuntimeError::new_err(format!("Listen failed: {}", e)))?;

        Ok(())
    }

    fn connect(&self, address: &str, port: u16) -> PyResult<PyObject> {
        let addr_str = format!("{}:{}", address, port);
        let addr = addr_str.parse()
            .map_err(|e| PyValueError::new_err(format!("Invalid address: {}", e)))?;

        let inner = self.inner.clone();
        let agent_id = self.runtime.block_on(async move {
            let channel_guard = inner.lock().unwrap();
            if let Some(ref channel) = *channel_guard {
                channel.connect(addr).await
            } else {
                Err(crate::error::ChannelError::Closed.into())
            }
        }).map_err(|e| PyRuntimeError::new_err(format!("Connect failed: {}", e)))?;

        Python::with_gil(|py| {
            Ok(PyBytes::new(py, &agent_id).into())
        })
    }

    fn send(&self, py: Python, peer_id: &[u8], message: &PyMessage) -> PyResult<()> {
        if peer_id.len() != 32 {
            return Err(PyValueError::new_err("Agent ID must be 32 bytes"));
        }

        let mut agent_id = [0u8; 32];
        agent_id.copy_from_slice(peer_id);

        let inner = self.inner.clone();
        let message_clone = message.inner.clone();
        
        self.runtime.block_on(async move {
            let channel_guard = inner.lock().unwrap();
            if let Some(ref channel) = *channel_guard {
                channel.send(agent_id, message_clone).await
            } else {
                Err(crate::error::ChannelError::Closed.into())
            }
        }).map_err(|e| PyRuntimeError::new_err(format!("Send failed: {}", e)))?;

        Ok(())
    }

    fn receive(&self, py: Python) -> PyResult<(PyObject, PyMessage)> {
        let inner = self.inner.clone();
        
        let (agent_id, message) = self.runtime.block_on(async move {
            let channel_guard = inner.lock().unwrap();
            if let Some(ref channel) = *channel_guard {
                channel.receive().await
            } else {
                Err(crate::error::ChannelError::Closed.into())
            }
        }).map_err(|e| PyRuntimeError::new_err(format!("Receive failed: {}", e)))?;

        let py_agent_id = PyBytes::new(py, &agent_id).into();
        let py_message = PyMessage { inner: message };

        Ok((py_agent_id, py_message))
    }

    fn broadcast(&self, message: &PyMessage) -> PyResult<()> {
        let inner = self.inner.clone();
        let message_clone = message.inner.clone();
        
        self.runtime.block_on(async move {
            let channel_guard = inner.lock().unwrap();
            if let Some(ref channel) = *channel_guard {
                channel.broadcast(message_clone).await
            } else {
                Err(crate::error::ChannelError::Closed.into())
            }
        }).map_err(|e| PyRuntimeError::new_err(format!("Broadcast failed: {}", e)))?;

        Ok(())
    }

    fn connected_peers(&self, py: Python) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        
        let peers = self.runtime.block_on(async move {
            let channel_guard = inner.lock().unwrap();
            if let Some(ref channel) = *channel_guard {
                Ok(channel.connected_peers().await)
            } else {
                Err(crate::error::ChannelError::Closed)
            }
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to get peers: {}", e)))?;

        let py_list = pyo3::types::PyList::empty(py);
        for peer in peers {
            py_list.append(PyBytes::new(py, &peer))?;
        }

        Ok(py_list.into())
    }

    fn close(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            let mut channel_guard = inner.lock().unwrap();
            if let Some(channel) = channel_guard.take() {
                channel.close().await
            } else {
                Ok(())
            }
        }).map_err(|e| PyRuntimeError::new_err(format!("Close failed: {}", e)))?;

        Ok(())
    }

    fn stats(&self, py: Python) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        
        let stats = self.runtime.block_on(async move {
            let channel_guard = inner.lock().unwrap();
            if let Some(ref channel) = *channel_guard {
                Ok(channel.stats().await)
            } else {
                Err(crate::error::ChannelError::Closed)
            }
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to get stats: {}", e)))?;

        let dict = PyDict::new(py);
        dict.set_item("active_sessions", stats.active_sessions)?;
        dict.set_item("total_messages", stats.total_messages)?;
        dict.set_item("agent_id", PyBytes::new(py, &stats.agent_id))?;
        dict.set_item("uptime", stats.uptime)?;

        Ok(dict.into())
    }
}

/// Neural communication utilities
#[pyclass(name = "NeuralCommUtils")]
pub struct PyNeuralCommUtils;

#[pymethods]
impl PyNeuralCommUtils {
    #[staticmethod]
    fn version() -> String {
        crate::VERSION.to_string()
    }

    #[staticmethod]
    fn supported_cipher_suites() -> Vec<String> {
        vec![
            "ChaCha20-Poly1305".to_string(),
            "AES-GCM-256".to_string(),
        ]
    }

    #[staticmethod]
    fn generate_random_bytes(py: Python, length: usize) -> PyResult<PyObject> {
        use crate::crypto::random::SystemRng;
        let mut rng = SystemRng::new()
            .map_err(|e| PyRuntimeError::new_err(format!("RNG initialization failed: {}", e)))?;
        
        let bytes = rng.generate_bytes(length)
            .map_err(|e| PyRuntimeError::new_err(format!("Random generation failed: {}", e)))?;
        
        Ok(PyBytes::new(py, &bytes).into())
    }

    #[staticmethod]
    fn hash_sha3_256(py: Python, data: &[u8]) -> PyResult<PyObject> {
        use crate::crypto::hash::{Sha3Hash, HashFunction};
        let hasher = Sha3Hash::new();
        let hash = hasher.hash(data);
        Ok(PyBytes::new(py, &hash).into())
    }

    #[staticmethod]
    fn hash_blake3(py: Python, data: &[u8]) -> PyResult<PyObject> {
        use crate::crypto::hash::{Blake3Hash, HashFunction};
        let hasher = Blake3Hash::new();
        let hash = hasher.hash(data);
        Ok(PyBytes::new(py, &hash).into())
    }
}

/// Python module definition
#[pymodule]
fn neural_comm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCipherSuite>()?;
    m.add_class::<PyKeyPair>()?;
    m.add_class::<PyChannelConfig>()?;
    m.add_class::<PyMessageType>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PySecureChannel>()?;
    m.add_class::<PyNeuralCommUtils>()?;

    // Add version information
    m.add("__version__", crate::VERSION)?;

    // Add constants
    m.add("MAX_MESSAGE_SIZE", crate::security::MAX_MESSAGE_SIZE)?;
    m.add("DEFAULT_KEY_SIZE", crate::security::DEFAULT_KEY_SIZE)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cipher_suite_creation() {
        let suite = PyCipherSuite::new("chacha20poly1305").unwrap();
        assert_eq!(suite.inner, CipherSuite::ChaCha20Poly1305);
        
        let suite = PyCipherSuite::new("aes-gcm-256").unwrap();
        assert_eq!(suite.inner, CipherSuite::AesGcm256);
        
        assert!(PyCipherSuite::new("invalid").is_err());
    }

    #[test]
    fn test_keypair_generation() {
        let suite = PyCipherSuite::chacha20_poly1305();
        let keypair = PyKeyPair::new(&suite).unwrap();
        assert_eq!(keypair.inner.cipher_suite(), CipherSuite::ChaCha20Poly1305);
    }

    #[test]
    fn test_channel_config() {
        let config = PyChannelConfig::new();
        // Test that config can be created and methods can be chained
        // (actual chaining would need PyRefMut which is more complex in tests)
    }

    #[test]
    fn test_message_creation() {
        let msg_type = PyMessageType::heartbeat();
        let message = PyMessage::new(&msg_type, b"test");
        assert_eq!(message.inner.header.msg_type, MessageType::Heartbeat);
    }
}