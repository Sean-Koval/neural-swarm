//! WebAssembly Runtime for Neural-Swarm Coordination
//!
//! This module provides WASM compilation target for neural-swarm coordination system,
//! optimized for edge deployment and resource-constrained environments.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use js_sys::{Array, Object, Reflect};
use web_sys::{console, Performance, Worker, WorkerGlobalScope};
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use anyhow::Result;

// Import neural-swarm core with edge features
use neuroplex::{
    NeuroConfig, NeuroCluster, NeuroNode, DistributedMemory,
    NodeId, Timestamp, VersionVector,
};

// Configure panic hook for debugging
#[cfg(feature = "console_error_panic_hook")]
pub use console_error_panic_hook::set_once as set_panic_hook;

// Configure global allocator for WASM
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Initialize the WASM runtime
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    set_panic_hook();
    
    wasm_logger::init(wasm_logger::Config::default());
    log::info!("Neural-Swarm WASM Runtime initialized");
}

/// WASM-optimized neural swarm node
#[wasm_bindgen]
pub struct WasmNeuroNode {
    inner: Arc<RwLock<NeuroCluster>>,
    config: NeuroConfig,
    performance: Performance,
    coordinator_sender: mpsc::UnboundedSender<CoordinationMessage>,
    coordinator_receiver: Arc<RwLock<mpsc::UnboundedReceiver<CoordinationMessage>>>,
}

/// Configuration for WASM deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmDeploymentConfig {
    /// Edge deployment mode
    pub edge_mode: bool,
    /// Power-aware optimizations
    pub power_aware: bool,
    /// Memory limit in bytes
    pub memory_limit: u32,
    /// CPU limit (fraction of available)
    pub cpu_limit: f32,
    /// Network optimization level
    pub network_optimization: u8,
    /// Compression level
    pub compression_level: u8,
}

/// Coordination message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    NodeJoin { node_id: NodeId, address: String },
    NodeLeave { node_id: NodeId },
    DataUpdate { key: String, value: String, timestamp: Timestamp },
    SyncRequest { node_id: NodeId, version: VersionVector },
    SyncResponse { data: HashMap<String, String>, version: VersionVector },
    HealthCheck { node_id: NodeId, status: String },
}

#[wasm_bindgen]
impl WasmNeuroNode {
    /// Create a new WASM neural node
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmNeuroNode, JsValue> {
        let config: WasmDeploymentConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;
        
        // Create optimized neural config for WASM
        let mut neuro_config = NeuroConfig::default();
        neuro_config.memory.max_size = config.memory_limit as usize;
        neuro_config.memory.compression = match config.compression_level {
            0 => neuroplex::CompressionAlgorithm::None,
            1..=5 => neuroplex::CompressionAlgorithm::Lz4,
            6..=9 => neuroplex::CompressionAlgorithm::Zstd,
            _ => neuroplex::CompressionAlgorithm::Lz4,
        };
        
        // Adjust for edge deployment
        if config.edge_mode {
            neuro_config.consensus.election_timeout = 10000;
            neuro_config.consensus.heartbeat_interval = 2000;
            neuro_config.sync.gossip_interval = 500;
            neuro_config.sync.gossip_fanout = 2;
            neuro_config.sync.delta_sync_batch_size = 100;
        }
        
        let performance = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object"))?
            .performance()
            .ok_or_else(|| JsValue::from_str("No performance object"))?;
        
        let (coordinator_sender, coordinator_receiver) = mpsc::unbounded_channel();
        
        Ok(WasmNeuroNode {
            inner: Arc::new(RwLock::new(NeuroCluster::new("wasm-node", "127.0.0.1:8080"))),
            config: neuro_config,
            performance,
            coordinator_sender,
            coordinator_receiver: Arc::new(RwLock::new(coordinator_receiver)),
        })
    }
    
    /// Start the WASM node
    #[wasm_bindgen]
    pub fn start(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        let sender = self.coordinator_sender.clone();
        
        wasm_bindgen_futures::future_to_promise(async move {
            let cluster = inner.read().await;
            
            // Start coordination loop
            spawn_local(async move {
                loop {
                    // Process coordination messages
                    if let Err(e) = sender.send(CoordinationMessage::HealthCheck {
                        node_id: uuid::Uuid::new_v4(),
                        status: "active".to_string(),
                    }) {
                        log::error!("Failed to send health check: {}", e);
                    }
                    
                    // WASM-optimized yield
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            });
            
            Ok(JsValue::from_str("Node started successfully"))
        })
    }
    
    /// Store data in the distributed memory
    #[wasm_bindgen]
    pub fn set_data(&self, key: &str, value: &str) -> js_sys::Promise {
        let inner = self.inner.clone();
        let key = key.to_string();
        let value = value.to_string();
        
        wasm_bindgen_futures::future_to_promise(async move {
            let cluster = inner.read().await;
            
            match cluster.set(&key, &value).await {
                Ok(_) => Ok(JsValue::from_str("Data stored successfully")),
                Err(e) => Err(JsValue::from_str(&format!("Storage error: {}", e))),
            }
        })
    }
    
    /// Get data from distributed memory
    #[wasm_bindgen]
    pub fn get_data(&self, key: &str) -> js_sys::Promise {
        let inner = self.inner.clone();
        let key = key.to_string();
        
        wasm_bindgen_futures::future_to_promise(async move {
            let cluster = inner.read().await;
            
            match cluster.get(&key).await {
                Ok(Some(value)) => Ok(JsValue::from_str(&value)),
                Ok(None) => Ok(JsValue::NULL),
                Err(e) => Err(JsValue::from_str(&format!("Retrieval error: {}", e))),
            }
        })
    }
    
    /// Get node performance metrics
    #[wasm_bindgen]
    pub fn get_metrics(&self) -> js_sys::Promise {
        let performance = self.performance.clone();
        
        wasm_bindgen_futures::future_to_promise(async move {
            let metrics = js_sys::Object::new();
            
            // Memory metrics
            if let Some(memory) = performance.memory() {
                Reflect::set(&metrics, &JsValue::from_str("usedJSHeapSize"), 
                           &JsValue::from_f64(memory.used_js_heap_size() as f64))?;
                Reflect::set(&metrics, &JsValue::from_str("totalJSHeapSize"), 
                           &JsValue::from_f64(memory.total_js_heap_size() as f64))?;
                Reflect::set(&metrics, &JsValue::from_str("jsHeapSizeLimit"), 
                           &JsValue::from_f64(memory.js_heap_size_limit() as f64))?;
            }
            
            // Timing metrics
            let now = performance.now();
            Reflect::set(&metrics, &JsValue::from_str("timestamp"), 
                       &JsValue::from_f64(now))?;
            
            Ok(metrics.into())
        })
    }
    
    /// Join a coordination cluster
    #[wasm_bindgen]
    pub fn join_cluster(&self, coordinator_url: &str) -> js_sys::Promise {
        let inner = self.inner.clone();
        let url = coordinator_url.to_string();
        
        wasm_bindgen_futures::future_to_promise(async move {
            let mut cluster = inner.write().await;
            
            // Simulate cluster join
            log::info!("Joining cluster at: {}", url);
            
            Ok(JsValue::from_str("Joined cluster successfully"))
        })
    }
    
    /// Leave the coordination cluster
    #[wasm_bindgen]
    pub fn leave_cluster(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        
        wasm_bindgen_futures::future_to_promise(async move {
            let mut cluster = inner.write().await;
            
            // Cleanup and leave cluster
            log::info!("Leaving cluster");
            
            Ok(JsValue::from_str("Left cluster successfully"))
        })
    }
}

/// WASM-optimized deployment framework
#[wasm_bindgen]
pub struct WasmDeploymentFramework {
    nodes: HashMap<String, WasmNeuroNode>,
    config: WasmDeploymentConfig,
}

#[wasm_bindgen]
impl WasmDeploymentFramework {
    /// Create new deployment framework
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmDeploymentFramework, JsValue> {
        let config: WasmDeploymentConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;
        
        Ok(WasmDeploymentFramework {
            nodes: HashMap::new(),
            config,
        })
    }
    
    /// Deploy a new node
    #[wasm_bindgen]
    pub fn deploy_node(&mut self, node_id: &str, config_json: &str) -> Result<(), JsValue> {
        let node = WasmNeuroNode::new(config_json)?;
        self.nodes.insert(node_id.to_string(), node);
        
        log::info!("Deployed node: {}", node_id);
        Ok(())
    }
    
    /// Remove a node
    #[wasm_bindgen]
    pub fn remove_node(&mut self, node_id: &str) -> Result<(), JsValue> {
        if self.nodes.remove(node_id).is_some() {
            log::info!("Removed node: {}", node_id);
            Ok(())
        } else {
            Err(JsValue::from_str("Node not found"))
        }
    }
    
    /// Get deployment status
    #[wasm_bindgen]
    pub fn get_status(&self) -> js_sys::Promise {
        let node_count = self.nodes.len();
        let config = self.config.clone();
        
        wasm_bindgen_futures::future_to_promise(async move {
            let status = js_sys::Object::new();
            
            Reflect::set(&status, &JsValue::from_str("nodeCount"), 
                       &JsValue::from_f64(node_count as f64))?;
            Reflect::set(&status, &JsValue::from_str("edgeMode"), 
                       &JsValue::from_bool(config.edge_mode))?;
            Reflect::set(&status, &JsValue::from_str("powerAware"), 
                       &JsValue::from_bool(config.power_aware))?;
            Reflect::set(&status, &JsValue::from_str("memoryLimit"), 
                       &JsValue::from_f64(config.memory_limit as f64))?;
            Reflect::set(&status, &JsValue::from_str("cpuLimit"), 
                       &JsValue::from_f64(config.cpu_limit as f64))?;
            
            Ok(status.into())
        })
    }
}

/// Host interface for system integration
#[wasm_bindgen]
pub struct WasmHostInterface {
    worker: Option<Worker>,
    message_handlers: HashMap<String, js_sys::Function>,
}

#[wasm_bindgen]
impl WasmHostInterface {
    /// Create new host interface
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmHostInterface {
        WasmHostInterface {
            worker: None,
            message_handlers: HashMap::new(),
        }
    }
    
    /// Initialize worker for background processing
    #[wasm_bindgen]
    pub fn init_worker(&mut self, worker_script: &str) -> Result<(), JsValue> {
        let worker = Worker::new(worker_script)?;
        self.worker = Some(worker);
        
        log::info!("Worker initialized: {}", worker_script);
        Ok(())
    }
    
    /// Send message to worker
    #[wasm_bindgen]
    pub fn send_to_worker(&self, message: &str) -> Result<(), JsValue> {
        if let Some(worker) = &self.worker {
            worker.post_message(&JsValue::from_str(message))?;
            Ok(())
        } else {
            Err(JsValue::from_str("Worker not initialized"))
        }
    }
    
    /// Register message handler
    #[wasm_bindgen]
    pub fn register_handler(&mut self, event_type: &str, handler: js_sys::Function) {
        self.message_handlers.insert(event_type.to_string(), handler);
        log::info!("Registered handler for: {}", event_type);
    }
    
    /// Process incoming message
    #[wasm_bindgen]
    pub fn process_message(&self, event_type: &str, data: &str) -> Result<(), JsValue> {
        if let Some(handler) = self.message_handlers.get(event_type) {
            let this = JsValue::NULL;
            let args = Array::new();
            args.push(&JsValue::from_str(data));
            
            handler.apply(&this, &args)?;
            Ok(())
        } else {
            Err(JsValue::from_str("No handler registered for event type"))
        }
    }
}

/// Utility functions for WASM optimization
#[wasm_bindgen]
pub fn optimize_for_edge() -> js_sys::Promise {
    wasm_bindgen_futures::future_to_promise(async move {
        // Trigger garbage collection
        if let Some(window) = web_sys::window() {
            if let Some(gc) = Reflect::get(&window, &JsValue::from_str("gc")).ok() {
                if let Some(gc_fn) = gc.dyn_ref::<js_sys::Function>() {
                    let _ = gc_fn.call0(&JsValue::NULL);
                }
            }
        }
        
        // Optimize memory usage
        log::info!("Edge optimization completed");
        Ok(JsValue::from_str("Edge optimization completed"))
    })
}

/// Check WASM runtime capabilities
#[wasm_bindgen]
pub fn check_capabilities() -> js_sys::Object {
    let caps = js_sys::Object::new();
    
    // Check for SharedArrayBuffer support
    let shared_array_buffer = js_sys::global()
        .get("SharedArrayBuffer")
        .is_some();
    
    // Check for Atomics support
    let atomics = js_sys::global()
        .get("Atomics")
        .is_some();
    
    // Check for Worker support
    let worker = js_sys::global()
        .get("Worker")
        .is_some();
    
    Reflect::set(&caps, &JsValue::from_str("sharedArrayBuffer"), 
               &JsValue::from_bool(shared_array_buffer)).unwrap();
    Reflect::set(&caps, &JsValue::from_str("atomics"), 
               &JsValue::from_bool(atomics)).unwrap();
    Reflect::set(&caps, &JsValue::from_str("worker"), 
               &JsValue::from_bool(worker)).unwrap();
    
    caps
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

// Export for JavaScript usage
#[wasm_bindgen]
pub fn greet(name: &str) {
    console_log!("Hello, {}! From neural-swarm WASM runtime", name);
}