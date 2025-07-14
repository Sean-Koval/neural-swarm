//! Integration APIs and Plugin Architecture
//!
//! Extensible integration framework for neural swarm ecosystem.

use super::{Integration, IntegrationInfo, IntegrationEvent, IntegrationStatus};
use crate::{Result, NeuroError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// APIs integration implementation
pub struct ApisIntegration {
    info: IntegrationInfo,
    status: IntegrationStatus,
    config: ApisConfig,
    plugin_manager: PluginManager,
    event_system: EventSystem,
    monitoring_system: MonitoringSystem,
    configuration_manager: ConfigurationManager,
}

/// APIs configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApisConfig {
    /// Plugin directory
    pub plugin_directory: String,
    /// Event buffer size
    pub event_buffer_size: usize,
    /// Monitoring interval in seconds
    pub monitoring_interval: u64,
    /// Configuration validation enabled
    pub config_validation: bool,
    /// Hot reload enabled
    pub hot_reload: bool,
}

/// Plugin manager
#[derive(Debug, Clone)]
pub struct PluginManager {
    /// Registered plugins
    plugins: HashMap<String, PluginInfo>,
    /// Active plugin instances
    active_plugins: HashMap<Uuid, Box<dyn Plugin>>,
    /// Plugin dependencies
    plugin_dependencies: HashMap<String, Vec<String>>,
}

/// Plugin information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin type
    pub plugin_type: PluginType,
    /// Plugin capabilities
    pub capabilities: Vec<String>,
    /// Plugin dependencies
    pub dependencies: Vec<String>,
    /// Configuration schema
    pub config_schema: serde_json::Value,
    /// Plugin metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Plugin types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginType {
    Communication,
    Consensus,
    Memory,
    Neural,
    Orchestration,
    Monitoring,
    Security,
}

/// Plugin trait
pub trait Plugin: Send + Sync {
    /// Initialize plugin
    fn initialize(&mut self, config: &serde_json::Value) -> Result<()>;
    
    /// Start plugin
    fn start(&mut self) -> Result<()>;
    
    /// Stop plugin
    fn stop(&mut self) -> Result<()>;
    
    /// Get plugin info
    fn info(&self) -> &PluginInfo;
    
    /// Handle plugin event
    fn handle_event(&mut self, event: PluginEvent) -> Result<()>;
    
    /// Get plugin status
    fn status(&self) -> PluginStatus;
    
    /// Get plugin metrics
    fn metrics(&self) -> HashMap<String, f64>;
}

/// Plugin event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginEvent {
    /// System event
    System {
        event_type: String,
        payload: serde_json::Value,
    },
    /// Configuration change
    ConfigurationChanged {
        key: String,
        old_value: serde_json::Value,
        new_value: serde_json::Value,
    },
    /// Resource allocation
    ResourceAllocated {
        resource_id: String,
        amount: f64,
    },
    /// Plugin lifecycle event
    LifecycleEvent {
        plugin_id: String,
        event: LifecycleEventType,
    },
}

/// Plugin lifecycle event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleEventType {
    Loaded,
    Initialized,
    Started,
    Stopped,
    Unloaded,
    Error,
}

/// Plugin status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginStatus {
    Loaded,
    Initialized,
    Running,
    Stopped,
    Error { message: String },
}

/// Event system
#[derive(Debug, Clone)]
pub struct EventSystem {
    /// Event subscribers
    subscribers: HashMap<String, Vec<EventSubscriber>>,
    /// Event buffer
    event_buffer: Vec<SystemEvent>,
    /// Event router
    event_router: EventRouter,
    /// Event persistence
    event_persistence: EventPersistence,
}

/// Event subscriber
#[derive(Debug, Clone)]
pub struct EventSubscriber {
    /// Subscriber ID
    pub id: Uuid,
    /// Event filter
    pub filter: EventFilter,
    /// Callback
    pub callback: fn(&SystemEvent) -> Result<()>,
}

/// Event filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    /// Event types to match
    pub event_types: Vec<String>,
    /// Content filters
    pub content_filters: HashMap<String, serde_json::Value>,
    /// Source filters
    pub source_filters: Vec<String>,
}

/// System event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEvent {
    /// Event ID
    pub id: Uuid,
    /// Event type
    pub event_type: String,
    /// Event source
    pub source: String,
    /// Event payload
    pub payload: serde_json::Value,
    /// Event timestamp
    pub timestamp: u64,
    /// Event metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Event router
#[derive(Debug, Clone)]
pub struct EventRouter {
    /// Routing rules
    routing_rules: Vec<RoutingRule>,
    /// Route statistics
    route_stats: HashMap<String, RouteStats>,
}

/// Routing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Target subscribers
    pub targets: Vec<String>,
    /// Priority
    pub priority: u32,
}

/// Route statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteStats {
    /// Messages routed
    pub messages_routed: u64,
    /// Routing errors
    pub routing_errors: u64,
    /// Average routing time
    pub avg_routing_time: f64,
}

/// Event persistence
#[derive(Debug, Clone)]
pub struct EventPersistence {
    /// Event log
    event_log: Vec<SystemEvent>,
    /// Persistence configuration
    config: PersistenceConfig,
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Enable persistence
    pub enabled: bool,
    /// Log file path
    pub log_file: String,
    /// Log rotation size
    pub rotation_size: usize,
    /// Retention period in days
    pub retention_days: u32,
}

/// Monitoring system
#[derive(Debug, Clone)]
pub struct MonitoringSystem {
    /// Metrics collectors
    collectors: HashMap<String, MetricsCollector>,
    /// Alerting rules
    alerting_rules: Vec<AlertingRule>,
    /// Alert notifications
    alert_notifications: Vec<AlertNotification>,
}

/// Metrics collector
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Collector name
    pub name: String,
    /// Metric definitions
    pub metrics: HashMap<String, MetricDefinition>,
    /// Collection interval
    pub collection_interval: u64,
    /// Last collection time
    pub last_collection: u64,
}

/// Metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Metric description
    pub description: String,
    /// Metric labels
    pub labels: HashMap<String, String>,
    /// Metric unit
    pub unit: String,
}

/// Metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Alerting rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert labels
    pub labels: HashMap<String, String>,
}

/// Alert severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertNotification {
    /// Notification ID
    pub id: Uuid,
    /// Alert rule name
    pub rule_name: String,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Notification timestamp
    pub timestamp: u64,
    /// Notification status
    pub status: NotificationStatus,
}

/// Notification status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationStatus {
    Pending,
    Sent,
    Failed,
    Acknowledged,
}

/// Configuration manager
#[derive(Debug, Clone)]
pub struct ConfigurationManager {
    /// Configuration sources
    sources: Vec<ConfigurationSource>,
    /// Configuration cache
    cache: HashMap<String, serde_json::Value>,
    /// Configuration validation
    validation: ConfigurationValidation,
}

/// Configuration source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationSource {
    /// Source name
    pub name: String,
    /// Source type
    pub source_type: ConfigurationSourceType,
    /// Source URI
    pub uri: String,
    /// Source priority
    pub priority: u32,
    /// Watch for changes
    pub watch: bool,
}

/// Configuration source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationSourceType {
    File,
    Environment,
    CommandLine,
    DistributedService,
    Api,
}

/// Configuration validation
#[derive(Debug, Clone)]
pub struct ConfigurationValidation {
    /// Schema registry
    schema_registry: HashMap<String, serde_json::Value>,
    /// Validation rules
    validation_rules: Vec<ValidationRule>,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Configuration path
    pub path: String,
    /// Validation expression
    pub expression: String,
    /// Error message
    pub error_message: String,
}

impl ApisIntegration {
    /// Create a new APIs integration
    pub fn new(info: IntegrationInfo) -> Self {
        Self {
            info,
            status: IntegrationStatus::Initializing,
            config: ApisConfig {
                plugin_directory: "./plugins".to_string(),
                event_buffer_size: 10000,
                monitoring_interval: 60,
                config_validation: true,
                hot_reload: true,
            },
            plugin_manager: PluginManager {
                plugins: HashMap::new(),
                active_plugins: HashMap::new(),
                plugin_dependencies: HashMap::new(),
            },
            event_system: EventSystem {
                subscribers: HashMap::new(),
                event_buffer: Vec::new(),
                event_router: EventRouter {
                    routing_rules: Vec::new(),
                    route_stats: HashMap::new(),
                },
                event_persistence: EventPersistence {
                    event_log: Vec::new(),
                    config: PersistenceConfig {
                        enabled: true,
                        log_file: "./logs/events.log".to_string(),
                        rotation_size: 100_000_000, // 100MB
                        retention_days: 30,
                    },
                },
            },
            monitoring_system: MonitoringSystem {
                collectors: HashMap::new(),
                alerting_rules: Vec::new(),
                alert_notifications: Vec::new(),
            },
            configuration_manager: ConfigurationManager {
                sources: Vec::new(),
                cache: HashMap::new(),
                validation: ConfigurationValidation {
                    schema_registry: HashMap::new(),
                    validation_rules: Vec::new(),
                },
            },
        }
    }
    
    /// Register plugin
    pub fn register_plugin(&mut self, plugin_info: PluginInfo) -> Result<()> {
        // Validate plugin dependencies
        for dep in &plugin_info.dependencies {
            if !self.plugin_manager.plugins.contains_key(dep) {
                return Err(NeuroError::integration(format!(
                    "Plugin dependency not found: {}", dep
                )));
            }
        }
        
        // Register plugin
        self.plugin_manager.plugins.insert(plugin_info.name.clone(), plugin_info.clone());
        self.plugin_manager.plugin_dependencies.insert(plugin_info.name.clone(), plugin_info.dependencies);
        
        Ok(())
    }
    
    /// Load plugin instance
    pub fn load_plugin(&mut self, plugin_name: &str, config: &serde_json::Value) -> Result<Uuid> {
        let plugin_info = self.plugin_manager.plugins.get(plugin_name)
            .ok_or_else(|| NeuroError::integration("Plugin not found"))?;
        
        // Create plugin instance (simplified - in real implementation would load from dynamic library)
        let plugin_instance = self.create_plugin_instance(plugin_info)?;
        
        let instance_id = Uuid::new_v4();
        self.plugin_manager.active_plugins.insert(instance_id, plugin_instance);
        
        // Initialize plugin
        if let Some(plugin) = self.plugin_manager.active_plugins.get_mut(&instance_id) {
            plugin.initialize(config)?;
        }
        
        Ok(instance_id)
    }
    
    /// Publish event
    pub fn publish_event(&mut self, event: SystemEvent) -> Result<()> {
        // Add to event buffer
        self.event_system.event_buffer.push(event.clone());
        
        // Route event to subscribers
        self.route_event(&event)?;
        
        // Persist event if enabled
        if self.event_system.event_persistence.config.enabled {
            self.event_system.event_persistence.event_log.push(event);
        }
        
        Ok(())
    }
    
    /// Subscribe to events
    pub fn subscribe_to_events(&mut self, subscriber: EventSubscriber) -> Result<()> {
        for event_type in &subscriber.filter.event_types {
            self.event_system.subscribers
                .entry(event_type.clone())
                .or_insert_with(Vec::new)
                .push(subscriber.clone());
        }
        
        Ok(())
    }
    
    /// Collect metrics
    pub fn collect_metrics(&mut self) -> Result<HashMap<String, f64>> {
        let mut all_metrics = HashMap::new();
        
        // Collect plugin metrics
        for (instance_id, plugin) in &self.plugin_manager.active_plugins {
            let plugin_metrics = plugin.metrics();
            for (metric_name, metric_value) in plugin_metrics {
                all_metrics.insert(format!("plugin.{}.{}", instance_id, metric_name), metric_value);
            }
        }
        
        // Collect system metrics
        all_metrics.insert("system.event_buffer_size".to_string(), self.event_system.event_buffer.len() as f64);
        all_metrics.insert("system.active_plugins".to_string(), self.plugin_manager.active_plugins.len() as f64);
        all_metrics.insert("system.registered_plugins".to_string(), self.plugin_manager.plugins.len() as f64);
        
        Ok(all_metrics)
    }
    
    /// Update configuration
    pub fn update_configuration(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        // Validate configuration if enabled
        if self.config.config_validation {
            self.validate_configuration(key, &value)?;
        }
        
        let old_value = self.configuration_manager.cache.get(key).cloned();
        self.configuration_manager.cache.insert(key.to_string(), value.clone());
        
        // Notify plugins of configuration change
        let event = PluginEvent::ConfigurationChanged {
            key: key.to_string(),
            old_value: old_value.unwrap_or(serde_json::Value::Null),
            new_value: value,
        };
        
        for plugin in self.plugin_manager.active_plugins.values_mut() {
            plugin.handle_event(event.clone())?;
        }
        
        Ok(())
    }
    
    /// Get configuration value
    pub fn get_configuration(&self, key: &str) -> Option<&serde_json::Value> {
        self.configuration_manager.cache.get(key)
    }
    
    /// Get integration statistics
    pub fn get_integration_stats(&self) -> IntegrationStats {
        IntegrationStats {
            total_plugins: self.plugin_manager.plugins.len(),
            active_plugins: self.plugin_manager.active_plugins.len(),
            total_events: self.event_system.event_buffer.len(),
            total_subscribers: self.event_system.subscribers.values().map(|v| v.len()).sum(),
            total_metrics: self.monitoring_system.collectors.len(),
            total_alerts: self.monitoring_system.alert_notifications.len(),
        }
    }
    
    /// Create plugin instance
    fn create_plugin_instance(&self, plugin_info: &PluginInfo) -> Result<Box<dyn Plugin>> {
        // Simplified plugin creation - in real implementation would use dynamic loading
        match plugin_info.plugin_type {
            PluginType::Communication => Ok(Box::new(CommunicationPlugin::new(plugin_info.clone()))),
            PluginType::Consensus => Ok(Box::new(ConsensusPlugin::new(plugin_info.clone()))),
            PluginType::Memory => Ok(Box::new(MemoryPlugin::new(plugin_info.clone()))),
            PluginType::Neural => Ok(Box::new(NeuralPlugin::new(plugin_info.clone()))),
            PluginType::Orchestration => Ok(Box::new(OrchestrationPlugin::new(plugin_info.clone()))),
            PluginType::Monitoring => Ok(Box::new(MonitoringPlugin::new(plugin_info.clone()))),
            PluginType::Security => Ok(Box::new(SecurityPlugin::new(plugin_info.clone()))),
        }
    }
    
    /// Route event to subscribers
    fn route_event(&mut self, event: &SystemEvent) -> Result<()> {
        if let Some(subscribers) = self.event_system.subscribers.get(&event.event_type) {
            for subscriber in subscribers {
                if self.event_matches_filter(&subscriber.filter, event) {
                    (subscriber.callback)(event)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if event matches filter
    fn event_matches_filter(&self, filter: &EventFilter, event: &SystemEvent) -> bool {
        // Check event type
        if !filter.event_types.contains(&event.event_type) {
            return false;
        }
        
        // Check source filter
        if !filter.source_filters.is_empty() && !filter.source_filters.contains(&event.source) {
            return false;
        }
        
        // Check content filters
        for (key, expected_value) in &filter.content_filters {
            if let Some(actual_value) = event.payload.get(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
    
    /// Validate configuration
    fn validate_configuration(&self, key: &str, value: &serde_json::Value) -> Result<()> {
        // Check validation rules
        for rule in &self.configuration_manager.validation.validation_rules {
            if rule.path == key {
                // Simplified validation - in real implementation would use expression evaluator
                if value.is_null() {
                    return Err(NeuroError::integration(&rule.error_message));
                }
            }
        }
        
        Ok(())
    }
    
    /// Get current timestamp
    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

/// Integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStats {
    pub total_plugins: usize,
    pub active_plugins: usize,
    pub total_events: usize,
    pub total_subscribers: usize,
    pub total_metrics: usize,
    pub total_alerts: usize,
}

/// Simplified plugin implementations for testing
pub struct CommunicationPlugin {
    info: PluginInfo,
    status: PluginStatus,
}

impl CommunicationPlugin {
    pub fn new(info: PluginInfo) -> Self {
        Self {
            info,
            status: PluginStatus::Loaded,
        }
    }
}

impl Plugin for CommunicationPlugin {
    fn initialize(&mut self, _config: &serde_json::Value) -> Result<()> {
        self.status = PluginStatus::Initialized;
        Ok(())
    }
    
    fn start(&mut self) -> Result<()> {
        self.status = PluginStatus::Running;
        Ok(())
    }
    
    fn stop(&mut self) -> Result<()> {
        self.status = PluginStatus::Stopped;
        Ok(())
    }
    
    fn info(&self) -> &PluginInfo {
        &self.info
    }
    
    fn handle_event(&mut self, _event: PluginEvent) -> Result<()> {
        Ok(())
    }
    
    fn status(&self) -> PluginStatus {
        self.status.clone()
    }
    
    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("messages_sent".to_string(), 100.0);
        metrics.insert("messages_received".to_string(), 95.0);
        metrics
    }
}

// Similar simplified implementations for other plugin types
pub struct ConsensusPlugin { info: PluginInfo, status: PluginStatus }
pub struct MemoryPlugin { info: PluginInfo, status: PluginStatus }
pub struct NeuralPlugin { info: PluginInfo, status: PluginStatus }
pub struct OrchestrationPlugin { info: PluginInfo, status: PluginStatus }
pub struct MonitoringPlugin { info: PluginInfo, status: PluginStatus }
pub struct SecurityPlugin { info: PluginInfo, status: PluginStatus }

// Implement Plugin trait for all simplified plugins with similar structure
macro_rules! impl_plugin {
    ($plugin_type:ident) => {
        impl $plugin_type {
            pub fn new(info: PluginInfo) -> Self {
                Self { info, status: PluginStatus::Loaded }
            }
        }
        
        impl Plugin for $plugin_type {
            fn initialize(&mut self, _config: &serde_json::Value) -> Result<()> {
                self.status = PluginStatus::Initialized;
                Ok(())
            }
            
            fn start(&mut self) -> Result<()> {
                self.status = PluginStatus::Running;
                Ok(())
            }
            
            fn stop(&mut self) -> Result<()> {
                self.status = PluginStatus::Stopped;
                Ok(())
            }
            
            fn info(&self) -> &PluginInfo {
                &self.info
            }
            
            fn handle_event(&mut self, _event: PluginEvent) -> Result<()> {
                Ok(())
            }
            
            fn status(&self) -> PluginStatus {
                self.status.clone()
            }
            
            fn metrics(&self) -> HashMap<String, f64> {
                HashMap::new()
            }
        }
    };
}

impl_plugin!(ConsensusPlugin);
impl_plugin!(MemoryPlugin);
impl_plugin!(NeuralPlugin);
impl_plugin!(OrchestrationPlugin);
impl_plugin!(MonitoringPlugin);
impl_plugin!(SecurityPlugin);

impl Integration for ApisIntegration {
    fn initialize(&mut self, config: &serde_json::Value) -> Result<()> {
        // Parse configuration
        if let Ok(apis_config) = serde_json::from_value::<ApisConfig>(config.clone()) {
            self.config = apis_config;
        }
        
        // Initialize configuration sources
        self.configuration_manager.sources.push(ConfigurationSource {
            name: "default".to_string(),
            source_type: ConfigurationSourceType::File,
            uri: "./config/default.json".to_string(),
            priority: 100,
            watch: true,
        });
        
        self.status = IntegrationStatus::Initializing;
        Ok(())
    }
    
    fn start(&mut self) -> Result<()> {
        self.status = IntegrationStatus::Running;
        Ok(())
    }
    
    fn stop(&mut self) -> Result<()> {
        // Stop all plugins
        for plugin in self.plugin_manager.active_plugins.values_mut() {
            plugin.stop()?;
        }
        
        self.status = IntegrationStatus::Stopped;
        Ok(())
    }
    
    fn info(&self) -> &IntegrationInfo {
        &self.info
    }
    
    fn handle_event(&mut self, event: IntegrationEvent) -> Result<()> {
        // Convert integration event to system event
        let system_event = SystemEvent {
            id: Uuid::new_v4(),
            event_type: match event {
                IntegrationEvent::TaskAssigned { .. } => "task_assigned".to_string(),
                IntegrationEvent::TaskCompleted { .. } => "task_completed".to_string(),
                IntegrationEvent::AgentRegistered { .. } => "agent_registered".to_string(),
                IntegrationEvent::AgentDeregistered { .. } => "agent_deregistered".to_string(),
                IntegrationEvent::ConfigurationChanged { .. } => "configuration_changed".to_string(),
            },
            source: "integration".to_string(),
            payload: serde_json::to_value(&event).unwrap_or(serde_json::Value::Null),
            timestamp: self.current_timestamp(),
            metadata: HashMap::new(),
        };
        
        self.publish_event(system_event)?;
        
        Ok(())
    }
    
    fn status(&self) -> IntegrationStatus {
        self.status.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_apis_integration_creation() {
        let info = IntegrationInfo {
            name: "apis".to_string(),
            version: "1.0.0".to_string(),
            description: "Test integration".to_string(),
            capabilities: vec![],
            dependencies: vec![],
            config_schema: serde_json::json!({}),
        };
        
        let integration = ApisIntegration::new(info);
        assert_eq!(integration.status(), IntegrationStatus::Initializing);
    }
    
    #[test]
    fn test_plugin_registration() {
        let info = IntegrationInfo {
            name: "apis".to_string(),
            version: "1.0.0".to_string(),
            description: "Test integration".to_string(),
            capabilities: vec![],
            dependencies: vec![],
            config_schema: serde_json::json!({}),
        };
        
        let mut integration = ApisIntegration::new(info);
        
        let plugin_info = PluginInfo {
            name: "test_plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Test plugin".to_string(),
            plugin_type: PluginType::Communication,
            capabilities: vec!["messaging".to_string()],
            dependencies: vec![],
            config_schema: serde_json::json!({}),
            metadata: HashMap::new(),
        };
        
        assert!(integration.register_plugin(plugin_info).is_ok());
        assert_eq!(integration.plugin_manager.plugins.len(), 1);
    }
}