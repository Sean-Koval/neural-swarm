//! Hybrid Deployment Framework for Neural-Swarm Coordination
//!
//! This framework provides automatic deployment decision making between container
//! and WASM targets based on resource constraints and deployment requirements.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use tracing::{info, warn, error};

/// Deployment target types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentTarget {
    Container,
    Wasm,
    Hybrid,
}

/// Resource constraints for deployment decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory in bytes
    pub memory_limit: u64,
    /// Maximum CPU cores
    pub cpu_limit: f64,
    /// Network bandwidth in bits per second
    pub network_bandwidth: u64,
    /// Power budget in watts (for edge devices)
    pub power_budget: Option<f64>,
    /// Storage space in bytes
    pub storage_limit: u64,
    /// Startup time requirement in milliseconds
    pub startup_time_limit: u64,
}

/// Deployment environment characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEnvironment {
    /// Environment type
    pub environment_type: EnvironmentType,
    /// Available resources
    pub resources: ResourceConstraints,
    /// Security requirements
    pub security_level: SecurityLevel,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Operational constraints
    pub operational_constraints: OperationalConstraints,
}

/// Environment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentType {
    Cloud,
    Edge,
    Mobile,
    Embedded,
    Serverless,
    Kubernetes,
    Container,
}

/// Security levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Latency requirement in milliseconds
    pub latency_requirement: u64,
    /// Throughput requirement in requests per second
    pub throughput_requirement: u64,
    /// Availability requirement (0.0-1.0)
    pub availability_requirement: f64,
    /// Scalability requirement (max instances)
    pub scalability_requirement: u32,
}

/// Operational constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalConstraints {
    /// Auto-scaling enabled
    pub auto_scaling: bool,
    /// Monitoring level
    pub monitoring_level: MonitoringLevel,
    /// Backup requirements
    pub backup_required: bool,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
}

/// Monitoring levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringLevel {
    Basic,
    Standard,
    Advanced,
    Custom,
}

/// Deployment decision result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentDecision {
    /// Recommended deployment target
    pub target: DeploymentTarget,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Reasoning for the decision
    pub reasoning: Vec<String>,
    /// Alternative options
    pub alternatives: Vec<(DeploymentTarget, f64)>,
    /// Configuration recommendations
    pub configuration: DeploymentConfiguration,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfiguration {
    /// Container-specific config
    pub container_config: Option<ContainerConfig>,
    /// WASM-specific config
    pub wasm_config: Option<WasmConfig>,
    /// Hybrid-specific config
    pub hybrid_config: Option<HybridConfig>,
}

/// Container deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Base image
    pub base_image: String,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
    /// Port mappings
    pub port_mappings: Vec<PortMapping>,
    /// Volume mounts
    pub volume_mounts: Vec<VolumeMount>,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
}

/// Resource limits for containers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// CPU limit in cores
    pub cpu_limit: f64,
    /// Memory limit in bytes
    pub memory_limit: u64,
    /// Network bandwidth limit
    pub network_limit: Option<u64>,
    /// Disk I/O limit
    pub disk_io_limit: Option<u64>,
}

/// Port mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    /// Container port
    pub container_port: u16,
    /// Host port
    pub host_port: u16,
    /// Protocol
    pub protocol: String,
}

/// Volume mount configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    /// Source path
    pub source: String,
    /// Target path
    pub target: String,
    /// Mount type
    pub mount_type: String,
    /// Read-only flag
    pub read_only: bool,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Command to run
    pub command: Vec<String>,
    /// Interval in seconds
    pub interval: u64,
    /// Timeout in seconds
    pub timeout: u64,
    /// Retries
    pub retries: u32,
    /// Start period in seconds
    pub start_period: u64,
}

/// WASM deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    /// WASM features to enable
    pub features: Vec<String>,
    /// Memory limit in bytes
    pub memory_limit: u64,
    /// CPU limit (0.0-1.0)
    pub cpu_limit: f64,
    /// Network optimization level
    pub network_optimization: u8,
    /// Compression level
    pub compression_level: u8,
    /// Edge optimizations
    pub edge_optimizations: bool,
    /// Power-aware mode
    pub power_aware: bool,
}

/// Hybrid deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Primary deployment target
    pub primary_target: DeploymentTarget,
    /// Fallback target
    pub fallback_target: DeploymentTarget,
    /// Migration triggers
    pub migration_triggers: Vec<MigrationTrigger>,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Migration trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationTrigger {
    /// Trigger type
    pub trigger_type: TriggerType,
    /// Threshold value
    pub threshold: f64,
    /// Duration in seconds
    pub duration: u64,
    /// Action to take
    pub action: MigrationAction,
}

/// Trigger types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    CpuUtilization,
    MemoryUtilization,
    NetworkLatency,
    PowerUsage,
    ErrorRate,
    ResponseTime,
}

/// Migration actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationAction {
    MigrateToContainer,
    MigrateToWasm,
    ScaleUp,
    ScaleDown,
    Restart,
    Alert,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRandom,
    ResourceBased,
    LatencyBased,
}

/// Deployment decision engine
pub struct DeploymentDecisionEngine {
    /// Decision rules
    rules: Vec<DecisionRule>,
    /// Historical data
    history: Arc<RwLock<Vec<DeploymentDecision>>>,
    /// Performance metrics
    metrics: Arc<RwLock<HashMap<String, f64>>>,
}

/// Decision rule
#[derive(Debug, Clone)]
pub struct DecisionRule {
    /// Rule name
    pub name: String,
    /// Rule weight
    pub weight: f64,
    /// Evaluation function
    pub evaluator: fn(&DeploymentEnvironment) -> f64,
    /// Target recommendation
    pub target: DeploymentTarget,
}

impl DeploymentDecisionEngine {
    /// Create a new decision engine
    pub fn new() -> Self {
        let mut engine = Self {
            rules: Vec::new(),
            history: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Add default rules
        engine.add_default_rules();
        engine
    }
    
    /// Add default decision rules
    fn add_default_rules(&mut self) {
        // Container rules
        self.rules.push(DecisionRule {
            name: "high_memory_container".to_string(),
            weight: 0.8,
            evaluator: |env| {
                if env.resources.memory_limit > 2 * 1024 * 1024 * 1024 { // > 2GB
                    0.9
                } else {
                    0.1
                }
            },
            target: DeploymentTarget::Container,
        });
        
        self.rules.push(DecisionRule {
            name: "kubernetes_environment".to_string(),
            weight: 0.9,
            evaluator: |env| {
                match env.environment_type {
                    EnvironmentType::Kubernetes => 0.95,
                    EnvironmentType::Cloud => 0.8,
                    _ => 0.2,
                }
            },
            target: DeploymentTarget::Container,
        });
        
        // WASM rules
        self.rules.push(DecisionRule {
            name: "low_memory_wasm".to_string(),
            weight: 0.9,
            evaluator: |env| {
                if env.resources.memory_limit < 128 * 1024 * 1024 { // < 128MB
                    0.9
                } else {
                    0.1
                }
            },
            target: DeploymentTarget::Wasm,
        });
        
        self.rules.push(DecisionRule {
            name: "edge_environment".to_string(),
            weight: 0.85,
            evaluator: |env| {
                match env.environment_type {
                    EnvironmentType::Edge => 0.9,
                    EnvironmentType::Mobile => 0.8,
                    EnvironmentType::Embedded => 0.95,
                    _ => 0.2,
                }
            },
            target: DeploymentTarget::Wasm,
        });
        
        self.rules.push(DecisionRule {
            name: "power_constraints".to_string(),
            weight: 0.7,
            evaluator: |env| {
                if let Some(power_budget) = env.resources.power_budget {
                    if power_budget < 5.0 { // < 5W
                        0.9
                    } else {
                        0.3
                    }
                } else {
                    0.0
                }
            },
            target: DeploymentTarget::Wasm,
        });
        
        self.rules.push(DecisionRule {
            name: "fast_startup".to_string(),
            weight: 0.6,
            evaluator: |env| {
                if env.resources.startup_time_limit < 1000 { // < 1 second
                    0.8
                } else {
                    0.2
                }
            },
            target: DeploymentTarget::Wasm,
        });
        
        // Hybrid rules
        self.rules.push(DecisionRule {
            name: "high_availability".to_string(),
            weight: 0.8,
            evaluator: |env| {
                if env.performance_requirements.availability_requirement > 0.99 {
                    0.9
                } else {
                    0.1
                }
            },
            target: DeploymentTarget::Hybrid,
        });
        
        self.rules.push(DecisionRule {
            name: "variable_load".to_string(),
            weight: 0.7,
            evaluator: |env| {
                if env.operational_constraints.auto_scaling {
                    0.8
                } else {
                    0.2
                }
            },
            target: DeploymentTarget::Hybrid,
        });
    }
    
    /// Make deployment decision
    pub async fn make_decision(&self, environment: &DeploymentEnvironment) -> Result<DeploymentDecision> {
        let mut scores = HashMap::new();
        let mut reasoning = Vec::new();
        
        // Evaluate all rules
        for rule in &self.rules {
            let score = (rule.evaluator)(environment) * rule.weight;
            *scores.entry(rule.target).or_insert(0.0) += score;
            
            if score > 0.5 {
                reasoning.push(format!("Rule '{}' supports {:?} (score: {:.2})", 
                                     rule.name, rule.target, score));
            }
        }
        
        // Find best target
        let best_target = scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(target, score)| (*target, *score))
            .unwrap_or((DeploymentTarget::Container, 0.0));
        
        // Create alternatives list
        let mut alternatives: Vec<(DeploymentTarget, f64)> = scores.into_iter()
            .filter(|(target, _)| *target != best_target.0)
            .collect();
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Generate configuration
        let configuration = self.generate_configuration(best_target.0, environment).await?;
        
        let decision = DeploymentDecision {
            target: best_target.0,
            confidence: best_target.1 / 10.0, // Normalize to 0-1
            reasoning,
            alternatives,
            configuration,
        };
        
        // Store decision in history
        self.history.write().await.push(decision.clone());
        
        Ok(decision)
    }
    
    /// Generate deployment configuration
    async fn generate_configuration(
        &self,
        target: DeploymentTarget,
        environment: &DeploymentEnvironment,
    ) -> Result<DeploymentConfiguration> {
        let mut config = DeploymentConfiguration {
            container_config: None,
            wasm_config: None,
            hybrid_config: None,
        };
        
        match target {
            DeploymentTarget::Container => {
                config.container_config = Some(self.generate_container_config(environment));
            }
            DeploymentTarget::Wasm => {
                config.wasm_config = Some(self.generate_wasm_config(environment));
            }
            DeploymentTarget::Hybrid => {
                config.hybrid_config = Some(self.generate_hybrid_config(environment));
                config.container_config = Some(self.generate_container_config(environment));
                config.wasm_config = Some(self.generate_wasm_config(environment));
            }
        }
        
        Ok(config)
    }
    
    /// Generate container configuration
    fn generate_container_config(&self, environment: &DeploymentEnvironment) -> ContainerConfig {
        let base_image = match environment.environment_type {
            EnvironmentType::Edge => "neural-swarm/neuroplex:edge".to_string(),
            _ => "neural-swarm/neuroplex:latest".to_string(),
        };
        
        ContainerConfig {
            base_image,
            resource_limits: ResourceLimits {
                cpu_limit: environment.resources.cpu_limit,
                memory_limit: environment.resources.memory_limit,
                network_limit: Some(environment.resources.network_bandwidth),
                disk_io_limit: None,
            },
            environment_variables: self.generate_environment_variables(environment),
            port_mappings: vec![
                PortMapping {
                    container_port: 8080,
                    host_port: 8080,
                    protocol: "tcp".to_string(),
                },
                PortMapping {
                    container_port: 8081,
                    host_port: 8081,
                    protocol: "tcp".to_string(),
                },
            ],
            volume_mounts: vec![
                VolumeMount {
                    source: "/var/lib/neuroplex".to_string(),
                    target: "/var/lib/neuroplex".to_string(),
                    mount_type: "bind".to_string(),
                    read_only: false,
                },
            ],
            health_check: HealthCheckConfig {
                command: vec!["curl".to_string(), "-f".to_string(), 
                             "http://localhost:8081/health".to_string()],
                interval: 30,
                timeout: 10,
                retries: 3,
                start_period: 30,
            },
        }
    }
    
    /// Generate WASM configuration
    fn generate_wasm_config(&self, environment: &DeploymentEnvironment) -> WasmConfig {
        let edge_optimizations = matches!(environment.environment_type, 
                                       EnvironmentType::Edge | EnvironmentType::Mobile | EnvironmentType::Embedded);
        
        let power_aware = environment.resources.power_budget.is_some();
        
        WasmConfig {
            features: if edge_optimizations {
                vec!["edge".to_string(), "minimal".to_string()]
            } else {
                vec!["default".to_string()]
            },
            memory_limit: environment.resources.memory_limit,
            cpu_limit: environment.resources.cpu_limit,
            network_optimization: if edge_optimizations { 8 } else { 5 },
            compression_level: if edge_optimizations { 9 } else { 6 },
            edge_optimizations,
            power_aware,
        }
    }
    
    /// Generate hybrid configuration
    fn generate_hybrid_config(&self, environment: &DeploymentEnvironment) -> HybridConfig {
        let primary_target = if environment.resources.memory_limit < 512 * 1024 * 1024 {
            DeploymentTarget::Wasm
        } else {
            DeploymentTarget::Container
        };
        
        let fallback_target = if primary_target == DeploymentTarget::Wasm {
            DeploymentTarget::Container
        } else {
            DeploymentTarget::Wasm
        };
        
        HybridConfig {
            primary_target,
            fallback_target,
            migration_triggers: vec![
                MigrationTrigger {
                    trigger_type: TriggerType::MemoryUtilization,
                    threshold: 0.8,
                    duration: 60,
                    action: MigrationAction::MigrateToContainer,
                },
                MigrationTrigger {
                    trigger_type: TriggerType::CpuUtilization,
                    threshold: 0.9,
                    duration: 30,
                    action: MigrationAction::ScaleUp,
                },
                MigrationTrigger {
                    trigger_type: TriggerType::PowerUsage,
                    threshold: 0.8,
                    duration: 120,
                    action: MigrationAction::MigrateToWasm,
                },
            ],
            load_balancing: LoadBalancingStrategy::ResourceBased,
        }
    }
    
    /// Generate environment variables
    fn generate_environment_variables(&self, environment: &DeploymentEnvironment) -> HashMap<String, String> {
        let mut env_vars = HashMap::new();
        
        env_vars.insert("NEUROPLEX_ENVIRONMENT".to_string(), 
                       format!("{:?}", environment.environment_type).to_lowercase());
        env_vars.insert("NEUROPLEX_MEMORY_LIMIT".to_string(), 
                       environment.resources.memory_limit.to_string());
        env_vars.insert("NEUROPLEX_CPU_LIMIT".to_string(), 
                       environment.resources.cpu_limit.to_string());
        env_vars.insert("NEUROPLEX_SECURITY_LEVEL".to_string(), 
                       format!("{:?}", environment.security_level).to_lowercase());
        
        if let Some(power_budget) = environment.resources.power_budget {
            env_vars.insert("NEUROPLEX_POWER_BUDGET".to_string(), 
                           power_budget.to_string());
        }
        
        env_vars
    }
    
    /// Get deployment history
    pub async fn get_history(&self) -> Vec<DeploymentDecision> {
        self.history.read().await.clone()
    }
    
    /// Update metrics
    pub async fn update_metrics(&self, metrics: HashMap<String, f64>) {
        let mut current_metrics = self.metrics.write().await;
        current_metrics.extend(metrics);
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.read().await.clone()
    }
}

impl Default for DeploymentDecisionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified deployment API
pub struct UnifiedDeploymentAPI {
    decision_engine: DeploymentDecisionEngine,
    deployments: Arc<RwLock<HashMap<String, DeploymentStatus>>>,
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentStatus {
    /// Deployment ID
    pub id: String,
    /// Current target
    pub target: DeploymentTarget,
    /// Status
    pub status: DeploymentState,
    /// Configuration
    pub configuration: DeploymentConfiguration,
    /// Metrics
    pub metrics: HashMap<String, f64>,
    /// Created timestamp
    pub created_at: u64,
    /// Updated timestamp
    pub updated_at: u64,
}

/// Deployment states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentState {
    Pending,
    Deploying,
    Running,
    Stopping,
    Stopped,
    Failed,
    Migrating,
}

impl UnifiedDeploymentAPI {
    /// Create new unified deployment API
    pub fn new() -> Self {
        Self {
            decision_engine: DeploymentDecisionEngine::new(),
            deployments: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Deploy with automatic target selection
    pub async fn deploy(&self, 
                       deployment_id: &str, 
                       environment: &DeploymentEnvironment) -> Result<DeploymentStatus> {
        
        // Make deployment decision
        let decision = self.decision_engine.make_decision(environment).await?;
        
        info!("Deployment decision for {}: {:?} (confidence: {:.2})", 
              deployment_id, decision.target, decision.confidence);
        
        // Create deployment status
        let status = DeploymentStatus {
            id: deployment_id.to_string(),
            target: decision.target,
            status: DeploymentState::Deploying,
            configuration: decision.configuration,
            metrics: HashMap::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        // Store deployment
        self.deployments.write().await.insert(deployment_id.to_string(), status.clone());
        
        // Trigger actual deployment (this would be implemented by specific deployers)
        self.trigger_deployment(&status).await?;
        
        Ok(status)
    }
    
    /// Trigger actual deployment
    async fn trigger_deployment(&self, status: &DeploymentStatus) -> Result<()> {
        match status.target {
            DeploymentTarget::Container => {
                info!("Triggering container deployment for {}", status.id);
                // Call container deployment logic
            }
            DeploymentTarget::Wasm => {
                info!("Triggering WASM deployment for {}", status.id);
                // Call WASM deployment logic
            }
            DeploymentTarget::Hybrid => {
                info!("Triggering hybrid deployment for {}", status.id);
                // Call hybrid deployment logic
            }
        }
        
        Ok(())
    }
    
    /// Get deployment status
    pub async fn get_status(&self, deployment_id: &str) -> Option<DeploymentStatus> {
        self.deployments.read().await.get(deployment_id).cloned()
    }
    
    /// List all deployments
    pub async fn list_deployments(&self) -> Vec<DeploymentStatus> {
        self.deployments.read().await.values().cloned().collect()
    }
    
    /// Update deployment
    pub async fn update_deployment(&self, deployment_id: &str, status: DeploymentState) -> Result<()> {
        let mut deployments = self.deployments.write().await;
        
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.status = status;
            deployment.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        } else {
            return Err(anyhow!("Deployment not found: {}", deployment_id));
        }
        
        Ok(())
    }
    
    /// Remove deployment
    pub async fn remove_deployment(&self, deployment_id: &str) -> Result<()> {
        let mut deployments = self.deployments.write().await;
        
        if deployments.remove(deployment_id).is_some() {
            info!("Removed deployment: {}", deployment_id);
            Ok(())
        } else {
            Err(anyhow!("Deployment not found: {}", deployment_id))
        }
    }
    
    /// Migrate deployment
    pub async fn migrate_deployment(&self, 
                                  deployment_id: &str, 
                                  new_target: DeploymentTarget) -> Result<()> {
        let mut deployments = self.deployments.write().await;
        
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.target = new_target;
            deployment.status = DeploymentState::Migrating;
            deployment.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            info!("Migrating deployment {} to {:?}", deployment_id, new_target);
        } else {
            return Err(anyhow!("Deployment not found: {}", deployment_id));
        }
        
        Ok(())
    }
}

impl Default for UnifiedDeploymentAPI {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_deployment_decision_engine() {
        let engine = DeploymentDecisionEngine::new();
        
        // Test edge environment
        let edge_env = DeploymentEnvironment {
            environment_type: EnvironmentType::Edge,
            resources: ResourceConstraints {
                memory_limit: 64 * 1024 * 1024, // 64MB
                cpu_limit: 0.5,
                network_bandwidth: 1_000_000, // 1Mbps
                power_budget: Some(2.0), // 2W
                storage_limit: 1024 * 1024 * 1024, // 1GB
                startup_time_limit: 500, // 500ms
            },
            security_level: SecurityLevel::Medium,
            performance_requirements: PerformanceRequirements {
                latency_requirement: 100,
                throughput_requirement: 1000,
                availability_requirement: 0.95,
                scalability_requirement: 10,
            },
            operational_constraints: OperationalConstraints {
                auto_scaling: false,
                monitoring_level: MonitoringLevel::Basic,
                backup_required: false,
                compliance_requirements: Vec::new(),
            },
        };
        
        let decision = engine.make_decision(&edge_env).await.unwrap();
        assert_eq!(decision.target, DeploymentTarget::Wasm);
        assert!(decision.confidence > 0.5);
        
        // Test cloud environment
        let cloud_env = DeploymentEnvironment {
            environment_type: EnvironmentType::Kubernetes,
            resources: ResourceConstraints {
                memory_limit: 4 * 1024 * 1024 * 1024, // 4GB
                cpu_limit: 4.0,
                network_bandwidth: 100_000_000, // 100Mbps
                power_budget: None,
                storage_limit: 100 * 1024 * 1024 * 1024, // 100GB
                startup_time_limit: 30_000, // 30s
            },
            security_level: SecurityLevel::High,
            performance_requirements: PerformanceRequirements {
                latency_requirement: 500,
                throughput_requirement: 10000,
                availability_requirement: 0.99,
                scalability_requirement: 100,
            },
            operational_constraints: OperationalConstraints {
                auto_scaling: true,
                monitoring_level: MonitoringLevel::Advanced,
                backup_required: true,
                compliance_requirements: vec!["SOC2".to_string()],
            },
        };
        
        let decision = engine.make_decision(&cloud_env).await.unwrap();
        assert!(decision.target == DeploymentTarget::Container || decision.target == DeploymentTarget::Hybrid);
    }
    
    #[tokio::test]
    async fn test_unified_deployment_api() {
        let api = UnifiedDeploymentAPI::new();
        
        let env = DeploymentEnvironment {
            environment_type: EnvironmentType::Edge,
            resources: ResourceConstraints {
                memory_limit: 128 * 1024 * 1024, // 128MB
                cpu_limit: 1.0,
                network_bandwidth: 10_000_000, // 10Mbps
                power_budget: Some(5.0), // 5W
                storage_limit: 2 * 1024 * 1024 * 1024, // 2GB
                startup_time_limit: 1000, // 1s
            },
            security_level: SecurityLevel::Medium,
            performance_requirements: PerformanceRequirements {
                latency_requirement: 200,
                throughput_requirement: 500,
                availability_requirement: 0.95,
                scalability_requirement: 5,
            },
            operational_constraints: OperationalConstraints {
                auto_scaling: false,
                monitoring_level: MonitoringLevel::Standard,
                backup_required: false,
                compliance_requirements: Vec::new(),
            },
        };
        
        let status = api.deploy("test-deployment", &env).await.unwrap();
        assert_eq!(status.id, "test-deployment");
        assert_eq!(status.status, DeploymentState::Deploying);
        
        let retrieved = api.get_status("test-deployment").await.unwrap();
        assert_eq!(retrieved.id, "test-deployment");
        
        let deployments = api.list_deployments().await;
        assert_eq!(deployments.len(), 1);
        
        api.update_deployment("test-deployment", DeploymentState::Running).await.unwrap();
        
        let updated = api.get_status("test-deployment").await.unwrap();
        assert_eq!(updated.status, DeploymentState::Running);
        
        api.remove_deployment("test-deployment").await.unwrap();
        let removed = api.get_status("test-deployment").await;
        assert!(removed.is_none());
    }
}