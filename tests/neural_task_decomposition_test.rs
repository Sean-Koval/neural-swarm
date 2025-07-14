//! Comprehensive Neural Task Decomposition Test Suite
//!
//! This test suite validates the core neural task decomposition functionality,
//! including heuristic methods, neural methods, hybrid strategies, and integration
//! with the swarm coordination system.

#[cfg(test)]
mod neural_task_decomposition_tests {
    use super::*;
    use crate::neural::{NeuralNetwork, XorNetwork, CascadeNetwork};
    use crate::agents::{Agent, AgentId, AgentMetadata, AgentConfig};
    use crate::coordination::{SwarmCoordinator, CoordinationStrategy};
    use crate::memory::{MemoryManager, MemoryConfig, MemoryEntry};
    use crate::test_utils::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use uuid::Uuid;
    use serde::{Deserialize, Serialize};

    /// Task decomposition strategy types
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum DecompositionStrategy {
        Heuristic,
        Neural,
        Hybrid,
    }

    /// Task representation for decomposition
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Task {
        pub id: String,
        pub name: String,
        pub description: String,
        pub complexity: f32,
        pub dependencies: Vec<String>,
        pub priority: TaskPriority,
        pub estimated_duration: u64,
        pub required_capabilities: Vec<String>,
        pub resources: TaskResources,
    }

    /// Task priority levels
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum TaskPriority {
        Critical,
        High,
        Medium,
        Low,
    }

    /// Task resource requirements
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskResources {
        pub memory_mb: u32,
        pub cpu_cores: f32,
        pub network_bandwidth: u32,
        pub storage_gb: f32,
    }

    /// Task decomposition result
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DecompositionResult {
        pub subtasks: Vec<Task>,
        pub execution_graph: TaskGraph,
        pub estimated_completion_time: u64,
        pub resource_allocation: HashMap<String, TaskResources>,
        pub strategy_used: DecompositionStrategy,
        pub confidence_score: f32,
    }

    /// Task execution graph (DAG)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskGraph {
        pub nodes: Vec<TaskNode>,
        pub edges: Vec<TaskEdge>,
        pub critical_path: Vec<String>,
        pub parallelizable_groups: Vec<Vec<String>>,
    }

    /// Task graph node
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskNode {
        pub task_id: String,
        pub level: u32,
        pub can_parallelize: bool,
        pub blocking_tasks: Vec<String>,
    }

    /// Task graph edge (dependency)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskEdge {
        pub from_task: String,
        pub to_task: String,
        pub dependency_type: DependencyType,
        pub strength: f32,
    }

    /// Dependency types between tasks
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum DependencyType {
        Sequential,
        DataFlow,
        Resource,
        Synchronization,
    }

    /// Neural task decomposer trait
    #[async_trait::async_trait]
    pub trait NeuralTaskDecomposer: Send + Sync {
        /// Decompose a task using the specified strategy
        async fn decompose_task(
            &self,
            task: &Task,
            strategy: DecompositionStrategy,
            context: &DecompositionContext,
        ) -> Result<DecompositionResult, DecompositionError>;

        /// Validate decomposition result
        async fn validate_decomposition(
            &self,
            original_task: &Task,
            result: &DecompositionResult,
        ) -> Result<ValidationResult, DecompositionError>;

        /// Optimize task graph for execution
        async fn optimize_task_graph(
            &self,
            graph: &TaskGraph,
            constraints: &OptimizationConstraints,
        ) -> Result<TaskGraph, DecompositionError>;

        /// Estimate execution time for task graph
        async fn estimate_execution_time(
            &self,
            graph: &TaskGraph,
            resource_pool: &ResourcePool,
        ) -> Result<u64, DecompositionError>;
    }

    /// Context for task decomposition
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DecompositionContext {
        pub available_agents: Vec<AgentId>,
        pub resource_constraints: ResourceConstraints,
        pub previous_decompositions: Vec<HistoricalDecomposition>,
        pub domain_knowledge: HashMap<String, String>,
        pub optimization_goals: Vec<OptimizationGoal>,
    }

    /// Resource constraints for decomposition
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ResourceConstraints {
        pub max_memory_mb: u32,
        pub max_cpu_cores: f32,
        pub max_network_bandwidth: u32,
        pub max_storage_gb: f32,
        pub max_execution_time: u64,
    }

    /// Historical decomposition data for learning
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HistoricalDecomposition {
        pub original_task: Task,
        pub decomposition_result: DecompositionResult,
        pub actual_execution_time: u64,
        pub success_rate: f32,
        pub performance_metrics: PerformanceMetrics,
    }

    /// Performance metrics for decomposition
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PerformanceMetrics {
        pub decomposition_time_ms: u64,
        pub memory_usage_mb: f32,
        pub cpu_utilization: f32,
        pub network_utilization: f32,
        pub accuracy_score: f32,
    }

    /// Optimization goals for decomposition
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum OptimizationGoal {
        MinimizeExecutionTime,
        MinimizeResourceUsage,
        MaximizeParallelism,
        MaximizeReliability,
        BalanceLoad,
    }

    /// Optimization constraints
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct OptimizationConstraints {
        pub max_parallel_tasks: u32,
        pub resource_limits: ResourceConstraints,
        pub deadline: Option<u64>,
        pub priority_weights: HashMap<TaskPriority, f32>,
    }

    /// Resource pool for execution
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ResourcePool {
        pub total_memory_mb: u32,
        pub total_cpu_cores: f32,
        pub total_network_bandwidth: u32,
        pub total_storage_gb: f32,
        pub available_agents: Vec<AgentId>,
    }

    /// Validation result for decomposition
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ValidationResult {
        pub is_valid: bool,
        pub issues: Vec<ValidationIssue>,
        pub completeness_score: f32,
        pub consistency_score: f32,
        pub feasibility_score: f32,
    }

    /// Validation issues
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ValidationIssue {
        pub issue_type: ValidationIssueType,
        pub severity: IssueSeverity,
        pub description: String,
        pub affected_tasks: Vec<String>,
        pub suggested_fix: Option<String>,
    }

    /// Types of validation issues
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum ValidationIssueType {
        MissingDependency,
        CircularDependency,
        ResourceOverallocation,
        TaskDuplication,
        IncompleteCoverage,
        InvalidGraph,
    }

    /// Issue severity levels
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum IssueSeverity {
        Critical,
        High,
        Medium,
        Low,
    }

    /// Decomposition errors
    #[derive(Debug, thiserror::Error)]
    pub enum DecompositionError {
        #[error("Task too complex for decomposition: {complexity}")]
        TaskTooComplex { complexity: f32 },
        
        #[error("Insufficient resources: required {required:?}, available {available:?}")]
        InsufficientResources { required: TaskResources, available: TaskResources },
        
        #[error("Circular dependency detected: {cycle:?}")]
        CircularDependency { cycle: Vec<String> },
        
        #[error("Neural network error: {message}")]
        NeuralNetworkError { message: String },
        
        #[error("Invalid task configuration: {message}")]
        InvalidTaskConfiguration { message: String },
        
        #[error("Decomposition timeout: {elapsed_ms}ms")]
        DecompositionTimeout { elapsed_ms: u64 },
        
        #[error("Memory allocation failed: {size_mb}MB")]
        MemoryAllocationFailed { size_mb: u32 },
        
        #[error("Agent assignment failed: {agent_id}")]
        AgentAssignmentFailed { agent_id: AgentId },
    }

    /// Mock neural task decomposer for testing
    pub struct MockNeuralTaskDecomposer {
        pub strategy: DecompositionStrategy,
        pub performance_metrics: Arc<RwLock<PerformanceMetrics>>,
        pub decomposition_history: Arc<RwLock<Vec<HistoricalDecomposition>>>,
    }

    impl MockNeuralTaskDecomposer {
        pub fn new(strategy: DecompositionStrategy) -> Self {
            Self {
                strategy,
                performance_metrics: Arc::new(RwLock::new(PerformanceMetrics {
                    decomposition_time_ms: 0,
                    memory_usage_mb: 0.0,
                    cpu_utilization: 0.0,
                    network_utilization: 0.0,
                    accuracy_score: 0.0,
                })),
                decomposition_history: Arc::new(RwLock::new(Vec::new())),
            }
        }

        /// Create a sample complex task for testing
        pub fn create_complex_task() -> Task {
            Task {
                id: "complex_task_001".to_string(),
                name: "Complex ML Pipeline".to_string(),
                description: "Build and deploy a complete machine learning pipeline with data preprocessing, model training, evaluation, and deployment".to_string(),
                complexity: 0.8,
                dependencies: vec!["data_source".to_string(), "compute_cluster".to_string()],
                priority: TaskPriority::High,
                estimated_duration: 3600, // 1 hour
                required_capabilities: vec![
                    "data_processing".to_string(),
                    "machine_learning".to_string(),
                    "model_deployment".to_string(),
                    "performance_monitoring".to_string(),
                ],
                resources: TaskResources {
                    memory_mb: 8192,
                    cpu_cores: 4.0,
                    network_bandwidth: 1000,
                    storage_gb: 100.0,
                },
            }
        }

        /// Create a sample simple task for testing
        pub fn create_simple_task() -> Task {
            Task {
                id: "simple_task_001".to_string(),
                name: "Data Validation".to_string(),
                description: "Validate input data format and quality".to_string(),
                complexity: 0.2,
                dependencies: vec!["input_data".to_string()],
                priority: TaskPriority::Medium,
                estimated_duration: 300, // 5 minutes
                required_capabilities: vec!["data_validation".to_string()],
                resources: TaskResources {
                    memory_mb: 512,
                    cpu_cores: 1.0,
                    network_bandwidth: 100,
                    storage_gb: 10.0,
                },
            }
        }

        /// Create default decomposition context
        pub fn create_default_context() -> DecompositionContext {
            DecompositionContext {
                available_agents: vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()],
                resource_constraints: ResourceConstraints {
                    max_memory_mb: 16384,
                    max_cpu_cores: 8.0,
                    max_network_bandwidth: 10000,
                    max_storage_gb: 1000.0,
                    max_execution_time: 7200,
                },
                previous_decompositions: vec![],
                domain_knowledge: HashMap::new(),
                optimization_goals: vec![
                    OptimizationGoal::MinimizeExecutionTime,
                    OptimizationGoal::MaximizeParallelism,
                ],
            }
        }

        /// Generate deterministic decomposition result for testing
        fn generate_test_decomposition(
            &self,
            task: &Task,
            strategy: DecompositionStrategy,
        ) -> DecompositionResult {
            let subtasks = match strategy {
                DecompositionStrategy::Heuristic => self.heuristic_decomposition(task),
                DecompositionStrategy::Neural => self.neural_decomposition(task),
                DecompositionStrategy::Hybrid => self.hybrid_decomposition(task),
            };

            let execution_graph = self.build_execution_graph(&subtasks);
            let resource_allocation = self.allocate_resources(&subtasks);

            DecompositionResult {
                subtasks,
                execution_graph,
                estimated_completion_time: task.estimated_duration,
                resource_allocation,
                strategy_used: strategy,
                confidence_score: 0.85,
            }
        }

        /// Heuristic-based decomposition
        fn heuristic_decomposition(&self, task: &Task) -> Vec<Task> {
            // Simple rule-based decomposition
            match task.complexity {
                x if x > 0.7 => {
                    // High complexity - break into multiple subtasks
                    vec![
                        Task {
                            id: format!("{}_subtask_1", task.id),
                            name: format!("{} - Setup", task.name),
                            description: "Initialize and setup task environment".to_string(),
                            complexity: 0.3,
                            dependencies: task.dependencies.clone(),
                            priority: task.priority.clone(),
                            estimated_duration: task.estimated_duration / 4,
                            required_capabilities: vec!["setup".to_string()],
                            resources: TaskResources {
                                memory_mb: task.resources.memory_mb / 4,
                                cpu_cores: task.resources.cpu_cores / 4.0,
                                network_bandwidth: task.resources.network_bandwidth / 4,
                                storage_gb: task.resources.storage_gb / 4.0,
                            },
                        },
                        Task {
                            id: format!("{}_subtask_2", task.id),
                            name: format!("{} - Processing", task.name),
                            description: "Main processing logic".to_string(),
                            complexity: 0.6,
                            dependencies: vec![format!("{}_subtask_1", task.id)],
                            priority: task.priority.clone(),
                            estimated_duration: task.estimated_duration / 2,
                            required_capabilities: task.required_capabilities.clone(),
                            resources: TaskResources {
                                memory_mb: task.resources.memory_mb / 2,
                                cpu_cores: task.resources.cpu_cores / 2.0,
                                network_bandwidth: task.resources.network_bandwidth / 2,
                                storage_gb: task.resources.storage_gb / 2.0,
                            },
                        },
                        Task {
                            id: format!("{}_subtask_3", task.id),
                            name: format!("{} - Finalization", task.name),
                            description: "Finalize and cleanup".to_string(),
                            complexity: 0.2,
                            dependencies: vec![format!("{}_subtask_2", task.id)],
                            priority: task.priority.clone(),
                            estimated_duration: task.estimated_duration / 4,
                            required_capabilities: vec!["cleanup".to_string()],
                            resources: TaskResources {
                                memory_mb: task.resources.memory_mb / 4,
                                cpu_cores: task.resources.cpu_cores / 4.0,
                                network_bandwidth: task.resources.network_bandwidth / 4,
                                storage_gb: task.resources.storage_gb / 4.0,
                            },
                        },
                    ]
                }
                _ => vec![task.clone()], // Simple task - no decomposition needed
            }
        }

        /// Neural network-based decomposition
        fn neural_decomposition(&self, task: &Task) -> Vec<Task> {
            // Mock neural network decomposition
            // In real implementation, this would use a trained transformer model
            let num_subtasks = ((task.complexity * 5.0) as usize).max(1);
            
            (0..num_subtasks)
                .map(|i| Task {
                    id: format!("{}_neural_subtask_{}", task.id, i),
                    name: format!("{} - Neural Subtask {}", task.name, i),
                    description: format!("Neural-generated subtask {}", i),
                    complexity: task.complexity / num_subtasks as f32,
                    dependencies: if i == 0 {
                        task.dependencies.clone()
                    } else {
                        vec![format!("{}_neural_subtask_{}", task.id, i - 1)]
                    },
                    priority: task.priority.clone(),
                    estimated_duration: task.estimated_duration / num_subtasks as u64,
                    required_capabilities: task.required_capabilities.clone(),
                    resources: TaskResources {
                        memory_mb: task.resources.memory_mb / num_subtasks as u32,
                        cpu_cores: task.resources.cpu_cores / num_subtasks as f32,
                        network_bandwidth: task.resources.network_bandwidth / num_subtasks as u32,
                        storage_gb: task.resources.storage_gb / num_subtasks as f32,
                    },
                })
                .collect()
        }

        /// Hybrid decomposition strategy
        fn hybrid_decomposition(&self, task: &Task) -> Vec<Task> {
            // Combine heuristic and neural approaches
            let heuristic_result = self.heuristic_decomposition(task);
            let neural_result = self.neural_decomposition(task);

            // Simple hybrid: take the better of the two based on complexity
            if task.complexity > 0.5 {
                neural_result
            } else {
                heuristic_result
            }
        }

        /// Build execution graph from subtasks
        fn build_execution_graph(&self, subtasks: &[Task]) -> TaskGraph {
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let mut level_map = HashMap::new();

            // Create nodes and calculate levels
            for task in subtasks {
                let level = self.calculate_task_level(task, subtasks);
                level_map.insert(task.id.clone(), level);
                
                nodes.push(TaskNode {
                    task_id: task.id.clone(),
                    level,
                    can_parallelize: task.dependencies.is_empty(),
                    blocking_tasks: task.dependencies.clone(),
                });
            }

            // Create edges based on dependencies
            for task in subtasks {
                for dep in &task.dependencies {
                    if subtasks.iter().any(|t| t.id == *dep) {
                        edges.push(TaskEdge {
                            from_task: dep.clone(),
                            to_task: task.id.clone(),
                            dependency_type: DependencyType::Sequential,
                            strength: 1.0,
                        });
                    }
                }
            }

            // Find critical path
            let critical_path = self.find_critical_path(subtasks);
            
            // Find parallelizable groups
            let parallelizable_groups = self.find_parallelizable_groups(subtasks, &level_map);

            TaskGraph {
                nodes,
                edges,
                critical_path,
                parallelizable_groups,
            }
        }

        /// Calculate task level in dependency graph
        fn calculate_task_level(&self, task: &Task, all_tasks: &[Task]) -> u32 {
            if task.dependencies.is_empty() {
                return 0;
            }

            let mut max_level = 0;
            for dep_id in &task.dependencies {
                if let Some(dep_task) = all_tasks.iter().find(|t| t.id == *dep_id) {
                    let dep_level = self.calculate_task_level(dep_task, all_tasks);
                    max_level = max_level.max(dep_level);
                }
            }

            max_level + 1
        }

        /// Find critical path in task graph
        fn find_critical_path(&self, tasks: &[Task]) -> Vec<String> {
            // Simple critical path: longest dependency chain
            let mut path = Vec::new();
            let mut current_task = tasks.iter().max_by_key(|t| t.estimated_duration);

            while let Some(task) = current_task {
                path.push(task.id.clone());
                current_task = task.dependencies.first()
                    .and_then(|dep_id| tasks.iter().find(|t| t.id == *dep_id));
            }

            path.reverse();
            path
        }

        /// Find parallelizable task groups
        fn find_parallelizable_groups(
            &self,
            tasks: &[Task],
            level_map: &HashMap<String, u32>,
        ) -> Vec<Vec<String>> {
            let mut groups = HashMap::new();
            
            for task in tasks {
                let level = level_map.get(&task.id).unwrap_or(&0);
                groups.entry(*level).or_insert_with(Vec::new).push(task.id.clone());
            }

            groups.into_values().collect()
        }

        /// Allocate resources to subtasks
        fn allocate_resources(&self, subtasks: &[Task]) -> HashMap<String, TaskResources> {
            subtasks.iter()
                .map(|task| (task.id.clone(), task.resources.clone()))
                .collect()
        }
    }

    #[async_trait::async_trait]
    impl NeuralTaskDecomposer for MockNeuralTaskDecomposer {
        async fn decompose_task(
            &self,
            task: &Task,
            strategy: DecompositionStrategy,
            _context: &DecompositionContext,
        ) -> Result<DecompositionResult, DecompositionError> {
            let start_time = std::time::Instant::now();
            
            // Simulate decomposition time based on complexity
            let delay_ms = (task.complexity * 100.0) as u64;
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            
            let result = self.generate_test_decomposition(task, strategy);
            
            // Update performance metrics
            let elapsed = start_time.elapsed();
            let mut metrics = self.performance_metrics.write().await;
            metrics.decomposition_time_ms = elapsed.as_millis() as u64;
            metrics.accuracy_score = result.confidence_score;
            
            Ok(result)
        }

        async fn validate_decomposition(
            &self,
            original_task: &Task,
            result: &DecompositionResult,
        ) -> Result<ValidationResult, DecompositionError> {
            let mut issues = Vec::new();
            
            // Check for circular dependencies
            if self.has_circular_dependencies(&result.execution_graph) {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::CircularDependency,
                    severity: IssueSeverity::Critical,
                    description: "Circular dependency detected in task graph".to_string(),
                    affected_tasks: vec![],
                    suggested_fix: Some("Remove circular dependencies".to_string()),
                });
            }

            // Check resource allocation
            let total_resources = self.calculate_total_resources(&result.subtasks);
            if total_resources.memory_mb > original_task.resources.memory_mb * 2 {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::ResourceOverallocation,
                    severity: IssueSeverity::High,
                    description: "Resource overallocation detected".to_string(),
                    affected_tasks: result.subtasks.iter().map(|t| t.id.clone()).collect(),
                    suggested_fix: Some("Optimize resource allocation".to_string()),
                });
            }

            // Check task completeness
            let completeness_score = self.calculate_completeness_score(original_task, &result.subtasks);
            let consistency_score = self.calculate_consistency_score(&result.subtasks);
            let feasibility_score = self.calculate_feasibility_score(&result.subtasks);

            Ok(ValidationResult {
                is_valid: issues.is_empty(),
                issues,
                completeness_score,
                consistency_score,
                feasibility_score,
            })
        }

        async fn optimize_task_graph(
            &self,
            graph: &TaskGraph,
            _constraints: &OptimizationConstraints,
        ) -> Result<TaskGraph, DecompositionError> {
            // Simple optimization: return the same graph
            // In real implementation, this would apply various optimization strategies
            Ok(graph.clone())
        }

        async fn estimate_execution_time(
            &self,
            graph: &TaskGraph,
            _resource_pool: &ResourcePool,
        ) -> Result<u64, DecompositionError> {
            // Simple estimation: sum of critical path durations
            // In real implementation, this would consider parallelism and resource constraints
            Ok(graph.critical_path.len() as u64 * 300) // 5 minutes per task
        }
    }

    impl MockNeuralTaskDecomposer {
        fn has_circular_dependencies(&self, graph: &TaskGraph) -> bool {
            // Simple cycle detection using DFS
            let mut visited = std::collections::HashSet::new();
            let mut rec_stack = std::collections::HashSet::new();
            
            for node in &graph.nodes {
                if !visited.contains(&node.task_id) {
                    if self.has_cycle_util(&node.task_id, graph, &mut visited, &mut rec_stack) {
                        return true;
                    }
                }
            }
            false
        }

        fn has_cycle_util(
            &self,
            task_id: &str,
            graph: &TaskGraph,
            visited: &mut std::collections::HashSet<String>,
            rec_stack: &mut std::collections::HashSet<String>,
        ) -> bool {
            visited.insert(task_id.to_string());
            rec_stack.insert(task_id.to_string());

            for edge in &graph.edges {
                if edge.from_task == task_id {
                    if !visited.contains(&edge.to_task) {
                        if self.has_cycle_util(&edge.to_task, graph, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack.contains(&edge.to_task) {
                        return true;
                    }
                }
            }

            rec_stack.remove(task_id);
            false
        }

        fn calculate_total_resources(&self, subtasks: &[Task]) -> TaskResources {
            subtasks.iter().fold(
                TaskResources {
                    memory_mb: 0,
                    cpu_cores: 0.0,
                    network_bandwidth: 0,
                    storage_gb: 0.0,
                },
                |acc, task| TaskResources {
                    memory_mb: acc.memory_mb + task.resources.memory_mb,
                    cpu_cores: acc.cpu_cores + task.resources.cpu_cores,
                    network_bandwidth: acc.network_bandwidth + task.resources.network_bandwidth,
                    storage_gb: acc.storage_gb + task.resources.storage_gb,
                },
            )
        }

        fn calculate_completeness_score(&self, original: &Task, subtasks: &[Task]) -> f32 {
            // Simple completeness: check if all capabilities are covered
            let original_caps: std::collections::HashSet<_> = original.required_capabilities.iter().collect();
            let subtask_caps: std::collections::HashSet<_> = subtasks.iter()
                .flat_map(|t| t.required_capabilities.iter())
                .collect();
            
            let covered = original_caps.intersection(&subtask_caps).count();
            covered as f32 / original_caps.len() as f32
        }

        fn calculate_consistency_score(&self, subtasks: &[Task]) -> f32 {
            // Simple consistency: check if resource allocation is reasonable
            let total_complexity: f32 = subtasks.iter().map(|t| t.complexity).sum();
            if total_complexity > 0.0 && total_complexity <= 1.0 {
                1.0
            } else {
                0.5
            }
        }

        fn calculate_feasibility_score(&self, subtasks: &[Task]) -> f32 {
            // Simple feasibility: check if all tasks have reasonable complexity
            let reasonable_tasks = subtasks.iter()
                .filter(|t| t.complexity > 0.0 && t.complexity <= 1.0)
                .count();
            
            reasonable_tasks as f32 / subtasks.len() as f32
        }
    }

    // Test helper functions
    impl Task {
        pub fn new_test_task(id: &str, complexity: f32) -> Self {
            Task {
                id: id.to_string(),
                name: format!("Test Task {}", id),
                description: format!("Test task with complexity {}", complexity),
                complexity,
                dependencies: vec![],
                priority: TaskPriority::Medium,
                estimated_duration: 600,
                required_capabilities: vec!["test".to_string()],
                resources: TaskResources {
                    memory_mb: 1024,
                    cpu_cores: 1.0,
                    network_bandwidth: 100,
                    storage_gb: 10.0,
                },
            }
        }
    }

    /// Test framework setup
    pub struct TestFramework {
        pub decomposer: MockNeuralTaskDecomposer,
        pub context: DecompositionContext,
        pub profiler: PerformanceProfiler,
    }

    impl TestFramework {
        pub fn new(strategy: DecompositionStrategy) -> Self {
            Self {
                decomposer: MockNeuralTaskDecomposer::new(strategy),
                context: MockNeuralTaskDecomposer::create_default_context(),
                profiler: PerformanceProfiler::new(),
            }
        }

        pub async fn run_decomposition_test(
            &mut self,
            task: &Task,
            strategy: DecompositionStrategy,
        ) -> Result<DecompositionResult, DecompositionError> {
            self.profiler.time_async("decomposition", || {
                self.decomposer.decompose_task(task, strategy, &self.context)
            }).await
        }

        pub async fn run_validation_test(
            &mut self,
            original_task: &Task,
            result: &DecompositionResult,
        ) -> Result<ValidationResult, DecompositionError> {
            self.profiler.time_async("validation", || {
                self.decomposer.validate_decomposition(original_task, result)
            }).await
        }
    }

    // Import test utilities
    use crate::test_utils::{
        FannDataParser, NetworkValidator, TestDataGenerator, PerformanceProfiler,
        TestFileManager, NeuralTestAssertions, PerformanceStats,
    };

    // Test implementation starts here
    // [Tests will be added in the next part due to length]
}