//! Task graph construction and dependency management

use crate::decomposer::{DecompositionResult, Subtask};
use crate::error::{SplinterError, Result};
use crate::TaskId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use petgraph::{Graph, Directed, Direction};
use petgraph::graph::{NodeIndex, EdgeIndex};
use std::fmt;

/// Task graph configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Maximum graph depth
    pub max_depth: usize,
    /// Maximum node count
    pub max_nodes: usize,
    /// Enable cycle detection
    pub enable_cycle_detection: bool,
    /// Enable critical path analysis
    pub enable_critical_path: bool,
    /// Enable resource optimization
    pub enable_resource_optimization: bool,
}

/// Task graph representing task dependencies
#[derive(Debug, Clone)]
pub struct TaskGraph {
    /// Internal graph structure
    graph: Graph<TaskNode, DependencyEdge, Directed>,
    /// Node mapping from task ID to graph node index
    node_map: HashMap<TaskId, NodeIndex>,
    /// Graph metadata
    metadata: GraphMetadata,
    /// Configuration
    config: GraphConfig,
}

/// Task node in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    /// Task ID
    pub id: TaskId,
    /// Task title
    pub title: String,
    /// Task description
    pub description: String,
    /// Task priority
    pub priority: u8,
    /// Estimated effort (person-hours)
    pub estimated_effort: f64,
    /// Estimated duration (hours)
    pub estimated_duration: f64,
    /// Required skills
    pub required_skills: Vec<String>,
    /// Required resources
    pub required_resources: Vec<String>,
    /// Acceptance criteria
    pub acceptance_criteria: Vec<String>,
    /// Task level in hierarchy
    pub level: usize,
    /// Task status
    pub status: TaskStatus,
    /// Node metadata
    pub metadata: NodeMetadata,
}

/// Task status in the graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    /// Task is pending
    Pending,
    /// Task is ready to start
    Ready,
    /// Task is in progress
    InProgress,
    /// Task is completed
    Completed,
    /// Task is blocked
    Blocked,
    /// Task is cancelled
    Cancelled,
}

/// Node metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Node confidence score
    pub confidence: f64,
    /// Success probability
    pub success_probability: f64,
    /// Risk factors
    pub risk_factors: Vec<String>,
    /// Critical path indicator
    pub is_critical_path: bool,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Resource requirements for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU requirements (cores)
    pub cpu_cores: Option<u32>,
    /// Memory requirements (MB)
    pub memory_mb: Option<u32>,
    /// Storage requirements (GB)
    pub storage_gb: Option<u32>,
    /// Network bandwidth (Mbps)
    pub network_mbps: Option<u32>,
    /// Human resources
    pub human_resources: Vec<HumanResource>,
}

/// Human resource requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanResource {
    /// Required skill
    pub skill: String,
    /// Required experience level
    pub experience_level: ExperienceLevel,
    /// Required time allocation (0.0-1.0)
    pub time_allocation: f64,
}

/// Experience levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperienceLevel {
    /// Junior level
    Junior,
    /// Mid level
    Mid,
    /// Senior level
    Senior,
    /// Expert level
    Expert,
}

/// Dependency edge in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Edge ID
    pub id: Uuid,
    /// Dependency type
    pub dependency_type: DependencyType,
    /// Edge weight (importance)
    pub weight: f64,
    /// Edge metadata
    pub metadata: EdgeMetadata,
}

/// Types of dependencies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DependencyType {
    /// Finish-to-start dependency
    FinishToStart,
    /// Start-to-start dependency
    StartToStart,
    /// Finish-to-finish dependency
    FinishToFinish,
    /// Start-to-finish dependency
    StartToFinish,
    /// Resource dependency
    Resource,
    /// Data dependency
    Data,
    /// Approval dependency
    Approval,
}

/// Edge metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Dependency lag time (hours)
    pub lag_time: f64,
    /// Dependency confidence
    pub confidence: f64,
    /// Dependency criticality
    pub criticality: f64,
    /// Dependency description
    pub description: String,
}

/// Graph metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Graph creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Graph version
    pub version: u32,
    /// Total nodes
    pub total_nodes: usize,
    /// Total edges
    pub total_edges: usize,
    /// Graph depth
    pub depth: usize,
    /// Critical path length
    pub critical_path_length: f64,
    /// Total estimated effort
    pub total_estimated_effort: f64,
    /// Total estimated duration
    pub total_estimated_duration: f64,
    /// Graph complexity score
    pub complexity_score: f64,
}

/// Graph builder for constructing task graphs
#[derive(Debug)]
pub struct GraphBuilder {
    /// Configuration
    config: GraphConfig,
}

/// Critical path analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    /// Path nodes
    pub nodes: Vec<TaskId>,
    /// Path edges
    pub edges: Vec<Uuid>,
    /// Total path duration
    pub total_duration: f64,
    /// Path criticality score
    pub criticality: f64,
    /// Path bottlenecks
    pub bottlenecks: Vec<TaskId>,
}

/// Graph analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysis {
    /// Critical path
    pub critical_path: CriticalPath,
    /// Resource conflicts
    pub resource_conflicts: Vec<ResourceConflict>,
    /// Scheduling recommendations
    pub scheduling_recommendations: Vec<SchedulingRecommendation>,
    /// Graph metrics
    pub metrics: GraphMetrics,
}

/// Resource conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConflict {
    /// Conflicting tasks
    pub tasks: Vec<TaskId>,
    /// Conflicting resource
    pub resource: String,
    /// Conflict severity
    pub severity: ConflictSeverity,
    /// Suggested resolution
    pub resolution: String,
}

/// Conflict severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Scheduling recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingRecommendation {
    /// Task to reschedule
    pub task_id: TaskId,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation description
    pub description: String,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation priority
    pub priority: u8,
}

/// Recommendation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecommendationType {
    /// Reorder tasks
    Reorder,
    /// Parallelize tasks
    Parallelize,
    /// Split task
    Split,
    /// Merge tasks
    Merge,
    /// Reallocate resources
    Reallocate,
    /// Adjust timeline
    AdjustTimeline,
}

/// Graph metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Node count
    pub node_count: usize,
    /// Edge count
    pub edge_count: usize,
    /// Graph density
    pub density: f64,
    /// Average degree
    pub average_degree: f64,
    /// Longest path length
    pub longest_path: usize,
    /// Parallelism factor
    pub parallelism_factor: f64,
    /// Resource utilization
    pub resource_utilization: f64,
}

/// Topological sort result
#[derive(Debug, Clone)]
pub struct TopologicalSort {
    /// Sorted task IDs
    pub sorted_tasks: Vec<TaskId>,
    /// Execution levels
    pub execution_levels: Vec<Vec<TaskId>>,
    /// Has cycles
    pub has_cycles: bool,
    /// Cycle nodes (if any)
    pub cycle_nodes: Vec<TaskId>,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            max_nodes: 1000,
            enable_cycle_detection: true,
            enable_critical_path: true,
            enable_resource_optimization: true,
        }
    }
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new(config: GraphConfig) -> Self {
        Self { config }
    }

    /// Build task graph from decomposition result
    pub async fn build(&self, decomposition: &DecompositionResult) -> Result<TaskGraph> {
        let mut graph = Graph::new();
        let mut node_map = HashMap::new();

        // Add nodes
        for subtask in &decomposition.subtasks {
            let task_node = TaskNode::from_subtask(subtask);
            let node_index = graph.add_node(task_node);
            node_map.insert(subtask.id, node_index);
        }

        // Add edges based on dependencies
        for subtask in &decomposition.subtasks {
            let target_index = node_map[&subtask.id];
            
            for dependency_id in &subtask.dependencies {
                if let Some(&source_index) = node_map.get(dependency_id) {
                    let edge = DependencyEdge {
                        id: Uuid::new_v4(),
                        dependency_type: DependencyType::FinishToStart,
                        weight: 1.0,
                        metadata: EdgeMetadata {
                            created_at: Utc::now(),
                            lag_time: 0.0,
                            confidence: 0.9,
                            criticality: 0.5,
                            description: format!("Dependency from {} to {}", dependency_id, subtask.id),
                        },
                    };
                    
                    graph.add_edge(source_index, target_index, edge);
                }
            }
        }

        // Detect cycles if enabled
        if self.config.enable_cycle_detection {
            if let Some(cycle) = self.detect_cycles(&graph, &node_map) {
                return Err(SplinterError::graph_error(format!("Cycle detected in graph: {:?}", cycle)));
            }
        }

        // Calculate graph metrics
        let total_nodes = graph.node_count();
        let total_edges = graph.edge_count();
        let depth = self.calculate_graph_depth(&graph);
        let total_estimated_effort = graph.node_weights().map(|n| n.estimated_effort).sum();
        let total_estimated_duration = graph.node_weights().map(|n| n.estimated_duration).sum();
        let complexity_score = self.calculate_complexity_score(&graph);

        let metadata = GraphMetadata {
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: 1,
            total_nodes,
            total_edges,
            depth,
            critical_path_length: 0.0, // Will be calculated later
            total_estimated_effort,
            total_estimated_duration,
            complexity_score,
        };

        let mut task_graph = TaskGraph {
            graph,
            node_map,
            metadata,
            config: self.config.clone(),
        };

        // Calculate critical path if enabled
        if self.config.enable_critical_path {
            let critical_path = task_graph.calculate_critical_path()?;
            task_graph.metadata.critical_path_length = critical_path.total_duration;
            task_graph.mark_critical_path(&critical_path);
        }

        Ok(task_graph)
    }

    /// Update configuration
    pub async fn update_config(&mut self, config: GraphConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Detect cycles in the graph
    fn detect_cycles(&self, graph: &Graph<TaskNode, DependencyEdge, Directed>, node_map: &HashMap<TaskId, NodeIndex>) -> Option<Vec<TaskId>> {
        use petgraph::algo::is_cyclic_directed;
        
        if is_cyclic_directed(graph) {
            // Find actual cycle nodes (simplified implementation)
            let mut visited = HashSet::new();
            let mut rec_stack = HashSet::new();
            
            for node_index in graph.node_indices() {
                if !visited.contains(&node_index) {
                    if let Some(cycle) = self.dfs_cycle_detection(graph, node_index, &mut visited, &mut rec_stack) {
                        // Convert node indices back to task IDs
                        let cycle_ids: Vec<TaskId> = cycle.iter()
                            .filter_map(|&idx| {
                                node_map.iter().find(|(_, &v)| v == idx).map(|(k, _)| *k)
                            })
                            .collect();
                        return Some(cycle_ids);
                    }
                }
            }
        }
        
        None
    }

    /// DFS-based cycle detection
    fn dfs_cycle_detection(
        &self,
        graph: &Graph<TaskNode, DependencyEdge, Directed>,
        node: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        rec_stack: &mut HashSet<NodeIndex>,
    ) -> Option<Vec<NodeIndex>> {
        visited.insert(node);
        rec_stack.insert(node);

        for edge in graph.edges(node) {
            let neighbor = edge.target();
            
            if !visited.contains(&neighbor) {
                if let Some(cycle) = self.dfs_cycle_detection(graph, neighbor, visited, rec_stack) {
                    return Some(cycle);
                }
            } else if rec_stack.contains(&neighbor) {
                // Found a cycle
                return Some(vec![node, neighbor]);
            }
        }

        rec_stack.remove(&node);
        None
    }

    /// Calculate graph depth
    fn calculate_graph_depth(&self, graph: &Graph<TaskNode, DependencyEdge, Directed>) -> usize {
        let mut max_depth = 0;
        
        for node_index in graph.node_indices() {
            let depth = self.calculate_node_depth(graph, node_index, &mut HashSet::new());
            max_depth = max_depth.max(depth);
        }
        
        max_depth
    }

    /// Calculate depth of a specific node
    fn calculate_node_depth(
        &self,
        graph: &Graph<TaskNode, DependencyEdge, Directed>,
        node: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
    ) -> usize {
        if visited.contains(&node) {
            return 0; // Avoid infinite recursion
        }
        
        visited.insert(node);
        
        let mut max_child_depth = 0;
        for edge in graph.edges(node) {
            let child_depth = self.calculate_node_depth(graph, edge.target(), visited);
            max_child_depth = max_child_depth.max(child_depth);
        }
        
        visited.remove(&node);
        max_child_depth + 1
    }

    /// Calculate complexity score
    fn calculate_complexity_score(&self, graph: &Graph<TaskNode, DependencyEdge, Directed>) -> f64 {
        let node_count = graph.node_count() as f64;
        let edge_count = graph.edge_count() as f64;
        
        if node_count == 0.0 {
            return 0.0;
        }
        
        // Complexity based on density and size
        let density = if node_count > 1.0 {
            edge_count / (node_count * (node_count - 1.0))
        } else {
            0.0
        };
        
        let size_factor = (node_count.log10() + 1.0) / 10.0;
        let density_factor = density;
        
        (size_factor * 0.6 + density_factor * 0.4).min(1.0)
    }
}

impl TaskGraph {
    /// Create a new empty task graph
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            node_map: HashMap::new(),
            metadata: GraphMetadata {
                created_at: Utc::now(),
                updated_at: Utc::now(),
                version: 1,
                total_nodes: 0,
                total_edges: 0,
                depth: 0,
                critical_path_length: 0.0,
                total_estimated_effort: 0.0,
                total_estimated_duration: 0.0,
                complexity_score: 0.0,
            },
            config: GraphConfig::default(),
        }
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get all task nodes
    pub fn nodes(&self) -> Vec<&TaskNode> {
        self.graph.node_weights().collect()
    }

    /// Get all dependency edges
    pub fn edges(&self) -> Vec<&DependencyEdge> {
        self.graph.edge_weights().collect()
    }

    /// Get task node by ID
    pub fn get_node(&self, task_id: &TaskId) -> Option<&TaskNode> {
        self.node_map.get(task_id)
            .and_then(|&index| self.graph.node_weight(index))
    }

    /// Get task dependencies
    pub fn get_dependencies(&self, task_id: &TaskId) -> Vec<&TaskNode> {
        if let Some(&node_index) = self.node_map.get(task_id) {
            self.graph.edges_directed(node_index, Direction::Incoming)
                .filter_map(|edge| self.graph.node_weight(edge.source()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get task dependents
    pub fn get_dependents(&self, task_id: &TaskId) -> Vec<&TaskNode> {
        if let Some(&node_index) = self.node_map.get(task_id) {
            self.graph.edges_directed(node_index, Direction::Outgoing)
                .filter_map(|edge| self.graph.node_weight(edge.target()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Calculate critical path
    pub fn calculate_critical_path(&self) -> Result<CriticalPath> {
        // Find all paths from root to leaf nodes
        let root_nodes = self.find_root_nodes();
        let leaf_nodes = self.find_leaf_nodes();
        
        let mut longest_path = Vec::new();
        let mut max_duration = 0.0;
        
        for root in root_nodes {
            for leaf in &leaf_nodes {
                if let Some(path) = self.find_path(root, *leaf) {
                    let path_duration = self.calculate_path_duration(&path);
                    if path_duration > max_duration {
                        max_duration = path_duration;
                        longest_path = path;
                    }
                }
            }
        }
        
        // Convert node indices to task IDs
        let nodes: Vec<TaskId> = longest_path.iter()
            .filter_map(|&index| {
                self.node_map.iter().find(|(_, &v)| v == index).map(|(k, _)| *k)
            })
            .collect();
        
        // Find bottlenecks (nodes with highest resource requirements)
        let bottlenecks = self.find_bottlenecks(&nodes);
        
        Ok(CriticalPath {
            nodes,
            edges: Vec::new(), // Could populate with actual edge IDs
            total_duration: max_duration,
            criticality: 1.0,
            bottlenecks,
        })
    }

    /// Find root nodes (no incoming edges)
    fn find_root_nodes(&self) -> Vec<NodeIndex> {
        self.graph.node_indices()
            .filter(|&node| self.graph.edges_directed(node, Direction::Incoming).count() == 0)
            .collect()
    }

    /// Find leaf nodes (no outgoing edges)
    fn find_leaf_nodes(&self) -> Vec<NodeIndex> {
        self.graph.node_indices()
            .filter(|&node| self.graph.edges_directed(node, Direction::Outgoing).count() == 0)
            .collect()
    }

    /// Find path between two nodes
    fn find_path(&self, start: NodeIndex, end: NodeIndex) -> Option<Vec<NodeIndex>> {
        use petgraph::algo::astar;
        
        let result = astar(
            &self.graph,
            start,
            |node| node == end,
            |edge| edge.weight().weight as u32,
            |_| 0,
        );
        
        result.map(|(_, path)| path)
    }

    /// Calculate path duration
    fn calculate_path_duration(&self, path: &[NodeIndex]) -> f64 {
        path.iter()
            .filter_map(|&index| self.graph.node_weight(index))
            .map(|node| node.estimated_duration)
            .sum()
    }

    /// Find bottlenecks in the path
    fn find_bottlenecks(&self, path: &[TaskId]) -> Vec<TaskId> {
        let mut bottlenecks = Vec::new();
        
        for &task_id in path {
            if let Some(node) = self.get_node(&task_id) {
                // Consider a task a bottleneck if it has high resource requirements
                let resource_score = self.calculate_resource_score(&node.metadata.resource_requirements);
                if resource_score > 0.7 {
                    bottlenecks.push(task_id);
                }
            }
        }
        
        bottlenecks
    }

    /// Calculate resource score
    fn calculate_resource_score(&self, requirements: &ResourceRequirements) -> f64 {
        let mut score = 0.0;
        
        if requirements.cpu_cores.unwrap_or(0) > 4 {
            score += 0.3;
        }
        if requirements.memory_mb.unwrap_or(0) > 8192 {
            score += 0.3;
        }
        if requirements.human_resources.len() > 2 {
            score += 0.4;
        }
        
        score.min(1.0)
    }

    /// Mark critical path nodes
    fn mark_critical_path(&mut self, critical_path: &CriticalPath) {
        for &task_id in &critical_path.nodes {
            if let Some(&node_index) = self.node_map.get(&task_id) {
                if let Some(node) = self.graph.node_weight_mut(node_index) {
                    node.metadata.is_critical_path = true;
                }
            }
        }
    }

    /// Perform topological sort
    pub fn topological_sort(&self) -> Result<TopologicalSort> {
        use petgraph::algo::toposort;
        
        match toposort(&self.graph, None) {
            Ok(sorted_indices) => {
                let sorted_tasks: Vec<TaskId> = sorted_indices.iter()
                    .filter_map(|&index| {
                        self.node_map.iter().find(|(_, &v)| v == index).map(|(k, _)| *k)
                    })
                    .collect();
                
                let execution_levels = self.calculate_execution_levels(&sorted_indices);
                
                Ok(TopologicalSort {
                    sorted_tasks,
                    execution_levels,
                    has_cycles: false,
                    cycle_nodes: Vec::new(),
                })
            }
            Err(cycle) => {
                let cycle_task_id = self.node_map.iter()
                    .find(|(_, &v)| v == cycle.node_id())
                    .map(|(k, _)| *k)
                    .unwrap_or_else(|| Uuid::new_v4());
                
                Ok(TopologicalSort {
                    sorted_tasks: Vec::new(),
                    execution_levels: Vec::new(),
                    has_cycles: true,
                    cycle_nodes: vec![cycle_task_id],
                })
            }
        }
    }

    /// Calculate execution levels for parallel execution
    fn calculate_execution_levels(&self, sorted_indices: &[NodeIndex]) -> Vec<Vec<TaskId>> {
        let mut levels = Vec::new();
        let mut node_levels = HashMap::new();
        
        // Calculate level for each node
        for &node_index in sorted_indices {
            let max_dependency_level = self.graph.edges_directed(node_index, Direction::Incoming)
                .map(|edge| node_levels.get(&edge.source()).unwrap_or(&0))
                .max()
                .unwrap_or(&0);
            
            let node_level = max_dependency_level + 1;
            node_levels.insert(node_index, node_level);
            
            // Ensure we have enough levels
            while levels.len() < node_level {
                levels.push(Vec::new());
            }
            
            // Add task to its level
            if let Some(task_id) = self.node_map.iter().find(|(_, &v)| v == node_index).map(|(k, _)| *k) {
                levels[node_level - 1].push(task_id);
            }
        }
        
        levels
    }

    /// Analyze graph for optimization opportunities
    pub fn analyze(&self) -> Result<GraphAnalysis> {
        let critical_path = self.calculate_critical_path()?;
        let resource_conflicts = self.detect_resource_conflicts();
        let scheduling_recommendations = self.generate_scheduling_recommendations(&critical_path, &resource_conflicts);
        let metrics = self.calculate_metrics();
        
        Ok(GraphAnalysis {
            critical_path,
            resource_conflicts,
            scheduling_recommendations,
            metrics,
        })
    }

    /// Detect resource conflicts
    fn detect_resource_conflicts(&self) -> Vec<ResourceConflict> {
        let mut conflicts = Vec::new();
        let mut resource_usage = HashMap::new();
        
        // Group tasks by resource requirements
        for node in self.graph.node_weights() {
            for skill in &node.required_skills {
                resource_usage.entry(skill.clone())
                    .or_insert_with(Vec::new)
                    .push(node.id);
            }
        }
        
        // Detect conflicts (simplified)
        for (resource, tasks) in resource_usage {
            if tasks.len() > 1 {
                conflicts.push(ResourceConflict {
                    tasks,
                    resource,
                    severity: ConflictSeverity::Medium,
                    resolution: "Consider task sequencing or resource allocation".to_string(),
                });
            }
        }
        
        conflicts
    }

    /// Generate scheduling recommendations
    fn generate_scheduling_recommendations(&self, critical_path: &CriticalPath, conflicts: &[ResourceConflict]) -> Vec<SchedulingRecommendation> {
        let mut recommendations = Vec::new();
        
        // Recommend parallelization for non-critical tasks
        let topo_sort = self.topological_sort().unwrap_or_else(|_| TopologicalSort {
            sorted_tasks: Vec::new(),
            execution_levels: Vec::new(),
            has_cycles: false,
            cycle_nodes: Vec::new(),
        });
        
        for level in &topo_sort.execution_levels {
            if level.len() > 1 {
                for &task_id in level {
                    if !critical_path.nodes.contains(&task_id) {
                        recommendations.push(SchedulingRecommendation {
                            task_id,
                            recommendation_type: RecommendationType::Parallelize,
                            description: "Task can be executed in parallel with other tasks at the same level".to_string(),
                            expected_impact: "Reduced overall project duration".to_string(),
                            priority: 7,
                        });
                    }
                }
            }
        }
        
        // Recommendations for resource conflicts
        for conflict in conflicts {
            if conflict.severity == ConflictSeverity::High || conflict.severity == ConflictSeverity::Critical {
                for &task_id in &conflict.tasks {
                    recommendations.push(SchedulingRecommendation {
                        task_id,
                        recommendation_type: RecommendationType::Reallocate,
                        description: format!("Reallocate {} resources to avoid conflicts", conflict.resource),
                        expected_impact: "Reduced resource contention".to_string(),
                        priority: 8,
                    });
                }
            }
        }
        
        recommendations
    }

    /// Calculate graph metrics
    fn calculate_metrics(&self) -> GraphMetrics {
        let node_count = self.graph.node_count();
        let edge_count = self.graph.edge_count();
        
        let density = if node_count > 1 {
            edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };
        
        let average_degree = if node_count > 0 {
            (edge_count * 2) as f64 / node_count as f64
        } else {
            0.0
        };
        
        let topo_sort = self.topological_sort().unwrap_or_else(|_| TopologicalSort {
            sorted_tasks: Vec::new(),
            execution_levels: Vec::new(),
            has_cycles: false,
            cycle_nodes: Vec::new(),
        });
        
        let longest_path = topo_sort.execution_levels.len();
        let parallelism_factor = if longest_path > 0 {
            node_count as f64 / longest_path as f64
        } else {
            0.0
        };
        
        // Calculate resource utilization (simplified)
        let total_resource_requirements = self.graph.node_weights()
            .map(|node| self.calculate_resource_score(&node.metadata.resource_requirements))
            .sum::<f64>();
        
        let resource_utilization = if node_count > 0 {
            total_resource_requirements / node_count as f64
        } else {
            0.0
        };
        
        GraphMetrics {
            node_count,
            edge_count,
            density,
            average_degree,
            longest_path,
            parallelism_factor,
            resource_utilization,
        }
    }

    /// Update task status
    pub fn update_task_status(&mut self, task_id: &TaskId, status: TaskStatus) -> Result<()> {
        if let Some(&node_index) = self.node_map.get(task_id) {
            if let Some(node) = self.graph.node_weight_mut(node_index) {
                node.status = status;
                node.metadata.updated_at = Utc::now();
                self.metadata.updated_at = Utc::now();
                self.metadata.version += 1;
                return Ok(());
            }
        }
        
        Err(SplinterError::graph_error(format!("Task {} not found in graph", task_id)))
    }

    /// Get ready tasks (tasks with all dependencies completed)
    pub fn get_ready_tasks(&self) -> Vec<&TaskNode> {
        self.graph.node_weights()
            .filter(|node| {
                node.status == TaskStatus::Pending &&
                self.get_dependencies(&node.id).iter().all(|dep| dep.status == TaskStatus::Completed)
            })
            .collect()
    }

    /// Validate graph consistency
    pub fn validate(&self) -> Result<()> {
        // Check for cycles
        let topo_sort = self.topological_sort()?;
        if topo_sort.has_cycles {
            return Err(SplinterError::graph_error("Graph contains cycles".to_string()));
        }
        
        // Check for orphaned nodes
        let orphaned_nodes: Vec<_> = self.graph.node_indices()
            .filter(|&node| {
                self.graph.edges_directed(node, Direction::Incoming).count() == 0 &&
                self.graph.edges_directed(node, Direction::Outgoing).count() == 0
            })
            .collect();
        
        if orphaned_nodes.len() > 1 {
            return Err(SplinterError::graph_error("Graph contains orphaned nodes".to_string()));
        }
        
        Ok(())
    }
}

impl TaskNode {
    /// Create task node from subtask
    pub fn from_subtask(subtask: &Subtask) -> Self {
        let resource_requirements = ResourceRequirements {
            cpu_cores: None,
            memory_mb: None,
            storage_gb: None,
            network_mbps: None,
            human_resources: subtask.required_skills.iter().map(|skill| HumanResource {
                skill: skill.clone(),
                experience_level: ExperienceLevel::Mid,
                time_allocation: 1.0,
            }).collect(),
        };
        
        Self {
            id: subtask.id,
            title: subtask.title.clone(),
            description: subtask.description.clone(),
            priority: subtask.priority,
            estimated_effort: subtask.estimated_effort,
            estimated_duration: subtask.estimated_duration,
            required_skills: subtask.required_skills.clone(),
            required_resources: subtask.required_resources.clone(),
            acceptance_criteria: subtask.acceptance_criteria.clone(),
            level: subtask.level,
            status: TaskStatus::Pending,
            metadata: NodeMetadata {
                created_at: subtask.created_at,
                updated_at: subtask.created_at,
                confidence: subtask.metadata.confidence,
                success_probability: subtask.metadata.success_probability,
                risk_factors: subtask.metadata.risk_factors.clone(),
                is_critical_path: false,
                resource_requirements,
            },
        }
    }
}

impl fmt::Display for TaskGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TaskGraph {{")?;
        writeln!(f, "  Nodes: {}", self.node_count())?;
        writeln!(f, "  Edges: {}", self.edge_count())?;
        writeln!(f, "  Depth: {}", self.metadata.depth)?;
        writeln!(f, "  Total Effort: {:.1} hours", self.metadata.total_estimated_effort)?;
        writeln!(f, "  Total Duration: {:.1} hours", self.metadata.total_estimated_duration)?;
        writeln!(f, "  Complexity: {:.2}", self.metadata.complexity_score)?;
        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decomposer::{DecompositionMetadata, SubtaskMetadata};
    use crate::parser::TaskType;

    fn create_test_subtask(id: TaskId, title: &str, dependencies: Vec<TaskId>) -> Subtask {
        Subtask {
            id,
            parent_id: Uuid::new_v4(),
            title: title.to_string(),
            description: format!("Test subtask: {}", title),
            task_type: TaskType::Development,
            priority: 5,
            estimated_effort: 8.0,
            estimated_duration: 8.0,
            required_skills: vec!["programming".to_string()],
            required_resources: vec!["computer".to_string()],
            dependencies,
            acceptance_criteria: vec!["Task completed successfully".to_string()],
            level: 1,
            created_at: Utc::now(),
            metadata: SubtaskMetadata {
                confidence: 0.8,
                rationale: "Test subtask".to_string(),
                alternatives: Vec::new(),
                risk_factors: Vec::new(),
                success_probability: 0.85,
            },
        }
    }

    #[tokio::test]
    async fn test_graph_builder() {
        let config = GraphConfig::default();
        let builder = GraphBuilder::new(config);
        
        let task1 = create_test_subtask(Uuid::new_v4(), "Task 1", Vec::new());
        let task2 = create_test_subtask(Uuid::new_v4(), "Task 2", vec![task1.id]);
        let task3 = create_test_subtask(Uuid::new_v4(), "Task 3", vec![task2.id]);
        
        let decomposition = DecompositionResult {
            task_id: Uuid::new_v4(),
            subtasks: vec![task1, task2, task3],
            strategy: crate::decomposer::DecompositionStrategy::Neural,
            metadata: DecompositionMetadata {
                decomposed_at: Utc::now(),
                decomposition_duration_ms: 100,
                strategy_confidence: 0.8,
                total_estimated_effort: 24.0,
                total_estimated_duration: 24.0,
                quality_score: 0.85,
                used_templates: Vec::new(),
            },
        };
        
        let graph = builder.build(&decomposition).await.unwrap();
        
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.metadata.depth, 3);
    }

    #[tokio::test]
    async fn test_critical_path() {
        let config = GraphConfig::default();
        let builder = GraphBuilder::new(config);
        
        let task1 = create_test_subtask(Uuid::new_v4(), "Task 1", Vec::new());
        let task2 = create_test_subtask(Uuid::new_v4(), "Task 2", vec![task1.id]);
        let task3 = create_test_subtask(Uuid::new_v4(), "Task 3", vec![task2.id]);
        
        let decomposition = DecompositionResult {
            task_id: Uuid::new_v4(),
            subtasks: vec![task1.clone(), task2.clone(), task3.clone()],
            strategy: crate::decomposer::DecompositionStrategy::Neural,
            metadata: DecompositionMetadata {
                decomposed_at: Utc::now(),
                decomposition_duration_ms: 100,
                strategy_confidence: 0.8,
                total_estimated_effort: 24.0,
                total_estimated_duration: 24.0,
                quality_score: 0.85,
                used_templates: Vec::new(),
            },
        };
        
        let graph = builder.build(&decomposition).await.unwrap();
        let critical_path = graph.calculate_critical_path().unwrap();
        
        assert_eq!(critical_path.nodes.len(), 3);
        assert_eq!(critical_path.total_duration, 24.0);
        assert!(critical_path.nodes.contains(&task1.id));
        assert!(critical_path.nodes.contains(&task2.id));
        assert!(critical_path.nodes.contains(&task3.id));
    }

    #[tokio::test]
    async fn test_topological_sort() {
        let config = GraphConfig::default();
        let builder = GraphBuilder::new(config);
        
        let task1 = create_test_subtask(Uuid::new_v4(), "Task 1", Vec::new());
        let task2 = create_test_subtask(Uuid::new_v4(), "Task 2", vec![task1.id]);
        let task3 = create_test_subtask(Uuid::new_v4(), "Task 3", vec![task1.id]);
        let task4 = create_test_subtask(Uuid::new_v4(), "Task 4", vec![task2.id, task3.id]);
        
        let decomposition = DecompositionResult {
            task_id: Uuid::new_v4(),
            subtasks: vec![task1.clone(), task2.clone(), task3.clone(), task4.clone()],
            strategy: crate::decomposer::DecompositionStrategy::Neural,
            metadata: DecompositionMetadata {
                decomposed_at: Utc::now(),
                decomposition_duration_ms: 100,
                strategy_confidence: 0.8,
                total_estimated_effort: 32.0,
                total_estimated_duration: 32.0,
                quality_score: 0.85,
                used_templates: Vec::new(),
            },
        };
        
        let graph = builder.build(&decomposition).await.unwrap();
        let topo_sort = graph.topological_sort().unwrap();
        
        assert!(!topo_sort.has_cycles);
        assert_eq!(topo_sort.sorted_tasks.len(), 4);
        assert_eq!(topo_sort.execution_levels.len(), 3);
        
        // Task 1 should be at level 0
        assert!(topo_sort.execution_levels[0].contains(&task1.id));
        // Tasks 2 and 3 should be at level 1
        assert!(topo_sort.execution_levels[1].contains(&task2.id));
        assert!(topo_sort.execution_levels[1].contains(&task3.id));
        // Task 4 should be at level 2
        assert!(topo_sort.execution_levels[2].contains(&task4.id));
    }

    #[tokio::test]
    async fn test_cycle_detection() {
        let config = GraphConfig::default();
        let builder = GraphBuilder::new(config);
        
        let task1 = create_test_subtask(Uuid::new_v4(), "Task 1", Vec::new());
        let task2 = create_test_subtask(Uuid::new_v4(), "Task 2", vec![task1.id]);
        
        // Create a cycle: task1 -> task2 -> task1
        let mut task1_with_cycle = task1.clone();
        task1_with_cycle.dependencies = vec![task2.id];
        
        let decomposition = DecompositionResult {
            task_id: Uuid::new_v4(),
            subtasks: vec![task1_with_cycle, task2],
            strategy: crate::decomposer::DecompositionStrategy::Neural,
            metadata: DecompositionMetadata {
                decomposed_at: Utc::now(),
                decomposition_duration_ms: 100,
                strategy_confidence: 0.8,
                total_estimated_effort: 16.0,
                total_estimated_duration: 16.0,
                quality_score: 0.85,
                used_templates: Vec::new(),
            },
        };
        
        let result = builder.build(&decomposition).await;
        assert!(result.is_err());
        
        // Check that the error message mentions cycle
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Cycle detected"));
    }

    #[tokio::test]
    async fn test_task_status_update() {
        let config = GraphConfig::default();
        let builder = GraphBuilder::new(config);
        
        let task1 = create_test_subtask(Uuid::new_v4(), "Task 1", Vec::new());
        let task_id = task1.id;
        
        let decomposition = DecompositionResult {
            task_id: Uuid::new_v4(),
            subtasks: vec![task1],
            strategy: crate::decomposer::DecompositionStrategy::Neural,
            metadata: DecompositionMetadata {
                decomposed_at: Utc::now(),
                decomposition_duration_ms: 100,
                strategy_confidence: 0.8,
                total_estimated_effort: 8.0,
                total_estimated_duration: 8.0,
                quality_score: 0.85,
                used_templates: Vec::new(),
            },
        };
        
        let mut graph = builder.build(&decomposition).await.unwrap();
        
        // Update task status
        graph.update_task_status(&task_id, TaskStatus::InProgress).unwrap();
        
        let node = graph.get_node(&task_id).unwrap();
        assert_eq!(node.status, TaskStatus::InProgress);
    }

    #[tokio::test]
    async fn test_ready_tasks() {
        let config = GraphConfig::default();
        let builder = GraphBuilder::new(config);
        
        let task1 = create_test_subtask(Uuid::new_v4(), "Task 1", Vec::new());
        let task2 = create_test_subtask(Uuid::new_v4(), "Task 2", vec![task1.id]);
        let task1_id = task1.id;
        let task2_id = task2.id;
        
        let decomposition = DecompositionResult {
            task_id: Uuid::new_v4(),
            subtasks: vec![task1, task2],
            strategy: crate::decomposer::DecompositionStrategy::Neural,
            metadata: DecompositionMetadata {
                decomposed_at: Utc::now(),
                decomposition_duration_ms: 100,
                strategy_confidence: 0.8,
                total_estimated_effort: 16.0,
                total_estimated_duration: 16.0,
                quality_score: 0.85,
                used_templates: Vec::new(),
            },
        };
        
        let mut graph = builder.build(&decomposition).await.unwrap();
        
        // Initially, only task1 should be ready
        let ready_tasks = graph.get_ready_tasks();
        assert_eq!(ready_tasks.len(), 1);
        assert_eq!(ready_tasks[0].id, task1_id);
        
        // Complete task1
        graph.update_task_status(&task1_id, TaskStatus::Completed).unwrap();
        
        // Now task2 should be ready
        let ready_tasks = graph.get_ready_tasks();
        assert_eq!(ready_tasks.len(), 1);
        assert_eq!(ready_tasks[0].id, task2_id);
    }

    #[tokio::test]
    async fn test_graph_analysis() {
        let config = GraphConfig::default();
        let builder = GraphBuilder::new(config);
        
        let task1 = create_test_subtask(Uuid::new_v4(), "Task 1", Vec::new());
        let task2 = create_test_subtask(Uuid::new_v4(), "Task 2", vec![task1.id]);
        let task3 = create_test_subtask(Uuid::new_v4(), "Task 3", vec![task1.id]);
        
        let decomposition = DecompositionResult {
            task_id: Uuid::new_v4(),
            subtasks: vec![task1, task2, task3],
            strategy: crate::decomposer::DecompositionStrategy::Neural,
            metadata: DecompositionMetadata {
                decomposed_at: Utc::now(),
                decomposition_duration_ms: 100,
                strategy_confidence: 0.8,
                total_estimated_effort: 24.0,
                total_estimated_duration: 24.0,
                quality_score: 0.85,
                used_templates: Vec::new(),
            },
        };
        
        let graph = builder.build(&decomposition).await.unwrap();
        let analysis = graph.analyze().unwrap();
        
        assert_eq!(analysis.metrics.node_count, 3);
        assert_eq!(analysis.metrics.edge_count, 2);
        assert!(analysis.metrics.parallelism_factor > 1.0);
        assert!(!analysis.scheduling_recommendations.is_empty());
    }
}