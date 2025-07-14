//! Task decomposition engine with multiple strategies

use crate::parser::{ParsedTask, TaskEntity, TaskType};
use crate::analyzer::{TaskContext, SemanticAnalysis};
use crate::neural::{NeuralBackend, NeuralConfig};
use crate::error::{SplinterError, Result};
use crate::graph::{TaskGraph, TaskNode, DependencyEdge};
use crate::TaskId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use std::sync::Arc;
use dashmap::DashMap;

/// Task decomposition strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecompositionStrategy {
    /// Neural network-based decomposition
    Neural,
    /// Rule-based decomposition
    RuleBased,
    /// Hybrid approach combining neural and rules
    Hybrid,
    /// Template-based decomposition
    Template,
    /// Hierarchical decomposition
    Hierarchical,
    /// Dependency-driven decomposition
    DependencyDriven,
}

/// Task decomposer with multiple strategies
#[derive(Debug)]
pub struct TaskDecomposer {
    /// Neural backend for AI-powered decomposition
    neural_backend: Arc<NeuralBackend>,
    /// Rule-based decomposition engine
    rule_engine: Arc<RuleEngine>,
    /// Template repository
    template_repo: Arc<TemplateRepository>,
    /// Decomposition cache
    decomposition_cache: Arc<DashMap<TaskId, DecompositionResult>>,
    /// Performance metrics
    metrics: Arc<RwLock<DecomposerMetrics>>,
    /// Configuration
    config: DecomposerConfig,
}

/// Decomposer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposerConfig {
    /// Maximum decomposition depth
    pub max_depth: usize,
    /// Minimum subtask size
    pub min_subtask_size: usize,
    /// Maximum subtask count
    pub max_subtask_count: usize,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Default decomposition strategy
    pub default_strategy: DecompositionStrategy,
    /// Strategy weights for hybrid approach
    pub strategy_weights: HashMap<DecompositionStrategy, f64>,
}

/// Decomposition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    /// Original task ID
    pub task_id: TaskId,
    /// Generated subtasks
    pub subtasks: Vec<Subtask>,
    /// Decomposition strategy used
    pub strategy: DecompositionStrategy,
    /// Decomposition metadata
    pub metadata: DecompositionMetadata,
}

/// Subtask generated from decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    /// Subtask ID
    pub id: TaskId,
    /// Parent task ID
    pub parent_id: TaskId,
    /// Subtask title
    pub title: String,
    /// Subtask description
    pub description: String,
    /// Subtask type
    pub task_type: TaskType,
    /// Priority level
    pub priority: u8,
    /// Estimated effort (person-hours)
    pub estimated_effort: f64,
    /// Estimated duration (hours)
    pub estimated_duration: f64,
    /// Required skills
    pub required_skills: Vec<String>,
    /// Required resources
    pub required_resources: Vec<String>,
    /// Dependencies on other subtasks
    pub dependencies: Vec<TaskId>,
    /// Acceptance criteria
    pub acceptance_criteria: Vec<String>,
    /// Decomposition level
    pub level: usize,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Task metadata
    pub metadata: SubtaskMetadata,
}

/// Subtask metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskMetadata {
    /// Confidence score for this subtask
    pub confidence: f64,
    /// Decomposition rationale
    pub rationale: String,
    /// Alternative approaches
    pub alternatives: Vec<String>,
    /// Risk factors
    pub risk_factors: Vec<String>,
    /// Success probability
    pub success_probability: f64,
}

/// Decomposition metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionMetadata {
    /// Decomposition timestamp
    pub decomposed_at: DateTime<Utc>,
    /// Decomposition duration
    pub decomposition_duration_ms: u64,
    /// Strategy confidence
    pub strategy_confidence: f64,
    /// Total estimated effort
    pub total_estimated_effort: f64,
    /// Total estimated duration
    pub total_estimated_duration: f64,
    /// Decomposition quality score
    pub quality_score: f64,
    /// Used templates
    pub used_templates: Vec<String>,
}

/// Rule-based decomposition engine
#[derive(Debug)]
pub struct RuleEngine {
    /// Decomposition rules
    rules: Vec<DecompositionRule>,
    /// Rule patterns
    patterns: HashMap<TaskType, Vec<RulePattern>>,
}

/// Decomposition rule
#[derive(Debug, Clone)]
pub struct DecompositionRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule conditions
    pub conditions: Vec<RuleCondition>,
    /// Rule actions
    pub actions: Vec<RuleAction>,
    /// Rule priority
    pub priority: u8,
    /// Rule confidence
    pub confidence: f64,
}

/// Rule condition
#[derive(Debug, Clone)]
pub struct RuleCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition value
    pub value: ConditionValue,
    /// Condition operator
    pub operator: ConditionOperator,
}

/// Condition types
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionType {
    /// Task type condition
    TaskType,
    /// Priority condition
    Priority,
    /// Complexity condition
    Complexity,
    /// Entity count condition
    EntityCount,
    /// Keyword condition
    Keyword,
    /// Dependency condition
    Dependency,
}

/// Condition value
#[derive(Debug, Clone)]
pub enum ConditionValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// List value
    List(Vec<String>),
}

/// Condition operator
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionOperator {
    /// Equals
    Equals,
    /// Not equals
    NotEquals,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Contains
    Contains,
    /// In list
    In,
    /// Matches regex
    Matches,
}

/// Rule action
#[derive(Debug, Clone)]
pub struct RuleAction {
    /// Action type
    pub action_type: ActionType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
}

/// Action types
#[derive(Debug, Clone, PartialEq)]
pub enum ActionType {
    /// Create subtask
    CreateSubtask,
    /// Set priority
    SetPriority,
    /// Add dependency
    AddDependency,
    /// Set estimated effort
    SetEstimatedEffort,
    /// Add skill requirement
    AddSkillRequirement,
    /// Add resource requirement
    AddResourceRequirement,
    /// Add acceptance criteria
    AddAcceptanceCriteria,
}

/// Rule pattern for task types
#[derive(Debug, Clone)]
pub struct RulePattern {
    /// Pattern name
    pub name: String,
    /// Pattern template
    pub template: String,
    /// Pattern subtasks
    pub subtasks: Vec<SubtaskTemplate>,
    /// Pattern dependencies
    pub dependencies: Vec<(usize, usize)>, // (from_index, to_index)
}

/// Subtask template
#[derive(Debug, Clone)]
pub struct SubtaskTemplate {
    /// Template title
    pub title: String,
    /// Template description
    pub description: String,
    /// Estimated effort range
    pub effort_range: (f64, f64),
    /// Required skills
    pub required_skills: Vec<String>,
    /// Acceptance criteria template
    pub acceptance_criteria: Vec<String>,
}

/// Template repository
#[derive(Debug)]
pub struct TemplateRepository {
    /// Task templates
    templates: HashMap<String, TaskTemplate>,
    /// Template usage statistics
    usage_stats: HashMap<String, TemplateUsageStats>,
}

/// Task template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTemplate {
    /// Template ID
    pub id: String,
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template type
    pub template_type: TaskType,
    /// Template subtasks
    pub subtasks: Vec<SubtaskTemplate>,
    /// Template dependencies
    pub dependencies: Vec<(usize, usize)>,
    /// Template metadata
    pub metadata: TemplateMetadata,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template version
    pub version: String,
    /// Template author
    pub author: String,
    /// Template created date
    pub created_at: DateTime<Utc>,
    /// Template usage count
    pub usage_count: usize,
    /// Template success rate
    pub success_rate: f64,
    /// Template tags
    pub tags: Vec<String>,
}

/// Template usage statistics
#[derive(Debug, Clone)]
pub struct TemplateUsageStats {
    /// Usage count
    pub usage_count: usize,
    /// Success count
    pub success_count: usize,
    /// Average execution time
    pub avg_execution_time: f64,
    /// Last used date
    pub last_used: DateTime<Utc>,
}

/// Decomposer performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposerMetrics {
    /// Total decompositions
    pub total_decompositions: u64,
    /// Average subtask count
    pub avg_subtask_count: f64,
    /// Average decomposition time
    pub avg_decomposition_time_ms: f64,
    /// Strategy usage statistics
    pub strategy_usage: HashMap<DecompositionStrategy, u64>,
    /// Success rate by strategy
    pub strategy_success_rate: HashMap<DecompositionStrategy, f64>,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

impl Default for DecomposerConfig {
    fn default() -> Self {
        let mut strategy_weights = HashMap::new();
        strategy_weights.insert(DecompositionStrategy::Neural, 0.4);
        strategy_weights.insert(DecompositionStrategy::RuleBased, 0.3);
        strategy_weights.insert(DecompositionStrategy::Template, 0.3);

        Self {
            max_depth: 5,
            min_subtask_size: 1,
            max_subtask_count: 20,
            enable_caching: true,
            cache_size_limit: 1000,
            default_strategy: DecompositionStrategy::Hybrid,
            strategy_weights,
        }
    }
}

impl TaskDecomposer {
    /// Create a new task decomposer
    pub async fn new(neural_config: NeuralConfig) -> Result<Self> {
        let config = DecomposerConfig::default();
        Self::with_config(neural_config, config).await
    }

    /// Create a new task decomposer with custom configuration
    pub async fn with_config(neural_config: NeuralConfig, config: DecomposerConfig) -> Result<Self> {
        let neural_backend = Arc::new(NeuralBackend::new(neural_config).await?);
        let rule_engine = Arc::new(RuleEngine::new());
        let template_repo = Arc::new(TemplateRepository::new());
        let decomposition_cache = Arc::new(DashMap::new());
        let metrics = Arc::new(RwLock::new(DecomposerMetrics::default()));

        Ok(Self {
            neural_backend,
            rule_engine,
            template_repo,
            decomposition_cache,
            metrics,
            config,
        })
    }

    /// Decompose a task using the specified strategy
    pub async fn decompose(
        &self,
        parsed_task: &ParsedTask,
        context: &TaskContext,
        strategy: DecompositionStrategy,
    ) -> Result<DecompositionResult> {
        let start_time = std::time::Instant::now();

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached_result) = self.decomposition_cache.get(&parsed_task.id) {
                tracing::debug!("Retrieved decomposition from cache for task {}", parsed_task.id);
                return Ok(cached_result.clone());
            }
        }

        // Perform decomposition based on strategy
        let result = match strategy {
            DecompositionStrategy::Neural => {
                self.neural_decompose(parsed_task, context).await?
            }
            DecompositionStrategy::RuleBased => {
                self.rule_based_decompose(parsed_task, context).await?
            }
            DecompositionStrategy::Template => {
                self.template_decompose(parsed_task, context).await?
            }
            DecompositionStrategy::Hybrid => {
                self.hybrid_decompose(parsed_task, context).await?
            }
            DecompositionStrategy::Hierarchical => {
                self.hierarchical_decompose(parsed_task, context).await?
            }
            DecompositionStrategy::DependencyDriven => {
                self.dependency_driven_decompose(parsed_task, context).await?
            }
        };

        // Update metadata
        let decomposition_duration = start_time.elapsed();
        let mut final_result = result;
        final_result.metadata.decomposition_duration_ms = decomposition_duration.as_millis() as u64;
        final_result.metadata.decomposed_at = Utc::now();

        // Cache the result
        if self.config.enable_caching {
            self.decomposition_cache.insert(parsed_task.id, final_result.clone());
        }

        // Update metrics
        self.update_metrics(&strategy, &final_result).await;

        tracing::info!(
            "Decomposed task {} into {} subtasks using {:?} strategy",
            parsed_task.id,
            final_result.subtasks.len(),
            strategy
        );

        Ok(final_result)
    }

    /// Neural network-based decomposition
    async fn neural_decompose(&self, parsed_task: &ParsedTask, context: &TaskContext) -> Result<DecompositionResult> {
        // Encode task and context for neural processing
        let task_encoding = self.neural_backend.encode_task(parsed_task).await?;
        let context_encoding = self.neural_backend.encode_context(context).await?;

        // Use neural network to generate subtasks
        let subtasks = self.generate_neural_subtasks(parsed_task, &task_encoding, &context_encoding).await?;

        // Calculate quality metrics
        let quality_score = self.calculate_quality_score(&subtasks);
        let strategy_confidence = self.calculate_strategy_confidence(&subtasks, &context);

        let metadata = DecompositionMetadata {
            decomposed_at: Utc::now(),
            decomposition_duration_ms: 0, // Will be set later
            strategy_confidence,
            total_estimated_effort: subtasks.iter().map(|s| s.estimated_effort).sum(),
            total_estimated_duration: subtasks.iter().map(|s| s.estimated_duration).sum(),
            quality_score,
            used_templates: Vec::new(),
        };

        Ok(DecompositionResult {
            task_id: parsed_task.id,
            subtasks,
            strategy: DecompositionStrategy::Neural,
            metadata,
        })
    }

    /// Rule-based decomposition
    async fn rule_based_decompose(&self, parsed_task: &ParsedTask, context: &TaskContext) -> Result<DecompositionResult> {
        let subtasks = self.rule_engine.decompose(parsed_task, context).await?;

        let quality_score = self.calculate_quality_score(&subtasks);
        let strategy_confidence = self.calculate_strategy_confidence(&subtasks, &context);

        let metadata = DecompositionMetadata {
            decomposed_at: Utc::now(),
            decomposition_duration_ms: 0,
            strategy_confidence,
            total_estimated_effort: subtasks.iter().map(|s| s.estimated_effort).sum(),
            total_estimated_duration: subtasks.iter().map(|s| s.estimated_duration).sum(),
            quality_score,
            used_templates: Vec::new(),
        };

        Ok(DecompositionResult {
            task_id: parsed_task.id,
            subtasks,
            strategy: DecompositionStrategy::RuleBased,
            metadata,
        })
    }

    /// Template-based decomposition
    async fn template_decompose(&self, parsed_task: &ParsedTask, context: &TaskContext) -> Result<DecompositionResult> {
        let (subtasks, used_templates) = self.template_repo.decompose(parsed_task, context).await?;

        let quality_score = self.calculate_quality_score(&subtasks);
        let strategy_confidence = self.calculate_strategy_confidence(&subtasks, &context);

        let metadata = DecompositionMetadata {
            decomposed_at: Utc::now(),
            decomposition_duration_ms: 0,
            strategy_confidence,
            total_estimated_effort: subtasks.iter().map(|s| s.estimated_effort).sum(),
            total_estimated_duration: subtasks.iter().map(|s| s.estimated_duration).sum(),
            quality_score,
            used_templates,
        };

        Ok(DecompositionResult {
            task_id: parsed_task.id,
            subtasks,
            strategy: DecompositionStrategy::Template,
            metadata,
        })
    }

    /// Hybrid decomposition combining multiple strategies
    async fn hybrid_decompose(&self, parsed_task: &ParsedTask, context: &TaskContext) -> Result<DecompositionResult> {
        // Get results from multiple strategies
        let neural_result = self.neural_decompose(parsed_task, context).await?;
        let rule_result = self.rule_based_decompose(parsed_task, context).await?;
        let template_result = self.template_decompose(parsed_task, context).await?;

        // Combine results using strategy weights
        let subtasks = self.combine_strategy_results(
            &neural_result,
            &rule_result,
            &template_result,
        );

        let quality_score = self.calculate_quality_score(&subtasks);
        let strategy_confidence = self.calculate_strategy_confidence(&subtasks, &context);

        let metadata = DecompositionMetadata {
            decomposed_at: Utc::now(),
            decomposition_duration_ms: 0,
            strategy_confidence,
            total_estimated_effort: subtasks.iter().map(|s| s.estimated_effort).sum(),
            total_estimated_duration: subtasks.iter().map(|s| s.estimated_duration).sum(),
            quality_score,
            used_templates: template_result.metadata.used_templates,
        };

        Ok(DecompositionResult {
            task_id: parsed_task.id,
            subtasks,
            strategy: DecompositionStrategy::Hybrid,
            metadata,
        })
    }

    /// Hierarchical decomposition
    async fn hierarchical_decompose(&self, parsed_task: &ParsedTask, context: &TaskContext) -> Result<DecompositionResult> {
        let mut subtasks = Vec::new();
        let mut current_level = 0;
        let mut tasks_to_decompose = vec![(parsed_task.clone(), current_level)];

        while !tasks_to_decompose.is_empty() && current_level < self.config.max_depth {
            let mut next_level_tasks = Vec::new();

            for (task, level) in tasks_to_decompose {
                // Decompose current task
                let level_subtasks = self.decompose_single_level(&task, context, level).await?;
                
                for subtask in level_subtasks {
                    // Check if subtask needs further decomposition
                    if self.should_decompose_further(&subtask, level) {
                        // Convert subtask to ParsedTask for further decomposition
                        let subtask_as_task = self.subtask_to_parsed_task(&subtask);
                        next_level_tasks.push((subtask_as_task, level + 1));
                    }
                    subtasks.push(subtask);
                }
            }

            tasks_to_decompose = next_level_tasks;
            current_level += 1;
        }

        let quality_score = self.calculate_quality_score(&subtasks);
        let strategy_confidence = self.calculate_strategy_confidence(&subtasks, &context);

        let metadata = DecompositionMetadata {
            decomposed_at: Utc::now(),
            decomposition_duration_ms: 0,
            strategy_confidence,
            total_estimated_effort: subtasks.iter().map(|s| s.estimated_effort).sum(),
            total_estimated_duration: subtasks.iter().map(|s| s.estimated_duration).sum(),
            quality_score,
            used_templates: Vec::new(),
        };

        Ok(DecompositionResult {
            task_id: parsed_task.id,
            subtasks,
            strategy: DecompositionStrategy::Hierarchical,
            metadata,
        })
    }

    /// Dependency-driven decomposition
    async fn dependency_driven_decompose(&self, parsed_task: &ParsedTask, context: &TaskContext) -> Result<DecompositionResult> {
        // Start with basic decomposition
        let mut subtasks = self.neural_decompose(parsed_task, context).await?.subtasks;

        // Analyze dependencies and create additional subtasks if needed
        let dependency_subtasks = self.analyze_and_create_dependency_subtasks(&subtasks, context).await?;
        subtasks.extend(dependency_subtasks);

        // Reorder subtasks based on dependencies
        subtasks = self.reorder_by_dependencies(subtasks);

        let quality_score = self.calculate_quality_score(&subtasks);
        let strategy_confidence = self.calculate_strategy_confidence(&subtasks, &context);

        let metadata = DecompositionMetadata {
            decomposed_at: Utc::now(),
            decomposition_duration_ms: 0,
            strategy_confidence,
            total_estimated_effort: subtasks.iter().map(|s| s.estimated_effort).sum(),
            total_estimated_duration: subtasks.iter().map(|s| s.estimated_duration).sum(),
            quality_score,
            used_templates: Vec::new(),
        };

        Ok(DecompositionResult {
            task_id: parsed_task.id,
            subtasks,
            strategy: DecompositionStrategy::DependencyDriven,
            metadata,
        })
    }

    /// Generate subtasks using neural network
    async fn generate_neural_subtasks(
        &self,
        parsed_task: &ParsedTask,
        task_encoding: &[f32],
        context_encoding: &[f32],
    ) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();

        // Use neural network to determine number of subtasks
        let subtask_count = self.predict_subtask_count(task_encoding, context_encoding).await?;

        // Generate subtasks based on task type and entities
        match parsed_task.task_type {
            TaskType::Development => {
                subtasks.extend(self.generate_development_subtasks(parsed_task, subtask_count).await?);
            }
            TaskType::Analysis => {
                subtasks.extend(self.generate_analysis_subtasks(parsed_task, subtask_count).await?);
            }
            TaskType::Testing => {
                subtasks.extend(self.generate_testing_subtasks(parsed_task, subtask_count).await?);
            }
            TaskType::Deployment => {
                subtasks.extend(self.generate_deployment_subtasks(parsed_task, subtask_count).await?);
            }
            _ => {
                subtasks.extend(self.generate_generic_subtasks(parsed_task, subtask_count).await?);
            }
        }

        Ok(subtasks)
    }

    /// Predict number of subtasks using neural network
    async fn predict_subtask_count(&self, task_encoding: &[f32], context_encoding: &[f32]) -> Result<usize> {
        // Simple prediction based on encoding features
        let complexity_score = task_encoding.iter().take(10).sum::<f32>() / 10.0;
        let context_score = context_encoding.iter().take(10).sum::<f32>() / 10.0;
        
        let predicted_count = ((complexity_score + context_score) * 10.0) as usize;
        Ok(predicted_count.clamp(2, self.config.max_subtask_count))
    }

    /// Generate development subtasks
    async fn generate_development_subtasks(&self, parsed_task: &ParsedTask, count: usize) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        // Standard development workflow
        let development_phases = vec![
            ("Requirements Analysis", "Analyze and document requirements", 4.0, 8.0),
            ("System Design", "Design system architecture and components", 6.0, 12.0),
            ("Implementation", "Implement core functionality", 16.0, 24.0),
            ("Testing", "Write and execute tests", 8.0, 16.0),
            ("Documentation", "Create technical documentation", 4.0, 8.0),
            ("Code Review", "Review code and address feedback", 2.0, 4.0),
            ("Deployment", "Deploy to production environment", 2.0, 4.0),
        ];

        for (i, (title, description, min_effort, max_effort)) in development_phases.iter().enumerate() {
            if i >= count {
                break;
            }

            let subtask = Subtask {
                id: Uuid::new_v4(),
                parent_id: parsed_task.id,
                title: title.to_string(),
                description: description.to_string(),
                task_type: TaskType::Development,
                priority: parsed_task.input.priority,
                estimated_effort: (min_effort + max_effort) / 2.0,
                estimated_duration: (min_effort + max_effort) / 2.0,
                required_skills: self.extract_required_skills(&parsed_task.entities),
                required_resources: self.extract_required_resources(&parsed_task.entities),
                dependencies: if i > 0 { vec![subtasks[i-1].id] } else { Vec::new() },
                acceptance_criteria: self.generate_acceptance_criteria(title, &parsed_task.entities),
                level: 1,
                created_at: Utc::now(),
                metadata: SubtaskMetadata {
                    confidence: 0.8,
                    rationale: format!("Standard development phase: {}", title),
                    alternatives: Vec::new(),
                    risk_factors: Vec::new(),
                    success_probability: 0.85,
                },
            };

            subtasks.push(subtask);
        }

        Ok(subtasks)
    }

    /// Generate analysis subtasks
    async fn generate_analysis_subtasks(&self, parsed_task: &ParsedTask, count: usize) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        let analysis_phases = vec![
            ("Data Collection", "Gather and prepare data for analysis", 4.0, 8.0),
            ("Data Exploration", "Explore and understand data structure", 6.0, 12.0),
            ("Analysis Implementation", "Implement analysis algorithms", 12.0, 20.0),
            ("Results Interpretation", "Interpret and validate results", 4.0, 8.0),
            ("Report Generation", "Create analysis report", 4.0, 8.0),
        ];

        for (i, (title, description, min_effort, max_effort)) in analysis_phases.iter().enumerate() {
            if i >= count {
                break;
            }

            let subtask = Subtask {
                id: Uuid::new_v4(),
                parent_id: parsed_task.id,
                title: title.to_string(),
                description: description.to_string(),
                task_type: TaskType::Analysis,
                priority: parsed_task.input.priority,
                estimated_effort: (min_effort + max_effort) / 2.0,
                estimated_duration: (min_effort + max_effort) / 2.0,
                required_skills: vec!["data analysis".to_string(), "statistics".to_string()],
                required_resources: vec!["data access".to_string(), "analysis tools".to_string()],
                dependencies: if i > 0 { vec![subtasks[i-1].id] } else { Vec::new() },
                acceptance_criteria: self.generate_acceptance_criteria(title, &parsed_task.entities),
                level: 1,
                created_at: Utc::now(),
                metadata: SubtaskMetadata {
                    confidence: 0.75,
                    rationale: format!("Standard analysis phase: {}", title),
                    alternatives: Vec::new(),
                    risk_factors: Vec::new(),
                    success_probability: 0.8,
                },
            };

            subtasks.push(subtask);
        }

        Ok(subtasks)
    }

    /// Generate testing subtasks
    async fn generate_testing_subtasks(&self, parsed_task: &ParsedTask, count: usize) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        let testing_phases = vec![
            ("Test Planning", "Plan testing strategy and approach", 2.0, 4.0),
            ("Unit Testing", "Write and execute unit tests", 8.0, 16.0),
            ("Integration Testing", "Test component integration", 6.0, 12.0),
            ("System Testing", "Test complete system functionality", 8.0, 16.0),
            ("Performance Testing", "Test system performance", 4.0, 8.0),
            ("Security Testing", "Test system security", 4.0, 8.0),
            ("User Acceptance Testing", "Validate user requirements", 4.0, 8.0),
        ];

        for (i, (title, description, min_effort, max_effort)) in testing_phases.iter().enumerate() {
            if i >= count {
                break;
            }

            let subtask = Subtask {
                id: Uuid::new_v4(),
                parent_id: parsed_task.id,
                title: title.to_string(),
                description: description.to_string(),
                task_type: TaskType::Testing,
                priority: parsed_task.input.priority,
                estimated_effort: (min_effort + max_effort) / 2.0,
                estimated_duration: (min_effort + max_effort) / 2.0,
                required_skills: vec!["testing".to_string(), "quality assurance".to_string()],
                required_resources: vec!["test environment".to_string(), "testing tools".to_string()],
                dependencies: if i > 0 { vec![subtasks[i-1].id] } else { Vec::new() },
                acceptance_criteria: self.generate_acceptance_criteria(title, &parsed_task.entities),
                level: 1,
                created_at: Utc::now(),
                metadata: SubtaskMetadata {
                    confidence: 0.85,
                    rationale: format!("Standard testing phase: {}", title),
                    alternatives: Vec::new(),
                    risk_factors: Vec::new(),
                    success_probability: 0.9,
                },
            };

            subtasks.push(subtask);
        }

        Ok(subtasks)
    }

    /// Generate deployment subtasks
    async fn generate_deployment_subtasks(&self, parsed_task: &ParsedTask, count: usize) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        let deployment_phases = vec![
            ("Environment Setup", "Setup deployment environment", 4.0, 8.0),
            ("Build Process", "Configure build and packaging", 2.0, 4.0),
            ("Deployment Automation", "Setup automated deployment", 6.0, 12.0),
            ("Monitoring Setup", "Configure monitoring and alerting", 4.0, 8.0),
            ("Production Deployment", "Deploy to production", 2.0, 4.0),
            ("Post-Deployment Testing", "Validate production deployment", 2.0, 4.0),
        ];

        for (i, (title, description, min_effort, max_effort)) in deployment_phases.iter().enumerate() {
            if i >= count {
                break;
            }

            let subtask = Subtask {
                id: Uuid::new_v4(),
                parent_id: parsed_task.id,
                title: title.to_string(),
                description: description.to_string(),
                task_type: TaskType::Deployment,
                priority: parsed_task.input.priority,
                estimated_effort: (min_effort + max_effort) / 2.0,
                estimated_duration: (min_effort + max_effort) / 2.0,
                required_skills: vec!["devops".to_string(), "system administration".to_string()],
                required_resources: vec!["deployment environment".to_string(), "deployment tools".to_string()],
                dependencies: if i > 0 { vec![subtasks[i-1].id] } else { Vec::new() },
                acceptance_criteria: self.generate_acceptance_criteria(title, &parsed_task.entities),
                level: 1,
                created_at: Utc::now(),
                metadata: SubtaskMetadata {
                    confidence: 0.8,
                    rationale: format!("Standard deployment phase: {}", title),
                    alternatives: Vec::new(),
                    risk_factors: Vec::new(),
                    success_probability: 0.85,
                },
            };

            subtasks.push(subtask);
        }

        Ok(subtasks)
    }

    /// Generate generic subtasks
    async fn generate_generic_subtasks(&self, parsed_task: &ParsedTask, count: usize) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        // Extract action entities to create subtasks
        let action_entities: Vec<_> = parsed_task.entities.iter()
            .filter(|e| e.entity_type == crate::parser::EntityType::Action)
            .collect();

        for (i, entity) in action_entities.iter().enumerate() {
            if i >= count {
                break;
            }

            let subtask = Subtask {
                id: Uuid::new_v4(),
                parent_id: parsed_task.id,
                title: format!("{} Task", entity.text),
                description: format!("Execute {} operation", entity.text),
                task_type: parsed_task.task_type.clone(),
                priority: parsed_task.input.priority,
                estimated_effort: 4.0,
                estimated_duration: 6.0,
                required_skills: self.extract_required_skills(&parsed_task.entities),
                required_resources: self.extract_required_resources(&parsed_task.entities),
                dependencies: Vec::new(),
                acceptance_criteria: self.generate_acceptance_criteria(&entity.text, &parsed_task.entities),
                level: 1,
                created_at: Utc::now(),
                metadata: SubtaskMetadata {
                    confidence: 0.7,
                    rationale: format!("Generated from action entity: {}", entity.text),
                    alternatives: Vec::new(),
                    risk_factors: Vec::new(),
                    success_probability: 0.75,
                },
            };

            subtasks.push(subtask);
        }

        // If no action entities, create generic subtasks
        if subtasks.is_empty() {
            let generic_subtasks = vec![
                "Planning and Analysis",
                "Implementation",
                "Testing and Validation",
                "Documentation",
                "Deployment",
            ];

            for (i, title) in generic_subtasks.iter().enumerate() {
                if i >= count {
                    break;
                }

                let subtask = Subtask {
                    id: Uuid::new_v4(),
                    parent_id: parsed_task.id,
                    title: title.to_string(),
                    description: format!("Execute {} phase", title),
                    task_type: parsed_task.task_type.clone(),
                    priority: parsed_task.input.priority,
                    estimated_effort: 4.0,
                    estimated_duration: 6.0,
                    required_skills: Vec::new(),
                    required_resources: Vec::new(),
                    dependencies: if i > 0 { vec![subtasks[i-1].id] } else { Vec::new() },
                    acceptance_criteria: Vec::new(),
                    level: 1,
                    created_at: Utc::now(),
                    metadata: SubtaskMetadata {
                        confidence: 0.6,
                        rationale: "Generic subtask".to_string(),
                        alternatives: Vec::new(),
                        risk_factors: Vec::new(),
                        success_probability: 0.7,
                    },
                };

                subtasks.push(subtask);
            }
        }

        Ok(subtasks)
    }

    /// Extract required skills from entities
    fn extract_required_skills(&self, entities: &[TaskEntity]) -> Vec<String> {
        let mut skills = Vec::new();
        
        for entity in entities {
            match entity.entity_type {
                crate::parser::EntityType::Technology => {
                    skills.push(entity.text.clone());
                }
                crate::parser::EntityType::Action => {
                    match entity.text.to_lowercase().as_str() {
                        "develop" | "build" | "create" | "implement" => {
                            skills.push("programming".to_string());
                        }
                        "test" | "verify" | "validate" => {
                            skills.push("testing".to_string());
                        }
                        "deploy" | "release" => {
                            skills.push("devops".to_string());
                        }
                        "analyze" | "research" => {
                            skills.push("analysis".to_string());
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        skills.sort();
        skills.dedup();
        skills
    }

    /// Extract required resources from entities
    fn extract_required_resources(&self, entities: &[TaskEntity]) -> Vec<String> {
        let mut resources = Vec::new();
        
        for entity in entities {
            match entity.entity_type {
                crate::parser::EntityType::Resource => {
                    resources.push(entity.text.clone());
                }
                crate::parser::EntityType::Technology => {
                    resources.push(format!("{} environment", entity.text));
                }
                _ => {}
            }
        }

        resources.sort();
        resources.dedup();
        resources
    }

    /// Generate acceptance criteria
    fn generate_acceptance_criteria(&self, title: &str, entities: &[TaskEntity]) -> Vec<String> {
        let mut criteria = Vec::new();
        
        // Generic criteria based on title
        match title.to_lowercase().as_str() {
            s if s.contains("test") => {
                criteria.push("All tests pass successfully".to_string());
                criteria.push("Test coverage meets requirements".to_string());
            }
            s if s.contains("deploy") => {
                criteria.push("Deployment completes without errors".to_string());
                criteria.push("System is accessible and functional".to_string());
            }
            s if s.contains("implement") || s.contains("develop") => {
                criteria.push("Code compiles without errors".to_string());
                criteria.push("Functionality works as specified".to_string());
            }
            s if s.contains("document") => {
                criteria.push("Documentation is complete and accurate".to_string());
                criteria.push("Documentation follows style guidelines".to_string());
            }
            _ => {
                criteria.push("Task completed successfully".to_string());
                criteria.push("Quality standards met".to_string());
            }
        }

        // Add technology-specific criteria
        for entity in entities {
            if entity.entity_type == crate::parser::EntityType::Technology {
                criteria.push(format!("{} integration working correctly", entity.text));
            }
        }

        criteria
    }

    /// Combine results from multiple strategies
    fn combine_strategy_results(
        &self,
        neural_result: &DecompositionResult,
        rule_result: &DecompositionResult,
        template_result: &DecompositionResult,
    ) -> Vec<Subtask> {
        let mut combined_subtasks = Vec::new();
        
        // Get weights from config
        let neural_weight = self.config.strategy_weights.get(&DecompositionStrategy::Neural).unwrap_or(&0.4);
        let rule_weight = self.config.strategy_weights.get(&DecompositionStrategy::RuleBased).unwrap_or(&0.3);
        let template_weight = self.config.strategy_weights.get(&DecompositionStrategy::Template).unwrap_or(&0.3);

        // Score each subtask from each strategy
        let mut all_subtasks: Vec<(Subtask, f64)> = Vec::new();
        
        for subtask in &neural_result.subtasks {
            let score = subtask.metadata.confidence * neural_weight;
            all_subtasks.push((subtask.clone(), score));
        }
        
        for subtask in &rule_result.subtasks {
            let score = subtask.metadata.confidence * rule_weight;
            all_subtasks.push((subtask.clone(), score));
        }
        
        for subtask in &template_result.subtasks {
            let score = subtask.metadata.confidence * template_weight;
            all_subtasks.push((subtask.clone(), score));
        }

        // Sort by score and deduplicate similar subtasks
        all_subtasks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for (subtask, _score) in all_subtasks {
            if !self.is_duplicate_subtask(&subtask, &combined_subtasks) {
                combined_subtasks.push(subtask);
            }
            
            if combined_subtasks.len() >= self.config.max_subtask_count {
                break;
            }
        }

        combined_subtasks
    }

    /// Check if subtask is duplicate
    fn is_duplicate_subtask(&self, subtask: &Subtask, existing: &[Subtask]) -> bool {
        for existing_subtask in existing {
            let title_similarity = self.calculate_string_similarity(&subtask.title, &existing_subtask.title);
            let desc_similarity = self.calculate_string_similarity(&subtask.description, &existing_subtask.description);
            
            if title_similarity > 0.8 || desc_similarity > 0.8 {
                return true;
            }
        }
        false
    }

    /// Calculate string similarity (simplified Jaccard similarity)
    fn calculate_string_similarity(&self, a: &str, b: &str) -> f64 {
        let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
        
        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Decompose single level for hierarchical approach
    async fn decompose_single_level(&self, task: &ParsedTask, context: &TaskContext, level: usize) -> Result<Vec<Subtask>> {
        // Use neural decomposition for single level
        let result = self.neural_decompose(task, context).await?;
        
        // Update level for all subtasks
        let mut subtasks = result.subtasks;
        for subtask in &mut subtasks {
            subtask.level = level + 1;
        }
        
        Ok(subtasks)
    }

    /// Check if subtask should be decomposed further
    fn should_decompose_further(&self, subtask: &Subtask, current_level: usize) -> bool {
        current_level < self.config.max_depth &&
        subtask.estimated_effort > 16.0 && // More than 2 person-days
        subtask.description.len() > 50 // Complex description
    }

    /// Convert subtask to ParsedTask for further decomposition
    fn subtask_to_parsed_task(&self, subtask: &Subtask) -> ParsedTask {
        use crate::parser::{TaskInput, InputFormat, DifficultyLevel, ParseMetadata};
        
        let input = TaskInput {
            description: subtask.description.clone(),
            priority: subtask.priority,
            deadline: None,
            context: HashMap::new(),
            format: InputFormat::PlainText,
            tags: Vec::new(),
            dependencies: Vec::new(),
            resources: subtask.required_resources.clone(),
            complexity: None,
        };

        ParsedTask {
            id: subtask.id,
            input,
            entities: Vec::new(), // Would need to re-parse in real implementation
            task_type: subtask.task_type.clone(),
            requirements: Vec::new(),
            difficulty: DifficultyLevel::Medium, // Would need to recalculate
            metadata: ParseMetadata {
                parsed_at: Utc::now(),
                parse_duration_ms: 0,
                confidence: subtask.metadata.confidence,
                language: Some("natural".to_string()),
                word_count: subtask.description.split_whitespace().count(),
                char_count: subtask.description.chars().count(),
            },
        }
    }

    /// Analyze and create dependency subtasks
    async fn analyze_and_create_dependency_subtasks(&self, subtasks: &[Subtask], context: &TaskContext) -> Result<Vec<Subtask>> {
        let mut dependency_subtasks = Vec::new();
        
        // Analyze relationships to identify missing dependencies
        for relationship in &context.relationships {
            if relationship.relationship_type == crate::analyzer::RelationshipType::Dependency {
                // Check if we need to create setup/configuration subtasks
                if relationship.source.contains("database") || relationship.target.contains("database") {
                    dependency_subtasks.push(self.create_database_setup_subtask(subtasks[0].parent_id));
                }
                
                if relationship.source.contains("docker") || relationship.target.contains("docker") {
                    dependency_subtasks.push(self.create_docker_setup_subtask(subtasks[0].parent_id));
                }
            }
        }

        Ok(dependency_subtasks)
    }

    /// Create database setup subtask
    fn create_database_setup_subtask(&self, parent_id: TaskId) -> Subtask {
        Subtask {
            id: Uuid::new_v4(),
            parent_id,
            title: "Database Setup".to_string(),
            description: "Setup and configure database environment".to_string(),
            task_type: TaskType::Administration,
            priority: 8,
            estimated_effort: 4.0,
            estimated_duration: 6.0,
            required_skills: vec!["database administration".to_string()],
            required_resources: vec!["database server".to_string()],
            dependencies: Vec::new(),
            acceptance_criteria: vec![
                "Database server is running".to_string(),
                "Database schema is created".to_string(),
                "Database is accessible".to_string(),
            ],
            level: 1,
            created_at: Utc::now(),
            metadata: SubtaskMetadata {
                confidence: 0.9,
                rationale: "Database dependency identified".to_string(),
                alternatives: Vec::new(),
                risk_factors: vec!["Database connectivity issues".to_string()],
                success_probability: 0.85,
            },
        }
    }

    /// Create docker setup subtask
    fn create_docker_setup_subtask(&self, parent_id: TaskId) -> Subtask {
        Subtask {
            id: Uuid::new_v4(),
            parent_id,
            title: "Docker Environment Setup".to_string(),
            description: "Setup Docker containerization environment".to_string(),
            task_type: TaskType::Administration,
            priority: 7,
            estimated_effort: 3.0,
            estimated_duration: 4.0,
            required_skills: vec!["docker".to_string(), "containerization".to_string()],
            required_resources: vec!["docker environment".to_string()],
            dependencies: Vec::new(),
            acceptance_criteria: vec![
                "Docker is installed and running".to_string(),
                "Docker images can be built".to_string(),
                "Containers can be started".to_string(),
            ],
            level: 1,
            created_at: Utc::now(),
            metadata: SubtaskMetadata {
                confidence: 0.85,
                rationale: "Docker dependency identified".to_string(),
                alternatives: Vec::new(),
                risk_factors: vec!["Docker setup complexity".to_string()],
                success_probability: 0.8,
            },
        }
    }

    /// Reorder subtasks based on dependencies
    fn reorder_by_dependencies(&self, mut subtasks: Vec<Subtask>) -> Vec<Subtask> {
        // Simple topological sort based on dependencies
        let mut ordered = Vec::new();
        let mut remaining = subtasks;
        
        while !remaining.is_empty() {
            // Find tasks with no unresolved dependencies
            let mut ready_indices = Vec::new();
            for (i, subtask) in remaining.iter().enumerate() {
                let has_unresolved_deps = subtask.dependencies.iter().any(|dep_id| {
                    !ordered.iter().any(|s: &Subtask| s.id == *dep_id)
                });
                
                if !has_unresolved_deps {
                    ready_indices.push(i);
                }
            }
            
            if ready_indices.is_empty() {
                // No more resolvable dependencies, add remaining tasks
                ordered.extend(remaining);
                break;
            }
            
            // Add ready tasks to ordered list
            for &i in ready_indices.iter().rev() {
                ordered.push(remaining.remove(i));
            }
        }
        
        ordered
    }

    /// Calculate quality score for subtasks
    fn calculate_quality_score(&self, subtasks: &[Subtask]) -> f64 {
        if subtasks.is_empty() {
            return 0.0;
        }

        let avg_confidence = subtasks.iter().map(|s| s.metadata.confidence).sum::<f64>() / subtasks.len() as f64;
        let avg_success_prob = subtasks.iter().map(|s| s.metadata.success_probability).sum::<f64>() / subtasks.len() as f64;
        
        // Consider subtask count (not too few, not too many)
        let count_score = match subtasks.len() {
            1..=2 => 0.6,
            3..=8 => 1.0,
            9..=15 => 0.8,
            _ => 0.5,
        };

        (avg_confidence * 0.4 + avg_success_prob * 0.4 + count_score * 0.2).min(1.0)
    }

    /// Calculate strategy confidence
    fn calculate_strategy_confidence(&self, subtasks: &[Subtask], context: &TaskContext) -> f64 {
        let subtask_confidence = if subtasks.is_empty() {
            0.0
        } else {
            subtasks.iter().map(|s| s.metadata.confidence).sum::<f64>() / subtasks.len() as f64
        };

        let context_confidence = context.semantic_analysis.confidence;
        
        (subtask_confidence * 0.7 + context_confidence * 0.3).min(1.0)
    }

    /// Update performance metrics
    async fn update_metrics(&self, strategy: &DecompositionStrategy, result: &DecompositionResult) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_decompositions += 1;
        
        // Update strategy usage
        *metrics.strategy_usage.entry(strategy.clone()).or_insert(0) += 1;
        
        // Update average subtask count
        let new_avg = (metrics.avg_subtask_count * (metrics.total_decompositions - 1) as f64 + result.subtasks.len() as f64) / metrics.total_decompositions as f64;
        metrics.avg_subtask_count = new_avg;
        
        // Update average decomposition time
        let new_time_avg = (metrics.avg_decomposition_time_ms * (metrics.total_decompositions - 1) as f64 + result.metadata.decomposition_duration_ms as f64) / metrics.total_decompositions as f64;
        metrics.avg_decomposition_time_ms = new_time_avg;
    }

    /// Get total decompositions count
    pub async fn total_decompositions(&self) -> u64 {
        self.metrics.read().await.total_decompositions
    }

    /// Get average subtask count
    pub async fn avg_subtask_count(&self) -> f64 {
        self.metrics.read().await.avg_subtask_count
    }

    /// Update configuration
    pub async fn update_config(&mut self, config: DecomposerConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Retrain neural models
    pub async fn retrain(&mut self) -> Result<()> {
        // In practice, would implement model retraining
        tracing::info!("Retraining neural models");
        Ok(())
    }
}

impl RuleEngine {
    /// Create a new rule engine
    pub fn new() -> Self {
        Self {
            rules: Self::create_default_rules(),
            patterns: Self::create_default_patterns(),
        }
    }

    /// Decompose task using rules
    pub async fn decompose(&self, parsed_task: &ParsedTask, context: &TaskContext) -> Result<Vec<Subtask>> {
        let mut subtasks = Vec::new();
        
        // Apply rules based on task type
        if let Some(patterns) = self.patterns.get(&parsed_task.task_type) {
            for pattern in patterns {
                if self.pattern_matches(pattern, parsed_task, context) {
                    let pattern_subtasks = self.apply_pattern(pattern, parsed_task);
                    subtasks.extend(pattern_subtasks);
                    break; // Use first matching pattern
                }
            }
        }

        // Apply general rules
        for rule in &self.rules {
            if self.rule_matches(rule, parsed_task, context) {
                let rule_subtasks = self.apply_rule(rule, parsed_task);
                subtasks.extend(rule_subtasks);
            }
        }

        Ok(subtasks)
    }

    /// Check if pattern matches task
    fn pattern_matches(&self, pattern: &RulePattern, parsed_task: &ParsedTask, context: &TaskContext) -> bool {
        // Simple pattern matching based on task type and keywords
        parsed_task.task_type == TaskType::Development || 
        context.semantic_analysis.key_concepts.iter().any(|c| c.name.contains(&pattern.name.to_lowercase()))
    }

    /// Apply pattern to create subtasks
    fn apply_pattern(&self, pattern: &RulePattern, parsed_task: &ParsedTask) -> Vec<Subtask> {
        let mut subtasks = Vec::new();
        
        for (i, template) in pattern.subtasks.iter().enumerate() {
            let subtask = Subtask {
                id: Uuid::new_v4(),
                parent_id: parsed_task.id,
                title: template.title.clone(),
                description: template.description.clone(),
                task_type: parsed_task.task_type.clone(),
                priority: parsed_task.input.priority,
                estimated_effort: (template.effort_range.0 + template.effort_range.1) / 2.0,
                estimated_duration: (template.effort_range.0 + template.effort_range.1) / 2.0,
                required_skills: template.required_skills.clone(),
                required_resources: Vec::new(),
                dependencies: pattern.dependencies.iter()
                    .filter_map(|(from, to)| if *to == i { subtasks.get(*from).map(|s| s.id) } else { None })
                    .collect(),
                acceptance_criteria: template.acceptance_criteria.clone(),
                level: 1,
                created_at: Utc::now(),
                metadata: SubtaskMetadata {
                    confidence: 0.8,
                    rationale: format!("Applied pattern: {}", pattern.name),
                    alternatives: Vec::new(),
                    risk_factors: Vec::new(),
                    success_probability: 0.85,
                },
            };
            
            subtasks.push(subtask);
        }
        
        subtasks
    }

    /// Check if rule matches task
    fn rule_matches(&self, rule: &DecompositionRule, parsed_task: &ParsedTask, context: &TaskContext) -> bool {
        for condition in &rule.conditions {
            if !self.evaluate_condition(condition, parsed_task, context) {
                return false;
            }
        }
        true
    }

    /// Evaluate rule condition
    fn evaluate_condition(&self, condition: &RuleCondition, parsed_task: &ParsedTask, context: &TaskContext) -> bool {
        match condition.condition_type {
            ConditionType::TaskType => {
                if let ConditionValue::String(ref task_type_str) = condition.value {
                    let expected_type = match task_type_str.as_str() {
                        "development" => TaskType::Development,
                        "analysis" => TaskType::Analysis,
                        "testing" => TaskType::Testing,
                        "deployment" => TaskType::Deployment,
                        _ => TaskType::Generic,
                    };
                    
                    match condition.operator {
                        ConditionOperator::Equals => parsed_task.task_type == expected_type,
                        ConditionOperator::NotEquals => parsed_task.task_type != expected_type,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            ConditionType::Priority => {
                if let ConditionValue::Integer(ref priority) = condition.value {
                    let task_priority = parsed_task.input.priority as i64;
                    match condition.operator {
                        ConditionOperator::Equals => task_priority == *priority,
                        ConditionOperator::GreaterThan => task_priority > *priority,
                        ConditionOperator::LessThan => task_priority < *priority,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            ConditionType::Complexity => {
                if let ConditionValue::Integer(ref complexity) = condition.value {
                    if let Some(task_complexity) = parsed_task.input.complexity {
                        let task_complexity = task_complexity as i64;
                        match condition.operator {
                            ConditionOperator::Equals => task_complexity == *complexity,
                            ConditionOperator::GreaterThan => task_complexity > *complexity,
                            ConditionOperator::LessThan => task_complexity < *complexity,
                            _ => false,
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            ConditionType::EntityCount => {
                if let ConditionValue::Integer(ref count) = condition.value {
                    let entity_count = parsed_task.entities.len() as i64;
                    match condition.operator {
                        ConditionOperator::Equals => entity_count == *count,
                        ConditionOperator::GreaterThan => entity_count > *count,
                        ConditionOperator::LessThan => entity_count < *count,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            ConditionType::Keyword => {
                if let ConditionValue::String(ref keyword) = condition.value {
                    match condition.operator {
                        ConditionOperator::Contains => parsed_task.input.description.to_lowercase().contains(&keyword.to_lowercase()),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            ConditionType::Dependency => {
                if let ConditionValue::Integer(ref count) = condition.value {
                    let dep_count = parsed_task.input.dependencies.len() as i64;
                    match condition.operator {
                        ConditionOperator::Equals => dep_count == *count,
                        ConditionOperator::GreaterThan => dep_count > *count,
                        ConditionOperator::LessThan => dep_count < *count,
                        _ => false,
                    }
                } else {
                    false
                }
            }
        }
    }

    /// Apply rule to create subtasks
    fn apply_rule(&self, rule: &DecompositionRule, parsed_task: &ParsedTask) -> Vec<Subtask> {
        let mut subtasks = Vec::new();
        
        for action in &rule.actions {
            if action.action_type == ActionType::CreateSubtask {
                let title = action.parameters.get("title").cloned().unwrap_or_else(|| "Generated Subtask".to_string());
                let description = action.parameters.get("description").cloned().unwrap_or_else(|| "Generated subtask from rule".to_string());
                
                let subtask = Subtask {
                    id: Uuid::new_v4(),
                    parent_id: parsed_task.id,
                    title,
                    description,
                    task_type: parsed_task.task_type.clone(),
                    priority: parsed_task.input.priority,
                    estimated_effort: 4.0,
                    estimated_duration: 6.0,
                    required_skills: Vec::new(),
                    required_resources: Vec::new(),
                    dependencies: Vec::new(),
                    acceptance_criteria: Vec::new(),
                    level: 1,
                    created_at: Utc::now(),
                    metadata: SubtaskMetadata {
                        confidence: rule.confidence,
                        rationale: format!("Applied rule: {}", rule.name),
                        alternatives: Vec::new(),
                        risk_factors: Vec::new(),
                        success_probability: 0.8,
                    },
                };
                
                subtasks.push(subtask);
            }
        }
        
        subtasks
    }

    /// Create default rules
    fn create_default_rules() -> Vec<DecompositionRule> {
        vec![
            DecompositionRule {
                id: "high_priority_rule".to_string(),
                name: "High Priority Task Rule".to_string(),
                conditions: vec![
                    RuleCondition {
                        condition_type: ConditionType::Priority,
                        value: ConditionValue::Integer(8),
                        operator: ConditionOperator::GreaterThan,
                    },
                ],
                actions: vec![
                    RuleAction {
                        action_type: ActionType::CreateSubtask,
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("title".to_string(), "Risk Assessment".to_string());
                            params.insert("description".to_string(), "Assess risks for high priority task".to_string());
                            params
                        },
                    },
                ],
                priority: 9,
                confidence: 0.9,
            },
            DecompositionRule {
                id: "complex_task_rule".to_string(),
                name: "Complex Task Rule".to_string(),
                conditions: vec![
                    RuleCondition {
                        condition_type: ConditionType::Complexity,
                        value: ConditionValue::Integer(7),
                        operator: ConditionOperator::GreaterThan,
                    },
                ],
                actions: vec![
                    RuleAction {
                        action_type: ActionType::CreateSubtask,
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("title".to_string(), "Complexity Analysis".to_string());
                            params.insert("description".to_string(), "Analyze complex task requirements".to_string());
                            params
                        },
                    },
                ],
                priority: 8,
                confidence: 0.85,
            },
        ]
    }

    /// Create default patterns
    fn create_default_patterns() -> HashMap<TaskType, Vec<RulePattern>> {
        let mut patterns = HashMap::new();
        
        // Development patterns
        patterns.insert(TaskType::Development, vec![
            RulePattern {
                name: "Standard Development".to_string(),
                template: "development".to_string(),
                subtasks: vec![
                    SubtaskTemplate {
                        title: "Requirements Analysis".to_string(),
                        description: "Analyze and document requirements".to_string(),
                        effort_range: (2.0, 8.0),
                        required_skills: vec!["analysis".to_string()],
                        acceptance_criteria: vec!["Requirements documented".to_string()],
                    },
                    SubtaskTemplate {
                        title: "System Design".to_string(),
                        description: "Design system architecture".to_string(),
                        effort_range: (4.0, 12.0),
                        required_skills: vec!["architecture".to_string()],
                        acceptance_criteria: vec!["Design approved".to_string()],
                    },
                    SubtaskTemplate {
                        title: "Implementation".to_string(),
                        description: "Implement system components".to_string(),
                        effort_range: (8.0, 32.0),
                        required_skills: vec!["programming".to_string()],
                        acceptance_criteria: vec!["Code implemented".to_string()],
                    },
                ],
                dependencies: vec![(0, 1), (1, 2)],
            },
        ]);
        
        patterns
    }
}

impl TemplateRepository {
    /// Create a new template repository
    pub fn new() -> Self {
        Self {
            templates: Self::create_default_templates(),
            usage_stats: HashMap::new(),
        }
    }

    /// Decompose task using templates
    pub async fn decompose(&self, parsed_task: &ParsedTask, context: &TaskContext) -> Result<(Vec<Subtask>, Vec<String>)> {
        let mut subtasks = Vec::new();
        let mut used_templates = Vec::new();
        
        // Find matching templates
        for (template_id, template) in &self.templates {
            if self.template_matches(template, parsed_task, context) {
                let template_subtasks = self.apply_template(template, parsed_task);
                subtasks.extend(template_subtasks);
                used_templates.push(template_id.clone());
                break; // Use first matching template
            }
        }
        
        // If no template found, use generic template
        if subtasks.is_empty() {
            if let Some(generic_template) = self.templates.get("generic") {
                let template_subtasks = self.apply_template(generic_template, parsed_task);
                subtasks.extend(template_subtasks);
                used_templates.push("generic".to_string());
            }
        }
        
        Ok((subtasks, used_templates))
    }

    /// Check if template matches task
    fn template_matches(&self, template: &TaskTemplate, parsed_task: &ParsedTask, context: &TaskContext) -> bool {
        // Match based on task type
        if template.template_type == parsed_task.task_type {
            return true;
        }
        
        // Match based on keywords in template tags
        for tag in &template.metadata.tags {
            if parsed_task.input.description.to_lowercase().contains(&tag.to_lowercase()) {
                return true;
            }
        }
        
        false
    }

    /// Apply template to create subtasks
    fn apply_template(&self, template: &TaskTemplate, parsed_task: &ParsedTask) -> Vec<Subtask> {
        let mut subtasks = Vec::new();
        
        for (i, template_subtask) in template.subtasks.iter().enumerate() {
            let subtask = Subtask {
                id: Uuid::new_v4(),
                parent_id: parsed_task.id,
                title: template_subtask.title.clone(),
                description: template_subtask.description.clone(),
                task_type: template.template_type.clone(),
                priority: parsed_task.input.priority,
                estimated_effort: (template_subtask.effort_range.0 + template_subtask.effort_range.1) / 2.0,
                estimated_duration: (template_subtask.effort_range.0 + template_subtask.effort_range.1) / 2.0,
                required_skills: template_subtask.required_skills.clone(),
                required_resources: Vec::new(),
                dependencies: template.dependencies.iter()
                    .filter_map(|(from, to)| if *to == i { subtasks.get(*from).map(|s| s.id) } else { None })
                    .collect(),
                acceptance_criteria: template_subtask.acceptance_criteria.clone(),
                level: 1,
                created_at: Utc::now(),
                metadata: SubtaskMetadata {
                    confidence: 0.85,
                    rationale: format!("Applied template: {}", template.name),
                    alternatives: Vec::new(),
                    risk_factors: Vec::new(),
                    success_probability: template.metadata.success_rate,
                },
            };
            
            subtasks.push(subtask);
        }
        
        subtasks
    }

    /// Create default templates
    fn create_default_templates() -> HashMap<String, TaskTemplate> {
        let mut templates = HashMap::new();
        
        // Generic template
        templates.insert("generic".to_string(), TaskTemplate {
            id: "generic".to_string(),
            name: "Generic Task Template".to_string(),
            description: "Generic template for any task".to_string(),
            template_type: TaskType::Generic,
            subtasks: vec![
                SubtaskTemplate {
                    title: "Planning".to_string(),
                    description: "Plan task execution".to_string(),
                    effort_range: (2.0, 4.0),
                    required_skills: vec!["planning".to_string()],
                    acceptance_criteria: vec!["Plan approved".to_string()],
                },
                SubtaskTemplate {
                    title: "Execution".to_string(),
                    description: "Execute planned task".to_string(),
                    effort_range: (4.0, 16.0),
                    required_skills: vec!["execution".to_string()],
                    acceptance_criteria: vec!["Task completed".to_string()],
                },
                SubtaskTemplate {
                    title: "Validation".to_string(),
                    description: "Validate task completion".to_string(),
                    effort_range: (1.0, 4.0),
                    required_skills: vec!["validation".to_string()],
                    acceptance_criteria: vec!["Results validated".to_string()],
                },
            ],
            dependencies: vec![(0, 1), (1, 2)],
            metadata: TemplateMetadata {
                version: "1.0.0".to_string(),
                author: "System".to_string(),
                created_at: Utc::now(),
                usage_count: 0,
                success_rate: 0.8,
                tags: vec!["generic".to_string()],
            },
        });
        
        templates
    }
}

impl Default for DecomposerMetrics {
    fn default() -> Self {
        Self {
            total_decompositions: 0,
            avg_subtask_count: 0.0,
            avg_decomposition_time_ms: 0.0,
            strategy_usage: HashMap::new(),
            strategy_success_rate: HashMap::new(),
            cache_hit_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{TaskInput, TaskParser};
    use crate::analyzer::ContextAnalyzer;
    use crate::neural::NeuralConfig;

    #[tokio::test]
    async fn test_decomposer_creation() {
        let config = NeuralConfig::default();
        let decomposer = TaskDecomposer::new(config).await;
        assert!(decomposer.is_ok());
    }

    #[tokio::test]
    async fn test_neural_decomposition() {
        let config = NeuralConfig::default();
        let decomposer = TaskDecomposer::new(config).await.unwrap();
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Create a REST API using Python and FastAPI")
            .priority(7);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        let result = decomposer.decompose(&parsed_task, &context, DecompositionStrategy::Neural).await.unwrap();
        
        assert!(!result.subtasks.is_empty());
        assert_eq!(result.strategy, DecompositionStrategy::Neural);
        assert!(result.metadata.total_estimated_effort > 0.0);
    }

    #[tokio::test]
    async fn test_rule_based_decomposition() {
        let config = NeuralConfig::default();
        let decomposer = TaskDecomposer::new(config).await.unwrap();
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Implement user authentication system")
            .priority(8);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        let result = decomposer.decompose(&parsed_task, &context, DecompositionStrategy::RuleBased).await.unwrap();
        
        assert!(!result.subtasks.is_empty());
        assert_eq!(result.strategy, DecompositionStrategy::RuleBased);
    }

    #[tokio::test]
    async fn test_template_decomposition() {
        let config = NeuralConfig::default();
        let decomposer = TaskDecomposer::new(config).await.unwrap();
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Generic task for testing")
            .priority(5);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        let result = decomposer.decompose(&parsed_task, &context, DecompositionStrategy::Template).await.unwrap();
        
        assert!(!result.subtasks.is_empty());
        assert_eq!(result.strategy, DecompositionStrategy::Template);
        assert!(!result.metadata.used_templates.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_decomposition() {
        let config = NeuralConfig::default();
        let decomposer = TaskDecomposer::new(config).await.unwrap();
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Complex system implementation")
            .priority(9)
            .complexity(8);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        let result = decomposer.decompose(&parsed_task, &context, DecompositionStrategy::Hybrid).await.unwrap();
        
        assert!(!result.subtasks.is_empty());
        assert_eq!(result.strategy, DecompositionStrategy::Hybrid);
    }

    #[tokio::test]
    async fn test_hierarchical_decomposition() {
        let config = NeuralConfig::default();
        let decomposer = TaskDecomposer::new(config).await.unwrap();
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Large scale system development")
            .priority(8)
            .complexity(9);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        let result = decomposer.decompose(&parsed_task, &context, DecompositionStrategy::Hierarchical).await.unwrap();
        
        assert!(!result.subtasks.is_empty());
        assert_eq!(result.strategy, DecompositionStrategy::Hierarchical);
        
        // Check that subtasks have different levels
        let levels: std::collections::HashSet<_> = result.subtasks.iter().map(|s| s.level).collect();
        assert!(levels.len() > 1 || result.subtasks.iter().all(|s| s.level == 1));
    }

    #[tokio::test]
    async fn test_caching() {
        let config = NeuralConfig::default();
        let decomposer = TaskDecomposer::new(config).await.unwrap();
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Test caching functionality")
            .priority(5);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        // First decomposition
        let result1 = decomposer.decompose(&parsed_task, &context, DecompositionStrategy::Neural).await.unwrap();
        
        // Second decomposition should use cache
        let result2 = decomposer.decompose(&parsed_task, &context, DecompositionStrategy::Neural).await.unwrap();
        
        assert_eq!(result1.task_id, result2.task_id);
        assert_eq!(result1.subtasks.len(), result2.subtasks.len());
    }

    #[test]
    fn test_rule_evaluation() {
        let engine = RuleEngine::new();
        
        // Test rule condition evaluation
        let condition = RuleCondition {
            condition_type: ConditionType::Priority,
            value: ConditionValue::Integer(8),
            operator: ConditionOperator::GreaterThan,
        };
        
        let input = TaskInput::new()
            .description("High priority task")
            .priority(9);
        
        let parsed_task = crate::parser::TaskParser::new().parse(input).unwrap();
        let context = crate::analyzer::TaskContext {
            task_id: parsed_task.id,
            semantic_analysis: crate::analyzer::SemanticAnalysis {
                domain: "test".to_string(),
                confidence: 0.8,
                key_concepts: Vec::new(),
                workflow_patterns: Vec::new(),
                tech_stack: crate::analyzer::TechStackAnalysis {
                    primary_technologies: Vec::new(),
                    supporting_technologies: Vec::new(),
                    compatibility_score: 0.8,
                    maturity_assessment: crate::analyzer::TechMaturity::Mature,
                    learning_curve: crate::analyzer::LearningCurve::Moderate,
                },
                complexity_analysis: crate::analyzer::ComplexityAnalysis {
                    overall_score: 0.5,
                    technical_complexity: 0.5,
                    integration_complexity: 0.5,
                    operational_complexity: 0.5,
                    maintenance_complexity: 0.5,
                    factors: Vec::new(),
                },
            },
            metadata: crate::analyzer::ContextMetadata {
                analyzed_at: Utc::now(),
                analysis_duration_ms: 100,
                analyzer_version: "1.0.0".to_string(),
                confidence: 0.8,
                cache_hit: false,
            },
            relationships: Vec::new(),
            challenges: Vec::new(),
            recommendations: Vec::new(),
        };
        
        assert!(engine.evaluate_condition(&condition, &parsed_task, &context));
    }
}