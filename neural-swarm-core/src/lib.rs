//! Splinter - Neural Task Decomposition Engine
//!
//! A high-performance task decomposition engine that uses neural networks
//! to intelligently break down complex tasks into manageable subtasks
//! for distributed swarm execution.
//!
//! ## Features
//!
//! - **Neural Decomposition**: BERT/GPT-based task analysis and breakdown
//! - **Graph Construction**: DAG-based task dependency modeling
//! - **Swarm Integration**: Seamless integration with neural-comm and neuroplex
//! - **Adaptive Learning**: Reinforcement learning for strategy optimization
//! - **Performance Optimization**: SIMD-accelerated neural operations
//! - **Python FFI**: Async Python integration for ML workflows
//!
//! ## Architecture
//!
//! The splinter engine consists of several key components:
//!
//! - **Parser**: Multi-format input parsing and validation
//! - **Analyzer**: Context analysis and semantic understanding
//! - **Decomposer**: Neural-powered task decomposition
//! - **Graph**: Task dependency graph construction
//! - **Dispatcher**: Swarm task distribution and coordination
//! - **Neural**: Neural network components and learning
//!
//! ## Quick Start
//!
//! ```rust
//! use splinter::{SplinterEngine, TaskInput, DecompositionStrategy};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let engine = SplinterEngine::new().await?;
//!     
//!     let task = TaskInput::new()
//!         .description("Build a REST API with authentication")
//!         .priority(5)
//!         .deadline(chrono::Utc::now() + chrono::Duration::hours(24));
//!     
//!     let decomposition = engine
//!         .decompose(task, DecompositionStrategy::Neural)
//!         .await?;
//!     
//!     for subtask in decomposition.subtasks {
//!         println!("Subtask: {}", subtask.description);
//!         println!("Dependencies: {:?}", subtask.dependencies);
//!     }
//!     
//!     Ok(())
//! }
//! ```

pub mod parser;
pub mod analyzer;
pub mod decomposer;
pub mod neural;
pub mod graph;
pub mod dispatcher;
pub mod error;

#[cfg(feature = "python-ffi")]
pub mod ffi;

// Re-export main types
pub use parser::{TaskInput, InputFormat, ParsedTask};
pub use analyzer::{ContextAnalyzer, TaskContext, SemanticAnalysis};
pub use decomposer::{TaskDecomposer, DecompositionStrategy, DecompositionResult};
pub use neural::{NeuralBackend, TransformerModel, ReinforcementLearner};
pub use graph::{TaskGraph, TaskNode, DependencyEdge};
pub use dispatcher::{SwarmDispatcher, TaskAssignment, ExecutionPlan};
pub use error::{SplinterError, Result};

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use std::sync::Arc;

/// Task identifier
pub type TaskId = Uuid;

/// Agent identifier
pub type AgentId = Uuid;

/// Priority level for tasks
pub type Priority = u8;

/// Main splinter engine for task decomposition
#[derive(Debug)]
pub struct SplinterEngine {
    /// Task parser for input processing
    parser: parser::TaskParser,
    /// Context analyzer for semantic understanding
    analyzer: analyzer::ContextAnalyzer,
    /// Task decomposer with neural backends
    decomposer: decomposer::TaskDecomposer,
    /// Task graph builder
    graph_builder: graph::GraphBuilder,
    /// Swarm dispatcher for task distribution
    dispatcher: dispatcher::SwarmDispatcher,
    /// Neural backend for AI operations
    neural: neural::NeuralBackend,
    /// Configuration settings
    config: SplinterConfig,
}

/// Configuration for the splinter engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplinterConfig {
    /// Maximum task depth for decomposition
    pub max_decomposition_depth: usize,
    /// Neural model configuration
    pub neural_config: neural::NeuralConfig,
    /// Graph construction settings
    pub graph_config: graph::GraphConfig,
    /// Swarm coordination settings
    pub swarm_config: dispatcher::SwarmConfig,
    /// Performance optimization settings
    pub performance_config: PerformanceConfig,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Neural model cache size
    pub model_cache_size: usize,
    /// Batch size for neural operations
    pub batch_size: usize,
    /// Async task pool size
    pub task_pool_size: usize,
}

impl Default for SplinterConfig {
    fn default() -> Self {
        Self {
            max_decomposition_depth: 10,
            neural_config: neural::NeuralConfig::default(),
            graph_config: graph::GraphConfig::default(),
            swarm_config: dispatcher::SwarmConfig::default(),
            performance_config: PerformanceConfig {
                enable_simd: true,
                model_cache_size: 1024,
                batch_size: 32,
                task_pool_size: 16,
            },
        }
    }
}

impl SplinterEngine {
    /// Create a new splinter engine with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(SplinterConfig::default()).await
    }

    /// Create a new splinter engine with custom configuration
    pub async fn with_config(config: SplinterConfig) -> Result<Self> {
        let parser = parser::TaskParser::new();
        let analyzer = analyzer::ContextAnalyzer::new().await?;
        let decomposer = decomposer::TaskDecomposer::new(config.neural_config.clone()).await?;
        let graph_builder = graph::GraphBuilder::new(config.graph_config.clone());
        let dispatcher = dispatcher::SwarmDispatcher::new(config.swarm_config.clone()).await?;
        let neural = neural::NeuralBackend::new(config.neural_config.clone()).await?;

        Ok(Self {
            parser,
            analyzer,
            decomposer,
            graph_builder,
            dispatcher,
            neural,
            config,
        })
    }

    /// Decompose a task using the specified strategy
    pub async fn decompose(
        &self,
        input: TaskInput,
        strategy: DecompositionStrategy,
    ) -> Result<DecompositionResult> {
        tracing::info!("Starting task decomposition for: {}", input.description);

        // 1. Parse the input task
        let parsed_task = self.parser.parse(input).await?;
        tracing::debug!("Task parsed successfully");

        // 2. Analyze context and semantics
        let context = self.analyzer.analyze(&parsed_task).await?;
        tracing::debug!("Context analysis complete");

        // 3. Decompose the task using neural networks
        let decomposition = self.decomposer.decompose(&parsed_task, &context, strategy).await?;
        tracing::debug!("Neural decomposition complete with {} subtasks", decomposition.subtasks.len());

        // 4. Build task dependency graph
        let graph = self.graph_builder.build(&decomposition).await?;
        tracing::debug!("Task graph constructed with {} nodes", graph.node_count());

        // 5. Create execution plan
        let execution_plan = self.dispatcher.create_plan(&graph, &context).await?;
        tracing::debug!("Execution plan created with {} phases", execution_plan.phases.len());

        Ok(DecompositionResult {
            task_id: decomposition.task_id,
            subtasks: decomposition.subtasks,
            graph,
            execution_plan,
            metadata: decomposition.metadata,
        })
    }

    /// Get current engine statistics
    pub async fn stats(&self) -> SplinterStats {
        SplinterStats {
            total_decompositions: self.decomposer.total_decompositions().await,
            avg_subtask_count: self.decomposer.avg_subtask_count().await,
            neural_model_accuracy: self.neural.model_accuracy().await,
            cache_hit_rate: self.neural.cache_hit_rate().await,
        }
    }

    /// Update engine configuration
    pub async fn update_config(&mut self, config: SplinterConfig) -> Result<()> {
        self.config = config.clone();
        self.decomposer.update_config(config.neural_config).await?;
        self.graph_builder.update_config(config.graph_config).await?;
        self.dispatcher.update_config(config.swarm_config).await?;
        Ok(())
    }

    /// Train the neural models with feedback
    pub async fn train_from_feedback(&mut self, feedback: TrainingFeedback) -> Result<()> {
        self.neural.train_from_feedback(feedback).await?;
        self.decomposer.retrain().await?;
        Ok(())
    }
}

/// Statistics for the splinter engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplinterStats {
    /// Total number of decompositions performed
    pub total_decompositions: u64,
    /// Average number of subtasks per decomposition
    pub avg_subtask_count: f64,
    /// Neural model accuracy score
    pub neural_model_accuracy: f64,
    /// Cache hit rate for neural operations
    pub cache_hit_rate: f64,
}

/// Training feedback for neural model improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFeedback {
    /// Task that was decomposed
    pub task_id: TaskId,
    /// Actual execution results
    pub execution_results: Vec<ExecutionResult>,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// User feedback
    pub user_feedback: Option<String>,
}

/// Result of task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Subtask that was executed
    pub subtask_id: TaskId,
    /// Whether execution was successful
    pub success: bool,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Error message if execution failed
    pub error_message: Option<String>,
}

/// Initialize tracing for the splinter engine
pub fn init_tracing() {
    use tracing_subscriber::prelude::*;
    
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = SplinterEngine::new().await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_basic_decomposition() {
        let engine = SplinterEngine::new().await.unwrap();
        
        let task = TaskInput::new()
            .description("Create a simple web server")
            .priority(5);
        
        let result = engine.decompose(task, DecompositionStrategy::Neural).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_default() {
        let config = SplinterConfig::default();
        assert_eq!(config.max_decomposition_depth, 10);
        assert!(config.performance_config.enable_simd);
    }
}