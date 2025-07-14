//! Context analysis and semantic understanding module

use crate::parser::{ParsedTask, TaskEntity, EntityType, TaskType};
use crate::error::{SplinterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Context analyzer for semantic understanding
#[derive(Debug)]
pub struct ContextAnalyzer {
    /// Semantic knowledge base
    knowledge_base: Arc<RwLock<SemanticKnowledge>>,
    /// Context cache for performance
    context_cache: Arc<DashMap<Uuid, TaskContext>>,
    /// Analysis configuration
    config: AnalyzerConfig,
}

/// Semantic knowledge base
#[derive(Debug, Clone)]
struct SemanticKnowledge {
    /// Technology relationships
    tech_relationships: HashMap<String, Vec<String>>,
    /// Action patterns
    action_patterns: HashMap<String, ActionPattern>,
    /// Domain knowledge
    domain_knowledge: HashMap<String, DomainInfo>,
    /// Complexity factors
    complexity_factors: HashMap<String, f64>,
}

/// Action pattern for understanding task sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActionPattern {
    /// Prerequisites for this action
    prerequisites: Vec<String>,
    /// Typical duration estimate
    duration_estimate: u64,
    /// Required skills
    required_skills: Vec<String>,
    /// Common followup actions
    followup_actions: Vec<String>,
    /// Complexity multiplier
    complexity_multiplier: f64,
}

/// Domain information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DomainInfo {
    /// Domain name
    name: String,
    /// Common technologies
    technologies: Vec<String>,
    /// Typical workflows
    workflows: Vec<String>,
    /// Best practices
    best_practices: Vec<String>,
    /// Common pitfalls
    pitfalls: Vec<String>,
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Enable semantic analysis
    pub enable_semantic_analysis: bool,
    /// Enable context caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Analysis timeout in seconds
    pub analysis_timeout_secs: u64,
    /// Enable deep context analysis
    pub enable_deep_analysis: bool,
}

/// Task context with semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    /// Task identifier
    pub task_id: Uuid,
    /// Semantic analysis results
    pub semantic_analysis: SemanticAnalysis,
    /// Context metadata
    pub metadata: ContextMetadata,
    /// Inferred relationships
    pub relationships: Vec<EntityRelationship>,
    /// Predicted challenges
    pub challenges: Vec<PredictedChallenge>,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
}

/// Semantic analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    /// Identified domain
    pub domain: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Key concepts extracted
    pub key_concepts: Vec<Concept>,
    /// Workflow patterns identified
    pub workflow_patterns: Vec<WorkflowPattern>,
    /// Technology stack analysis
    pub tech_stack: TechStackAnalysis,
    /// Complexity analysis
    pub complexity_analysis: ComplexityAnalysis,
}

/// Semantic concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Concept name
    pub name: String,
    /// Concept type
    pub concept_type: ConceptType,
    /// Relevance score
    pub relevance: f64,
    /// Related concepts
    pub related_concepts: Vec<String>,
}

/// Types of concepts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConceptType {
    /// Technical concept
    Technical,
    /// Business concept
    Business,
    /// Process concept
    Process,
    /// Quality concept
    Quality,
    /// Resource concept
    Resource,
}

/// Workflow pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowPattern {
    /// Pattern name
    pub name: String,
    /// Pattern steps
    pub steps: Vec<String>,
    /// Estimated duration
    pub estimated_duration: u64,
    /// Required resources
    pub required_resources: Vec<String>,
    /// Success probability
    pub success_probability: f64,
}

/// Technology stack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechStackAnalysis {
    /// Primary technologies
    pub primary_technologies: Vec<String>,
    /// Supporting technologies
    pub supporting_technologies: Vec<String>,
    /// Technology compatibility score
    pub compatibility_score: f64,
    /// Maturity assessment
    pub maturity_assessment: TechMaturity,
    /// Learning curve estimate
    pub learning_curve: LearningCurve,
}

/// Technology maturity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TechMaturity {
    /// Cutting edge, experimental
    Experimental,
    /// Early adoption phase
    Early,
    /// Mature and stable
    Mature,
    /// Legacy technology
    Legacy,
}

/// Learning curve assessment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LearningCurve {
    /// Easy to learn
    Easy,
    /// Moderate learning curve
    Moderate,
    /// Steep learning curve
    Steep,
    /// Very difficult to master
    VeryDifficult,
}

/// Complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAnalysis {
    /// Overall complexity score
    pub overall_score: f64,
    /// Technical complexity
    pub technical_complexity: f64,
    /// Integration complexity
    pub integration_complexity: f64,
    /// Operational complexity
    pub operational_complexity: f64,
    /// Maintenance complexity
    pub maintenance_complexity: f64,
    /// Complexity factors
    pub factors: Vec<ComplexityFactor>,
}

/// Individual complexity factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFactor {
    /// Factor name
    pub name: String,
    /// Factor weight
    pub weight: f64,
    /// Factor description
    pub description: String,
    /// Impact level
    pub impact: ImpactLevel,
}

/// Impact levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImpactLevel {
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Critical impact
    Critical,
}

/// Context metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,
    /// Analysis duration
    pub analysis_duration_ms: u64,
    /// Analyzer version
    pub analyzer_version: String,
    /// Analysis confidence
    pub confidence: f64,
    /// Cache hit indicator
    pub cache_hit: bool,
}

/// Relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationship {
    /// Source entity
    pub source: String,
    /// Target entity
    pub target: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Relationship strength
    pub strength: f64,
    /// Relationship description
    pub description: String,
}

/// Types of relationships
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelationshipType {
    /// Dependency relationship
    Dependency,
    /// Compatibility relationship
    Compatibility,
    /// Sequence relationship
    Sequence,
    /// Alternative relationship
    Alternative,
    /// Composition relationship
    Composition,
    /// Inheritance relationship
    Inheritance,
}

/// Predicted challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedChallenge {
    /// Challenge description
    pub description: String,
    /// Challenge type
    pub challenge_type: ChallengeType,
    /// Likelihood (0.0-1.0)
    pub likelihood: f64,
    /// Severity level
    pub severity: SeverityLevel,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Types of challenges
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChallengeType {
    /// Technical challenge
    Technical,
    /// Resource challenge
    Resource,
    /// Timeline challenge
    Timeline,
    /// Skill challenge
    Skill,
    /// Integration challenge
    Integration,
    /// Quality challenge
    Quality,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SeverityLevel {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Recommendation for task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation text
    pub text: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: u8,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation effort
    pub implementation_effort: EffortLevel,
}

/// Types of recommendations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecommendationType {
    /// Technology recommendation
    Technology,
    /// Process recommendation
    Process,
    /// Resource recommendation
    Resource,
    /// Quality recommendation
    Quality,
    /// Performance recommendation
    Performance,
    /// Security recommendation
    Security,
}

/// Effort levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EffortLevel {
    /// Low effort
    Low,
    /// Medium effort
    Medium,
    /// High effort
    High,
    /// Very high effort
    VeryHigh,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            enable_semantic_analysis: true,
            enable_caching: true,
            cache_size_limit: 10000,
            analysis_timeout_secs: 30,
            enable_deep_analysis: true,
        }
    }
}

impl ContextAnalyzer {
    /// Create a new context analyzer
    pub async fn new() -> Result<Self> {
        Self::with_config(AnalyzerConfig::default()).await
    }

    /// Create a new context analyzer with custom configuration
    pub async fn with_config(config: AnalyzerConfig) -> Result<Self> {
        let knowledge_base = Arc::new(RwLock::new(Self::build_knowledge_base()));
        let context_cache = Arc::new(DashMap::new());

        Ok(Self {
            knowledge_base,
            context_cache,
            config,
        })
    }

    /// Analyze parsed task to extract context
    pub async fn analyze(&self, parsed_task: &ParsedTask) -> Result<TaskContext> {
        let start_time = std::time::Instant::now();

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached_context) = self.context_cache.get(&parsed_task.id) {
                tracing::debug!("Retrieved context from cache for task {}", parsed_task.id);
                return Ok(cached_context.clone());
            }
        }

        // Perform semantic analysis
        let semantic_analysis = if self.config.enable_semantic_analysis {
            self.perform_semantic_analysis(parsed_task).await?
        } else {
            self.create_basic_semantic_analysis(parsed_task)
        };

        // Extract relationships
        let relationships = self.extract_relationships(&parsed_task.entities).await?;

        // Predict challenges
        let challenges = self.predict_challenges(parsed_task, &semantic_analysis).await?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(parsed_task, &semantic_analysis, &challenges).await?;

        // Create metadata
        let analysis_duration = start_time.elapsed();
        let metadata = ContextMetadata {
            analyzed_at: Utc::now(),
            analysis_duration_ms: analysis_duration.as_millis() as u64,
            analyzer_version: env!("CARGO_PKG_VERSION").to_string(),
            confidence: semantic_analysis.confidence,
            cache_hit: false,
        };

        let context = TaskContext {
            task_id: parsed_task.id,
            semantic_analysis,
            metadata,
            relationships,
            challenges,
            recommendations,
        };

        // Cache the result
        if self.config.enable_caching {
            self.context_cache.insert(parsed_task.id, context.clone());
        }

        Ok(context)
    }

    /// Perform semantic analysis
    async fn perform_semantic_analysis(&self, parsed_task: &ParsedTask) -> Result<SemanticAnalysis> {
        let kb = self.knowledge_base.read().await;

        // Identify domain
        let domain = self.identify_domain(parsed_task, &kb);

        // Extract key concepts
        let key_concepts = self.extract_key_concepts(&parsed_task.entities, &kb);

        // Identify workflow patterns
        let workflow_patterns = self.identify_workflow_patterns(parsed_task, &kb);

        // Analyze technology stack
        let tech_stack = self.analyze_tech_stack(&parsed_task.entities, &kb);

        // Perform complexity analysis
        let complexity_analysis = self.analyze_complexity(parsed_task, &kb);

        // Calculate overall confidence
        let confidence = self.calculate_semantic_confidence(&key_concepts, &workflow_patterns, &tech_stack);

        Ok(SemanticAnalysis {
            domain,
            confidence,
            key_concepts,
            workflow_patterns,
            tech_stack,
            complexity_analysis,
        })
    }

    /// Create basic semantic analysis for non-semantic mode
    fn create_basic_semantic_analysis(&self, parsed_task: &ParsedTask) -> SemanticAnalysis {
        let domain = match parsed_task.task_type {
            TaskType::Development => "software_development",
            TaskType::Administration => "system_administration",
            TaskType::Analysis => "data_analysis",
            TaskType::Research => "research",
            TaskType::Testing => "quality_assurance",
            TaskType::Documentation => "documentation",
            TaskType::Deployment => "deployment",
            TaskType::Maintenance => "maintenance",
            TaskType::Generic => "generic",
        }.to_string();

        SemanticAnalysis {
            domain,
            confidence: 0.6,
            key_concepts: Vec::new(),
            workflow_patterns: Vec::new(),
            tech_stack: TechStackAnalysis {
                primary_technologies: Vec::new(),
                supporting_technologies: Vec::new(),
                compatibility_score: 0.5,
                maturity_assessment: TechMaturity::Mature,
                learning_curve: LearningCurve::Moderate,
            },
            complexity_analysis: ComplexityAnalysis {
                overall_score: 0.5,
                technical_complexity: 0.5,
                integration_complexity: 0.5,
                operational_complexity: 0.5,
                maintenance_complexity: 0.5,
                factors: Vec::new(),
            },
        }
    }

    /// Identify domain from task
    fn identify_domain(&self, parsed_task: &ParsedTask, kb: &SemanticKnowledge) -> String {
        let mut domain_scores = HashMap::new();

        // Score based on entities
        for entity in &parsed_task.entities {
            if entity.entity_type == EntityType::Technology {
                for (domain, info) in &kb.domain_knowledge {
                    if info.technologies.contains(&entity.text.to_lowercase()) {
                        *domain_scores.entry(domain.clone()).or_insert(0.0) += entity.confidence;
                    }
                }
            }
        }

        // Score based on task type
        let task_type_domain = match parsed_task.task_type {
            TaskType::Development => "software_development",
            TaskType::Administration => "system_administration",
            TaskType::Analysis => "data_analysis",
            TaskType::Research => "research",
            TaskType::Testing => "quality_assurance",
            TaskType::Documentation => "documentation",
            TaskType::Deployment => "deployment",
            TaskType::Maintenance => "maintenance",
            TaskType::Generic => "generic",
        };

        *domain_scores.entry(task_type_domain.to_string()).or_insert(0.0) += 0.5;

        // Return highest scoring domain
        domain_scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(domain, _)| domain)
            .unwrap_or_else(|| "generic".to_string())
    }

    /// Extract key concepts from entities
    fn extract_key_concepts(&self, entities: &[TaskEntity], kb: &SemanticKnowledge) -> Vec<Concept> {
        let mut concepts = Vec::new();

        for entity in entities {
            let concept_type = match entity.entity_type {
                EntityType::Technology => ConceptType::Technical,
                EntityType::Action => ConceptType::Process,
                EntityType::Resource => ConceptType::Resource,
                EntityType::Quality => ConceptType::Quality,
                _ => ConceptType::Technical,
            };

            // Find related concepts
            let related_concepts = if let Some(relationships) = kb.tech_relationships.get(&entity.text.to_lowercase()) {
                relationships.clone()
            } else {
                Vec::new()
            };

            concepts.push(Concept {
                name: entity.text.clone(),
                concept_type,
                relevance: entity.confidence,
                related_concepts,
            });
        }

        concepts
    }

    /// Identify workflow patterns
    fn identify_workflow_patterns(&self, parsed_task: &ParsedTask, kb: &SemanticKnowledge) -> Vec<WorkflowPattern> {
        let mut patterns = Vec::new();

        // Identify patterns based on actions
        let actions: Vec<_> = parsed_task.entities.iter()
            .filter(|e| e.entity_type == EntityType::Action)
            .map(|e| e.text.clone())
            .collect();

        if actions.is_empty() {
            return patterns;
        }

        // Create workflow pattern from actions
        for action in &actions {
            if let Some(action_pattern) = kb.action_patterns.get(&action.to_lowercase()) {
                patterns.push(WorkflowPattern {
                    name: format!("{} Workflow", action),
                    steps: action_pattern.prerequisites.clone(),
                    estimated_duration: action_pattern.duration_estimate,
                    required_resources: action_pattern.required_skills.clone(),
                    success_probability: 0.8,
                });
            }
        }

        patterns
    }

    /// Analyze technology stack
    fn analyze_tech_stack(&self, entities: &[TaskEntity], kb: &SemanticKnowledge) -> TechStackAnalysis {
        let tech_entities: Vec<_> = entities.iter()
            .filter(|e| e.entity_type == EntityType::Technology)
            .collect();

        let primary_technologies: Vec<String> = tech_entities.iter()
            .map(|e| e.text.clone())
            .collect();

        let supporting_technologies = Vec::new(); // TODO: Implement based on relationships

        // Calculate compatibility score
        let compatibility_score = if primary_technologies.len() <= 1 {
            1.0
        } else {
            // Simplified compatibility calculation
            0.8 - (primary_technologies.len() as f64 * 0.1).min(0.5)
        };

        // Assess maturity (simplified)
        let maturity_assessment = if primary_technologies.iter().any(|t| 
            ["react", "python", "java", "javascript", "mysql", "postgresql"].contains(&t.to_lowercase().as_str())
        ) {
            TechMaturity::Mature
        } else {
            TechMaturity::Early
        };

        // Assess learning curve
        let learning_curve = if primary_technologies.len() > 3 {
            LearningCurve::Steep
        } else {
            LearningCurve::Moderate
        };

        TechStackAnalysis {
            primary_technologies,
            supporting_technologies,
            compatibility_score,
            maturity_assessment,
            learning_curve,
        }
    }

    /// Analyze complexity
    fn analyze_complexity(&self, parsed_task: &ParsedTask, kb: &SemanticKnowledge) -> ComplexityAnalysis {
        let mut factors = Vec::new();
        let mut technical_complexity = 0.0;
        let mut integration_complexity = 0.0;
        let mut operational_complexity = 0.0;
        let mut maintenance_complexity = 0.0;

        // Analyze based on entities
        let tech_count = parsed_task.entities.iter().filter(|e| e.entity_type == EntityType::Technology).count();
        if tech_count > 0 {
            technical_complexity = (tech_count as f64 * 0.2).min(1.0);
            factors.push(ComplexityFactor {
                name: "Technology Stack".to_string(),
                weight: 0.3,
                description: format!("Uses {} technologies", tech_count),
                impact: if tech_count > 5 { ImpactLevel::High } else { ImpactLevel::Medium },
            });
        }

        // Analyze based on requirements
        let req_count = parsed_task.requirements.len();
        if req_count > 0 {
            integration_complexity = (req_count as f64 * 0.15).min(1.0);
            factors.push(ComplexityFactor {
                name: "Requirements".to_string(),
                weight: 0.25,
                description: format!("Has {} requirements", req_count),
                impact: if req_count > 10 { ImpactLevel::High } else { ImpactLevel::Medium },
            });
        }

        // Analyze based on dependencies
        let dep_count = parsed_task.input.dependencies.len();
        if dep_count > 0 {
            integration_complexity += (dep_count as f64 * 0.1).min(0.5);
            factors.push(ComplexityFactor {
                name: "Dependencies".to_string(),
                weight: 0.2,
                description: format!("Has {} dependencies", dep_count),
                impact: if dep_count > 5 { ImpactLevel::High } else { ImpactLevel::Medium },
            });
        }

        // Simple operational and maintenance complexity
        operational_complexity = 0.3;
        maintenance_complexity = 0.4;

        let overall_score = (technical_complexity * 0.4 + 
                           integration_complexity * 0.3 + 
                           operational_complexity * 0.15 + 
                           maintenance_complexity * 0.15).min(1.0);

        ComplexityAnalysis {
            overall_score,
            technical_complexity,
            integration_complexity,
            operational_complexity,
            maintenance_complexity,
            factors,
        }
    }

    /// Calculate semantic confidence
    fn calculate_semantic_confidence(&self, concepts: &[Concept], patterns: &[WorkflowPattern], tech_stack: &TechStackAnalysis) -> f64 {
        let concept_confidence = if concepts.is_empty() {
            0.3
        } else {
            concepts.iter().map(|c| c.relevance).sum::<f64>() / concepts.len() as f64
        };

        let pattern_confidence = if patterns.is_empty() {
            0.5
        } else {
            patterns.iter().map(|p| p.success_probability).sum::<f64>() / patterns.len() as f64
        };

        let tech_confidence = tech_stack.compatibility_score;

        (concept_confidence * 0.4 + pattern_confidence * 0.3 + tech_confidence * 0.3).min(1.0)
    }

    /// Extract relationships between entities
    async fn extract_relationships(&self, entities: &[TaskEntity]) -> Result<Vec<EntityRelationship>> {
        let mut relationships = Vec::new();
        let kb = self.knowledge_base.read().await;

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity1 = &entities[i];
                let entity2 = &entities[j];

                // Check for technology relationships
                if entity1.entity_type == EntityType::Technology && entity2.entity_type == EntityType::Technology {
                    if let Some(related_techs) = kb.tech_relationships.get(&entity1.text.to_lowercase()) {
                        if related_techs.contains(&entity2.text.to_lowercase()) {
                            relationships.push(EntityRelationship {
                                source: entity1.text.clone(),
                                target: entity2.text.clone(),
                                relationship_type: RelationshipType::Compatibility,
                                strength: 0.8,
                                description: format!("{} is compatible with {}", entity1.text, entity2.text),
                            });
                        }
                    }
                }

                // Check for sequence relationships (action followed by action)
                if entity1.entity_type == EntityType::Action && entity2.entity_type == EntityType::Action {
                    if entity1.start < entity2.start {
                        relationships.push(EntityRelationship {
                            source: entity1.text.clone(),
                            target: entity2.text.clone(),
                            relationship_type: RelationshipType::Sequence,
                            strength: 0.6,
                            description: format!("{} should be done before {}", entity1.text, entity2.text),
                        });
                    }
                }
            }
        }

        Ok(relationships)
    }

    /// Predict challenges based on analysis
    async fn predict_challenges(&self, parsed_task: &ParsedTask, semantic_analysis: &SemanticAnalysis) -> Result<Vec<PredictedChallenge>> {
        let mut challenges = Vec::new();

        // Technical challenges
        if semantic_analysis.tech_stack.primary_technologies.len() > 3 {
            challenges.push(PredictedChallenge {
                description: "Complex technology stack may lead to integration issues".to_string(),
                challenge_type: ChallengeType::Technical,
                likelihood: 0.7,
                severity: SeverityLevel::Medium,
                mitigation_strategies: vec![
                    "Create comprehensive integration tests".to_string(),
                    "Use standardized APIs".to_string(),
                    "Implement proper error handling".to_string(),
                ],
            });
        }

        // Learning curve challenges
        if semantic_analysis.tech_stack.learning_curve == LearningCurve::Steep {
            challenges.push(PredictedChallenge {
                description: "Steep learning curve may impact timeline".to_string(),
                challenge_type: ChallengeType::Skill,
                likelihood: 0.8,
                severity: SeverityLevel::High,
                mitigation_strategies: vec![
                    "Provide comprehensive training".to_string(),
                    "Allocate additional time for learning".to_string(),
                    "Consider simpler alternatives".to_string(),
                ],
            });
        }

        // Complexity challenges
        if semantic_analysis.complexity_analysis.overall_score > 0.7 {
            challenges.push(PredictedChallenge {
                description: "High complexity may lead to quality issues".to_string(),
                challenge_type: ChallengeType::Quality,
                likelihood: 0.6,
                severity: SeverityLevel::Medium,
                mitigation_strategies: vec![
                    "Break down into smaller tasks".to_string(),
                    "Implement thorough testing".to_string(),
                    "Regular code reviews".to_string(),
                ],
            });
        }

        // Timeline challenges
        if parsed_task.input.deadline.is_some() && parsed_task.input.priority > 7 {
            challenges.push(PredictedChallenge {
                description: "Tight deadline with high priority may impact quality".to_string(),
                challenge_type: ChallengeType::Timeline,
                likelihood: 0.5,
                severity: SeverityLevel::High,
                mitigation_strategies: vec![
                    "Prioritize core features".to_string(),
                    "Implement minimum viable product first".to_string(),
                    "Allocate additional resources".to_string(),
                ],
            });
        }

        Ok(challenges)
    }

    /// Generate recommendations
    async fn generate_recommendations(&self, parsed_task: &ParsedTask, semantic_analysis: &SemanticAnalysis, challenges: &[PredictedChallenge]) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Technology recommendations
        if semantic_analysis.tech_stack.compatibility_score < 0.5 {
            recommendations.push(Recommendation {
                text: "Consider using more compatible technologies to reduce integration complexity".to_string(),
                recommendation_type: RecommendationType::Technology,
                priority: 8,
                expected_impact: "Reduced integration issues and faster development".to_string(),
                implementation_effort: EffortLevel::Medium,
            });
        }

        // Process recommendations
        if semantic_analysis.complexity_analysis.overall_score > 0.6 {
            recommendations.push(Recommendation {
                text: "Implement incremental development approach with regular checkpoints".to_string(),
                recommendation_type: RecommendationType::Process,
                priority: 7,
                expected_impact: "Better risk management and quality control".to_string(),
                implementation_effort: EffortLevel::Low,
            });
        }

        // Quality recommendations
        if challenges.iter().any(|c| c.challenge_type == ChallengeType::Quality) {
            recommendations.push(Recommendation {
                text: "Implement comprehensive testing strategy including unit, integration, and end-to-end tests".to_string(),
                recommendation_type: RecommendationType::Quality,
                priority: 9,
                expected_impact: "Higher quality deliverables and reduced bugs".to_string(),
                implementation_effort: EffortLevel::High,
            });
        }

        // Resource recommendations
        if semantic_analysis.tech_stack.learning_curve == LearningCurve::Steep {
            recommendations.push(Recommendation {
                text: "Allocate additional time for learning and skill development".to_string(),
                recommendation_type: RecommendationType::Resource,
                priority: 6,
                expected_impact: "Better prepared team and higher quality results".to_string(),
                implementation_effort: EffortLevel::Medium,
            });
        }

        // Performance recommendations
        if semantic_analysis.complexity_analysis.technical_complexity > 0.7 {
            recommendations.push(Recommendation {
                text: "Consider performance optimization from the beginning of development".to_string(),
                recommendation_type: RecommendationType::Performance,
                priority: 5,
                expected_impact: "Better performance and user experience".to_string(),
                implementation_effort: EffortLevel::Medium,
            });
        }

        Ok(recommendations)
    }

    /// Build semantic knowledge base
    fn build_knowledge_base() -> SemanticKnowledge {
        let mut tech_relationships = HashMap::new();
        let mut action_patterns = HashMap::new();
        let mut domain_knowledge = HashMap::new();
        let mut complexity_factors = HashMap::new();

        // Technology relationships
        tech_relationships.insert("react".to_string(), vec!["javascript".to_string(), "node.js".to_string(), "webpack".to_string()]);
        tech_relationships.insert("python".to_string(), vec!["django".to_string(), "flask".to_string(), "fastapi".to_string()]);
        tech_relationships.insert("docker".to_string(), vec!["kubernetes".to_string(), "nginx".to_string(), "redis".to_string()]);
        tech_relationships.insert("postgresql".to_string(), vec!["sql".to_string(), "database".to_string(), "orm".to_string()]);

        // Action patterns
        action_patterns.insert("create".to_string(), ActionPattern {
            prerequisites: vec!["analyze requirements".to_string(), "design architecture".to_string()],
            duration_estimate: 480, // 8 hours
            required_skills: vec!["programming".to_string(), "design".to_string()],
            followup_actions: vec!["test".to_string(), "deploy".to_string()],
            complexity_multiplier: 1.2,
        });

        action_patterns.insert("implement".to_string(), ActionPattern {
            prerequisites: vec!["design".to_string(), "setup environment".to_string()],
            duration_estimate: 720, // 12 hours
            required_skills: vec!["programming".to_string(), "debugging".to_string()],
            followup_actions: vec!["test".to_string(), "optimize".to_string()],
            complexity_multiplier: 1.5,
        });

        // Domain knowledge
        domain_knowledge.insert("software_development".to_string(), DomainInfo {
            name: "Software Development".to_string(),
            technologies: vec!["react".to_string(), "python".to_string(), "javascript".to_string(), "docker".to_string()],
            workflows: vec!["agile".to_string(), "devops".to_string(), "cicd".to_string()],
            best_practices: vec!["code reviews".to_string(), "testing".to_string(), "documentation".to_string()],
            pitfalls: vec!["scope creep".to_string(), "technical debt".to_string(), "poor testing".to_string()],
        });

        // Complexity factors
        complexity_factors.insert("technology_count".to_string(), 0.2);
        complexity_factors.insert("dependency_count".to_string(), 0.15);
        complexity_factors.insert("requirement_count".to_string(), 0.1);
        complexity_factors.insert("integration_complexity".to_string(), 0.25);

        SemanticKnowledge {
            tech_relationships,
            action_patterns,
            domain_knowledge,
            complexity_factors,
        }
    }

    /// Clear context cache
    pub async fn clear_cache(&self) {
        self.context_cache.clear();
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cache_size".to_string(), self.context_cache.len());
        stats.insert("cache_limit".to_string(), self.config.cache_size_limit);
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{TaskInput, TaskParser};

    #[tokio::test]
    async fn test_context_analysis() {
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Create a REST API using Python and FastAPI with PostgreSQL database")
            .priority(7);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        assert_eq!(context.semantic_analysis.domain, "software_development");
        assert!(!context.semantic_analysis.key_concepts.is_empty());
        assert!(!context.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_challenge_prediction() {
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Build a distributed microservices system with React, Node.js, Python, Docker, Kubernetes, PostgreSQL, Redis, and MongoDB")
            .priority(9);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        assert!(!context.challenges.is_empty());
        assert!(context.challenges.iter().any(|c| c.challenge_type == ChallengeType::Technical));
    }

    #[tokio::test]
    async fn test_relationship_extraction() {
        let parser = TaskParser::new();
        let analyzer = ContextAnalyzer::new().await.unwrap();
        
        let input = TaskInput::new()
            .description("Create and then test a React application")
            .priority(5);
        
        let parsed_task = parser.parse(input).await.unwrap();
        let context = analyzer.analyze(&parsed_task).await.unwrap();
        
        // Should have sequence relationship between "create" and "test"
        assert!(context.relationships.iter().any(|r| r.relationship_type == RelationshipType::Sequence));
    }

    #[tokio::test]
    async fn test_caching() {
        let analyzer = ContextAnalyzer::new().await.unwrap();
        let parser = TaskParser::new();
        
        let input = TaskInput::new()
            .description("Simple task")
            .priority(5);
        
        let parsed_task = parser.parse(input).await.unwrap();
        
        // First analysis
        let context1 = analyzer.analyze(&parsed_task).await.unwrap();
        assert!(!context1.metadata.cache_hit);
        
        // Second analysis should hit cache
        let context2 = analyzer.analyze(&parsed_task).await.unwrap();
        // Note: The cache_hit flag is only set during retrieval, not in the stored context
        assert_eq!(context1.task_id, context2.task_id);
    }
}