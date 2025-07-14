//! Task input parsing and validation module

use crate::error::{SplinterError, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use regex::Regex;
use lazy_static::lazy_static;

/// Task input formats supported by the parser
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InputFormat {
    /// Plain text description
    PlainText,
    /// Markdown formatted text
    Markdown,
    /// JSON structured data
    Json,
    /// YAML structured data
    Yaml,
    /// Code snippet (auto-detected language)
    Code,
    /// Natural language query
    NaturalLanguage,
}

/// Input task structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInput {
    /// Task description
    pub description: String,
    /// Task priority (1-10)
    pub priority: u8,
    /// Task deadline (optional)
    pub deadline: Option<DateTime<Utc>>,
    /// Task context/metadata
    pub context: HashMap<String, String>,
    /// Input format
    pub format: InputFormat,
    /// Task tags for categorization
    pub tags: Vec<String>,
    /// Dependencies on other tasks
    pub dependencies: Vec<String>,
    /// Required resources
    pub resources: Vec<String>,
    /// Estimated complexity (1-10)
    pub complexity: Option<u8>,
}

/// Parsed task with extracted information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedTask {
    /// Unique task identifier
    pub id: Uuid,
    /// Original input
    pub input: TaskInput,
    /// Extracted entities
    pub entities: Vec<TaskEntity>,
    /// Identified task type
    pub task_type: TaskType,
    /// Extracted requirements
    pub requirements: Vec<Requirement>,
    /// Estimated difficulty
    pub difficulty: DifficultyLevel,
    /// Parsing metadata
    pub metadata: ParseMetadata,
}

/// Extracted entities from task description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEntity {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
}

/// Types of entities that can be extracted
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntityType {
    /// Technology stack (e.g., "React", "Python")
    Technology,
    /// Action verb (e.g., "create", "implement")
    Action,
    /// Resource (e.g., "database", "API")
    Resource,
    /// Time reference (e.g., "tomorrow", "next week")
    Time,
    /// Quality requirement (e.g., "secure", "fast")
    Quality,
    /// Quantity (e.g., "10 users", "100MB")
    Quantity,
    /// Location (e.g., "server", "client")
    Location,
}

/// Task type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskType {
    /// Software development task
    Development,
    /// System administration task
    Administration,
    /// Data analysis task
    Analysis,
    /// Research task
    Research,
    /// Testing task
    Testing,
    /// Documentation task
    Documentation,
    /// Deployment task
    Deployment,
    /// Maintenance task
    Maintenance,
    /// Generic task
    Generic,
}

/// Task requirement extracted from description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Requirement {
    /// Requirement description
    pub description: String,
    /// Requirement type
    pub req_type: RequirementType,
    /// Priority level
    pub priority: u8,
    /// Whether requirement is mandatory
    pub mandatory: bool,
}

/// Types of requirements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RequirementType {
    /// Functional requirement
    Functional,
    /// Non-functional requirement
    NonFunctional,
    /// Performance requirement
    Performance,
    /// Security requirement
    Security,
    /// Usability requirement
    Usability,
    /// Compatibility requirement
    Compatibility,
    /// Scalability requirement
    Scalability,
}

/// Difficulty level assessment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DifficultyLevel {
    /// Very easy (1-2)
    VeryEasy,
    /// Easy (3-4)
    Easy,
    /// Medium (5-6)
    Medium,
    /// Hard (7-8)
    Hard,
    /// Very hard (9-10)
    VeryHard,
}

/// Metadata from parsing process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseMetadata {
    /// Parsing timestamp
    pub parsed_at: DateTime<Utc>,
    /// Parsing duration in milliseconds
    pub parse_duration_ms: u64,
    /// Confidence score for parsing
    pub confidence: f64,
    /// Detected language
    pub language: Option<String>,
    /// Word count
    pub word_count: usize,
    /// Character count
    pub char_count: usize,
}

/// Task parser for input processing
#[derive(Debug)]
pub struct TaskParser {
    /// Entity extraction patterns
    patterns: EntityPatterns,
    /// Task type classifiers
    classifiers: TaskClassifiers,
}

/// Entity extraction patterns
#[derive(Debug)]
struct EntityPatterns {
    technology: Regex,
    action: Regex,
    resource: Regex,
    time: Regex,
    quality: Regex,
    quantity: Regex,
    location: Regex,
}

/// Task type classifiers
#[derive(Debug)]
struct TaskClassifiers {
    development_keywords: Vec<&'static str>,
    administration_keywords: Vec<&'static str>,
    analysis_keywords: Vec<&'static str>,
    research_keywords: Vec<&'static str>,
    testing_keywords: Vec<&'static str>,
    documentation_keywords: Vec<&'static str>,
    deployment_keywords: Vec<&'static str>,
    maintenance_keywords: Vec<&'static str>,
}

lazy_static! {
    static ref TECHNOLOGY_PATTERN: Regex = Regex::new(
        r"(?i)\b(react|vue|angular|python|javascript|typescript|rust|java|go|docker|kubernetes|aws|azure|gcp|postgresql|mysql|mongodb|redis|nginx|apache|node\.js|express|django|flask|spring|rails)\b"
    ).unwrap();
    
    static ref ACTION_PATTERN: Regex = Regex::new(
        r"(?i)\b(create|build|implement|develop|design|deploy|test|analyze|research|document|maintain|update|fix|optimize|refactor|migrate|integrate|configure|setup|install)\b"
    ).unwrap();
    
    static ref RESOURCE_PATTERN: Regex = Regex::new(
        r"(?i)\b(database|api|server|client|frontend|backend|service|microservice|application|system|infrastructure|network|storage|cache|queue|pipeline|workflow)\b"
    ).unwrap();
    
    static ref TIME_PATTERN: Regex = Regex::new(
        r"(?i)\b(today|tomorrow|yesterday|next week|last week|next month|last month|deadline|asap|urgent|immediately|soon|later|eventually)\b"
    ).unwrap();
    
    static ref QUALITY_PATTERN: Regex = Regex::new(
        r"(?i)\b(secure|fast|scalable|reliable|maintainable|efficient|performant|robust|stable|responsive|accessible|usable|testable|modular|reusable)\b"
    ).unwrap();
    
    static ref QUANTITY_PATTERN: Regex = Regex::new(
        r"\b(\d+(?:\.\d+)?)\s*(users?|requests?|transactions?|mb|gb|tb|kb|ms|seconds?|minutes?|hours?|days?|weeks?|months?|years?|percent|%)\b"
    ).unwrap();
    
    static ref LOCATION_PATTERN: Regex = Regex::new(
        r"(?i)\b(server|client|cloud|edge|mobile|desktop|web|browser|container|vm|cluster|node|region|zone|datacenter|production|staging|development|local)\b"
    ).unwrap();
}

impl TaskInput {
    /// Create a new task input
    pub fn new() -> Self {
        Self {
            description: String::new(),
            priority: 5,
            deadline: None,
            context: HashMap::new(),
            format: InputFormat::PlainText,
            tags: Vec::new(),
            dependencies: Vec::new(),
            resources: Vec::new(),
            complexity: None,
        }
    }

    /// Set task description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set task priority
    pub fn priority(mut self, priority: u8) -> Self {
        self.priority = priority.clamp(1, 10);
        self
    }

    /// Set task deadline
    pub fn deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add context metadata
    pub fn context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Set input format
    pub fn format(mut self, format: InputFormat) -> Self {
        self.format = format;
        self
    }

    /// Add tags
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add dependencies
    pub fn dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Add required resources
    pub fn resources(mut self, resources: Vec<String>) -> Self {
        self.resources = resources;
        self
    }

    /// Set complexity estimate
    pub fn complexity(mut self, complexity: u8) -> Self {
        self.complexity = Some(complexity.clamp(1, 10));
        self
    }
}

impl TaskParser {
    /// Create a new task parser
    pub fn new() -> Self {
        Self {
            patterns: EntityPatterns {
                technology: TECHNOLOGY_PATTERN.clone(),
                action: ACTION_PATTERN.clone(),
                resource: RESOURCE_PATTERN.clone(),
                time: TIME_PATTERN.clone(),
                quality: QUALITY_PATTERN.clone(),
                quantity: QUANTITY_PATTERN.clone(),
                location: LOCATION_PATTERN.clone(),
            },
            classifiers: TaskClassifiers {
                development_keywords: vec![
                    "build", "create", "implement", "develop", "code", "program", "software", "application", "system", "api", "frontend", "backend", "database", "web", "mobile"
                ],
                administration_keywords: vec![
                    "deploy", "configure", "setup", "install", "maintain", "monitor", "backup", "restore", "update", "patch", "server", "infrastructure", "network", "security"
                ],
                analysis_keywords: vec![
                    "analyze", "report", "dashboard", "metrics", "data", "statistics", "insights", "visualization", "query", "research", "investigate", "study"
                ],
                research_keywords: vec![
                    "research", "investigate", "study", "explore", "evaluate", "compare", "benchmark", "prototype", "experiment", "proof of concept"
                ],
                testing_keywords: vec![
                    "test", "verify", "validate", "check", "qa", "quality", "unit test", "integration test", "performance test", "load test", "security test"
                ],
                documentation_keywords: vec![
                    "document", "write", "manual", "guide", "tutorial", "readme", "specification", "design", "architecture", "wiki", "help"
                ],
                deployment_keywords: vec![
                    "deploy", "release", "publish", "distribute", "launch", "rollout", "cicd", "pipeline", "containerize", "orchestrate"
                ],
                maintenance_keywords: vec![
                    "maintain", "fix", "bug", "issue", "support", "troubleshoot", "debug", "optimize", "refactor", "upgrade", "migrate"
                ],
            },
        }
    }

    /// Parse task input into structured format
    pub async fn parse(&self, input: TaskInput) -> Result<ParsedTask> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        if input.description.is_empty() {
            return Err(SplinterError::parse_error("Task description cannot be empty"));
        }

        if input.priority < 1 || input.priority > 10 {
            return Err(SplinterError::parse_error("Priority must be between 1 and 10"));
        }

        // Extract entities
        let entities = self.extract_entities(&input.description)?;

        // Classify task type
        let task_type = self.classify_task_type(&input.description, &entities);

        // Extract requirements
        let requirements = self.extract_requirements(&input.description, &entities)?;

        // Assess difficulty
        let difficulty = self.assess_difficulty(&input, &entities, &requirements);

        // Create metadata
        let parse_duration = start_time.elapsed();
        let metadata = ParseMetadata {
            parsed_at: Utc::now(),
            parse_duration_ms: parse_duration.as_millis() as u64,
            confidence: self.calculate_confidence(&entities, &requirements),
            language: self.detect_language(&input.description),
            word_count: input.description.split_whitespace().count(),
            char_count: input.description.chars().count(),
        };

        Ok(ParsedTask {
            id: Uuid::new_v4(),
            input,
            entities,
            task_type,
            requirements,
            difficulty,
            metadata,
        })
    }

    /// Extract entities from text
    fn extract_entities(&self, text: &str) -> Result<Vec<TaskEntity>> {
        let mut entities = Vec::new();

        // Extract technology entities
        for cap in self.patterns.technology.find_iter(text) {
            entities.push(TaskEntity {
                text: cap.as_str().to_string(),
                entity_type: EntityType::Technology,
                confidence: 0.9,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Extract action entities
        for cap in self.patterns.action.find_iter(text) {
            entities.push(TaskEntity {
                text: cap.as_str().to_string(),
                entity_type: EntityType::Action,
                confidence: 0.85,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Extract resource entities
        for cap in self.patterns.resource.find_iter(text) {
            entities.push(TaskEntity {
                text: cap.as_str().to_string(),
                entity_type: EntityType::Resource,
                confidence: 0.8,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Extract time entities
        for cap in self.patterns.time.find_iter(text) {
            entities.push(TaskEntity {
                text: cap.as_str().to_string(),
                entity_type: EntityType::Time,
                confidence: 0.75,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Extract quality entities
        for cap in self.patterns.quality.find_iter(text) {
            entities.push(TaskEntity {
                text: cap.as_str().to_string(),
                entity_type: EntityType::Quality,
                confidence: 0.7,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Extract quantity entities
        for cap in self.patterns.quantity.find_iter(text) {
            entities.push(TaskEntity {
                text: cap.as_str().to_string(),
                entity_type: EntityType::Quantity,
                confidence: 0.95,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Extract location entities
        for cap in self.patterns.location.find_iter(text) {
            entities.push(TaskEntity {
                text: cap.as_str().to_string(),
                entity_type: EntityType::Location,
                confidence: 0.8,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Sort entities by position
        entities.sort_by_key(|e| e.start);

        Ok(entities)
    }

    /// Classify task type based on content
    fn classify_task_type(&self, text: &str, entities: &[TaskEntity]) -> TaskType {
        let text_lower = text.to_lowercase();
        
        // Count keyword matches for each category
        let mut scores = HashMap::new();
        
        // Development keywords
        let dev_score = self.classifiers.development_keywords.iter()
            .filter(|&keyword| text_lower.contains(keyword))
            .count();
        scores.insert(TaskType::Development, dev_score);
        
        // Administration keywords
        let admin_score = self.classifiers.administration_keywords.iter()
            .filter(|&keyword| text_lower.contains(keyword))
            .count();
        scores.insert(TaskType::Administration, admin_score);
        
        // Analysis keywords
        let analysis_score = self.classifiers.analysis_keywords.iter()
            .filter(|&keyword| text_lower.contains(keyword))
            .count();
        scores.insert(TaskType::Analysis, analysis_score);
        
        // Research keywords
        let research_score = self.classifiers.research_keywords.iter()
            .filter(|&keyword| text_lower.contains(keyword))
            .count();
        scores.insert(TaskType::Research, research_score);
        
        // Testing keywords
        let testing_score = self.classifiers.testing_keywords.iter()
            .filter(|&keyword| text_lower.contains(keyword))
            .count();
        scores.insert(TaskType::Testing, testing_score);
        
        // Documentation keywords
        let doc_score = self.classifiers.documentation_keywords.iter()
            .filter(|&keyword| text_lower.contains(keyword))
            .count();
        scores.insert(TaskType::Documentation, doc_score);
        
        // Deployment keywords
        let deploy_score = self.classifiers.deployment_keywords.iter()
            .filter(|&keyword| text_lower.contains(keyword))
            .count();
        scores.insert(TaskType::Deployment, deploy_score);
        
        // Maintenance keywords
        let maint_score = self.classifiers.maintenance_keywords.iter()
            .filter(|&keyword| text_lower.contains(keyword))
            .count();
        scores.insert(TaskType::Maintenance, maint_score);
        
        // Also consider entities
        let tech_entities = entities.iter()
            .filter(|e| e.entity_type == EntityType::Technology)
            .count();
        
        if tech_entities > 0 {
            let current_dev_score = scores.get(&TaskType::Development).unwrap_or(&0);
            scores.insert(TaskType::Development, current_dev_score + tech_entities);
        }
        
        // Find the highest scoring category
        scores.into_iter()
            .max_by_key(|(_, score)| *score)
            .map(|(task_type, _)| task_type)
            .unwrap_or(TaskType::Generic)
    }

    /// Extract requirements from text
    fn extract_requirements(&self, text: &str, entities: &[TaskEntity]) -> Result<Vec<Requirement>> {
        let mut requirements = Vec::new();
        
        // Extract quality requirements
        for entity in entities {
            if entity.entity_type == EntityType::Quality {
                requirements.push(Requirement {
                    description: format!("System must be {}", entity.text),
                    req_type: RequirementType::NonFunctional,
                    priority: 7,
                    mandatory: true,
                });
            }
        }
        
        // Extract performance requirements
        for entity in entities {
            if entity.entity_type == EntityType::Quantity {
                requirements.push(Requirement {
                    description: format!("System must handle {}", entity.text),
                    req_type: RequirementType::Performance,
                    priority: 8,
                    mandatory: true,
                });
            }
        }
        
        // Extract security requirements
        if text.to_lowercase().contains("secure") || text.to_lowercase().contains("authentication") {
            requirements.push(Requirement {
                description: "System must implement proper security measures".to_string(),
                req_type: RequirementType::Security,
                priority: 9,
                mandatory: true,
            });
        }
        
        // Extract functional requirements based on actions
        for entity in entities {
            if entity.entity_type == EntityType::Action {
                requirements.push(Requirement {
                    description: format!("System must {} the specified components", entity.text),
                    req_type: RequirementType::Functional,
                    priority: 8,
                    mandatory: true,
                });
            }
        }
        
        Ok(requirements)
    }

    /// Assess task difficulty
    fn assess_difficulty(&self, input: &TaskInput, entities: &[TaskEntity], requirements: &[Requirement]) -> DifficultyLevel {
        let mut score = 0;
        
        // Base score from user complexity if provided
        if let Some(complexity) = input.complexity {
            score += complexity as i32;
        } else {
            score += 5; // Default medium complexity
        }
        
        // Adjust based on entities
        let tech_count = entities.iter().filter(|e| e.entity_type == EntityType::Technology).count();
        score += (tech_count as i32).min(5);
        
        let action_count = entities.iter().filter(|e| e.entity_type == EntityType::Action).count();
        score += (action_count as i32).min(3);
        
        // Adjust based on requirements
        let mandatory_reqs = requirements.iter().filter(|r| r.mandatory).count();
        score += (mandatory_reqs as i32).min(4);
        
        // Adjust based on dependencies
        score += (input.dependencies.len() as i32).min(3);
        
        // Adjust based on word count
        let word_count = input.description.split_whitespace().count();
        if word_count > 100 {
            score += 2;
        } else if word_count > 50 {
            score += 1;
        }
        
        // Clamp score to valid range
        score = score.clamp(1, 10);
        
        match score {
            1..=2 => DifficultyLevel::VeryEasy,
            3..=4 => DifficultyLevel::Easy,
            5..=6 => DifficultyLevel::Medium,
            7..=8 => DifficultyLevel::Hard,
            9..=10 => DifficultyLevel::VeryHard,
            _ => DifficultyLevel::Medium,
        }
    }

    /// Calculate parsing confidence
    fn calculate_confidence(&self, entities: &[TaskEntity], requirements: &[Requirement]) -> f64 {
        if entities.is_empty() {
            return 0.3; // Low confidence if no entities found
        }
        
        let avg_entity_confidence: f64 = entities.iter()
            .map(|e| e.confidence)
            .sum::<f64>() / entities.len() as f64;
        
        let req_bonus = if requirements.is_empty() {
            0.0
        } else {
            0.1 * (requirements.len() as f64).min(5.0)
        };
        
        (avg_entity_confidence + req_bonus).min(1.0)
    }

    /// Detect language (simplified)
    fn detect_language(&self, text: &str) -> Option<String> {
        // Simple language detection based on common patterns
        if text.contains("def ") || text.contains("import ") || text.contains("class ") {
            Some("python".to_string())
        } else if text.contains("function ") || text.contains("const ") || text.contains("let ") {
            Some("javascript".to_string())
        } else if text.contains("fn ") || text.contains("struct ") || text.contains("impl ") {
            Some("rust".to_string())
        } else if text.contains("public class ") || text.contains("public static") {
            Some("java".to_string())
        } else {
            Some("natural".to_string())
        }
    }
}

impl Default for TaskParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_parsing() {
        let parser = TaskParser::new();
        let input = TaskInput::new()
            .description("Create a REST API using Python and FastAPI")
            .priority(7);
        
        let result = parser.parse(input).await.unwrap();
        
        assert_eq!(result.task_type, TaskType::Development);
        assert!(result.entities.len() > 0);
        assert!(result.requirements.len() > 0);
    }

    #[tokio::test]
    async fn test_entity_extraction() {
        let parser = TaskParser::new();
        let input = TaskInput::new()
            .description("Build a secure React application with authentication")
            .priority(8);
        
        let result = parser.parse(input).await.unwrap();
        
        let tech_entities: Vec<_> = result.entities.iter()
            .filter(|e| e.entity_type == EntityType::Technology)
            .collect();
        
        assert!(!tech_entities.is_empty());
        assert!(tech_entities.iter().any(|e| e.text.to_lowercase().contains("react")));
    }

    #[tokio::test]
    async fn test_task_classification() {
        let parser = TaskParser::new();
        
        // Test development classification
        let dev_input = TaskInput::new()
            .description("Implement user authentication system")
            .priority(6);
        let dev_result = parser.parse(dev_input).await.unwrap();
        assert_eq!(dev_result.task_type, TaskType::Development);
        
        // Test testing classification
        let test_input = TaskInput::new()
            .description("Write unit tests for the API endpoints")
            .priority(5);
        let test_result = parser.parse(test_input).await.unwrap();
        assert_eq!(test_result.task_type, TaskType::Testing);
    }

    #[tokio::test]
    async fn test_difficulty_assessment() {
        let parser = TaskParser::new();
        
        // Simple task
        let simple_input = TaskInput::new()
            .description("Fix a typo in documentation")
            .priority(2);
        let simple_result = parser.parse(simple_input).await.unwrap();
        assert!(matches!(simple_result.difficulty, DifficultyLevel::VeryEasy | DifficultyLevel::Easy));
        
        // Complex task
        let complex_input = TaskInput::new()
            .description("Build a distributed microservices architecture with Docker, Kubernetes, PostgreSQL, Redis, and implement authentication, authorization, monitoring, and logging")
            .priority(9);
        let complex_result = parser.parse(complex_input).await.unwrap();
        assert!(matches!(complex_result.difficulty, DifficultyLevel::Hard | DifficultyLevel::VeryHard));
    }

    #[test]
    fn test_task_input_builder() {
        let task = TaskInput::new()
            .description("Test task")
            .priority(7)
            .context("project", "test")
            .tags(vec!["test".to_string(), "example".to_string()]);
        
        assert_eq!(task.description, "Test task");
        assert_eq!(task.priority, 7);
        assert_eq!(task.context.get("project"), Some(&"test".to_string()));
        assert_eq!(task.tags.len(), 2);
    }
}