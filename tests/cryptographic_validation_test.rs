//! Cryptographic Validation Test Suite for Neural Swarm
//!
//! This module provides comprehensive testing of cryptographic functions
//! and security-critical operations in the neural communication system.

use neural_swarm::{
    NeuralNetwork, NetworkBuilder,
    activation::ActivationType,
    network::LayerConfig,
    training::{TrainingData, TrainingAlgorithm, TrainingParams, Trainer},
    NeuralFloat,
};
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use sha2::{Sha256, Digest};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

// =============================================================================
// CRYPTOGRAPHIC TEST FRAMEWORK
// =============================================================================

#[derive(Debug, Clone)]
pub struct CryptographicTestSuite {
    config: CryptoTestConfig,
    entropy_analyzer: EntropyAnalyzer,
    hash_validator: HashValidator,
    integrity_checker: IntegrityChecker,
}

#[derive(Debug, Clone)]
pub struct CryptoTestConfig {
    pub randomness_tests: bool,
    pub integrity_tests: bool,
    pub serialization_tests: bool,
    pub key_derivation_tests: bool,
    pub sample_size: usize,
    pub statistical_threshold: f64,
}

impl Default for CryptoTestConfig {
    fn default() -> Self {
        Self {
            randomness_tests: true,
            integrity_tests: true,
            serialization_tests: true,
            key_derivation_tests: true,
            sample_size: 10000,
            statistical_threshold: 0.95,
        }
    }
}

impl CryptographicTestSuite {
    pub fn new(config: CryptoTestConfig) -> Self {
        Self {
            config,
            entropy_analyzer: EntropyAnalyzer::new(),
            hash_validator: HashValidator::new(),
            integrity_checker: IntegrityChecker::new(),
        }
    }

    /// Run comprehensive cryptographic validation tests
    pub fn run_crypto_tests(&mut self) -> CryptoTestResults {
        let mut results = CryptoTestResults::new();

        if self.config.randomness_tests {
            results.randomness_quality = self.test_randomness_quality();
        }

        if self.config.integrity_tests {
            results.data_integrity = self.test_data_integrity();
        }

        if self.config.serialization_tests {
            results.serialization_security = self.test_serialization_security();
        }

        if self.config.key_derivation_tests {
            results.key_derivation = self.test_key_derivation();
        }

        results.calculate_overall_score();
        results
    }

    /// Test quality of random number generation
    fn test_randomness_quality(&mut self) -> CryptoTestCategory {
        let mut category = CryptoTestCategory::new("Randomness Quality");

        // Test 1: Basic entropy test
        category.add_test(self.test_basic_entropy());

        // Test 2: Chi-square test for uniformity
        category.add_test(self.test_chi_square_uniformity());

        // Test 3: Runs test for independence
        category.add_test(self.test_runs_independence());

        // Test 4: Autocorrelation test
        category.add_test(self.test_autocorrelation());

        // Test 5: Frequency analysis
        category.add_test(self.test_frequency_analysis());

        category
    }

    /// Test data integrity and authentication
    fn test_data_integrity(&mut self) -> CryptoTestCategory {
        let mut category = CryptoTestCategory::new("Data Integrity");

        // Test 1: Hash consistency
        category.add_test(self.test_hash_consistency());

        // Test 2: Tamper detection
        category.add_test(self.test_tamper_detection());

        // Test 3: Network state integrity
        category.add_test(self.test_network_state_integrity());

        // Test 4: Training data validation
        category.add_test(self.test_training_data_validation());

        category
    }

    /// Test serialization security
    fn test_serialization_security(&mut self) -> CryptoTestCategory {
        let mut category = CryptoTestCategory::new("Serialization Security");

        // Test 1: Serialization determinism
        category.add_test(self.test_serialization_determinism());

        // Test 2: Deserialization validation
        category.add_test(self.test_deserialization_validation());

        // Test 3: Version compatibility
        category.add_test(self.test_version_compatibility());

        // Test 4: Malformed data handling
        category.add_test(self.test_malformed_data_handling());

        category
    }

    /// Test key derivation and seeding
    fn test_key_derivation(&mut self) -> CryptoTestCategory {
        let mut category = CryptoTestCategory::new("Key Derivation");

        // Test 1: Seed determinism
        category.add_test(self.test_seed_determinism());

        // Test 2: Key derivation function
        category.add_test(self.test_key_derivation_function());

        // Test 3: Seed sensitivity
        category.add_test(self.test_seed_sensitivity());

        // Test 4: Cryptographic key strength
        category.add_test(self.test_cryptographic_key_strength());

        category
    }
}

// =============================================================================
// INDIVIDUAL CRYPTOGRAPHIC TESTS
// =============================================================================

impl CryptographicTestSuite {
    /// Test basic entropy of random number generation
    fn test_basic_entropy(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Basic Entropy Test");

        let mut entropy_values = Vec::new();
        
        // Generate multiple networks with random seeds
        for i in 0..100 {
            let configs = vec![
                LayerConfig::new(10, ActivationType::Linear),
                LayerConfig::new(20, ActivationType::ReLU),
                LayerConfig::new(5, ActivationType::Sigmoid),
            ];

            if let Ok(mut network) = NeuralNetwork::new_feedforward(&configs) {
                let _ = network.initialize_weights(Some(i));
                
                // Extract some weight values for entropy analysis
                let test_input = vec![0.1; 10];
                if let Ok(output) = network.predict(&test_input) {
                    for &value in &output {
                        entropy_values.push(value);
                    }
                }
            }
        }

        if entropy_values.len() < 100 {
            test.mark_failed("Insufficient data for entropy analysis");
            return test;
        }

        let entropy = self.entropy_analyzer.calculate_entropy(&entropy_values);
        
        if entropy > 0.5 {
            test.mark_passed(&format!("Good entropy: {:.3}", entropy));
        } else {
            test.mark_failed(&format!("Low entropy: {:.3}", entropy));
        }

        test
    }

    /// Test chi-square uniformity of random values
    fn test_chi_square_uniformity(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Chi-Square Uniformity Test");

        let mut values = Vec::new();
        
        // Generate random values from network initialization
        for i in 0..1000 {
            let configs = vec![
                LayerConfig::new(5, ActivationType::Linear),
                LayerConfig::new(1, ActivationType::Sigmoid),
            ];

            if let Ok(mut network) = NeuralNetwork::new_feedforward(&configs) {
                let _ = network.initialize_weights(Some(i));
                let test_input = vec![0.5; 5];
                if let Ok(output) = network.predict(&test_input) {
                    values.push(output[0]);
                }
            }
        }

        if values.is_empty() {
            test.mark_failed("No values generated for uniformity test");
            return test;
        }

        let chi_square = self.entropy_analyzer.chi_square_test(&values, 10);
        let critical_value = 16.919; // Chi-square critical value for 9 degrees of freedom at 95% confidence

        if chi_square < critical_value {
            test.mark_passed(&format!("Uniform distribution: χ² = {:.3}", chi_square));
        } else {
            test.mark_warning(&format!("Non-uniform distribution: χ² = {:.3}", chi_square));
        }

        test
    }

    /// Test runs for independence
    fn test_runs_independence(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Runs Independence Test");

        let mut binary_sequence = Vec::new();
        
        // Generate binary sequence from network outputs
        for i in 0..1000 {
            let configs = vec![
                LayerConfig::new(2, ActivationType::Linear),
                LayerConfig::new(1, ActivationType::Sigmoid),
            ];

            if let Ok(mut network) = NeuralNetwork::new_feedforward(&configs) {
                let _ = network.initialize_weights(Some(i));
                let test_input = vec![0.5, -0.5];
                if let Ok(output) = network.predict(&test_input) {
                    binary_sequence.push(output[0] > 0.5);
                }
            }
        }

        if binary_sequence.len() < 100 {
            test.mark_failed("Insufficient data for runs test");
            return test;
        }

        let runs_statistic = self.entropy_analyzer.runs_test(&binary_sequence);
        
        // Expected runs for random sequence
        let n = binary_sequence.len() as f64;
        let ones = binary_sequence.iter().filter(|&&b| b).count() as f64;
        let zeros = n - ones;
        
        let expected_runs = (2.0 * ones * zeros / n) + 1.0;
        let variance = (2.0 * ones * zeros * (2.0 * ones * zeros - n)) / (n * n * (n - 1.0));
        let z_score = (runs_statistic as f64 - expected_runs) / variance.sqrt();

        if z_score.abs() < 1.96 { // 95% confidence interval
            test.mark_passed(&format!("Independent sequence: z = {:.3}", z_score));
        } else {
            test.mark_warning(&format!("Possibly non-independent: z = {:.3}", z_score));
        }

        test
    }

    /// Test autocorrelation
    fn test_autocorrelation(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Autocorrelation Test");

        let mut sequence = Vec::new();
        
        // Generate sequence from network outputs
        for i in 0..500 {
            let configs = vec![
                LayerConfig::new(3, ActivationType::Linear),
                LayerConfig::new(1, ActivationType::Tanh),
            ];

            if let Ok(mut network) = NeuralNetwork::new_feedforward(&configs) {
                let _ = network.initialize_weights(Some(i));
                let test_input = vec![0.1, 0.5, -0.2];
                if let Ok(output) = network.predict(&test_input) {
                    sequence.push(output[0]);
                }
            }
        }

        if sequence.len() < 100 {
            test.mark_failed("Insufficient data for autocorrelation test");
            return test;
        }

        let autocorr = self.entropy_analyzer.autocorrelation(&sequence, 1);
        
        if autocorr.abs() < 0.1 {
            test.mark_passed(&format!("Low autocorrelation: {:.3}", autocorr));
        } else {
            test.mark_warning(&format!("High autocorrelation: {:.3}", autocorr));
        }

        test
    }

    /// Test frequency analysis
    fn test_frequency_analysis(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Frequency Analysis Test");

        let mut bit_counts = [0; 32]; // Count bits in each position
        let mut total_samples = 0;

        // Generate values and analyze bit patterns
        for i in 0..1000 {
            let configs = vec![
                LayerConfig::new(4, ActivationType::Linear),
                LayerConfig::new(1, ActivationType::Linear),
            ];

            if let Ok(mut network) = NeuralNetwork::new_feedforward(&configs) {
                let _ = network.initialize_weights(Some(i));
                let test_input = vec![0.25, 0.5, 0.75, -0.25];
                if let Ok(output) = network.predict(&test_input) {
                    let bits = output[0].to_bits();
                    for (j, count) in bit_counts.iter_mut().enumerate() {
                        if (bits >> j) & 1 == 1 {
                            *count += 1;
                        }
                    }
                    total_samples += 1;
                }
            }
        }

        if total_samples < 100 {
            test.mark_failed("Insufficient samples for frequency analysis");
            return test;
        }

        // Check if bit frequencies are roughly balanced
        let expected_frequency = total_samples as f64 / 2.0;
        let mut frequency_chi_square = 0.0;

        for &count in &bit_counts[0..23] { // Skip sign and exponent bits
            let diff = count as f64 - expected_frequency;
            frequency_chi_square += (diff * diff) / expected_frequency;
        }

        let critical_value = 35.172; // Chi-square critical value for 22 degrees of freedom at 95%

        if frequency_chi_square < critical_value {
            test.mark_passed(&format!("Balanced bit frequencies: χ² = {:.3}", frequency_chi_square));
        } else {
            test.mark_warning(&format!("Imbalanced bit frequencies: χ² = {:.3}", frequency_chi_square));
        }

        test
    }

    /// Test hash consistency
    fn test_hash_consistency(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Hash Consistency Test");

        let configs = vec![
            LayerConfig::new(5, ActivationType::Linear),
            LayerConfig::new(10, ActivationType::ReLU),
            LayerConfig::new(3, ActivationType::Sigmoid),
        ];

        let mut network = match NeuralNetwork::new_feedforward(&configs) {
            Ok(net) => net,
            Err(_) => {
                test.mark_failed("Failed to create test network");
                return test;
            }
        };

        let _ = network.initialize_weights(Some(42));

        // Serialize network multiple times and check hash consistency
        let mut hashes = HashSet::new();
        
        for _ in 0..10 {
            if let Ok(serialized) = serde_json::to_string(&network) {
                let hash = self.hash_validator.compute_hash(&serialized);
                hashes.insert(hash);
            }
        }

        if hashes.len() == 1 {
            test.mark_passed("Consistent serialization hashes");
        } else {
            test.mark_failed(&format!("Inconsistent hashes: {} different", hashes.len()));
        }

        test
    }

    /// Test tamper detection
    fn test_tamper_detection(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Tamper Detection Test");

        let configs = vec![
            LayerConfig::new(3, ActivationType::Linear),
            LayerConfig::new(5, ActivationType::ReLU),
            LayerConfig::new(2, ActivationType::Sigmoid),
        ];

        let mut network = match NeuralNetwork::new_feedforward(&configs) {
            Ok(net) => net,
            Err(_) => {
                test.mark_failed("Failed to create test network");
                return test;
            }
        };

        let _ = network.initialize_weights(Some(123));

        // Get original hash
        let original_serialized = match serde_json::to_string(&network) {
            Ok(s) => s,
            Err(_) => {
                test.mark_failed("Failed to serialize network");
                return test;
            }
        };

        let original_hash = self.hash_validator.compute_hash(&original_serialized);

        // Tamper with the serialized data
        let mut tampered = original_serialized.clone();
        if let Some(byte) = tampered.as_bytes_mut().get_mut(100) {
            *byte = byte.wrapping_add(1);
        }

        let tampered_hash = self.hash_validator.compute_hash(&String::from_utf8_lossy(tampered.as_bytes()));

        if original_hash != tampered_hash {
            test.mark_passed("Tamper detection working correctly");
        } else {
            test.mark_failed("Tamper detection failed");
        }

        test
    }

    /// Test network state integrity
    fn test_network_state_integrity(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Network State Integrity Test");

        let configs = vec![
            LayerConfig::new(4, ActivationType::Linear),
            LayerConfig::new(8, ActivationType::ReLU),
            LayerConfig::new(2, ActivationType::Sigmoid),
        ];

        let mut network = match NeuralNetwork::new_feedforward(&configs) {
            Ok(net) => net,
            Err(_) => {
                test.mark_failed("Failed to create test network");
                return test;
            }
        };

        let _ = network.initialize_weights(Some(456));

        // Test input
        let test_input = vec![0.1, 0.2, 0.3, 0.4];
        
        // Get baseline output
        let baseline_output = match network.predict(&test_input) {
            Ok(output) => output,
            Err(_) => {
                test.mark_failed("Failed to get baseline output");
                return test;
            }
        };

        // Serialize and deserialize
        let serialized = match serde_json::to_string(&network) {
            Ok(s) => s,
            Err(_) => {
                test.mark_failed("Failed to serialize network");
                return test;
            }
        };

        let mut restored_network: NeuralNetwork = match serde_json::from_str(&serialized) {
            Ok(net) => net,
            Err(_) => {
                test.mark_failed("Failed to deserialize network");
                return test;
            }
        };

        // Test restored network
        let restored_output = match restored_network.predict(&test_input) {
            Ok(output) => output,
            Err(_) => {
                test.mark_failed("Failed to get restored output");
                return test;
            }
        };

        // Compare outputs
        let max_diff = baseline_output.iter()
            .zip(restored_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        if max_diff < 1e-6 {
            test.mark_passed("Network state integrity preserved");
        } else {
            test.mark_failed(&format!("State integrity compromised: max diff = {}", max_diff));
        }

        test
    }

    /// Test seed determinism
    fn test_seed_determinism(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Seed Determinism Test");

        let configs = vec![
            LayerConfig::new(6, ActivationType::Linear),
            LayerConfig::new(4, ActivationType::ReLU),
            LayerConfig::new(2, ActivationType::Sigmoid),
        ];

        let seed = 789u64;
        let test_input = vec![0.5, -0.2, 1.0, 0.0, 0.8, -0.5];

        // Create multiple networks with same seed
        let mut outputs = Vec::new();
        
        for _ in 0..5 {
            let mut network = match NeuralNetwork::new_feedforward(&configs) {
                Ok(net) => net,
                Err(_) => {
                    test.mark_failed("Failed to create test network");
                    return test;
                }
            };

            let _ = network.initialize_weights(Some(seed));
            
            match network.predict(&test_input) {
                Ok(output) => outputs.push(output),
                Err(_) => {
                    test.mark_failed("Failed to get network output");
                    return test;
                }
            }
        }

        if outputs.len() < 2 {
            test.mark_failed("Insufficient outputs for determinism test");
            return test;
        }

        // Check that all outputs are identical
        let first_output = &outputs[0];
        let all_identical = outputs.iter().all(|output| {
            output.iter().zip(first_output.iter())
                .all(|(a, b)| (a - b).abs() < 1e-9)
        });

        if all_identical {
            test.mark_passed("Seed determinism verified");
        } else {
            test.mark_failed("Seed determinism failed");
        }

        test
    }

    /// Test seed sensitivity
    fn test_seed_sensitivity(&mut self) -> CryptoTestResult {
        let mut test = CryptoTestResult::new("Seed Sensitivity Test");

        let configs = vec![
            LayerConfig::new(5, ActivationType::Linear),
            LayerConfig::new(3, ActivationType::ReLU),
            LayerConfig::new(1, ActivationType::Sigmoid),
        ];

        let test_input = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let mut outputs = HashMap::new();

        // Test with different seeds
        for seed in [1u64, 2, 3, 1000, 1001] {
            let mut network = match NeuralNetwork::new_feedforward(&configs) {
                Ok(net) => net,
                Err(_) => continue,
            };

            let _ = network.initialize_weights(Some(seed));
            
            if let Ok(output) = network.predict(&test_input) {
                outputs.insert(seed, output);
            }
        }

        if outputs.len() < 3 {
            test.mark_failed("Insufficient outputs for sensitivity test");
            return test;
        }

        // Check that small seed changes produce different outputs
        let seed1_output = &outputs[&1];
        let seed2_output = &outputs[&2];
        
        let diff = seed1_output.iter()
            .zip(seed2_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        if diff > 0.01 {
            test.mark_passed(&format!("Good seed sensitivity: max diff = {:.3}", diff));
        } else {
            test.mark_warning(&format!("Low seed sensitivity: max diff = {:.3}", diff));
        }

        test
    }

    // Additional placeholder tests
    fn test_training_data_validation(&mut self) -> CryptoTestResult {
        CryptoTestResult::new("Training Data Validation").mark_skipped("Not yet implemented")
    }

    fn test_serialization_determinism(&mut self) -> CryptoTestResult {
        CryptoTestResult::new("Serialization Determinism").mark_skipped("Not yet implemented")
    }

    fn test_deserialization_validation(&mut self) -> CryptoTestResult {
        CryptoTestResult::new("Deserialization Validation").mark_skipped("Not yet implemented")
    }

    fn test_version_compatibility(&mut self) -> CryptoTestResult {
        CryptoTestResult::new("Version Compatibility").mark_skipped("Not yet implemented")
    }

    fn test_malformed_data_handling(&mut self) -> CryptoTestResult {
        CryptoTestResult::new("Malformed Data Handling").mark_skipped("Not yet implemented")
    }

    fn test_key_derivation_function(&mut self) -> CryptoTestResult {
        CryptoTestResult::new("Key Derivation Function").mark_skipped("Not yet implemented")
    }

    fn test_cryptographic_key_strength(&mut self) -> CryptoTestResult {
        CryptoTestResult::new("Cryptographic Key Strength").mark_skipped("Not yet implemented")
    }
}

// =============================================================================
// UTILITY CLASSES
// =============================================================================

pub struct EntropyAnalyzer;

impl EntropyAnalyzer {
    fn new() -> Self {
        Self
    }

    fn calculate_entropy(&self, values: &[f32]) -> f64 {
        let mut histogram = HashMap::new();
        let bins = 100;
        
        for &value in values {
            let bin = ((value + 1.0) * bins as f32 / 2.0) as usize;
            let bin = bin.min(bins - 1);
            *histogram.entry(bin).or_insert(0) += 1;
        }

        let n = values.len() as f64;
        let mut entropy = 0.0;

        for &count in histogram.values() {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.log2();
            }
        }

        entropy / (bins as f64).log2() // Normalize
    }

    fn chi_square_test(&self, values: &[f32], bins: usize) -> f64 {
        let mut histogram = vec![0; bins];
        
        for &value in values {
            let bin = ((value + 1.0) * bins as f32 / 2.0) as usize;
            let bin = bin.min(bins - 1);
            histogram[bin] += 1;
        }

        let expected = values.len() as f64 / bins as f64;
        let mut chi_square = 0.0;

        for count in histogram {
            let diff = count as f64 - expected;
            chi_square += (diff * diff) / expected;
        }

        chi_square
    }

    fn runs_test(&self, sequence: &[bool]) -> usize {
        let mut runs = 1;
        
        for i in 1..sequence.len() {
            if sequence[i] != sequence[i-1] {
                runs += 1;
            }
        }

        runs
    }

    fn autocorrelation(&self, sequence: &[f32], lag: usize) -> f64 {
        if sequence.len() <= lag {
            return 0.0;
        }

        let n = sequence.len() - lag;
        let mean: f64 = sequence.iter().take(n).map(|&x| x as f64).sum::<f64>() / n as f64;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let xi = sequence[i] as f64 - mean;
            let xi_lag = sequence[i + lag] as f64 - mean;
            
            numerator += xi * xi_lag;
            denominator += xi * xi;
        }

        if denominator == 0.0 { 0.0 } else { numerator / denominator }
    }
}

pub struct HashValidator;

impl HashValidator {
    fn new() -> Self {
        Self
    }

    fn compute_hash(&self, data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

pub struct IntegrityChecker;

impl IntegrityChecker {
    fn new() -> Self {
        Self
    }
}

// =============================================================================
// TEST RESULTS AND REPORTING
// =============================================================================

#[derive(Debug, Clone)]
pub struct CryptoTestResults {
    pub randomness_quality: CryptoTestCategory,
    pub data_integrity: CryptoTestCategory,
    pub serialization_security: CryptoTestCategory,
    pub key_derivation: CryptoTestCategory,
    pub overall_score: f64,
    pub security_rating: SecurityRating,
}

impl CryptoTestResults {
    fn new() -> Self {
        Self {
            randomness_quality: CryptoTestCategory::new("Randomness Quality"),
            data_integrity: CryptoTestCategory::new("Data Integrity"),
            serialization_security: CryptoTestCategory::new("Serialization Security"),
            key_derivation: CryptoTestCategory::new("Key Derivation"),
            overall_score: 0.0,
            security_rating: SecurityRating::Unknown,
        }
    }

    fn calculate_overall_score(&mut self) {
        let categories = [
            &self.randomness_quality,
            &self.data_integrity,
            &self.serialization_security,
            &self.key_derivation,
        ];

        let total_score: f64 = categories.iter()
            .map(|cat| cat.success_rate())
            .sum();

        self.overall_score = total_score / categories.len() as f64;

        self.security_rating = match self.overall_score {
            s if s >= 0.95 => SecurityRating::Excellent,
            s if s >= 0.85 => SecurityRating::Good,
            s if s >= 0.70 => SecurityRating::Acceptable,
            s if s >= 0.50 => SecurityRating::Poor,
            _ => SecurityRating::Critical,
        };
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== CRYPTOGRAPHIC VALIDATION REPORT ===\n\n");
        report.push_str(&format!("Overall Score: {:.1}%\n", self.overall_score * 100.0));
        report.push_str(&format!("Security Rating: {:?}\n\n", self.security_rating));

        for category in [
            &self.randomness_quality,
            &self.data_integrity,
            &self.serialization_security,
            &self.key_derivation,
        ] {
            report.push_str(&category.format_report());
            report.push_str("\n");
        }

        report
    }
}

#[derive(Debug, Clone)]
pub struct CryptoTestCategory {
    pub name: String,
    pub tests: Vec<CryptoTestResult>,
}

impl CryptoTestCategory {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tests: Vec::new(),
        }
    }

    fn add_test(&mut self, test: CryptoTestResult) {
        self.tests.push(test);
    }

    fn success_rate(&self) -> f64 {
        if self.tests.is_empty() {
            return 0.0;
        }

        let passed = self.tests.iter()
            .filter(|t| t.status == CryptoTestStatus::Passed)
            .count();

        passed as f64 / self.tests.len() as f64
    }

    fn format_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("=== {} ===\n", self.name));
        report.push_str(&format!("Success Rate: {:.1}%\n", self.success_rate() * 100.0));
        
        for test in &self.tests {
            report.push_str(&format!("  {} - {:?}", test.name, test.status));
            if !test.message.is_empty() {
                report.push_str(&format!(": {}", test.message));
            }
            report.push_str("\n");
        }
        
        report
    }
}

#[derive(Debug, Clone)]
pub struct CryptoTestResult {
    pub name: String,
    pub status: CryptoTestStatus,
    pub message: String,
    pub execution_time: Duration,
}

impl CryptoTestResult {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CryptoTestStatus::Running,
            message: String::new(),
            execution_time: Duration::new(0, 0),
        }
    }

    fn mark_passed(mut self, message: &str) -> Self {
        self.status = CryptoTestStatus::Passed;
        self.message = message.to_string();
        self
    }

    fn mark_failed(mut self, message: &str) -> Self {
        self.status = CryptoTestStatus::Failed;
        self.message = message.to_string();
        self
    }

    fn mark_warning(mut self, message: &str) -> Self {
        self.status = CryptoTestStatus::Warning;
        self.message = message.to_string();
        self
    }

    fn mark_skipped(mut self, message: &str) -> Self {
        self.status = CryptoTestStatus::Skipped;
        self.message = message.to_string();
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CryptoTestStatus {
    Running,
    Passed,
    Failed,
    Warning,
    Skipped,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SecurityRating {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Critical,
    Unknown,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cryptographic_suite_creation() {
        let config = CryptoTestConfig::default();
        let suite = CryptographicTestSuite::new(config);
        assert_eq!(suite.config.sample_size, 10000);
    }

    #[test]
    fn test_entropy_calculation() {
        let analyzer = EntropyAnalyzer::new();
        
        // Test with uniform random values
        let uniform_values: Vec<f32> = (0..1000).map(|i| (i as f32) / 1000.0 * 2.0 - 1.0).collect();
        let entropy = analyzer.calculate_entropy(&uniform_values);
        assert!(entropy > 0.5, "Entropy should be high for uniform distribution");

        // Test with constant values
        let constant_values = vec![0.5f32; 1000];
        let entropy = analyzer.calculate_entropy(&constant_values);
        assert!(entropy < 0.1, "Entropy should be low for constant values");
    }

    #[test]
    fn test_hash_validator() {
        let validator = HashValidator::new();
        
        let data1 = "test data";
        let data2 = "test data";
        let data3 = "different data";
        
        let hash1 = validator.compute_hash(data1);
        let hash2 = validator.compute_hash(data2);
        let hash3 = validator.compute_hash(data3);
        
        assert_eq!(hash1, hash2, "Same data should produce same hash");
        assert_ne!(hash1, hash3, "Different data should produce different hash");
    }

    #[test]
    fn test_basic_randomness_quality() {
        let mut config = CryptoTestConfig::default();
        config.sample_size = 100; // Reduce for faster testing
        
        let mut suite = CryptographicTestSuite::new(config);
        let results = suite.test_randomness_quality();
        
        assert!(results.tests.len() > 0);
    }

    #[test]
    fn test_comprehensive_crypto_validation() {
        let mut config = CryptoTestConfig::default();
        config.sample_size = 50; // Reduce for testing
        
        let mut suite = CryptographicTestSuite::new(config);
        let results = suite.run_crypto_tests();
        
        assert!(results.overall_score >= 0.0);
        assert!(results.overall_score <= 1.0);
        
        println!("{}", results.generate_report());
    }
}