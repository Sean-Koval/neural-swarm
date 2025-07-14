# Contributing to FANN-Rust-Core

We welcome contributions to FANN-Rust-Core! This document provides guidelines for contributing to the project, whether you're fixing bugs, adding features, or improving documentation.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contributing Process](#contributing-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Performance Benchmarks](#performance-benchmarks)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- **Rust**: 1.70.0 or later
- **Git**: For version control
- **Python**: 3.8+ (for Python bindings development)
- **C Compiler**: GCC or Clang (for C API compatibility)

### Development Dependencies

```bash
# Install development tools
cargo install cargo-criterion cargo-flamegraph cargo-watch
cargo install cargo-tarpaulin  # For code coverage

# Python development (optional)
pip install maturin pytest black isort mypy

# Documentation tools
cargo install mdbook
```

## Development Environment

### Clone and Setup

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/fann-rust-core
cd fann-rust-core

# Add upstream remote
git remote add upstream https://github.com/neural-swarm/fann-rust-core

# Create development branch
git checkout -b feature/your-feature-name
```

### Build and Test

```bash
# Build in debug mode
cargo build

# Run all tests
cargo test

# Run tests with coverage
cargo tarpaulin --out Html

# Run benchmarks
cargo bench

# Build documentation
cargo doc --open
```

### Development Workflow

```bash
# Start continuous testing
cargo watch -x test

# Format code
cargo fmt

# Check for issues
cargo clippy

# Build release version
cargo build --release
```

## Contributing Process

### 1. Issue First

Before starting work:

1. Check [existing issues](https://github.com/neural-swarm/fann-rust-core/issues)
2. Create a new issue if needed
3. Discuss the approach in the issue
4. Get approval for significant changes

### 2. Branch Naming

Use descriptive branch names:

```bash
feature/add-gpu-acceleration
bugfix/fix-simd-alignment
docs/improve-getting-started
refactor/optimize-memory-layout
test/add-edge-deployment-tests
```

### 3. Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Detailed explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `perf`: Performance improvements

**Examples:**
```
feat(quantization): add mixed-precision quantization support

Implements dynamic quantization with sensitivity analysis for optimal
precision selection per layer. Includes benchmarks showing 4-10x
compression with <1% accuracy loss.

Fixes #45
```

### 4. Pull Request Process

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure all checks pass**:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

3. **Create pull request** with:
   - Clear title and description
   - Link to related issues
   - List of changes
   - Testing information
   - Performance impact (if applicable)

4. **Address review feedback**
5. **Ensure CI passes**
6. **Wait for approval and merge**

## Code Style

### Rust Code Style

We follow the Rust standard style with some additions:

```rust
// Use explicit types for public APIs
pub fn create_network(layers: &[usize]) -> Result<Network, NetworkError> {
    // Implementation
}

// Document all public functions
/// Creates a neural network with the specified layer sizes.
///
/// # Arguments
/// * `layers` - Slice containing the size of each layer
///
/// # Returns
/// * `Result<Network, NetworkError>` - The created network or an error
///
/// # Examples
/// ```
/// let network = create_network(&[784, 128, 10])?;
/// ```
pub fn create_network(layers: &[usize]) -> Result<Network, NetworkError> {
    // Implementation
}

// Use descriptive error types
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("Invalid architecture: {0}")]
    InvalidArchitecture(String),
    #[error("Training failed: {0}")]
    TrainingError(String),
}

// Performance-critical code should be well-documented
/// SIMD-optimized matrix multiplication using AVX2 instructions.
/// 
/// # Safety
/// This function uses unsafe code for SIMD operations. Ensure:
/// - Input slices have correct alignment
/// - Vector lengths are multiples of 8
/// - AVX2 feature is available on target CPU
#[target_feature(enable = "avx2")]
unsafe fn matrix_multiply_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    // Implementation with safety comments
}
```

### Code Organization

```
src/
├── lib.rs              # Public API exports
├── prelude.rs          # Common imports
├── core/               # Core neural network functionality
│   ├── mod.rs
│   ├── network.rs      # Network implementations
│   ├── layer.rs        # Layer types
│   └── activation.rs   # Activation functions
├── training/           # Training algorithms
├── optimization/       # SIMD, quantization, etc.
├── serialization/      # Model persistence
├── integration/        # Swarm and MCP integration
├── utils/              # Utilities and helpers
└── error.rs           # Error types
```

### Documentation Standards

```rust
//! Module-level documentation
//! 
//! This module provides neural network training algorithms including
//! backpropagation, RPROP, and genetic algorithms.

/// Struct documentation with examples
/// 
/// # Examples
/// ```
/// use fann_rust_core::NetworkBuilder;
/// 
/// let network = NetworkBuilder::new()
///     .layers(&[784, 128, 10])
///     .build()?;
/// ```
pub struct NetworkBuilder {
    // Fields should be documented if public
    /// Layer configuration for the network
    pub layers: Vec<LayerConfig>,
}

impl NetworkBuilder {
    /// Method documentation
    /// 
    /// # Arguments
    /// * `layers` - Layer sizes
    /// 
    /// # Errors
    /// Returns `NetworkError::InvalidArchitecture` if layers is empty
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = layers.iter()
            .map(|&size| LayerConfig::new(size))
            .collect();
        self
    }
}
```

## Testing

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical paths
4. **Property Tests**: Use `proptest` for property-based testing

### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_network_creation() {
        let network = NetworkBuilder::new()
            .layers(&[2, 3, 1])
            .build()
            .unwrap();
        
        assert_eq!(network.input_size(), 2);
        assert_eq!(network.output_size(), 1);
    }

    #[test]
    fn test_invalid_architecture() {
        let result = NetworkBuilder::new()
            .layers(&[])  // Empty layers should fail
            .build();
        
        assert!(result.is_err());
        match result.unwrap_err() {
            NetworkError::InvalidArchitecture(_) => {},
            _ => panic!("Wrong error type"),
        }
    }

    // Property-based test
    proptest! {
        #[test]
        fn test_forward_pass_output_size(
            input_size in 1..1000usize,
            output_size in 1..100usize,
            input in prop::collection::vec(-10.0f32..10.0, 1..1000)
        ) {
            let network = NetworkBuilder::new()
                .layers(&[input_size, output_size])
                .build()
                .unwrap();
            
            let input_trimmed = input.into_iter().take(input_size).collect::<Vec<_>>();
            let output = network.forward(&input_trimmed).unwrap();
            
            prop_assert_eq!(output.len(), output_size);
        }
    }
}

// Integration tests in tests/ directory
#[test]
fn test_mnist_training_integration() {
    let (train_data, test_data) = load_test_data();
    
    let mut network = NetworkBuilder::new()
        .layers(&[784, 128, 10])
        .build()
        .unwrap();
    
    let results = network.train(&train_data, TrainingConfig::default()).unwrap();
    
    // Should achieve reasonable accuracy on test data
    let accuracy = evaluate_network(&network, &test_data);
    assert!(accuracy > 0.8, "Accuracy too low: {}", accuracy);
}
```

### Performance Testing

```rust
// Benchmark critical functions
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_matrix_multiply(c: &mut Criterion) {
    let a = vec![1.0f32; 1000];
    let b = vec![1.0f32; 1000];
    let mut c_out = vec![0.0f32; 1000];
    
    c.bench_function("matrix_multiply_simd", |bencher| {
        bencher.iter(|| {
            matrix_multiply_simd(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_out)
            )
        })
    });
}

criterion_group!(benches, benchmark_matrix_multiply);
criterion_main!(benches);
```

### Test Data

```rust
// Create deterministic test data
pub fn create_test_network() -> Network {
    let mut network = NetworkBuilder::new()
        .layers(&[2, 3, 1])
        .seed(42)  // Deterministic initialization
        .build()
        .unwrap();
    
    // Set known weights for reproducible tests
    let weights = vec![0.1, 0.2, 0.3, /* ... */];
    network.set_weights(&weights).unwrap();
    
    network
}

// Use small datasets for fast tests
pub fn load_test_data() -> (TrainingData, TrainingData) {
    let train_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let train_targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];
    
    let train_data = TrainingData::new(train_inputs, train_targets).unwrap();
    let test_data = train_data.clone(); // Small test for speed
    
    (train_data, test_data)
}
```

## Performance Benchmarks

### Benchmark Requirements

All performance-critical changes must include benchmarks:

1. **Before/After Comparisons**: Show performance impact
2. **Cross-Platform**: Test on different architectures
3. **Regression Prevention**: Ensure no performance degradation
4. **Real-World Scenarios**: Use realistic workloads

### Benchmark Structure

```rust
// benchmarks/neural_operations.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use fann_rust_core::*;

fn benchmark_inference_batch_sizes(c: &mut Criterion) {
    let network = create_benchmark_network();
    
    let mut group = c.benchmark_group("inference_scaling");
    
    for batch_size in [1, 10, 100, 1000].iter() {
        let inputs = create_random_inputs(*batch_size, 784);
        
        group.bench_with_input(
            BenchmarkId::new("batch_inference", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    network.predict_batch(black_box(&inputs))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_comparison");
    
    let input = vec![1.0f32; 1024];
    
    group.bench_function("simd_enabled", |b| {
        let network = create_network_with_simd(true);
        b.iter(|| network.forward(black_box(&input)))
    });
    
    group.bench_function("simd_disabled", |b| {
        let network = create_network_with_simd(false);
        b.iter(|| network.forward(black_box(&input)))
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_inference_batch_sizes, benchmark_simd_vs_scalar);
criterion_main!(benches);
```

### Performance Targets

Maintain these performance characteristics:

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Forward Pass (MNIST) | <1ms | 784->128->10 network |
| Training Epoch | <10s | MNIST, 60k samples |
| SIMD Speedup | >2x | vs scalar implementation |
| Memory Usage | <100MB | Large networks |
| Quantization | <5% accuracy loss | Int8 quantization |

## Documentation

### Documentation Types

1. **API Documentation**: Rust docs with examples
2. **User Guides**: Getting started, tutorials
3. **Design Documents**: Architecture decisions
4. **Examples**: Working code samples

### Writing Guidelines

```rust
/// Brief one-line description.
///
/// Longer description explaining the purpose, behavior, and any important
/// details about the function or type.
///
/// # Arguments
/// * `param1` - Description of first parameter
/// * `param2` - Description of second parameter
///
/// # Returns
/// Description of return value and what it represents.
///
/// # Errors
/// Describe when and why this function might return an error.
///
/// # Examples
/// ```
/// use fann_rust_core::NetworkBuilder;
/// 
/// let network = NetworkBuilder::new()
///     .layers(&[784, 128, 10])
///     .build()?;
/// 
/// let output = network.forward(&input_data)?;
/// assert_eq!(output.len(), 10);
/// ```
///
/// # Panics
/// Describe any conditions that would cause a panic.
///
/// # Safety
/// If using unsafe code, explain safety requirements.
pub fn example_function(param1: &[f32], param2: usize) -> Result<Vec<f32>, Error> {
    // Implementation
}
```

### Documentation Testing

```bash
# Test all documentation examples
cargo test --doc

# Build and check documentation
cargo doc --no-deps --open

# Validate markdown links
mdbook test docs/
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive
- Help others learn and contribute
- Focus on what's best for the community
- Show empathy towards other contributors

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcase
- **Pull Requests**: Code review and discussion

### Getting Help

If you need help:

1. Check existing documentation
2. Search issues and discussions
3. Ask questions in GitHub discussions
4. Reach out to maintainers if needed

### Recognition

Contributors are recognized through:

- Author credit in release notes
- Contributor list in README
- Special recognition for significant contributions

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. Update CHANGELOG.md
2. Bump version numbers
3. Update documentation
4. Run full test suite
5. Create release PR
6. Tag release
7. Publish to crates.io
8. Update GitHub release

## Development Tips

### Performance Development

```bash
# Profile hot paths
cargo flamegraph --bin your_benchmark

# Check assembly output
cargo rustc --release -- --emit asm

# Memory profiling with valgrind
cargo build --release
valgrind --tool=massif target/release/your_app
```

### Debugging

```rust
// Use debug assertions for expensive checks
debug_assert!(input.len() == expected_size);

// Add tracing for complex algorithms
use tracing::{debug, info, span, Level};

let span = span!(Level::DEBUG, "matrix_multiply", 
                 rows = m, cols = n);
let _enter = span.enter();

debug!("Starting matrix multiplication");
// ... implementation
info!("Matrix multiplication completed in {:?}", elapsed);
```

### Testing on Different Platforms

```bash
# Cross-compilation testing
cargo build --target x86_64-pc-windows-gnu
cargo build --target aarch64-unknown-linux-gnu

# WASM testing
cargo build --target wasm32-unknown-unknown
```

Thank you for contributing to FANN-Rust-Core! Your contributions help make neural computing faster, safer, and more accessible.