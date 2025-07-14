# Neural Swarm FANN Test Suite

This directory contains comprehensive tests for the Neural Swarm FANN-compatible implementation, based on canonical FANN examples and extended with Rust safety and performance features.

## Test Structure

### Core Test Files

- **`xor_test.rs`** - Comprehensive XOR neural network tests based on canonical FANN XOR examples
- **`cascade_training_test.rs`** - Cascade training validation tests with architecture evolution
- **`integration_tests.rs`** - Full system integration tests validating FANN compatibility
- **`test_utils.rs`** - Common utilities and helpers for neural network testing

### Test Categories

#### Unit Tests
- Network creation and configuration
- Training data validation
- Forward pass correctness
- Basic training functionality
- Network serialization/deserialization

#### Integration Tests
- Full FANN compatibility validation
- Cross-platform compatibility
- Stress testing under load
- Error recovery and resilience
- Performance characteristics

#### Compatibility Tests
- FANN data format parsing
- Network parameter validation
- Expected behavior verification
- Cross-library consistency

## Test Data

### Standard Test Datasets

- **XOR Problem**: Classic non-linearly separable binary classification
- **Logic Functions**: AND, OR, NAND, NOR validation
- **Parity Problems**: N-bit parity for cascade training
- **Spiral Classification**: Complex non-linear boundary problem

### Data Formats

All test data is provided in FANN-compatible format:

```
num_patterns num_inputs num_outputs
input1 input2 ... inputN
output1 output2 ... outputM
input1 input2 ... inputN
output1 output2 ... outputM
...
```

Example XOR data (`data/xor.data`):
```
4 2 1
0 0
0
0 1
1
1 0
1
1 1
0
```

## Running Tests

### All Tests
```bash
cargo test
```

### Specific Test Categories
```bash
# XOR-specific tests
cargo test xor

# Cascade training tests
cargo test cascade

# Integration tests
cargo test integration

# Performance tests
cargo test performance
```

### Test Output Verbosity
```bash
# Detailed output
cargo test -- --nocapture

# Show performance metrics
cargo test performance -- --nocapture
```

## Test Features

### FANN Compatibility Validation

The test suite validates compatibility with canonical FANN behavior:

1. **XOR Learning**: Standard 2-input XOR problem solving
2. **Cascade Training**: Architecture evolution during training
3. **Data Format**: FANN training file format support
4. **Network Serialization**: Save/load network state
5. **Parameter Compatibility**: FANN-compatible configuration options

### Performance Testing

- **Training Speed**: Measure iterations per second
- **Inference Speed**: Forward pass performance
- **Memory Usage**: Resource consumption monitoring
- **Scalability**: Performance under load

### Error Handling

- **Invalid Data**: NaN, infinity, empty datasets
- **Network Errors**: Invalid architectures, parameters
- **Resource Limits**: Memory, computation constraints
- **Recovery**: Graceful error handling and recovery

## Test Utilities

### `FannDataParser`
- Parse FANN training data format
- Generate FANN-compatible data files
- Validate data consistency
- Scale data between ranges

### `NetworkValidator`
- Validate network architectures
- Check training parameters
- Calculate accuracy metrics
- Compute MSE and bit failures

### `TestDataGenerator`
- Generate standard test problems
- Create synthetic datasets
- Parameterized problem generation
- Deterministic test data

### `PerformanceProfiler`
- Time async operations
- Collect performance statistics
- Generate performance reports
- Compare execution times

## Expected Results

### XOR Problem
- **Accuracy**: >75% after sufficient training
- **Convergence**: MSE < 0.1 within reasonable iterations
- **Architecture**: 2 inputs, 1 output, 3-8 hidden neurons

### Cascade Training
- **Architecture Growth**: Network should add neurons for complex problems
- **Convergence**: Better performance with added complexity
- **Configuration**: Support for cascade-specific parameters

### Performance Benchmarks
- **Training**: <10 seconds for 10,000 XOR iterations
- **Inference**: <1ms for single forward pass
- **Memory**: <100MB for standard networks

## Test Configuration

### Mock Implementations

The test suite includes mock implementations for development and testing:

- `XorNetwork`: Basic feedforward network simulator
- `CascadeNetwork`: Cascade training simulator
- Deterministic outputs for reproducible testing
- Configurable behavior for different test scenarios

### Real Implementation Integration

When the actual FANN integration is implemented:

1. Replace mock implementations with real neural networks
2. Update trait implementations for actual FANN calls
3. Adjust test expectations based on real performance
4. Validate against original FANN library behavior

## Test Maintenance

### Adding New Tests

1. Follow existing patterns in test organization
2. Use `test_utils` for common functionality
3. Include both positive and negative test cases
4. Document expected behavior and edge cases

### Performance Baselines

- Update performance expectations as implementation improves
- Maintain backward compatibility with test interfaces
- Document performance regressions and improvements
- Use criterion benchmarks for detailed performance analysis

### Compatibility Updates

- Verify tests against new FANN versions
- Update test data formats as needed
- Maintain compatibility with existing test interfaces
- Document breaking changes and migration paths

## Contributing

When adding tests:

1. **Follow Test Structure**: Use existing patterns and utilities
2. **Document Behavior**: Clear descriptions of what is being tested
3. **Include Edge Cases**: Test boundary conditions and error cases
4. **Performance Aware**: Consider performance implications of tests
5. **FANN Compatible**: Ensure tests validate FANN compatibility

## References

- [FANN Library Documentation](http://leenissen.dk/fann/wp/)
- [FANN GitHub Repository](https://github.com/libfann/fann)
- [XOR Problem Analysis](https://en.wikipedia.org/wiki/XOR_gate)
- [Cascade Correlation Algorithm](https://en.wikipedia.org/wiki/Cascade_correlation)

---

This test suite ensures the Neural Swarm implementation maintains full compatibility with FANN while adding Rust safety and performance benefits.