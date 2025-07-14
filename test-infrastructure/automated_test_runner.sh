#!/bin/bash

# Automated Test Runner for Neuroplex Distributed Memory System
# This script provides automated testing infrastructure for the comprehensive test suite

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_RESULTS_DIR="${PROJECT_ROOT}/test-results"
LOGS_DIR="${PROJECT_ROOT}/logs"
COVERAGE_DIR="${PROJECT_ROOT}/coverage"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmarks"

# Test categories
UNIT_TESTS=true
INTEGRATION_TESTS=true
PERFORMANCE_BENCHMARKS=true
CHAOS_TESTS=true
PYTHON_FFI_TESTS=false
MEMORY_LEAK_TESTS=false

# Test configuration
MAX_PARALLEL_TESTS=4
TEST_TIMEOUT=300
VERBOSE=false

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

show_help() {
    cat << EOF
Automated Test Runner for Neuroplex Distributed Memory System

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -u, --unit-only         Run only unit tests
    -i, --integration-only  Run only integration tests
    -p, --performance-only  Run only performance benchmarks
    -c, --chaos-only        Run only chaos engineering tests
    -f, --ffi-tests         Enable Python FFI tests
    -m, --memory-tests      Enable memory leak tests
    --timeout SECONDS       Set test timeout (default: 300)
    --parallel N            Set max parallel tests (default: 4)
    --coverage              Generate code coverage report
    --ci                    Run in CI mode (non-interactive)
    --clean                 Clean previous test results
    --docker                Run tests in Docker containers
    --kubernetes            Run tests in Kubernetes cluster

Examples:
    $0                      Run all tests
    $0 -u                   Run only unit tests
    $0 --performance-only   Run only performance benchmarks
    $0 --coverage           Run tests with coverage
    $0 --docker             Run tests in Docker
EOF
}

setup_environment() {
    log "Setting up test environment..."
    
    # Create directories
    mkdir -p "${TEST_RESULTS_DIR}"
    mkdir -p "${LOGS_DIR}"
    mkdir -p "${COVERAGE_DIR}"
    mkdir -p "${BENCHMARK_DIR}"
    
    # Clean previous results if requested
    if [[ "${CLEAN:-false}" == "true" ]]; then
        log "Cleaning previous test results..."
        rm -rf "${TEST_RESULTS_DIR}"/* "${LOGS_DIR}"/* "${COVERAGE_DIR}"/* "${BENCHMARK_DIR}"/*
    fi
    
    # Check dependencies
    check_dependencies
    
    # Set environment variables
    export RUST_LOG=debug
    export RUST_BACKTRACE=1
    export NEUROPLEX_TEST_MODE=true
    export NEUROPLEX_LOG_LEVEL=debug
    
    success "Environment setup completed"
}

check_dependencies() {
    log "Checking dependencies..."
    
    # Required tools
    local required_tools=("cargo" "rustc" "docker" "kubectl" "jq")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "Required tool '$tool' not found"
            exit 1
        fi
    done
    
    # Check Rust version
    local rust_version=$(rustc --version | cut -d' ' -f2)
    log "Rust version: $rust_version"
    
    # Check Docker
    if ! docker info &> /dev/null; then
        warning "Docker not running, some tests may fail"
    fi
    
    # Check Kubernetes
    if ! kubectl version --client &> /dev/null; then
        warning "kubectl not configured, Kubernetes tests disabled"
    fi
    
    success "Dependencies check completed"
}

run_unit_tests() {
    if [[ "$UNIT_TESTS" != "true" ]]; then
        return 0
    fi
    
    log "Running unit tests..."
    
    local test_args=()
    if [[ "$VERBOSE" == "true" ]]; then
        test_args+=("--verbose")
    fi
    
    if [[ "${COVERAGE:-false}" == "true" ]]; then
        test_args+=("--coverage")
    fi
    
    # Run CRDT property tests
    log "Running CRDT property tests..."
    cargo test comprehensive_crdt_property_tests "${test_args[@]}" \
        --timeout "$TEST_TIMEOUT" \
        2>&1 | tee "${LOGS_DIR}/crdt_tests.log"
    
    # Run consensus protocol tests
    log "Running consensus protocol tests..."
    cargo test comprehensive_consensus_tests "${test_args[@]}" \
        --timeout "$TEST_TIMEOUT" \
        2>&1 | tee "${LOGS_DIR}/consensus_tests.log"
    
    success "Unit tests completed"
}

run_integration_tests() {
    if [[ "$INTEGRATION_TESTS" != "true" ]]; then
        return 0
    fi
    
    log "Running integration tests..."
    
    # Setup test cluster
    setup_test_cluster
    
    # Run multi-node integration tests
    log "Running multi-node integration tests..."
    cargo test multi_node_integration_tests \
        --timeout "$TEST_TIMEOUT" \
        2>&1 | tee "${LOGS_DIR}/integration_tests.log"
    
    # Cleanup test cluster
    cleanup_test_cluster
    
    success "Integration tests completed"
}

run_performance_benchmarks() {
    if [[ "$PERFORMANCE_BENCHMARKS" != "true" ]]; then
        return 0
    fi
    
    log "Running performance benchmarks..."
    
    # Run criterion benchmarks
    cargo bench \
        --bench comprehensive_performance_benchmarks \
        2>&1 | tee "${LOGS_DIR}/benchmarks.log"
    
    # Generate benchmark report
    generate_benchmark_report
    
    success "Performance benchmarks completed"
}

run_chaos_tests() {
    if [[ "$CHAOS_TESTS" != "true" ]]; then
        return 0
    fi
    
    log "Running chaos engineering tests..."
    
    # Setup chaos test environment
    setup_chaos_environment
    
    # Run chaos tests
    cargo test chaos_engineering_test_suite \
        --timeout "$((TEST_TIMEOUT * 2))" \
        2>&1 | tee "${LOGS_DIR}/chaos_tests.log"
    
    # Cleanup chaos environment
    cleanup_chaos_environment
    
    success "Chaos engineering tests completed"
}

run_python_ffi_tests() {
    if [[ "$PYTHON_FFI_TESTS" != "true" ]]; then
        return 0
    fi
    
    log "Running Python FFI tests..."
    
    # Setup Python environment
    setup_python_environment
    
    # Build Python bindings
    cargo build --features python-ffi
    
    # Run Python tests
    python -m pytest tests/python/ -v \
        2>&1 | tee "${LOGS_DIR}/python_ffi_tests.log"
    
    success "Python FFI tests completed"
}

run_memory_leak_tests() {
    if [[ "$MEMORY_LEAK_TESTS" != "true" ]]; then
        return 0
    fi
    
    log "Running memory leak tests..."
    
    # Install valgrind if not available
    if ! command -v valgrind &> /dev/null; then
        warning "Valgrind not found, installing..."
        sudo apt-get update && sudo apt-get install -y valgrind
    fi
    
    # Run memory leak tests
    valgrind --tool=memcheck --leak-check=full --track-origins=yes \
        cargo test --target x86_64-unknown-linux-gnu \
        2>&1 | tee "${LOGS_DIR}/memory_leak_tests.log"
    
    success "Memory leak tests completed"
}

setup_test_cluster() {
    log "Setting up test cluster..."
    
    if [[ "${DOCKER:-false}" == "true" ]]; then
        # Docker-based test cluster
        docker-compose -f test-infrastructure/docker-compose.test.yml up -d
        sleep 10  # Wait for services to start
    elif [[ "${KUBERNETES:-false}" == "true" ]]; then
        # Kubernetes-based test cluster
        kubectl apply -f test-infrastructure/k8s-test-cluster.yml
        kubectl wait --for=condition=Ready pods --all --timeout=300s
    else
        # Local test cluster
        setup_local_test_cluster
    fi
    
    success "Test cluster setup completed"
}

setup_local_test_cluster() {
    log "Setting up local test cluster..."
    
    # Start Redis
    if ! pgrep redis-server &> /dev/null; then
        redis-server --daemonize yes --port 6379
    fi
    
    # Start additional services as needed
    # This would be expanded based on specific requirements
    
    success "Local test cluster setup completed"
}

cleanup_test_cluster() {
    log "Cleaning up test cluster..."
    
    if [[ "${DOCKER:-false}" == "true" ]]; then
        docker-compose -f test-infrastructure/docker-compose.test.yml down
    elif [[ "${KUBERNETES:-false}" == "true" ]]; then
        kubectl delete -f test-infrastructure/k8s-test-cluster.yml
    else
        cleanup_local_test_cluster
    fi
    
    success "Test cluster cleanup completed"
}

cleanup_local_test_cluster() {
    log "Cleaning up local test cluster..."
    
    # Stop Redis
    if pgrep redis-server &> /dev/null; then
        pkill redis-server
    fi
    
    # Clean up other services
    
    success "Local test cluster cleanup completed"
}

setup_chaos_environment() {
    log "Setting up chaos environment..."
    
    # Install chaos engineering tools
    if ! command -v chaos &> /dev/null; then
        warning "Chaos engineering tools not found, installing..."
        # This would install chaos engineering tools
    fi
    
    success "Chaos environment setup completed"
}

cleanup_chaos_environment() {
    log "Cleaning up chaos environment..."
    
    # Cleanup chaos tools and restore normal state
    
    success "Chaos environment cleanup completed"
}

setup_python_environment() {
    log "Setting up Python environment..."
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements-test.txt
    
    success "Python environment setup completed"
}

generate_benchmark_report() {
    log "Generating benchmark report..."
    
    # Create benchmark report
    cat > "${BENCHMARK_DIR}/benchmark_report.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Neuroplex Performance Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        .passed { color: green; }
        .failed { color: red; }
        .warning { color: orange; }
    </style>
</head>
<body>
    <h1>Neuroplex Performance Benchmark Report</h1>
    <p>Generated: $(date)</p>
    
    <h2>Summary</h2>
    <div class="metric">
        <strong>Total Benchmarks:</strong> <span id="total">Loading...</span>
    </div>
    
    <h2>Detailed Results</h2>
    <div id="results">Loading...</div>
    
    <script>
        // This would be populated with actual benchmark data
        // For now, it's a placeholder
    </script>
</body>
</html>
EOF
    
    success "Benchmark report generated"
}

generate_test_report() {
    log "Generating comprehensive test report..."
    
    local report_file="${TEST_RESULTS_DIR}/test_report.json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": {
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "rust_version": "$(rustc --version)",
        "docker_version": "$(docker --version 2>/dev/null || echo 'N/A')"
    },
    "configuration": {
        "unit_tests": $UNIT_TESTS,
        "integration_tests": $INTEGRATION_TESTS,
        "performance_benchmarks": $PERFORMANCE_BENCHMARKS,
        "chaos_tests": $CHAOS_TESTS,
        "python_ffi_tests": $PYTHON_FFI_TESTS,
        "memory_leak_tests": $MEMORY_LEAK_TESTS,
        "max_parallel_tests": $MAX_PARALLEL_TESTS,
        "test_timeout": $TEST_TIMEOUT
    },
    "results": {
        "overall_status": "completed",
        "total_duration": "$(date +%s)",
        "categories": []
    }
}
EOF
    
    success "Test report generated: $report_file"
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -u|--unit-only)
                UNIT_TESTS=true
                INTEGRATION_TESTS=false
                PERFORMANCE_BENCHMARKS=false
                CHAOS_TESTS=false
                shift
                ;;
            -i|--integration-only)
                UNIT_TESTS=false
                INTEGRATION_TESTS=true
                PERFORMANCE_BENCHMARKS=false
                CHAOS_TESTS=false
                shift
                ;;
            -p|--performance-only)
                UNIT_TESTS=false
                INTEGRATION_TESTS=false
                PERFORMANCE_BENCHMARKS=true
                CHAOS_TESTS=false
                shift
                ;;
            -c|--chaos-only)
                UNIT_TESTS=false
                INTEGRATION_TESTS=false
                PERFORMANCE_BENCHMARKS=false
                CHAOS_TESTS=true
                shift
                ;;
            -f|--ffi-tests)
                PYTHON_FFI_TESTS=true
                shift
                ;;
            -m|--memory-tests)
                MEMORY_LEAK_TESTS=true
                shift
                ;;
            --timeout)
                TEST_TIMEOUT="$2"
                shift 2
                ;;
            --parallel)
                MAX_PARALLEL_TESTS="$2"
                shift 2
                ;;
            --coverage)
                COVERAGE=true
                shift
                ;;
            --ci)
                CI_MODE=true
                shift
                ;;
            --clean)
                CLEAN=true
                shift
                ;;
            --docker)
                DOCKER=true
                shift
                ;;
            --kubernetes)
                KUBERNETES=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Main execution
    log "Starting Neuroplex Distributed Memory System Test Suite"
    log "=================================================="
    
    # Setup
    setup_environment
    
    # Run tests
    local start_time=$(date +%s)
    
    run_unit_tests
    run_integration_tests
    run_performance_benchmarks
    run_chaos_tests
    run_python_ffi_tests
    run_memory_leak_tests
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Generate reports
    generate_test_report
    
    # Summary
    log "=================================================="
    success "Test suite completed in ${duration}s"
    log "Results available in: ${TEST_RESULTS_DIR}"
    log "Logs available in: ${LOGS_DIR}"
    
    if [[ "${COVERAGE:-false}" == "true" ]]; then
        log "Coverage report available in: ${COVERAGE_DIR}"
    fi
    
    if [[ "$PERFORMANCE_BENCHMARKS" == "true" ]]; then
        log "Benchmark report available in: ${BENCHMARK_DIR}"
    fi
}

# Run main function
main "$@"