# Makefile for Neuroplex Distributed Memory System Testing Infrastructure

# Variables
PROJECT_NAME := neuroplex
RUST_VERSION := 1.70.0
DOCKER_REGISTRY := localhost:5000
TEST_TIMEOUT := 300
MAX_PARALLEL_TESTS := 4

# Directories
BUILD_DIR := target
TEST_RESULTS_DIR := test-results
LOGS_DIR := logs
COVERAGE_DIR := coverage
BENCHMARK_DIR := benchmarks

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.PHONY: all
all: clean build test

# Help target
.PHONY: help
help:
	@echo "$(BLUE)Neuroplex Distributed Memory System - Test Infrastructure$(NC)"
	@echo "=========================================================="
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  all               - Clean, build, and test"
	@echo "  help              - Show this help message"
	@echo "  clean             - Clean build artifacts and test results"
	@echo "  build             - Build the project"
	@echo "  test              - Run all tests"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-performance  - Run performance benchmarks only"
	@echo "  test-chaos        - Run chaos engineering tests only"
	@echo "  test-python       - Run Python FFI tests"
	@echo "  test-memory       - Run memory leak tests"
	@echo "  coverage          - Generate code coverage report"
	@echo "  benchmark         - Run performance benchmarks"
	@echo "  docker-build      - Build Docker images"
	@echo "  docker-test       - Run tests in Docker"
	@echo "  k8s-test          - Run tests in Kubernetes"
	@echo "  setup             - Setup development environment"
	@echo "  deps              - Install dependencies"
	@echo "  format            - Format code"
	@echo "  lint              - Run linting"
	@echo "  security-audit    - Run security audit"
	@echo "  docs              - Generate documentation"
	@echo "  release           - Build release version"

# Clean targets
.PHONY: clean
clean:
	@echo "$(BLUE)Cleaning build artifacts and test results...$(NC)"
	cargo clean
	rm -rf $(TEST_RESULTS_DIR)
	rm -rf $(LOGS_DIR)
	rm -rf $(COVERAGE_DIR)
	rm -rf $(BENCHMARK_DIR)
	docker system prune -f
	@echo "$(GREEN)Clean completed$(NC)"

.PHONY: clean-all
clean-all: clean
	@echo "$(BLUE)Cleaning all artifacts including dependencies...$(NC)"
	rm -rf target/
	rm -rf venv/
	rm -rf node_modules/
	docker system prune -af
	@echo "$(GREEN)Deep clean completed$(NC)"

# Build targets
.PHONY: build
build: setup-dirs
	@echo "$(BLUE)Building project...$(NC)"
	cargo build --release
	@echo "$(GREEN)Build completed$(NC)"

.PHONY: build-debug
build-debug: setup-dirs
	@echo "$(BLUE)Building project in debug mode...$(NC)"
	cargo build
	@echo "$(GREEN)Debug build completed$(NC)"

.PHONY: build-python
build-python: setup-dirs
	@echo "$(BLUE)Building Python bindings...$(NC)"
	cargo build --features python-ffi
	@echo "$(GREEN)Python bindings built$(NC)"

# Test targets
.PHONY: test
test: build
	@echo "$(BLUE)Running comprehensive test suite...$(NC)"
	./test-infrastructure/automated_test_runner.sh
	@echo "$(GREEN)All tests completed$(NC)"

.PHONY: test-unit
test-unit: build
	@echo "$(BLUE)Running unit tests...$(NC)"
	./test-infrastructure/automated_test_runner.sh --unit-only
	@echo "$(GREEN)Unit tests completed$(NC)"

.PHONY: test-integration
test-integration: build
	@echo "$(BLUE)Running integration tests...$(NC)"
	./test-infrastructure/automated_test_runner.sh --integration-only
	@echo "$(GREEN)Integration tests completed$(NC)"

.PHONY: test-performance
test-performance: build
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	./test-infrastructure/automated_test_runner.sh --performance-only
	@echo "$(GREEN)Performance benchmarks completed$(NC)"

.PHONY: test-chaos
test-chaos: build
	@echo "$(BLUE)Running chaos engineering tests...$(NC)"
	./test-infrastructure/automated_test_runner.sh --chaos-only
	@echo "$(GREEN)Chaos engineering tests completed$(NC)"

.PHONY: test-python
test-python: build-python
	@echo "$(BLUE)Running Python FFI tests...$(NC)"
	./test-infrastructure/automated_test_runner.sh --ffi-tests
	@echo "$(GREEN)Python FFI tests completed$(NC)"

.PHONY: test-memory
test-memory: build-debug
	@echo "$(BLUE)Running memory leak tests...$(NC)"
	./test-infrastructure/automated_test_runner.sh --memory-tests
	@echo "$(GREEN)Memory leak tests completed$(NC)"

.PHONY: test-ci
test-ci: build
	@echo "$(BLUE)Running CI test suite...$(NC)"
	./test-infrastructure/automated_test_runner.sh --ci --coverage
	@echo "$(GREEN)CI tests completed$(NC)"

# Coverage targets
.PHONY: coverage
coverage: build
	@echo "$(BLUE)Generating code coverage report...$(NC)"
	./test-infrastructure/automated_test_runner.sh --coverage
	@echo "$(GREEN)Coverage report generated$(NC)"

.PHONY: coverage-html
coverage-html: coverage
	@echo "$(BLUE)Generating HTML coverage report...$(NC)"
	cargo tarpaulin --out Html --output-dir $(COVERAGE_DIR)
	@echo "$(GREEN)HTML coverage report generated in $(COVERAGE_DIR)$(NC)"

# Benchmark targets
.PHONY: benchmark
benchmark: build
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	cargo bench
	@echo "$(GREEN)Benchmarks completed$(NC)"

.PHONY: benchmark-baseline
benchmark-baseline: build
	@echo "$(BLUE)Setting benchmark baseline...$(NC)"
	cargo bench -- --save-baseline main
	@echo "$(GREEN)Benchmark baseline set$(NC)"

.PHONY: benchmark-compare
benchmark-compare: build
	@echo "$(BLUE)Comparing benchmarks with baseline...$(NC)"
	cargo bench -- --baseline main
	@echo "$(GREEN)Benchmark comparison completed$(NC)"

# Docker targets
.PHONY: docker-build
docker-build:
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker build -t $(PROJECT_NAME):latest .
	docker build -t $(PROJECT_NAME)-test:latest -f test-infrastructure/Dockerfile.test-coordinator .
	docker build -t $(PROJECT_NAME)-node:latest -f test-infrastructure/Dockerfile.neuroplex-node .
	docker build -t $(PROJECT_NAME)-load-tester:latest -f test-infrastructure/Dockerfile.load-tester .
	docker build -t $(PROJECT_NAME)-chaos:latest -f test-infrastructure/Dockerfile.chaos-monkey .
	@echo "$(GREEN)Docker images built$(NC)"

.PHONY: docker-test
docker-test: docker-build
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	./test-infrastructure/automated_test_runner.sh --docker
	@echo "$(GREEN)Docker tests completed$(NC)"

.PHONY: docker-push
docker-push: docker-build
	@echo "$(BLUE)Pushing Docker images...$(NC)"
	docker tag $(PROJECT_NAME):latest $(DOCKER_REGISTRY)/$(PROJECT_NAME):latest
	docker push $(DOCKER_REGISTRY)/$(PROJECT_NAME):latest
	@echo "$(GREEN)Docker images pushed$(NC)"

# Kubernetes targets
.PHONY: k8s-test
k8s-test: docker-build
	@echo "$(BLUE)Running tests in Kubernetes...$(NC)"
	./test-infrastructure/automated_test_runner.sh --kubernetes
	@echo "$(GREEN)Kubernetes tests completed$(NC)"

.PHONY: k8s-deploy
k8s-deploy: docker-push
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	kubectl apply -f test-infrastructure/k8s-test-cluster.yml
	kubectl wait --for=condition=Ready pods --all --timeout=300s
	@echo "$(GREEN)Kubernetes deployment completed$(NC)"

.PHONY: k8s-cleanup
k8s-cleanup:
	@echo "$(BLUE)Cleaning up Kubernetes resources...$(NC)"
	kubectl delete -f test-infrastructure/k8s-test-cluster.yml
	@echo "$(GREEN)Kubernetes cleanup completed$(NC)"

# Setup and dependency targets
.PHONY: setup
setup: setup-dirs deps
	@echo "$(BLUE)Setting up development environment...$(NC)"
	chmod +x test-infrastructure/automated_test_runner.sh
	@echo "$(GREEN)Development environment setup completed$(NC)"

.PHONY: setup-dirs
setup-dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(TEST_RESULTS_DIR)
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(COVERAGE_DIR)
	@mkdir -p $(BENCHMARK_DIR)

.PHONY: deps
deps:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	cargo install cargo-tarpaulin
	cargo install criterion
	cargo install cargo-audit
	cargo install cargo-outdated
	cargo install cargo-expand
	@echo "$(GREEN)Dependencies installed$(NC)"

.PHONY: deps-python
deps-python:
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	python -m venv venv
	source venv/bin/activate && pip install -r requirements-test.txt
	@echo "$(GREEN)Python dependencies installed$(NC)"

# Code quality targets
.PHONY: format
format:
	@echo "$(BLUE)Formatting code...$(NC)"
	cargo fmt --all
	@echo "$(GREEN)Code formatting completed$(NC)"

.PHONY: lint
lint:
	@echo "$(BLUE)Running linting...$(NC)"
	cargo clippy --all-targets --all-features -- -D warnings
	@echo "$(GREEN)Linting completed$(NC)"

.PHONY: security-audit
security-audit:
	@echo "$(BLUE)Running security audit...$(NC)"
	cargo audit
	@echo "$(GREEN)Security audit completed$(NC)"

.PHONY: outdated
outdated:
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	cargo outdated
	@echo "$(GREEN)Outdated dependencies check completed$(NC)"

# Documentation targets
.PHONY: docs
docs:
	@echo "$(BLUE)Generating documentation...$(NC)"
	cargo doc --all-features --no-deps
	@echo "$(GREEN)Documentation generated$(NC)"

.PHONY: docs-open
docs-open: docs
	@echo "$(BLUE)Opening documentation...$(NC)"
	cargo doc --all-features --no-deps --open

# Release targets
.PHONY: release
release: clean security-audit lint test
	@echo "$(BLUE)Building release version...$(NC)"
	cargo build --release
	@echo "$(GREEN)Release build completed$(NC)"

.PHONY: release-patch
release-patch: release
	@echo "$(BLUE)Creating patch release...$(NC)"
	cargo release patch
	@echo "$(GREEN)Patch release completed$(NC)"

.PHONY: release-minor
release-minor: release
	@echo "$(BLUE)Creating minor release...$(NC)"
	cargo release minor
	@echo "$(GREEN)Minor release completed$(NC)"

.PHONY: release-major
release-major: release
	@echo "$(BLUE)Creating major release...$(NC)"
	cargo release major
	@echo "$(GREEN)Major release completed$(NC)"

# Utility targets
.PHONY: watch
watch:
	@echo "$(BLUE)Starting file watcher...$(NC)"
	cargo watch -x "test"

.PHONY: expand
expand:
	@echo "$(BLUE)Expanding macros...$(NC)"
	cargo expand

.PHONY: size
size:
	@echo "$(BLUE)Analyzing binary size...$(NC)"
	cargo bloat --release

.PHONY: deps-graph
deps-graph:
	@echo "$(BLUE)Generating dependency graph...$(NC)"
	cargo deps --all-deps | dot -Tpng > deps-graph.png
	@echo "$(GREEN)Dependency graph generated: deps-graph.png$(NC)"

# Performance monitoring targets
.PHONY: perf-record
perf-record:
	@echo "$(BLUE)Recording performance profile...$(NC)"
	cargo build --release
	perf record --call-graph=dwarf target/release/neuroplex
	@echo "$(GREEN)Performance profile recorded$(NC)"

.PHONY: perf-report
perf-report:
	@echo "$(BLUE)Generating performance report...$(NC)"
	perf report
	@echo "$(GREEN)Performance report generated$(NC)"

# Continuous integration targets
.PHONY: ci-setup
ci-setup:
	@echo "$(BLUE)Setting up CI environment...$(NC)"
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	source ~/.cargo/env
	rustup component add rustfmt clippy
	@echo "$(GREEN)CI environment setup completed$(NC)"

.PHONY: ci-test
ci-test: ci-setup lint security-audit test-ci
	@echo "$(GREEN)CI pipeline completed successfully$(NC)"

# Database targets
.PHONY: db-setup
db-setup:
	@echo "$(BLUE)Setting up test database...$(NC)"
	docker run -d --name postgres-test -p 5432:5432 -e POSTGRES_DB=neuroplex_test -e POSTGRES_USER=neuroplex -e POSTGRES_PASSWORD=test_password postgres:15-alpine
	@echo "$(GREEN)Test database setup completed$(NC)"

.PHONY: db-cleanup
db-cleanup:
	@echo "$(BLUE)Cleaning up test database...$(NC)"
	docker stop postgres-test || true
	docker rm postgres-test || true
	@echo "$(GREEN)Test database cleanup completed$(NC)"

# Monitoring targets
.PHONY: monitor-setup
monitor-setup:
	@echo "$(BLUE)Setting up monitoring stack...$(NC)"
	docker-compose -f test-infrastructure/docker-compose.monitoring.yml up -d
	@echo "$(GREEN)Monitoring stack setup completed$(NC)"

.PHONY: monitor-cleanup
monitor-cleanup:
	@echo "$(BLUE)Cleaning up monitoring stack...$(NC)"
	docker-compose -f test-infrastructure/docker-compose.monitoring.yml down
	@echo "$(GREEN)Monitoring stack cleanup completed$(NC)"

# Status and information targets
.PHONY: status
status:
	@echo "$(BLUE)Project Status$(NC)"
	@echo "=============="
	@echo "Rust version: $(shell rustc --version)"
	@echo "Cargo version: $(shell cargo --version)"
	@echo "Docker version: $(shell docker --version)"
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Test results: $(TEST_RESULTS_DIR)"
	@echo "Logs: $(LOGS_DIR)"
	@echo "Coverage: $(COVERAGE_DIR)"
	@echo "Benchmarks: $(BENCHMARK_DIR)"

.PHONY: info
info:
	@echo "$(BLUE)Project Information$(NC)"
	@echo "==================="
	@echo "Project: $(PROJECT_NAME)"
	@echo "Rust version: $(RUST_VERSION)"
	@echo "Docker registry: $(DOCKER_REGISTRY)"
	@echo "Test timeout: $(TEST_TIMEOUT)s"
	@echo "Max parallel tests: $(MAX_PARALLEL_TESTS)"