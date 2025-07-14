# Neural-Swarm Hybrid Deployment Framework

A comprehensive deployment framework for the neural-swarm coordination system, supporting automatic deployment decision-making between container and WebAssembly (WASM) targets based on resource constraints and deployment requirements.

## 🚀 Features

### **Container Deployment**
- **Production-ready Dockerfiles** with multi-stage builds
- **Kubernetes manifests** with HPA (Horizontal Pod Autoscaler)
- **Helm charts** for easy deployment management
- **Docker Compose** for local development
- **Resource optimization** based on deployment environment

### **WASM Deployment**
- **Edge-optimized WASM** compilation targets
- **Resource-constrained** environment support
- **Power-aware** coordination protocols
- **Host interface APIs** for system integration
- **Minimal footprint** for IoT and mobile devices

### **Hybrid Deployment Framework**
- **Automatic target selection** based on resource analysis
- **Seamless migration** between container and WASM
- **Unified deployment API** across all targets
- **Real-time monitoring** and adaptive scaling
- **Decision engine** with machine learning insights

## 📁 Directory Structure

```
deployment/
├── docker/                    # Container deployment
│   ├── Dockerfile            # Production container
│   ├── Dockerfile.edge       # Edge-optimized container
│   └── docker-compose.yml    # Multi-service setup
├── kubernetes/               # Kubernetes deployment
│   ├── namespace.yaml        # Namespace definition
│   ├── configmap.yaml        # Configuration management
│   ├── deployment.yaml       # Deployment specs
│   ├── service.yaml          # Service definitions
│   └── hpa.yaml             # Auto-scaling configuration
├── wasm/                     # WASM deployment
│   ├── Cargo.toml           # WASM build configuration
│   ├── src/lib.rs           # WASM runtime implementation
│   └── build.sh             # WASM build script
├── hybrid/                   # Hybrid deployment framework
│   ├── src/lib.rs           # Core framework
│   ├── src/bin/deployer.rs  # CLI deployment tool
│   └── src/bin/controller.rs # Deployment controller
└── README.md                # This file
```

## 🛠️ Quick Start

### 1. Container Deployment

#### Local Development with Docker Compose
```bash
cd deployment/docker
docker-compose up -d
```

#### Production Container Build
```bash
# Build production image
docker build -f deployment/docker/Dockerfile -t neural-swarm/neuroplex:latest .

# Build edge-optimized image
docker build -f deployment/docker/Dockerfile.edge -t neural-swarm/neuroplex:edge .

# Run container
docker run -p 8080:8080 -p 8081:8081 neural-swarm/neuroplex:latest
```

### 2. Kubernetes Deployment

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Deploy configuration
kubectl apply -f deployment/kubernetes/configmap.yaml

# Deploy services
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/hpa.yaml

# Check status
kubectl get pods -n neural-swarm
kubectl get services -n neural-swarm
```

### 3. WASM Deployment

#### Build WASM Runtime
```bash
cd deployment/wasm
./build.sh
```

#### Use in Browser
```html
<script src="dist/neuroplex-swarm.js"></script>
<script>
async function deployWasm() {
    const runtime = new NeuroSwarmRuntime();
    await runtime.init({
        edge_mode: true,
        power_aware: true,
        memory_limit: 64 * 1024 * 1024 // 64MB
    });
    
    await runtime.deployNode('node-1');
    console.log('WASM node deployed');
}
deployWasm();
</script>
```

#### Use in Node.js
```javascript
const NeuroSwarmRuntime = require('./dist/neuroplex-swarm.js');

async function main() {
    const runtime = new NeuroSwarmRuntime();
    await runtime.init();
    
    await runtime.deployNode('node-1');
    const status = await runtime.getStatus();
    console.log('Status:', status);
}
main();
```

### 4. Hybrid Deployment with CLI

#### Install CLI Tool
```bash
cd deployment/hybrid
cargo build --release
cp target/release/neural-swarm-deployer /usr/local/bin/
```

#### Generate Environment Configuration
```bash
neural-swarm-deployer generate environment ./config
```

Edit `config/environment.toml`:
```toml
environment_type = "Edge"
[resources]
memory_limit = 134217728  # 128MB
cpu_limit = 1.0
power_budget = 5.0
```

#### Deploy with Automatic Target Selection
```bash
# Analyze environment
neural-swarm-deployer analyze -e config/environment.toml

# Deploy with automatic target selection
neural-swarm-deployer deploy -i my-deployment -e config/environment.toml

# Check status
neural-swarm-deployer status my-deployment

# List all deployments
neural-swarm-deployer list --detailed
```

## 🔧 Configuration

### Environment Configuration

The hybrid deployment framework uses environment configuration to make deployment decisions:

```toml
# Environment type affects deployment decision
environment_type = "Edge"  # Cloud, Edge, Mobile, Embedded, Kubernetes

[resources]
memory_limit = 134217728      # 128MB - affects container vs WASM decision
cpu_limit = 1.0              # CPU cores available
network_bandwidth = 10000000  # 10Mbps
power_budget = 5.0           # 5W - enables power-aware optimizations
storage_limit = 1073741824   # 1GB
startup_time_limit = 1000    # 1 second - affects WASM preference

[security]
security_level = "Medium"    # Low, Medium, High, Critical

[performance]
latency_requirement = 100    # milliseconds
throughput_requirement = 1000 # requests per second
availability_requirement = 0.95 # 95%
scalability_requirement = 10   # max instances

[operations]
auto_scaling = false
monitoring_level = "Standard"
backup_required = false
compliance_requirements = []
```

### Deployment Decision Logic

The framework uses weighted rules to make deployment decisions:

| Condition | Container Score | WASM Score | Hybrid Score |
|-----------|----------------|------------|--------------|
| Memory > 2GB | High | Low | Medium |
| Memory < 128MB | Low | High | Medium |
| Edge Environment | Low | High | Medium |
| Power Budget < 5W | Low | High | Medium |
| Kubernetes Environment | High | Low | Medium |
| High Availability Required | Medium | Low | High |
| Auto-scaling Enabled | Medium | Low | High |

## 🔍 Monitoring and Migration

### Automatic Migration Triggers

The deployment controller monitors these metrics and triggers migrations:

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU Utilization | > 80% | WASM → Container |
| Memory Utilization | > 80% | WASM → Container |
| Power Usage | > 8W | Container → WASM |
| Network Latency | > 1000ms | Container → WASM |
| Error Rate | > 10% | Any → Hybrid |
| Response Time | > 5000ms | WASM → Container |

### Start Deployment Controller

```bash
# Start controller for automatic management
deployment-controller

# Controller monitors deployments and triggers migrations
# based on resource usage and performance metrics
```

## 📊 Performance Characteristics

### Container Deployment
- **Memory Usage**: 512MB - 2GB
- **Startup Time**: 5-30 seconds
- **CPU Efficiency**: High for complex workloads
- **Network Throughput**: Excellent
- **Best For**: Production, high-performance, scalable deployments

### WASM Deployment
- **Memory Usage**: 32MB - 256MB
- **Startup Time**: 100ms - 1 second
- **CPU Efficiency**: Good for lightweight workloads
- **Network Throughput**: Good with optimization
- **Best For**: Edge, mobile, IoT, resource-constrained environments

### Hybrid Deployment
- **Memory Usage**: Adaptive based on load
- **Startup Time**: Fast failover between targets
- **CPU Efficiency**: Optimal for variable workloads
- **Network Throughput**: Load-balanced across targets
- **Best For**: High availability, variable load, mixed environments

## 🔄 Migration Scenarios

### 1. Edge to Cloud Migration
```bash
# Start with edge WASM deployment
neural-swarm-deployer deploy -i edge-node -e edge-config.toml

# Monitor shows high CPU usage
# Controller automatically migrates to container
# Or manually trigger:
neural-swarm-deployer migrate edge-node container
```

### 2. Power-Aware Migration
```bash
# Deploy in power-aware mode
neural-swarm-deployer deploy -i mobile-node -e mobile-config.toml

# When power budget exceeded, automatically migrates to
# more efficient WASM target or reduces functionality
```

### 3. Load-Based Scaling
```bash
# Deploy hybrid configuration
neural-swarm-deployer deploy -i hybrid-node -e hybrid-config.toml

# Controller automatically scales between:
# - WASM for low load (power efficient)
# - Container for high load (performance)
# - Hybrid for variable load (balanced)
```

## 🧪 Testing

### Unit Tests
```bash
cd deployment/hybrid
cargo test
```

### Integration Tests
```bash
# Test container deployment
docker-compose -f deployment/docker/docker-compose.yml up -d
curl http://localhost:8080/health

# Test WASM deployment
cd deployment/wasm
./build.sh
python3 -m http.server 8000 -d dist
# Open http://localhost:8000/demo.html

# Test hybrid deployment
cd deployment/hybrid
cargo run --bin deployer -- deploy -i test -e examples/edge.toml --dry-run
```

### Performance Benchmarks
```bash
# Container performance
docker run --rm neural-swarm/neuroplex:latest benchmark

# WASM performance
cd deployment/wasm/dist
node node-example.js

# Hybrid performance comparison
neural-swarm-deployer analyze -e config/environment.toml
```

## 📖 API Reference

### Unified Deployment API

```rust
use neural_swarm_hybrid_deployment::*;

// Create deployment API
let api = UnifiedDeploymentAPI::new();

// Deploy with automatic target selection
let status = api.deploy("my-deployment", &environment).await?;

// Get deployment status
let status = api.get_status("my-deployment").await;

// Migrate deployment
api.migrate_deployment("my-deployment", DeploymentTarget::Wasm).await?;

// Remove deployment
api.remove_deployment("my-deployment").await?;
```

### WASM Runtime API

```javascript
// Initialize WASM runtime
const runtime = new NeuroSwarmRuntime();
await runtime.init(config);

// Deploy nodes
await runtime.deployNode('node-1', nodeConfig);

// Get status
const status = await runtime.getStatus();

// Optimize for edge
await runtime.optimizeForEdge();
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT OR Apache-2.0 License - see the [LICENSE](../LICENSE) file for details.

## 🔗 Links

- [Neural-Swarm Main Repository](https://github.com/neural-swarm/neural-swarm)
- [Documentation](https://neural-swarm.github.io/docs)
- [Examples](https://github.com/neural-swarm/neural-swarm/tree/main/examples)
- [Issue Tracker](https://github.com/neural-swarm/neural-swarm/issues)

---

**Made with ❤️ by the Neural-Swarm Team**