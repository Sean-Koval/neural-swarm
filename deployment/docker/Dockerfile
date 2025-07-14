# Production-ready Dockerfile for neural-swarm coordination system
# Multi-stage build for optimized deployment

# Build stage
FROM rust:1.74-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libfann-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/neural-swarm

# Copy source files
COPY . .

# Create optimized release build
RUN cargo build --release --bin neuroplex-node

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    libfann2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false -m -d /var/lib/neuroplex neuroplex

# Copy binary from builder
COPY --from=builder /usr/src/neural-swarm/target/release/neuroplex-node /usr/local/bin/

# Copy configuration
COPY --from=builder /usr/src/neural-swarm/config/ /etc/neuroplex/

# Set ownership
RUN chown -R neuroplex:neuroplex /var/lib/neuroplex /etc/neuroplex

# Switch to non-root user
USER neuroplex

# Expose ports
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV NEUROPLEX_CONFIG=/etc/neuroplex/config.toml

# Default command
CMD ["neuroplex-node", "--config", "/etc/neuroplex/config.toml"]