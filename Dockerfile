# --- Tessella Worker Dockerfile ---
# Use NVIDIA CUDA base for GPU support
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as builder

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Rust and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/tessella
COPY . .

# Build the worker with CUDA support
# Note: Ensure the 'cuda' feature is enabled in Cargo.toml or passed here
RUN cargo build --release --bin tessella-worker --features cuda

# --- Runtime stage ---
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/tessella/target/release/tessella-worker /app/tessella-worker

# Persistent directories (mount these to keep the keypair and model cache)
RUN mkdir -p /app/models /app/data

# Ports: 8080 = HTTP API, 9080 = P2P swarm
EXPOSE 8080 9080

# Environment variables
ENV MUSIVUM_URL="http://musivum:8080"
ENV VRAM_LIMIT="8.0"
ENV TESSELLA_NON_INTERACTIVE="1"
ENV TESSELLA_MODELS_DIR="/app/models"
ENV TESSELLA_KEYPAIR_PATH="/app/data/node.key"
ENV TESSELLA_API_PORT="8080"
ENV TESSELLA_P2P_PORT="9080"
# Public address overrides (set these behind NAT/proxies, e.g. RunPod)
# TESSELLA_PUBLIC_IP / TESSELLA_PUBLIC_API_PORT / TESSELLA_PUBLIC_P2P_PORT
# TESSELLA_PUBLIC_API_URL (full URL, takes precedence over the ip/port pair)

CMD ["/app/tessella-worker"]
