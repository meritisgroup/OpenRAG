#!/bin/bash

# Source the detection script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_os.sh"

# Detect system configuration
echo "========================================="
echo "🔍 Detecting system configuration..."
echo "========================================="
detect_system

# Navigate to docker directory
cd "$SCRIPT_DIR/../docker"

# Get the appropriate docker-compose command
COMPOSE_CMD=$(get_docker_compose_cmd "all")

# Start Elasticsearch service
echo "========================================="
echo "🚀 Starting Elasticsearch service..."
echo "========================================="
cd ../infrastructure/elasticsearch
if [[ "$OS_TYPE" != "macos" ]]; then
    chown -R 1000:1000 ../../volumes/dev-elasticsearch
fi
docker compose up -d
cd ../../docker

# Start all services (Ollama + VLLM) with detected configuration
echo "========================================="
echo "🚀 Starting all services (Ollama + VLLM)..."
echo "========================================="
$COMPOSE_CMD up -d --build

echo "========================================="
echo "✅ All services started successfully!"
if [ "$HAS_NVIDIA_GPU" = false ] && [ "$OS_TYPE" = "macos" ]; then
    echo "⚠️  Note: VLLM may not work properly without GPU support"
fi
echo "========================================="