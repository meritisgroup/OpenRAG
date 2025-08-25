#!/bin/bash

# Source the detection script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/detect_os.sh"

# Detect system configuration
echo "========================================="
echo "🔍 Detecting system configuration..."
echo "========================================="
detect_system

# Check if GPU is available for VLLM
if [ "$HAS_NVIDIA_GPU" = false ]; then
    echo "========================================="
    echo "⚠️  WARNING: VLLM requires NVIDIA GPU!"
    echo "⚠️  The service may not work properly on CPU-only systems."
    echo "========================================="
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Navigate to docker directory
cd "$SCRIPT_DIR/../docker"

# Get the appropriate docker-compose command
COMPOSE_CMD=$(get_docker_compose_cmd "vllm")

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

# Start VLLM service with detected configuration
echo "========================================="
echo "🚀 Starting VLLM service..."
echo "========================================="
$COMPOSE_CMD up -d --build

echo "========================================="
echo "✅ Services started!"
if [ "$HAS_NVIDIA_GPU" = false ]; then
    echo "⚠️  Remember: VLLM may not function correctly without GPU"
fi
echo "========================================="