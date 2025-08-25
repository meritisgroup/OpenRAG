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
COMPOSE_CMD=$(get_docker_compose_cmd "ollama")

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

# Start Ollama service with detected configuration
echo "========================================="
echo "🚀 Starting Ollama service..."
echo "========================================="
$COMPOSE_CMD up -d --build

echo "========================================="
echo "✅ Services started successfully!"
echo "========================================="