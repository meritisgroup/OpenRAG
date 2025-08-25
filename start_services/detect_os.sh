#!/bin/bash

# Function to detect OS and system capabilities
detect_system() {
    OS_TYPE=""
    HAS_NVIDIA_GPU=false
    DOCKER_COMPOSE_CMD=""
    
    # Detect operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"
        
        # Check for NVIDIA GPU on Linux
        if command -v nvidia-smi &> /dev/null; then
            if nvidia-smi &> /dev/null; then
                HAS_NVIDIA_GPU=true
                echo "✅ Linux system with NVIDIA GPU detected"
            else
                echo "✅ Linux system without NVIDIA GPU detected"
            fi
        else
            echo "✅ Linux system without NVIDIA GPU detected"
        fi
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        echo "✅ macOS system detected (no NVIDIA GPU support)"
        
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS_TYPE="windows"
        
        # Check for NVIDIA GPU on Windows
        if command -v nvidia-smi &> /dev/null; then
            if nvidia-smi &> /dev/null; then
                HAS_NVIDIA_GPU=true
                echo "✅ Windows system with NVIDIA GPU detected"
            else
                echo "✅ Windows system without NVIDIA GPU detected"
            fi
        else
            echo "✅ Windows system without NVIDIA GPU detected"
        fi
    else
        echo "⚠️  Unrecognized operating system: $OSTYPE"
        OS_TYPE="unknown"
    fi
    
    # Export variables for later use
    export OS_TYPE
    export HAS_NVIDIA_GPU
}

# Function to get docker-compose command for a service
get_docker_compose_cmd() {
    local service=$1
    local compose_cmd=""
    
    case $service in
        "ollama")
            compose_cmd="docker-compose -f docker-compose-ollama.yml"
            if [ "$HAS_NVIDIA_GPU" = true ]; then
                compose_cmd="$compose_cmd -f docker-compose-ollama.gpu.yml"
                echo "📁 Using Ollama with GPU support"
            else
                echo "📁 Using Ollama with CPU only"
            fi
            ;;
        "all")
            compose_cmd="docker-compose -f docker-compose-all.yml"
            if [ "$HAS_NVIDIA_GPU" = true ]; then
                compose_cmd="$compose_cmd -f docker-compose-all.gpu.yml"
                echo "📁 Using All services with GPU support"
            else
                echo "📁 Using All services with CPU only (VLLM may not work properly)"
            fi
            ;;
        "vllm")
            compose_cmd="docker-compose -f docker-compose-vllm.yml"
            if [ "$HAS_NVIDIA_GPU" = true ]; then
                compose_cmd="$compose_cmd -f docker-compose-vllm.gpu.yml"
                echo "📁 Using VLLM with GPU support"
            else
                echo "⚠️  WARNING: VLLM requires GPU for proper operation"
                echo "📁 Using VLLM with CPU only (may not work)"
            fi
            ;;
        "api")
            compose_cmd="docker-compose -f docker-compose-api.yml"
            echo "📁 Using API configuration"
            ;;
        *)
            echo "❌ Unknown service: $service"
            return 1
            ;;
    esac
    
    echo "$compose_cmd"
}

# Run detection if script is executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    detect_system
fi