# 🚀 OpenRAG by Meritis

**Welcome to OpenRAG** - An open-source, user-friendly RAG (Retrieval-Augmented Generation) benchmark tool that helps you find the perfect RAG method for your specific use case and data.

![OpenRAG Interface](streamlit_/images/screen_rag_maker.png)

---

## 📋 Table of Contents

- [🎯 What is OpenRAG?](#-what-is-openrag)
- [✨ Key Features](#-key-features)
- [🔧 Prerequisites](#-prerequisites)
- [⚡ Quick Start](#-quick-start)
- [🛠️ Installation & Setup](#️-installation--setup)
- [🚀 Launch Methods](#-launch-methods)
- [📊 Available Configurations](#-available-configurations)
- [💻 System Requirements](#-system-requirements)
- [📱 Using the Application](#-using-the-application)
- [🔧 Troubleshooting](#-troubleshooting)
- [📞 Support & Contact](#-support--contact)

---

## 🎯 What is OpenRAG?

OpenRAG is a comprehensive benchmark tool designed to help users evaluate and compare different RAG (Retrieval-Augmented Generation) methods. With over a dozen implemented RAG techniques, OpenRAG provides:

- **Quantitative analysis** of RAG performance
- **Customizable configurations** for each RAG method
- **Energy consumption and carbon footprint** tracking
- **Token usage analysis** and cost estimation
- **Response time benchmarking**

The goal is to help you decide which RAG method is most appropriate for your specific use case and data.

---

## ✨ Key Features

- 🤖 **12+ RAG Methods**: Naive RAG, Corrective RAG, Self RAG, Graph RAG, and more
- 🔌 **Multiple LLM Providers**: Ollama, VLLM, OpenAI, Mistral
- 📊 **Comprehensive Benchmarking**: Performance, speed, energy, and cost analysis
- 🎨 **User-Friendly Interface**: Streamlit-based web interface
- 🔧 **Full Customization**: Tailor each RAG method to your needs
- 📁 **Data Upload**: Support for various document formats
- 🌍 **Cross-Platform**: Works on Windows, macOS, and Linux
- 🎮 **Auto-Detection**: Automatic OS and GPU detection

---

## 🔧 Prerequisites

Before getting started, make sure you have the following installed:

### Required Software
- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **Git** (for cloning the repository)

### System Requirements
- **RAM**: Minimum 8GB (16GB+ recommended for GPU usage)
- **Storage**: At least 10GB free space
- **Operating System**: Windows 10+, macOS 10.14+, or Linux

### For GPU Usage (Optional but Recommended)
- **NVIDIA GPU** with CUDA support
- **NVIDIA Docker** runtime installed
- **CUDA 12.1+** for VLLM support

---

## ⚡ Quick Start

Get OpenRAG running in under 5 minutes:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-org/OpenRAG.git
cd OpenRAG
```

### 2️⃣ Choose Your Launch Method

**Option A: Automatic Detection (Recommended)**
```bash
cd docker
./launch.sh ollama up -d
```

**Option B: Full Service with Elasticsearch**
```bash
./start_services/ollama.sh
```

### 3️⃣ Access the Application
Open your browser and navigate to: **[http://localhost:8506](http://localhost:8506)**

---

## 🛠️ Installation & Setup

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/your-org/OpenRAG.git
cd OpenRAG

# Make scripts executable (Linux/macOS)
chmod +x docker/launch.sh
chmod +x start_services/*.sh
```

### Step 2: Configure Environment (Optional)
Create a `.env` file in the root directory for custom configurations:
```bash
# Example .env file
FRONT_LOCAL_PORT=8506
ES_LOCAL_PORT=9200
ES_LOCAL_PASSWORD=changeme
ollama_LOCAL_CONTAINER_NAME=openrag_ollama
```

### Step 3: Choose Your Configuration
Based on your system and requirements, choose one of the launch methods below.

---

## 🚀 Launch Methods

OpenRAG offers three different ways to launch the application, each suited for different use cases:

### 🎯 Method 1: Auto-Detection Script (Recommended)

**Best for**: Most users who want simplicity with automatic system detection.

```bash
cd docker
./launch.sh [service] [docker-compose-commands]
```

**Examples:**
```bash
# Start Ollama (automatically detects CPU/GPU)
./launch.sh ollama up -d

# Stop all services
./launch.sh ollama down

# View live logs
./launch.sh ollama logs -f

# Restart services
./launch.sh ollama restart

# Check service status
./launch.sh ollama ps
```

### 🔧 Method 2: Manual Docker Compose

**Best for**: Advanced users who want full control over configurations.

#### For CPU-Only Systems (macOS, systems without NVIDIA GPU):
```bash
cd docker

# Launch individual services
docker compose -f docker-compose-ollama.yml up -d
docker compose -f docker-compose-api.yml up -d

# Launch all services (Note: VLLM may not work without GPU)
docker compose -f docker-compose-all.yml up -d
```

#### For GPU Systems (Linux/Windows with NVIDIA GPU):
```bash
cd docker

# Launch with GPU acceleration
docker compose -f docker-compose-ollama.yml -f docker-compose-ollama.gpu.yml up -d
docker compose -f docker-compose-vllm.yml -f docker-compose-vllm.gpu.yml up -d
docker compose -f docker-compose-all.yml -f docker-compose-all.gpu.yml up -d
```

### 🎪 Method 3: Full Service Scripts

**Best for**: Users who want complete service management including Elasticsearch setup.

```bash
# Start Ollama with Elasticsearch
./start_services/ollama.sh

# Start VLLM with Elasticsearch (requires GPU)
./start_services/vllm.sh

# Start all services with Elasticsearch
./start_services/all.sh

# Start API-only mode (Elasticsearch + Frontend)
./start_services/api.sh
```

---

## 📊 Available Configurations

Choose the configuration that matches your needs:

| Configuration | Services Included | Use Case | GPU Required |
|---------------|-------------------|----------|--------------|
| **🤖 ollama** | Elasticsearch + Frontend + Ollama | Local LLM with Ollama | Optional (recommended) |
| **⚡ vllm** | Elasticsearch + Frontend + VLLM | High-performance inference | **Yes** |
| **🚀 all** | Elasticsearch + Frontend + Ollama + VLLM | Full local setup | Optional |
| **🌐 api** | Elasticsearch + Frontend only | Use with API keys (OpenAI, Mistral) | No |

### System Detection Logic

OpenRAG automatically detects your system configuration:

- **🍎 macOS**: Always uses CPU-only configuration (Apple Silicon/Intel)
- **🐧 Linux**: Checks for NVIDIA GPU with `nvidia-smi`
- **🪟 Windows**: Checks for NVIDIA GPU with `nvidia-smi`

### LLM Provider Support

| Provider | API Key Required | Local Hardware | GPU Recommended |
|----------|------------------|----------------|-----------------|
| **Ollama** | ❌ No | ✅ Yes | 🟡 Optional |
| **VLLM** | ❌ No | ✅ Yes | ✅ Required |
| **OpenAI** | ✅ Yes | ❌ No | ❌ No |
| **Mistral** | ✅ Yes | ❌ No | ❌ No |

---

## 💻 System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended for GPU Usage
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 50GB+ free space (for models)

### VLLM Specific Requirements
- **CUDA**: Version 12.1 or higher
- **GPU Memory**: 8GB+ recommended
- **OS**: Linux or Windows with WSL2

---

## 📱 Using the Application

Once OpenRAG is running, access it at **[http://localhost:8506](http://localhost:8506)**

### 📂 1. Upload Your Data
![Database Interface](streamlit_/images/screen_db.png)

- Navigate to the **"🌐 Databases"** page
- Upload documents in various formats (PDF, TXT, DOCX, etc.)
- Configure indexing settings for your data

### 💬 2. Chat & Test RAG Methods
- Go to **"💬 Chat"** to interact with different RAG methods
- Select your preferred RAG technique
- Test responses to assess performance qualitatively

### ⚙️ 3. Customize RAG Methods
![RAG Maker Interface](streamlit_/images/screen_rag_maker.png)

- Visit **"🔧 RAG Maker"** for advanced configurations
- Tune parameters for each RAG method
- Create custom RAG pipelines

### 📊 4. Benchmark & Compare
![Report Interface](streamlit_/images/screen_report.png)

- Use **"📚 Benchmark"** to run quantitative comparisons
- Generate detailed reports on:
  - **Performance metrics** (accuracy, relevance)
  - **Response times**
  - **Energy consumption**
  - **Token usage and costs**
  - **Carbon footprint**

### 🛠️ 5. Advanced Configuration
- Access **"🛠️ Advanced Configuration"** for system-level settings
- Configure model parameters, memory usage, and more

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 🚨 "Permission denied" when running scripts
```bash
# Make scripts executable
chmod +x docker/launch.sh
chmod +x start_services/*.sh
```

#### 🚨 Port already in use (8506)
```bash
# Check what's using the port
lsof -i :8506

# Kill the process or change port in docker-compose files
# Edit the FRONT_LOCAL_PORT in your .env file
```

#### 🚨 Docker: "no such file or directory"
```bash
# Make sure you're in the correct directory
cd OpenRAG/docker

# Verify files exist
ls -la docker-compose-*.yml
```

#### 🚨 Elasticsearch fails to start
```bash
# Check available memory
free -m

# Ensure port 9200 is available
lsof -i :9200

# On Linux, fix permissions
sudo chown -R 1000:1000 volumes/dev-elasticsearch
```

#### 🚨 VLLM not working on macOS
**Issue**: VLLM requires NVIDIA GPU, which is not available on macOS.

**Solutions**:
- Use Ollama configuration instead: `./launch.sh ollama up -d`
- Use API-only mode: `./launch.sh api up -d`
- Consider using OpenAI or Mistral APIs

#### 🚨 GPU not detected on Linux
```bash
# Check NVIDIA drivers
nvidia-smi

# Install NVIDIA Docker runtime
sudo apt install nvidia-docker2
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 🚨 Out of memory errors
```bash
# Check Docker memory limits
docker system info

# Reduce model size or batch size in configurations
# Close other applications to free up RAM
```

### 📝 Getting Help

If you encounter issues not covered here:

1. **Check Docker logs**:
   ```bash
   docker logs [container_name]
   ```

2. **Restart services**:
   ```bash
   ./launch.sh [service] restart
   ```

3. **Clean restart**:
   ```bash
   ./launch.sh [service] down
   docker system prune -f
   ./launch.sh [service] up -d
   ```

---

## 📞 Support & Contact

### 🤝 Getting Help

- **Issues & Bugs**: [GitHub Issues](https://github.com/your-org/OpenRAG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/OpenRAG/discussions)
- **Documentation**: This README and in-app help

### 💼 Professional Support

For business inquiries, enterprise support, or custom implementations:

**Meritis Innovation IA Team**
- 🌐 Website: [https://meritis.fr/expertise/innovation-ia/](https://meritis.fr/expertise/innovation-ia/)
- 📧 Contact: [Contact Form](https://meritis.fr/expertise/innovation-ia/#block-form)

### 🤖 Community

- Star ⭐ the repository if OpenRAG helps your projects
- Share your use cases and feedback
- Contribute to the project development

---

## 🙏 Acknowledgments

OpenRAG is developed and maintained by the Innovation IA team at **Meritis**. Special thanks to all contributors and the open-source community for making this project possible.

---

**Happy RAG Benchmarking! 🚀**