# Isaac Sim 4.5 Docker Container

A Docker containerized solution for running NVIDIA Isaac Sim 4.5 with ROS2 Humble integration.

## 🚀 Features

- **Isaac Sim 4.5**: Latest version support
- **ROS2 Humble**: Fully integrated ROS2 environment in the Container
- **SmartX Extensions**: Auto-registration of NetAI group custom extensions
- **Nucleus Server**: Remote connection to Omniverse Nucleus server
- **Headless Mode**: Run simulations without GUI
- **Cache Optimization**: Volume mounts for faster startup

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 (recommended)
- NVIDIA Container Toolkit
- NVIDIA GPU Driver (535)

## 🔧 Installation

### 1. Install NVIDIA Container Toolkit
```bash
# Add package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Build Docker Image
```bash
git clone <this-repository>
cd isaac-sim-container
docker build -t isaac-sim:4.5 .
```

## 🖥️ Usage

### Basic Run (Headless Mode Example)
```bash
sudo docker run --name isaac-sim \
--detach \
--restart unless-stopped \
--runtime=nvidia --gpus all \
-e "ACCEPT_EULA=Y" \
-e "PRIVACY_CONSENT=Y" \
-e "OMNI_SERVER=omniverse://YOUR_NUCLEUS_SERVER/NVIDIA/Assets/Isaac/4.5" \
-e "OMNI_USER=your_username" \
-e "OMNI_PASS=your_password" \
-e "OMNI_KIT_ALLOW_ROOT=1" \
--network=host \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
-v /path/to/your/extensions:/isaac-sim/Extension:rw \
isaac-sim:4.5
```

### Environment Variables
```bash
# GUI mode
-e "START_GUI=true"

# Python REPL mode
-e "START_PYTHON=true" 

# Interactive bash session
-e "START_BASH=true"

# Re-register extensions
-e "REREGISTER_EXTENSIONS=true"
```

## 📁 Volume Mount Guide

### Cache Directories (Performance Boost)
```bash
# Create local cache directories
mkdir -p ~/docker/isaac-sim/{cache/{kit,ov,pip,glcache,computecache},logs,data,documents}
```

### Custom Paths
- `~/docker/isaac-sim/cache/*`: Cache storage (faster startup)
- `/path/to/your/extensions`: User extension folder
- `/path/to/your/scripts`: Python scripts folder
- `/path/to/your/assets`: 3D models and assets folder

> **Note**: Adjust all paths according to your environment.

## 🔗 Nucleus Server Connection

### Environment Variables
```bash
-e "OMNI_SERVER=omniverse://YOUR_SERVER_IP/PATH"
-e "OMNI_USER=your_username"
-e "OMNI_PASS=your_password"
```

### Connection Verification
```bash
# Check connection status in logs
docker logs isaac-sim | grep -i "omni\|nucleus"
```

## 🛠️ Development & Debugging

### Container Access
```bash
# Access running container
docker exec -it isaac-sim /bin/bash

# Start new bash session
docker run -e START_BASH=true -it isaac-sim:4.5
```

### Log Monitoring
```bash
# Real-time logs
docker logs -f isaac-sim

# Extension registration logs
docker logs isaac-sim | grep -i extension
```

### Extension Development
```bash
# Re-register extensions
docker exec isaac-sim /isaac-sim/entrypoint.sh -e REREGISTER_EXTENSIONS=true

# List extensions
docker exec isaac-sim ls -la /isaac-sim/exts/
```

## 🐛 Troubleshooting

### GPU Recognition Issues
bash# Check GPU status and RT core availability
nvidia-smi
nvidia-smi --query-gpu=name,driver_version --format=csv

#### Test with specific GPU
docker run --rm --gpus '"device=1"' nvidia/cuda:11.8-base nvidia-smi
Mixed GPU Environment Issues
bash# ❌ Common error with mixed GPUs:
## "RTX not supported" or rendering failures

✅ Solution: Always specify RT-capable GPU IDs
--gpus '"device=1,2"'  # Use specific GPU IDs with RT cores

✅ Check which GPUs have RT cores
nvidia-smi --query-gpu=name --format=csv,noheader | grep -i rtx

### Permission Issues
```bash
# Run with sudo
sudo docker run ...

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Memory Issues
```bash
# Increase shared memory
--shm-size=2g
```

## 📚 Useful Commands

```bash
# Start Isaac Sim GUI
/isaac-sim/isaac-sim.sh

# Start Headless mode
/isaac-sim/runheadless.sh -v

# Start Python environment
/isaac-sim/python.sh

# Run Python script
/isaac-sim/python.sh /path/to/script.py

# List extensions
ls /isaac-sim/exts/

# Check ROS environment
printenv | grep ROS
```