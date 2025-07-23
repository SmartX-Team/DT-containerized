# Isaac Sim 4.5 + Isaac Lab Docker Container

A Docker containerized solution for running NVIDIA Isaac Sim 4.5 with Isaac Lab framework and ROS2 Humble integration.

## 🚀 Features

- **Isaac Sim 4.5**: Latest version support
- **Isaac Lab**: GPU-accelerated robot learning framework
- **ROS2 Humble**: Fully integrated ROS2 environment
- **Learning Frameworks**: Pre-installed rl_games, rsl_rl, sb3, skrl, robomimic
- **SmartX Extensions**: Auto-registration of NetAI group custom extensions
- **Nucleus Server**: Remote connection to Omniverse Nucleus server
- **Headless Mode**: Run simulations without GUI
- **Cache Optimization**: Volume mounts for faster startup

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 (recommended)
- **RAM**: 32GB+ recommended for Isaac Lab
- **VRAM**: 16GB+ recommended for Isaac Lab
- NVIDIA Container Toolkit
- NVIDIA GPU Driver (535+)
- RT-capable GPU (RTX series) for optimal performance

## 🔧 Installation

### 1. Install NVIDIA Container Toolkit
```bash
# Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update

# Install the NVIDIA Container Toolkit packages
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Configure the container runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Build Docker Image
```bash
git clone <this-repository>
cd isaac-sim-lab_container_4_5
docker build -t isaac-sim-lab:4.5 .
```

## 🖥️ Usage

### Basic Run (Headless Mode)
```bash
sudo docker run --name isaac-sim-lab \
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
-v ~/docker/user/my_rl_projects:/mnt/rl_code:ro \
isaac-sim-lab:4.5
```

### Inter-Container Communication (with ROS2 containers)
To enable communication with other ROS2-based containers on the same host machine:

```bash
sudo docker run --name isaac-sim-lab \
--detach \
--restart unless-stopped \
--runtime=nvidia --gpus all \
--network=host \
--ipc=host \
--privileged \
-e "ACCEPT_EULA=Y" \
-e "PRIVACY_CONSENT=Y" \
-e "OMNI_SERVER=omniverse://YOUR_NUCLEUS_SERVER/NVIDIA/Assets/Isaac/4.5" \
-e "OMNI_USER=your_username" \
-e "OMNI_PASS=your_password" \
-e "OMNI_KIT_ALLOW_ROOT=1" \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
-v /path/to/your/isaac_lab_projects:/opt/IsaacLab/projects:rw \
isaac-sim-lab:4.5
```

##### --ipc=host: Required for efficient inter-process communication used by ROS2 middleware
##### --privileged: Needed for accessing host devices (cameras, LiDARs, sensors)

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

### Isaac Lab Specific Mounts
- `/opt/IsaacLab/projects`: Isaac Lab project workspace
- `/opt/IsaacLab/logs`: Training logs and checkpoints
- `/opt/SmartX_Omniverse_Extensions`: SmartX custom extensions
- `/root/isaac_sim_ros_ws`: ROS2 workspace

### Custom Paths
- `~/docker/isaac-sim/cache/*`: Cache storage (faster startup)
- `/path/to/your/isaac_lab_projects`: Isaac Lab projects and experiments
- `/path/to/your/extensions`: User extension folder
- `/path/to/your/assets`: 3D models and assets folder

> **Note**: Adjust all paths according to your environment.

## Isaac Lab Usage

### Running Isaac Lab Scripts
```bash
# Access container
docker exec -it isaac-sim-lab /bin/bash

# Run Isaac Lab training example
cd /opt/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Reach-Franka-v0 --headless

# Run Isaac Lab demo
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# List available environments
./isaaclab.sh -p source/standalone/environments/list_envs.py
```


## 🛠️ Development & Debugging

### Container Access
```bash
# Access running container
docker exec -it isaac-sim-lab /bin/bash

# Start new bash session
docker run -e START_BASH=true -it isaac-sim-lab:4.5
```

### Log Monitoring
```bash
# Real-time logs
docker logs -f isaac-sim-lab

# Isaac Lab training logs
docker exec isaac-sim-lab ls -la /opt/IsaacLab/logs/

# Extension registration logs
docker logs isaac-sim-lab | grep -i extension
```

### Extension Development
```bash
# Re-register extensions
docker exec isaac-sim-lab /isaac-sim/entrypoint.sh -e REREGISTER_EXTENSIONS=true

# List extensions
docker exec isaac-sim-lab ls -la /isaac-sim/exts/
docker exec isaac-sim-lab ls -la /opt/SmartX_Omniverse_Extensions/
```

### ROS2 Workspace Issues
```bash
# Check ROS2 environment
docker exec isaac-sim-lab bash -c "source /opt/ros/humble/setup.sh && printenv | grep ROS"

# Rebuild ROS workspace
docker exec isaac-sim-lab bash -c "cd /root/isaac_sim_ros_ws && source /opt/ros/humble/setup.sh && colcon build --symlink-install"
```

## Useful Commands


### Isaac Lab Commands
```bash
# Isaac Lab Python interpreter
/opt/IsaacLab/isaaclab.sh -p

# Run Isaac Lab script
/opt/IsaacLab/isaaclab.sh -p <script_path>

# List available tasks
/opt/IsaacLab/isaaclab.sh -p source/standalone/environments/list_envs.py

# Generate VSCode settings
/opt/IsaacLab/isaaclab.sh -v
```

### ROS2 Commands
```bash
# Source ROS2 environment
source /opt/ros/humble/setup.sh
source /root/isaac_sim_ros_ws/install/setup.sh

# List ROS2 topics
ros2 topic list

# Check ROS2 nodes
ros2 node list
```

## Project Structure

```
isaac-sim-lab_container_4_5/
├── Dockerfile                    # Main container definition
├── entrypoint.sh                 # Container startup script
├── README.md                     # This file
└── docker-compose.yml            # (Optional) Docker Compose configuration
```

## Additional Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)
- [SmartX Omniverse Extensions](https://github.com/SmartX-Team/Omniverse)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)

## Version Info

- **Isaac Sim**: 4.5.0
- **Isaac Lab**: Latest (isaac-sim organization)
- **ROS2**: Humble
- **Ubuntu**: 22.04
- **Python**: 3.10 (Isaac Sim bundled)