# Isaac Sim 5.0 Docker Container

Digital Twin display container based on `DT-containerized/Isaac Sim 4.5.0 Docker Container`.

# Added Feature

Automatically sets the viewport to a specific USD camera perspective upon Isaac Sim initialization.
check `6. Create Startup Stage and Camera Script` on `entrypoint.sh`

## Installation

### 1.Docker installation
```bash
# Docker installation using the convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Post-install steps for Docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker
docker run hello-world
```

### 2.NVIDIA Container Toolkit installation

```bash
# Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && \
    sudo apt-get update

# Install the NVIDIA Container Toolkit packages
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Configure the container runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify NVIDIA Container Toolkit
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### 3. Build Docker Image
```bash
git clone <this-repository>
cd isaac-sim-container_5_0
docker build -t isaac-sim:5.0 .
```


### Basic Run (Docker Container)

```bash
sudo docker run --name isaac-sim \
--detach \
--restart unless-stopped \
--runtime=nvidia --gpus all \
--ipc=host \
-e "ACCEPT_EULA=Y" \
-e "PRIVACY_CONSENT=Y" \
-e "OMNI_SERVER=omniverse://YOUR_NUCLEUS_SERVER/NVIDIA/Assets/Isaac/5.0" \
-e "OMNI_USER=your_username" \
-e "OMNI_PASS=your_password" \
-e "OMNI_KIT_ALLOW_ROOT=1" \
-e "STARTUP_USD_STAGE=omniverse://YOUR_NUCLEUS_SERVER/Projects/Display/Dream-AI_Plus_Twin/Dream-AI_Plus_Twin.usd" \
-e "STARTUP_CAMERA_PATH=/World/Display1" \
-e XAUTHORITY=${XAUTHORITY}
--network=host \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-v ${XAUTHORITY}:${XAUTHORITY}:rw \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
-v /path/to/your/extensions:/isaac-sim/Extension:rw \
isaac-sim:5.0
```

### Multi-Run (Docker Compose)
**Current Method for Managing Hallway Displays** (2026-01-30)
check `example_docker-compose.yml` file
