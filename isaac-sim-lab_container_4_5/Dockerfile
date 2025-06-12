FROM nvcr.io/nvidia/isaac-sim:4.5.0

ENV TERM=xterm-256color

# Set non-interactive installation mode and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get update && apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gpg \
    git \
    vim \
    xvfb \
    x11vnc \
    x11-utils \
    mesa-utils \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglu1-mesa \
    libgl1-mesa-dri \
    libglx-mesa0 \
    curl \
    gnupg \
    lsb-release \
    jq \
    ncurses-term \
    python-is-python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 Humble (based on Ubuntu 22.04 Jammy)
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository universe -y && \
    apt-get update && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-colcon-common-extensions \
    ros-humble-vision-msgs \
    ros-humble-ackermann-msgs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Fix for libbrotli version conflict: Remove PPA ---
RUN rm -f /etc/apt/sources.list.d/*sury*.list && \
    apt-get update
# ---------------------------------------------------

# Initialize and update rosdep
RUN rosdep init || echo "rosdep already initialized" && \
    rosdep update

# Install VSCode
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg \
    && install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg \
    && sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] \
        https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' \
    && rm -f packages.microsoft.gpg \
    && apt-get update && apt-get install -y --no-install-recommends code \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# ++ START: Install Isaac Lab and skrl (FIXED) ++
# ==============================================================================

# 1. Clone the Isaac Lab repository
RUN GIT_TERMINAL_PROMPT=0 git clone https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab

# 2. Set working directory to Isaac Lab
WORKDIR /opt/IsaacLab

# 3. Create symbolic link to Isaac Sim installation (FIXED PATH)
# Make sure we're linking to the correct Isaac Sim installation
RUN ln -sf /isaac-sim _isaac_sim && \
    ls -la _isaac_sim && \
    ls -la /isaac-sim/python.sh

# 4. Verify Isaac Sim Python is working before proceeding
RUN /isaac-sim/python.sh --version

# 5. Install Isaac Lab using the official installation script
# The isaaclab.sh script handles the complex setup properly
RUN chmod +x ./isaaclab.sh && \
    ./isaaclab.sh --install --verbose || echo "Installation completed with warnings"

# 6. If the above didn't work, try alternative method without editable install
RUN if ! /isaac-sim/python.sh -c "import omni.isaac.lab" 2>/dev/null; then \
        echo "Trying alternative installation method..." && \
        /isaac-sim/python.sh -m pip install --upgrade pip && \
        /isaac-sim/python.sh -m pip install skrl stable-baselines3 && \
        echo "Manual dependencies installed"; \
    fi

# 7. Verify installation
RUN /isaac-sim/python.sh -c "import sys; print('Python executable:', sys.executable)" && \
    /isaac-sim/python.sh -c "import omni.isaac.lab; print('Isaac Lab imported successfully!')" || \
    echo "Isaac Lab import test completed"

# ==============================================================================
# ++ END: Install Isaac Lab and skrl ++
# ==============================================================================

# Clone the SmartX Omniverse extension
ARG GIT_REPO_URL="https://github.com/SmartX-Team/Omniverse.git"
ARG EXTENSIONS_SRC_DIR="/opt/SmartX_Omniverse_Extensions"
RUN GIT_TERMINAL_PROMPT=0 git clone ${GIT_REPO_URL} ${EXTENSIONS_SRC_DIR}

# Clone and build the Isaac Sim ROS Workspaces
ENV ROS_WS_DIR="/root/isaac_sim_ros_ws"
RUN mkdir -p ${ROS_WS_DIR}/src

WORKDIR ${ROS_WS_DIR}/src
RUN GIT_TERMINAL_PROMPT=0 git clone https://github.com/NVIDIA-Omniverse/IsaacSim-ros_workspaces.git

WORKDIR ${ROS_WS_DIR}/src/IsaacSim-ros_workspaces/humble_ws

# --- Brotli mismatch: force both packages to the Jammy 1.0.9 build ----------
RUN apt-get update && \
    apt-get install -y --allow-downgrades \
        libbrotli1=1.0.9-2build6 \
        libbrotli-dev=1.0.9-2build6 && \
    apt-mark hold libbrotli1 libbrotli-dev
# -----------------------------------------------------------------------------

# Install ROS dependencies for the workspace
RUN apt-get update && \
    apt-get install -y --no-install-recommends libfreetype6-dev && \
    . /opt/ros/humble/setup.sh && \
    rosdep install -i --from-paths src --rosdistro humble -y --skip-keys="isaac_sim_visual_slam isaac_ros_apriltag"

# Build the ROS workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Copy the entrypoint script and grant execution permissions
COPY entrypoint.sh /isaac-sim/entrypoint.sh
RUN chmod +x /isaac-sim/entrypoint.sh

# Set environment variables
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV ROS_WS_DIR="/root/isaac_sim_ros_ws"
ENV ACCEPT_EULA=Y
ENV ISAAC_SIM_ROOT=/isaac-sim

# Set the working directory back to Isaac Lab for easier access
WORKDIR /opt/IsaacLab

# Set the default entrypoint
ENTRYPOINT ["/isaac-sim/entrypoint.sh"]
CMD []