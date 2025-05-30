FROM nvcr.io/nvidia/isaac-sim:4.5.0

# 비대화형 설치 모드 설정 및 타임존 미리 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get update && apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# 기본 패키지 설치 (Xvfb 관련 패키지 제거)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gpg \
    git \
    vim \
    # xvfb \ # Xvfb 제거
    # x11vnc \ # X11VNC 제거
    # x11-utils \ # X11 유틸리티 제거 (필요 시 주석 해제)
    # xterm \ # Xterm 제거 (필요 시 주석 해제)
    mesa-utils \
    libgl1-mesa-dri \
    libglx-mesa0 \
    curl \
    gnupg \
    lsb-release \
    jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ROS2 Humble 설치 (Ubuntu 22.04 Jammy 기준)
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

# --- libbrotli 버전 충돌 해결을 위해 PPA 제거 ---
RUN rm -f /etc/apt/sources.list.d/*sury*.list && \
    apt-get update
# ---------------------------------------------------

# rosdep 초기화 및 업데이트
RUN rosdep init || echo "rosdep already initialized" && \
    rosdep update

# VSCode 설치
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg \
    && install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg \
    && sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] \
       https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' \
    && rm -f packages.microsoft.gpg \
    && apt-get update && apt-get install -y --no-install-recommends code \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# SmartX Omniverse 확장 기능 클론
ARG GIT_REPO_URL="https://github.com/SmartX-Team/Omniverse.git"
ARG EXTENSIONS_SRC_DIR="/opt/SmartX_Omniverse_Extensions"
RUN git clone ${GIT_REPO_URL} ${EXTENSIONS_SRC_DIR}

# Isaac Sim ROS Workspaces 저장소 복제 및 빌드
ENV ROS_WS_DIR="/root/isaac_sim_ros_ws"
RUN mkdir -p ${ROS_WS_DIR}/src
WORKDIR ${ROS_WS_DIR}/src
RUN git clone https://github.com/NVIDIA-Omniverse/IsaacSim-ros_workspaces.git
WORKDIR ${ROS_WS_DIR}/src/IsaacSim-ros_workspaces/humble_ws

# --- Brotli mismatch: force both packages to the Jammy 1.0.9 build ----------
RUN apt-get update && \
    # allow-downgrades is needed because 1.1.0 -> 1.0.9 is a version decrease
    apt-get install -y --allow-downgrades \
        libbrotli1=1.0.9-2build6 \
        libbrotli-dev=1.0.9-2build6 && \
    # optional but recommended: freeze them so later apt operations don't pull 1.1.0 back in
    apt-mark hold libbrotli1 libbrotli-dev
# -----------------------------------------------------------------------------

# PPA가 제거된 상태의 최신 패키지 목록을 확실히 사용하도록 apt-get update를 먼저 실행
# 필요한 의존성 패키지(libfreetype6-dev)도 여기서 함께 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends libfreetype6-dev && \
    . /opt/ros/humble/setup.sh && \
    rosdep install -i --from-paths src --rosdistro humble -y --skip-keys="isaac_sim_visual_slam isaac_ros_apriltag"

RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# entrypoint.sh 파일 복사 및 실행 권한 부여
COPY entrypoint.sh /isaac-sim/entrypoint.sh
RUN chmod +x /isaac-sim/entrypoint.sh

# 환경 변수 설정
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV ROS_WS_DIR="/root/isaac_sim_ros_ws"
ENV ACCEPT_EULA=Y
# ENV DISPLAY=:1 # 필요시 호스트의 DISPLAY 환경변수 사용

# 작업 디렉토리 설정
WORKDIR /isaac-sim

# 기본 진입점 설정
ENTRYPOINT ["/isaac-sim/entrypoint.sh"]
CMD []