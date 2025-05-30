#!/bin/bash
set -e

echo "=== Isaac Sim Container Starting ==="

# --- 1. 가상 디스플레이(Xvfb) 시작 (안정성 강화) ---
# 이전에 비정상 종료로 남은 X-lock 파일을 제거하여 충돌 방지
rm -f /tmp/.X1-lock
echo "Starting Xvfb on display :1"
Xvfb :1 -screen 0 1920x1080x24 &
export DISPLAY=:1

# --- 2. ROS 환경 설정 ---
echo "Setting up ROS environment..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "  ✓ ROS Humble loaded"
fi
ROS_WS_SETUP="/root/isaac_sim_ros_ws/install/setup.bash"
if [ -f "${ROS_WS_SETUP}" ]; then
    source "${ROS_WS_SETUP}"
    echo "  ✓ Isaac Sim ROS workspace loaded"
fi

# --- 3. 확장 프로그램 경로를 환경 변수로 설정 ---
echo "=== Setting up Extension Search Paths via Environment Variable ==="
SRC_DIR="/opt/SmartX_Omniverse_Extensions/Extension"
CUSTOM_EXT_PATHS=""

if [ -d "$SRC_DIR" ]; then
    find "$SRC_DIR" -type f -name "extension.toml" | while read -r toml_file; do
        config_dir=$(dirname "$toml_file")
        ext_path=$(dirname "$config_dir")
        
        if [ -z "$CUSTOM_EXT_PATHS" ]; then
            CUSTOM_EXT_PATHS="$ext_path"
        else
            CUSTOM_EXT_PATHS="$CUSTOM_EXT_PATHS:$ext_path"
        fi
        echo "  ✓ Added to path: $ext_path"
    done
fi

if [ -n "$CUSTOM_EXT_PATHS" ]; then
    export OMNI_KIT_EXTENSION_PATH="${CUSTOM_EXT_PATHS}:${OMNI_KIT_EXTENSION_PATH:-}"
    echo "✅ Final Extension Path Variable Set."
fi
echo "=== Path Setup Complete ==="

# --- 4. Isaac Sim 실행 ---
echo "=== Container Ready ==="

# 실행 모드 결정
ARGS=("$@")
if [ "${START_GUI}" = "true" ]; then
    echo "🚀 Starting Isaac Sim GUI..."
    exec /isaac-sim/isaac-sim.sh "${ARGS[@]}"
elif [ "${START_PYTHON}" = "true" ]; then
    echo "🐍 Starting Isaac Sim Python..."
    exec /isaac-sim/python.sh "${ARGS[@]}"
elif [ "${START_BASH}" = "true" ]; then
    echo "🖥️ Starting bash session..."
    exec /bin/bash
elif [ ${#ARGS[@]} -eq 0 ]; then
    echo "🖥️ Starting Isaac Sim Headless (default)..."
    exec /isaac-sim/runheadless.sh
else
    echo "Executing: ${ARGS[@]}"
    exec "${ARGS[@]}"
fi