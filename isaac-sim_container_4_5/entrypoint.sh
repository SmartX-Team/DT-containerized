#!/bin/bash
set -e

echo "=== Isaac Sim Container Starting ==="

# NVIDIA EULA 자동 수락 (Isaac Sim 실행을 위해 필요)
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ALLOW_ROOT=1

# Extension 검색 경로 추가
export OMNI_KIT_EXTENSION_SEARCH_PATHS="/isaac-sim/exts:${OMNI_KIT_EXTENSION_SEARCH_PATHS:-}"
export CARB_APP_EXTENSION_FOLDERS="/isaac-sim/exts"

# ROS 환경 설정
echo "Setting up ROS environment..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "  ✓ ROS Humble loaded"
fi

# Isaac Sim ROS Workspace 환경 설정
ROS_WS_SETUP="${ROS_WS_DIR}/src/IsaacSim-ros_workspaces/humble_ws/install/setup.bash"
if [ -f "${ROS_WS_SETUP}" ]; then
    source "${ROS_WS_SETUP}"
    echo "  ✓ Isaac Sim ROS workspace loaded"
fi

# SmartX Extension 등록
register_extensions() {
    echo "=== SmartX Extension Registration Started ==="
    
    EXTENSIONS_SRC_DIR="/opt/SmartX_Omniverse_Extensions"
    ISAAC_EXTS_DIR="/isaac-sim/exts"
    
    # Extension 디렉토리 생성
    mkdir -p "${ISAAC_EXTS_DIR}"
    
    echo "Source directory: ${EXTENSIONS_SRC_DIR}"
    echo "Target directory: ${ISAAC_EXTS_DIR}"
    
    # 디렉토리 존재 확인
    if [ ! -d "${EXTENSIONS_SRC_DIR}/Extension" ]; then
        echo "❌ Extension directory not found: ${EXTENSIONS_SRC_DIR}/Extension"
        return 1
    fi
    
    echo "Searching for [NetAI] groups..."
    find "${EXTENSIONS_SRC_DIR}/Extension/" -maxdepth 1 -type d -name "\[NetAI\]*" | while read netai_group; do
        group_name=$(basename "$netai_group")
        exts_path="${netai_group}/exts"
        
        echo "Processing NetAI group: ${group_name}"
        echo "  Group path: ${netai_group}"
        echo "  Exts path: ${exts_path}"
        
        if [ -d "$exts_path" ]; then
            echo "  ✓ Found exts folder"
            
            # exts 폴더 내의 각 Extension 처리
            find "$exts_path" -maxdepth 1 -type d ! -path "$exts_path" | while read ext_folder; do
                ext_name=$(basename "$ext_folder")
                config_file="${ext_folder}/config/extension.toml"
                
                echo "    Checking extension: ${ext_name}"
                echo "    Extension path: ${ext_folder}"
                echo "    Config file: ${config_file}"
                
                if [ -f "$config_file" ]; then
                    target_link="${ISAAC_EXTS_DIR}/${ext_name}"
                    
                    echo "    ✓ Valid extension found"
                    
                    # 기존 링크/폴더 제거
                    if [ -L "$target_link" ] || [ -e "$target_link" ]; then
                        echo "    Removing existing: ${target_link}"
                        rm -rf "$target_link"
                    fi
                    
                    # 심볼릭 링크 생성
                    if ln -s "$ext_folder" "$target_link"; then
                        echo "    ✅ Successfully linked: ${ext_name}"
                        echo "      ${ext_folder} -> ${target_link}"
                    else
                        echo "    ❌ Failed to link: ${ext_name}"
                    fi
                else
                    echo "    ⚠️ No extension.toml found, skipping: ${ext_name}"
                fi
            done
        else
            echo "  ❌ No exts folder found in: ${group_name}"
            echo "  Available subdirectories:"
            ls -la "$netai_group" 2>/dev/null || echo "  Cannot list group directory"
        fi
        echo ""
    done
    
    # 실제 등록된 extension 확인
    echo "=== Final Registration Results ==="
    if [ -d "${ISAAC_EXTS_DIR}" ]; then
        actual_count=$(find "${ISAAC_EXTS_DIR}" -maxdepth 1 -type l 2>/dev/null | wc -l)
        echo "Total extensions registered: ${actual_count}"
        
        if [ ${actual_count} -gt 0 ]; then
            echo "Successfully registered extensions:"
            find "${ISAAC_EXTS_DIR}" -maxdepth 1 -type l | while read link; do
                link_name=$(basename "$link")
                link_target=$(readlink "$link")
                echo "  ✓ ${link_name} -> ${link_target}"
            done
        else
            echo "⚠️ No extensions were successfully registered"
            echo "Extensions directory contents:"
            ls -la "${ISAAC_EXTS_DIR}/" 2>/dev/null || echo "Cannot list extensions directory"
        fi
    fi
    
    echo "=== Extension Registration Complete ==="
}
# Extension Registry 업데이트 함수
update_extension_registry() {
    echo "=== Updating Extension Registry ==="
    
    REGISTRY_FILE="/isaac-sim/kit/kernel/py/omni/kit/registry/extension_registry.py"
    USER_CONFIG="/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/user.config.json"
    
    # 사용자 설정에서 Extension path 추가
    if [ -f "$USER_CONFIG" ]; then
        echo "Updating user config: $USER_CONFIG"
        # JSON에 extension search path 추가
        python3 -c "
import json
import os

config_file = '$USER_CONFIG'
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    config = {}

# Extension search paths 추가
if 'exts' not in config:
    config['exts'] = {}

if 'folders' not in config['exts']:
    config['exts']['folders'] = []

# /isaac-sim/exts 경로 추가
ext_path = '/isaac-sim/exts'
if ext_path not in config['exts']['folders']:
    config['exts']['folders'].append(ext_path)
    print(f'Added {ext_path} to extension search paths')

# 저장
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
    
print('Extension registry updated')
"
    else
        echo "Creating user config directory..."
        mkdir -p "$(dirname "$USER_CONFIG")"
        echo '{
  "exts": {
    "folders": ["/isaac-sim/exts"]
  }
}' > "$USER_CONFIG"
    fi
    
    # Extension 활성화를 위한 추가 설정
    EXTENSION_CONFIG="/isaac-sim/kit/configs/extensions.config.json"
    if [ -f "$EXTENSION_CONFIG" ]; then
        echo "Updating extension config: $EXTENSION_CONFIG"
        cp "$EXTENSION_CONFIG" "${EXTENSION_CONFIG}.backup"
        
        python3 -c "
import json

with open('$EXTENSION_CONFIG', 'r') as f:
    config = json.load(f)

# Extension 경로 추가
if 'extension_folders' not in config:
    config['extension_folders'] = []

ext_path = '/isaac-sim/exts'
if ext_path not in config['extension_folders']:
    config['extension_folders'].append(ext_path)
    print(f'Added {ext_path} to extension folders')

with open('$EXTENSION_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
"
    fi
}

# Extension 등록 (강제 실행)
echo "=== Forcing Extension Registration ==="
register_extensions

echo "=== Container Ready ==="

# 실행 모드 결정 (runheadless.sh 우선)
if [ "${START_GUI}" = "true" ]; then
    echo "🚀 Starting Isaac Sim GUI..."
    exec /isaac-sim/isaac-sim.sh
elif [ "${START_PYTHON}" = "true" ]; then
    echo "🐍 Starting Isaac Sim Python..."
    exec /isaac-sim/python.sh
elif [ "${START_BASH}" = "true" ]; then
    echo "🖥️ Starting bash session..."
    exec /bin/bash
elif [ $# -eq 0 ]; then
    # 기본값: runheadless.sh 실행
    echo "🖥️ Starting Isaac Sim Headless (default)..."
    echo "Use Ctrl+C to stop, or set START_BASH=true for interactive mode"
    exec /isaac-sim/runheadless.sh
else
    echo "Executing: $@"
    exec "$@"
fi