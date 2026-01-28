#!/bin/bash
# =================================================================
#  Enhanced Isaac Sim Entrypoint Script
#
#  - Initializes virtual display (Xvfb)
#  - Updates custom extensions from Git
#  - Sets up ROS environment
#  - Registers extensions using a robust dual-method approach
#  - **NEW**: Automatically opens a USD stage on startup if the
#    `STARTUP_USD_STAGE` environment variable is set.
#  - **NEW**: Automatically sets camera view if the
#    `STARTUP_CAMERA_PATH` environment variable is set.
# =================================================================
set -e

echo "=== Isaac Sim Container Starting ==="

# --- 1. Start virtual display (Xvfb) with enhanced stability ---
# Remove any leftover X-lock files from abnormal termination to prevent conflicts
if [ "${START_GUI}" != "true" ]; then
    # [Headless 모드] 가상 화면이 필요할 때만 실행
    
    # 여기서 청소를 먼저 하고!
    rm -f /tmp/.X1-lock
    
    # 가상 화면(Xvfb)을 켭니다
    echo "Starting Xvfb on display :1"
    Xvfb :1 -screen 0 1920x1080x24 &
    export DISPLAY=:1
else
    # [GUI 모드] 청소도, 가상 화면도 필요 없음
    echo "=== GUI MODE DETECTED ==="
    echo "Skipping Xvfb setup. Using Host Display: $DISPLAY"
fi

# --- 2. Update SmartX Omniverse Extensions ---
echo "=== Updating SmartX Omniverse Extensions ==="
SRC_DIR="/opt/SmartX_Omniverse_Extensions"
if [ -d "$SRC_DIR" ]; then
    cd "$SRC_DIR"
    echo "Pulling latest changes from SmartX Omniverse repository..."
    
    # Check and clean git status
    if git status --porcelain | grep -q .; then
        echo "Local changes detected. Stashing them..."
        git stash push -m "Auto-stash before pull at $(date)"
    fi
    
    # Pull latest changes
    if git pull origin main 2>/dev/null || git pull origin master 2>/dev/null; then
        echo "Successfully updated SmartX Omniverse extensions"
    else
        echo "Failed to pull updates, continuing with existing version"
    fi
    
    # Restore working directory
    cd /isaac-sim
else
    echo "SmartX extensions directory not found at $SRC_DIR"
fi

# --- 3. Setup ROS environment ---
echo "Setting up ROS environment..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "ROS Humble loaded"
fi
ROS_WS_SETUP="/root/isaac_sim_ros_ws/install/setup.bash"
if [ -f "${ROS_WS_SETUP}" ]; then
    source "${ROS_WS_SETUP}"
    echo "Isaac Sim ROS workspace loaded"
fi

# --- 4. Register extensions using COMBINED approach ---
echo "=== Combined Extension Registration ==="

# Extension paths array (declared globally for use in execution section)
declare -a extension_paths
registered_count=0

if [ -d "$SRC_DIR" ]; then
    echo "Scanning for extensions in $SRC_DIR..."
    
    # METHOD 1: Create symbolic links in /isaac-sim/exts (like working version)
    ISAAC_EXTS_DIR="/isaac-sim/exts"
    mkdir -p "$ISAAC_EXTS_DIR"
    echo "Creating symbolic links in Isaac Sim exts directory: $ISAAC_EXTS_DIR"
    
    # Find and process each extension
    while IFS= read -r -d '' toml_file; do
        config_dir=$(dirname "$toml_file")
        ext_path=$(dirname "$config_dir")
        ext_name=$(basename "$ext_path")
        
        echo "Found extension: $ext_name"
        echo "  Path: $ext_path"
        
        # Validate extension structure
        if [ -f "$ext_path/config/extension.toml" ]; then
            extension_paths+=("$ext_path")
            registered_count=$((registered_count + 1))
            echo "  ✓ Valid extension structure"
            
            # Create symbolic link in Isaac Sim exts directory (like working version)
            target_link="$ISAAC_EXTS_DIR/$ext_name"
            if [ -L "$target_link" ] || [ -e "$target_link" ]; then
                echo "  Removing existing: $target_link"
                rm -rf "$target_link"
            fi
            
            if ln -s "$ext_path" "$target_link"; then
                echo "  Successfully linked: $ext_name"
                echo "    $ext_path -> $target_link"
            else
                echo "  Failed to link: $ext_name"
            fi
        else
            echo "  Invalid extension structure (missing config/extension.toml)"
        fi
        
    done < <(find "$SRC_DIR" -type f -name "extension.toml" -not -path "*/deprecated/*" -print0)
    
    # METHOD 2: Update user config (like working version)
    if [ ${#extension_paths[@]} -gt 0 ]; then
        echo "Updating user configuration..."
        USER_CONFIG_DIR="/root/.local/share/ov/data/Kit/Isaac-Sim/4.5"
        mkdir -p "$USER_CONFIG_DIR"
        USER_CONFIG="$USER_CONFIG_DIR/user.config.json"
        
        python3 -c "
import json
import os

config_file = '$USER_CONFIG'
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        try:
            config = json.load(f)
        except:
            config = {}
else:
    config = {}

# Add extension search paths
if 'exts' not in config:
    config['exts'] = {}

if 'folders' not in config['exts']:
    config['exts']['folders'] = []

# Add /isaac-sim/exts path (where symlinks are)
ext_path = '/isaac-sim/exts'
if ext_path not in config['exts']['folders']:
    config['exts']['folders'].append(ext_path)
    print(f'Added {ext_path} to extension search paths')

# Save
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print('Extension registry updated')
"
        echo "  ✓ Updated user config: $USER_CONFIG"
    fi
    
    echo "=== Extension Registration Summary ==="
    echo "Total valid extensions found: $registered_count"
    
    if [ ${#extension_paths[@]} -gt 0 ]; then
        echo "Registered extensions (dual method):"
        for i in "${!extension_paths[@]}"; do
            ext_name=$(basename "${extension_paths[$i]}")
            echo "  $((i+1)). $ext_name: ${extension_paths[$i]}"
        done
        
        # Final check of symbolic links
        echo ""
        echo "Symbolic links created:"
        if [ -d "$ISAAC_EXTS_DIR" ]; then
            find "$ISAAC_EXTS_DIR" -maxdepth 1 -type l | while read link; do
                link_name=$(basename "$link")
                link_target=$(readlink "$link")
                echo "  ✓ $link_name -> $link_target"
            done
        fi
    else
        echo "No valid extensions found"
    fi
    
else
    echo "No SmartX extensions directory found at $SRC_DIR"
fi

echo "=== Extension Setup Complete ==="

# --- 5. Nucleus 연결 설정 ---
echo "=== Configuring Nucleus Connection ==="

# Omniverse 설정 파일 생성
OMNIVERSE_CONFIG="/root/.nvidia-omniverse/config/omniverse.toml"
mkdir -p "$(dirname "$OMNIVERSE_CONFIG")"

cat > "$OMNIVERSE_CONFIG" << EOF
[library_root]
default = "${OMNI_SERVER}"

[settings]
privacy_consent = "Y"
accept_eula = "Y"

[servers."${OMNI_SERVER}"]
enabled = true
username = "${OMNI_USER}"
password = "${OMNI_PASS}"
EOF

# Isaac Sim 설정에 assets root 추가 ( 다음 스크립트 버전에서 환경 변수는 위쪽으로 명시할 것)
ISAAC_VERSION="5.0"
if [ -n "${OMNI_SERVER}" ]; then
    ASSET_ROOT="${OMNI_SERVER}/NVIDIA/Assets/Isaac/${ISAAC_VERSION}"
    EXTRA_ARGS+=" --/persistent/isaac/asset_root/default=${ASSET_ROOT}"
fi

echo "✓ Nucleus connection configured"

# --- 6. Create Startup Stage and Camera Script ---
echo "=== Preparing Startup Stage and Camera Script ==="
SCRIPT_DIR="/isaac-sim/scripts"
OPEN_STAGE_SCRIPT="${SCRIPT_DIR}/open_stage_with_camera.py"
mkdir -p "$SCRIPT_DIR"

cat <<EOF > "$OPEN_STAGE_SCRIPT"
import sys
import omni.usd
import omni.kit.viewport.utility as vp_utils
import carb
import asyncio
from pxr import UsdGeom

async def setup_stage_and_camera(stage_path, camera_path=None):
    """Open USD stage and optionally set camera view"""
    try:
        # Open the stage
        print(f"[setup_stage_and_camera] Opening stage: {stage_path}")
        omni.usd.get_context().open_stage(stage_path)
        
        # Wait for stage to load
        await asyncio.sleep(2.0)
        
        # Set camera if specified
        if camera_path:
            print(f"[setup_stage_and_camera] Setting camera: {camera_path}")
            
            stage = omni.usd.get_context().get_stage()
            if stage:
                camera_prim = stage.GetPrimAtPath(camera_path)
                
                if camera_prim and camera_prim.IsValid():
                    # Check if it's a camera prim
                    if camera_prim.GetTypeName() == "Camera":
                        # Set as active camera in viewport
                        viewport_api = vp_utils.get_active_viewport()
                        if viewport_api:
                            viewport_api.set_active_camera(camera_path)
                            print(f"[setup_stage_and_camera] Successfully activated camera: {camera_path}")
                        else:
                            print("[setup_stage_and_camera] Could not get active viewport")
                    else:
                        print(f"[setup_stage_and_camera] Prim at {camera_path} is not a Camera type: {camera_prim.GetTypeName()}")
                else:
                    print(f"[setup_stage_and_camera] Camera prim not found at: {camera_path}")
                    
                    # List available cameras for debugging
                    print("Available cameras in scene:")
                    for prim in stage.Traverse():
                        if prim.GetTypeName() == "Camera":
                            print(f"  - {prim.GetPath()}")
            else:
                print("[setup_stage_and_camera] Stage not available")
        
        carb.log_info(f"Stage setup completed: {stage_path}")
        
    except Exception as e:
        carb.log_error(f"Error in setup_stage_and_camera: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if len(sys.argv) > 1:
    stage_path = sys.argv[1]
    camera_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"[open_stage_with_camera.py] Stage: {stage_path}")
    if camera_path:
        print(f"[open_stage_with_camera.py] Camera: {camera_path}")
    
    # Run the async function
    asyncio.ensure_future(setup_stage_and_camera(stage_path, camera_path))
else:
    carb.log_warn("No stage path provided to open_stage_with_camera.py script.")
EOF

chmod +x "$OPEN_STAGE_SCRIPT"
echo "✓ Startup script created at ${OPEN_STAGE_SCRIPT}"

# --- 7. Start Isaac Sim ---
echo "=== Container Ready ==="

# Build Kit settings arguments for extension paths
if [ ${#extension_paths[@]} -gt 0 ]; then
    echo "Building Kit settings arguments for ${#extension_paths[@]} extensions..."
    for ext_path in "${extension_paths[@]}"; do
        EXTRA_ARGS+=" --/app/exts/folders+=${ext_path}"
    done
    echo "Final extension Kit arguments: $EXTRA_ARGS"
fi

# Build execution command for opening a startup stage and camera
if [ -n "${STARTUP_USD_STAGE}" ]; then
    if [ -n "${STARTUP_CAMERA_PATH}" ]; then
        EXEC_CMD="--exec \"${OPEN_STAGE_SCRIPT} ${STARTUP_USD_STAGE} ${STARTUP_CAMERA_PATH}\""
        echo "Will open stage: ${STARTUP_USD_STAGE} with camera: ${STARTUP_CAMERA_PATH}"
    else
        EXEC_CMD="--exec \"${OPEN_STAGE_SCRIPT} ${STARTUP_USD_STAGE}\""
        echo "Will open stage: ${STARTUP_USD_STAGE} (no camera specified)"
    fi
fi

# Determine execution mode and inject arguments
ARGS=("$@")
if [ "${START_GUI}" = "true" ]; then
    echo "Starting Isaac Sim GUI..."
    exec /isaac-sim/isaac-sim.sh ${EXTRA_ARGS} ${EXEC_CMD} "${ARGS[@]}"
elif [ "${START_PYTHON}" = "true" ]; then
    echo "Starting Isaac Sim Python..."
    exec /isaac-sim/python.sh ${EXTRA_ARGS} ${EXEC_CMD} "${ARGS[@]}"
elif [ "${START_BASH}" = "true" ]; then
    echo "Starting bash session..."
    exec /bin/bash
elif [ ${#ARGS[@]} -eq 0 ]; then
    echo "Starting Isaac Sim Headless..."
    if [ -n "${STARTUP_USD_STAGE}" ]; then
        if [ -n "${STARTUP_CAMERA_PATH}" ]; then
            exec /isaac-sim/runapp.sh \
            ${EXTRA_ARGS} \
            --exec "${OPEN_STAGE_SCRIPT} ${STARTUP_USD_STAGE} ${STARTUP_CAMERA_PATH}"
        else
            exec /isaac-sim/runapp.sh \
            ${EXTRA_ARGS} \
            --exec "${OPEN_STAGE_SCRIPT} ${STARTUP_USD_STAGE}"
        fi
    else
        exec /isaac-sim/runapp.sh ${EXTRA_ARGS}
    fi
else
    # 기본적으로 runheadless.sh 사용하되, webrtc 요청시에도 처리 ; 다음 버전에서 지워버릴 예정
    if [[ "${ARGS[0]}" == *"webrtc"* ]]; then
        echo "WebRTC streaming requested, using runheadless.sh with streaming support..."
        exec /isaac-sim/runapp.sh --enable-kit-web-ui ${EXTRA_ARGS} ${EXEC_CMD}
    else
        echo "Executing: ${ARGS[@]} with extensions and stage loading..."
        exec "${ARGS[@]}" ${EXTRA_ARGS} ${EXEC_CMD}
    fi
fi

