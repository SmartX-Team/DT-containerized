#!/bin/bash
set -e

echo "=== Isaac Sim Container Starting ==="

# --- 1. Start virtual display (Xvfb) with enhanced stability ---
# Remove any leftover X-lock files from abnormal termination to prevent conflicts
rm -f /tmp/.X1-lock
echo "Starting Xvfb on display :1"
Xvfb :1 -screen 0 1920x1080x24 &
export DISPLAY=:1

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
               echo "   Successfully linked: $ext_name"
               echo "    $ext_path -> $target_link"
           else
               echo "   Failed to link: $ext_name"
           fi
       else
           echo "   Invalid extension structure (missing config/extension.toml)"
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

# Extension search paths 추가
if 'exts' not in config:
    config['exts'] = {}

if 'folders' not in config['exts']:
    config['exts']['folders'] = []

# /isaac-sim/exts 경로 추가 (심볼릭 링크들이 있는 곳)
ext_path = '/isaac-sim/exts'
if ext_path not in config['exts']['folders']:
    config['exts']['folders'].append(ext_path)
    print(f'Added {ext_path} to extension search paths')

# 저장
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

# --- 5. Start Isaac Sim with Kit settings for extensions ---
echo "=== Container Ready ==="

# Build Kit settings arguments for extension paths
EXTRA_ARGS=""
if [ ${#extension_paths[@]} -gt 0 ]; then
    echo "Building Kit settings arguments for ${#extension_paths[@]} extensions..."
    for ext_path in "${extension_paths[@]}"; do
        EXTRA_ARGS+=" --/app/exts/folders+=${ext_path}"
        echo "  Added Kit setting: --/app/exts/folders+=${ext_path}"
    done
    echo "Final Kit arguments: $EXTRA_ARGS"
else
    echo "No extensions to register via Kit settings"
fi

# Determine execution mode and inject extension paths via Kit settings
ARGS=("$@")
if [ "${START_GUI}" = "true" ]; then
    echo "Starting Isaac Sim GUI with extension Kit settings..."
    exec /isaac-sim/isaac-sim.sh ${EXTRA_ARGS} "${ARGS[@]}"
elif [ "${START_PYTHON}" = "true" ]; then
    echo "Starting Isaac Sim Python with extension Kit settings..."
    exec /isaac-sim/python.sh ${EXTRA_ARGS} "${ARGS[@]}"
elif [ "${START_BASH}" = "true" ]; then
    echo "Starting bash session..."
    exec /bin/bash
elif [ ${#ARGS[@]} -eq 0 ]; then
    echo "Starting Isaac Sim Headless with extension Kit settings..."
    exec /isaac-sim/runheadless.sh ${EXTRA_ARGS}
else
    echo "Executing with extension Kit settings: ${ARGS[@]}"
    exec "${ARGS[@]}" ${EXTRA_ARGS}
fi