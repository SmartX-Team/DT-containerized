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

# --- 3. Copy local RL code to Isaac Lab ---
echo "=== Copying Local RL Code to Isaac Lab ==="

# Define source and destination paths
RL_CODE_SOURCE="/mnt/rl_code"
ISAAC_LAB_SCRIPTS="/opt/IsaacLab/source/standalone/workflows/custom"
ISAAC_LAB_PROJECTS="/opt/IsaacLab/projects"

# Create destination directories if they don't exist
mkdir -p "$ISAAC_LAB_SCRIPTS"
mkdir -p "$ISAAC_LAB_PROJECTS"

# Copy RL training scripts
if [ -d "$RL_CODE_SOURCE/scripts" ]; then
    echo "Copying RL training scripts..."
    cp -r "$RL_CODE_SOURCE/scripts/"* "$ISAAC_LAB_SCRIPTS/" 2>/dev/null || true
    echo "RL scripts copied to $ISAAC_LAB_SCRIPTS"
fi

# Copy RL projects/experiments
if [ -d "$RL_CODE_SOURCE/projects" ]; then
    echo "Copying RL projects..."
    cp -r "$RL_CODE_SOURCE/projects/"* "$ISAAC_LAB_PROJECTS/" 2>/dev/null || true
    echo "RL projects copied to $ISAAC_LAB_PROJECTS"
fi

# Copy entire RL code directory if it exists
if [ -d "$RL_CODE_SOURCE" ] && [ "$(ls -A $RL_CODE_SOURCE 2>/dev/null)" ]; then
    echo "Copying all RL code from $RL_CODE_SOURCE..."
    
    # Copy Python files to custom workflows
    find "$RL_CODE_SOURCE" -name "*.py" -type f | while read -r py_file; do
        rel_path=$(realpath --relative-to="$RL_CODE_SOURCE" "$py_file")
        dest_dir="$ISAAC_LAB_SCRIPTS/$(dirname "$rel_path")"
        mkdir -p "$dest_dir"
        cp "$py_file" "$dest_dir/"
        echo "Copied: $rel_path"
    done
    
    # Copy configuration files
    find "$RL_CODE_SOURCE" -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.toml" -type f | while read -r config_file; do
        rel_path=$(realpath --relative-to="$RL_CODE_SOURCE" "$config_file")
        dest_dir="$ISAAC_LAB_PROJECTS/$(dirname "$rel_path")"
        mkdir -p "$dest_dir"
        cp "$config_file" "$dest_dir/"
        echo "Copied config: $rel_path"
    done
    
    # Set proper permissions
    chmod -R 755 "$ISAAC_LAB_SCRIPTS"
    chmod -R 755 "$ISAAC_LAB_PROJECTS"
    
    echo "RL code copy completed successfully!"
else
    echo "No RL code found at $RL_CODE_SOURCE (this is normal if not mounting RL code)"
fi

# --- 4. Setup ROS environment ---
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

# --- 5. Set extension paths via environment variable ---
echo "=== Setting up Extension Search Paths via Environment Variable ==="
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
       echo "Added to path: $ext_path"
   done
fi

if [ -n "$CUSTOM_EXT_PATHS" ]; then
   export OMNI_KIT_EXTENSION_PATH="${CUSTOM_EXT_PATHS}:${OMNI_KIT_EXTENSION_PATH:-}"
   echo "Final Extension Path Variable Set."
fi
echo "=== Path Setup Complete ==="

# --- 6. Display useful information ---
echo "=== Container Information ==="
echo "Isaac Lab path: /opt/IsaacLab"
echo "Custom RL scripts: $ISAAC_LAB_SCRIPTS"
echo "RL projects: $ISAAC_LAB_PROJECTS"
echo "SmartX extensions: $SRC_DIR"
echo ""
echo "Quick commands:"
echo "  - Run Isaac Lab training: /opt/IsaacLab/isaaclab.sh -p source/standalone/workflows/custom/your_script.py"
echo "  - List environments: /opt/IsaacLab/isaaclab.sh -p source/standalone/environments/list_envs.py"
echo "  - Access Isaac Lab: cd /opt/IsaacLab"
echo ""

# --- 7. Start Isaac Sim ---
echo "=== Container Ready ==="

# Determine execution mode
ARGS=("$@")
if [ "${START_GUI}" = "true" ]; then
   echo "Starting Isaac Sim GUI..."
   exec /isaac-sim/isaac-sim.sh "${ARGS[@]}"
elif [ "${START_PYTHON}" = "true" ]; then
   echo "Starting Isaac Sim Python..."
   exec /isaac-sim/python.sh "${ARGS[@]}"
elif [ "${START_BASH}" = "true" ]; then
   echo "Starting bash session..."
   exec /bin/bash
elif [ ${#ARGS[@]} -eq 0 ]; then
   echo "Starting Isaac Sim Headless (default)..."
   exec /isaac-sim/runheadless.sh
else
   echo "Executing: ${ARGS[@]}"
   exec "${ARGS[@]}"
fi