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

# --- 4. Set extension paths via environment variable ---
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

# --- 5. Start Isaac Sim ---
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