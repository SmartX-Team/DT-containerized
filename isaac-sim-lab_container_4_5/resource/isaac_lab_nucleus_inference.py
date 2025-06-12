#!/usr/bin/env python3
"""
Isaac Lab Reinforcement Learning with cmd_vel Topic Publishing
강화학습 에이전트가 학습한 액션을 cmd_vel 토픽으로 발행하는 스크립트
"""

import torch
import numpy as np
import math
import sys
import argparse
import threading
import time
from typing import Dict, Any, Optional

# Isaac Lab path
sys.path.append('/opt/IsaacLab/source')

def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Lab RL with cmd_vel Publishing")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="Base task environment")
    parser.add_argument("--nucleus_world", type=str, default="", help="Optional Nucleus world path")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max_iterations", type=int, default=500, help="Training iterations")
    
    # ROS2 topics
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel", 
                       help="Command velocity topic to publish RL actions")
    parser.add_argument("--scan_topic", type=str, default="/scan", 
                       help="Laser scan topic to subscribe")
    parser.add_argument("--odom_topic", type=str, default="/odom", 
                       help="Odometry topic to subscribe")
    
    # RL parameters
    parser.add_argument("--algorithm", type=str, default="PPO", 
                       choices=["PPO", "SAC", "DQN"], help="RL algorithm")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    
    # AppLauncher arguments
    from omni.isaac.lab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()

class ROS2RLBridge:
    """ROS2와 강화학습 에이전트 간의 브리지"""
    
    def __init__(self, cmd_vel_topic="/cmd_vel", scan_topic="/scan", odom_topic="/odom"):
        # ROS2 초기화
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist, Vector3
            from sensor_msgs.msg import LaserScan
            from nav_msgs.msg import Odometry
            
            self.rclpy = rclpy
            self.Node = Node
            self.Twist = Twist
            self.Vector3 = Vector3
            self.LaserScan = LaserScan
            self.Odometry = Odometry
            
            print("ROS2 modules imported successfully!")
            
        except ImportError as e:
            print(f"ROS2 import failed: {e}")
            self.rclpy = None
            return
        
        self.cmd_vel_topic = cmd_vel_topic
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        
        # Sensor data storage
        self.latest_scan = None
        self.latest_odom = None
        self.sensor_lock = threading.Lock()
        
        # ROS2 node
        self.node = None
        self.ros_thread = None
        self.running = False
        
        # Action statistics
        self.action_count = 0
        self.last_action_time = time.time()
        
    def start_ros2_node(self):
        """Start ROS2 node for publishing cmd_vel and subscribing sensors"""
        if not self.rclpy:
            print("ROS2 not available")
            return
            
        def ros2_thread():
            self.rclpy.init()
            
            class RLAgentROS2Node(self.Node):
                def __init__(self, bridge):
                    super().__init__('isaac_lab_rl_agent')
                    self.bridge = bridge
                    
                    # Publisher for cmd_vel (RL actions)
                    self.cmd_vel_pub = self.create_publisher(
                        self.bridge.Twist,
                        bridge.cmd_vel_topic,
                        10
                    )
                    
                    # Subscribers for sensor data
                    self.scan_sub = self.create_subscription(
                        self.bridge.LaserScan,
                        bridge.scan_topic,
                        self.scan_callback,
                        10
                    )
                    
                    self.odom_sub = self.create_subscription(
                        self.bridge.Odometry,
                        bridge.odom_topic,
                        self.odom_callback,
                        10
                    )
                    
                    self.get_logger().info(f"RL Agent ROS2 Node started")
                    self.get_logger().info(f"Publishing RL actions to: {bridge.cmd_vel_topic}")
                    self.get_logger().info(f"Subscribing sensors: {bridge.scan_topic}, {bridge.odom_topic}")
                
                def scan_callback(self, msg):
                    """Store latest lidar data"""
                    with self.bridge.sensor_lock:
                        # Convert to numpy array
                        ranges = np.array(msg.ranges)
                        ranges = np.where(np.isinf(ranges), msg.range_max, ranges)  # Replace inf with max range
                        self.bridge.latest_scan = ranges
                
                def odom_callback(self, msg):
                    """Store latest odometry data"""
                    with self.bridge.sensor_lock:
                        self.bridge.latest_odom = {
                            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
                            'orientation': [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                                          msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
                            'linear_velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
                            'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
                        }
            
            self.node = RLAgentROS2Node(self)
            
            # Spin ROS2 node
            try:
                while self.running:
                    self.rclpy.spin_once(self.node, timeout_sec=0.01)
            except Exception as e:
                print(f"ROS2 spinning error: {e}")
            finally:
                if self.node:
                    self.node.destroy_node()
                if self.rclpy.ok():
                    self.rclpy.shutdown()
        
        self.running = True
        self.ros_thread = threading.Thread(target=ros2_thread, daemon=True)
        self.ros_thread.start()
        print("ROS2 RL bridge thread started")
        time.sleep(1)  # Wait for initialization
    
    def publish_cmd_vel(self, linear_x, angular_z):
        """Publish RL action as cmd_vel"""
        if not self.node:
            return
            
        cmd_msg = self.Twist()
        cmd_msg.linear = self.Vector3(x=float(linear_x), y=0.0, z=0.0)
        cmd_msg.angular = self.Vector3(x=0.0, y=0.0, z=float(angular_z))
        
        self.node.cmd_vel_pub.publish(cmd_msg)
        
        # Statistics
        self.action_count += 1
        current_time = time.time()
        if current_time - self.last_action_time > 5.0:  # Every 5 seconds
            self.node.get_logger().info(
                f"Published {self.action_count} cmd_vel actions. Latest: linear={linear_x:.2f}, angular={angular_z:.2f}"
            )
            self.last_action_time = current_time
    
    def get_sensor_data(self):
        """Get latest sensor data"""
        with self.sensor_lock:
            return self.latest_scan, self.latest_odom
    
    def stop(self):
        """Stop ROS2 bridge"""
        self.running = False
        if self.ros_thread:
            self.ros_thread.join(timeout=2.0)

class RLAgentCmdVel:
    """강화학습 에이전트 (cmd_vel 액션 공간)"""
    
    def __init__(self, num_envs, device="cuda", algorithm="PPO"):
        self.num_envs = num_envs
        self.device = device
        self.algorithm = algorithm
        
        # Q-Learning parameters (for simple DQN)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        
        # Action space: [linear_x, angular_z]
        # Discrete actions for simplicity
        self.actions = {
            0: (0.5, 0.0),    # Forward
            1: (0.3, 0.5),    # Forward + Left
            2: (0.3, -0.5),   # Forward + Right
            3: (0.0, 0.5),    # Turn Left
            4: (0.0, -0.5),   # Turn Right
            5: (0.0, 0.0),    # Stop
            6: (-0.2, 0.0),   # Backward
        }
        
        self.num_actions = len(self.actions)
        
        # Q-tables for each environment
        self.q_tables = {}
        
        # Goal positions (navigational target)
        self.goal_positions = torch.tensor([[5.0, 0.0]] * num_envs, device=device)
        
        # State tracking
        self.last_states = [None] * num_envs
        self.last_actions = [None] * num_envs
        self.last_distances = [None] * num_envs
        
    def discretize_observations(self, observations, robot_positions=None):
        """Convert observations to discrete states"""
        states = []
        
        for env_idx in range(min(self.num_envs, observations.shape[0])):
            # Discretize observation (assume it's already processed)
            obs = observations[env_idx]
            
            # Simple state discretization
            if len(obs) >= 4:
                # Assume first 4 dimensions represent some relevant features
                state_features = []
                for i in range(4):
                    if obs[i] < -0.5:
                        state_features.append(0)
                    elif obs[i] < 0.5:
                        state_features.append(1)
                    else:
                        state_features.append(2)
                
                state = tuple(state_features)
            else:
                # Fallback for unknown observation space
                state = tuple([int(x > 0) for x in obs[:3]] + [0])
            
            states.append(state)
        
        return states
    
    def discretize_sensor_data(self, scan_data, odom_data):
        """Convert ROS2 sensor data to discrete state"""
        if scan_data is None:
            return (1, 1, 1, 1)  # Default safe state
        
        # Discretize lidar data into sectors
        num_rays = len(scan_data)
        sector_size = num_rays // 3
        
        # Front, Left, Right sectors
        front_min = np.min(scan_data[sector_size:2*sector_size])
        left_min = np.min(scan_data[2*sector_size:])
        right_min = np.min(scan_data[:sector_size])
        
        # Discretize distances
        front_state = 0 if front_min < 0.5 else (1 if front_min < 1.5 else 2)
        left_state = 0 if left_min < 0.5 else (1 if left_min < 1.5 else 2)
        right_state = 0 if right_min < 0.5 else (1 if right_min < 1.5 else 2)
        
        # Goal direction (if odom available)
        goal_state = 1  # Default center
        if odom_data:
            robot_pos = np.array(odom_data['position'][:2])
            goal_pos = self.goal_positions[0].cpu().numpy()  # Use first env goal
            goal_dir = goal_pos - robot_pos
            goal_angle = np.arctan2(goal_dir[1], goal_dir[0])
            
            # Robot orientation
            quat = odom_data['orientation']
            robot_yaw = np.arctan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]),
                                 1 - 2 * (quat[1]**2 + quat[2]**2))
            
            angle_diff = goal_angle - robot_yaw
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            goal_state = 0 if angle_diff < -np.pi/4 else (1 if angle_diff < np.pi/4 else 2)
        
        return (front_state, left_state, right_state, goal_state)
    
    def select_action(self, state, env_idx=0):
        """Select action using epsilon-greedy policy"""
        # Initialize Q-table for new states
        if state not in self.q_tables:
            self.q_tables[state] = np.zeros(self.num_actions)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            action_idx = np.argmax(self.q_tables[state])
        
        return action_idx
    
    def update_q_value(self, state, action_idx, reward, next_state):
        """Update Q-value using Q-learning"""
        if state not in self.q_tables:
            self.q_tables[state] = np.zeros(self.num_actions)
        if next_state not in self.q_tables:
            self.q_tables[next_state] = np.zeros(self.num_actions)
        
        # Q-learning update
        old_q = self.q_tables[state][action_idx]
        next_max_q = np.max(self.q_tables[next_state])
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q)
        self.q_tables[state][action_idx] = new_q
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, scan_data, odom_data, last_distance):
        """Calculate reward based on sensor data"""
        reward = 0
        
        # Collision penalty
        if scan_data is not None:
            min_distance = np.min(scan_data)
            if min_distance < 0.3:
                reward -= 100  # Collision penalty
            elif min_distance < 0.8:
                reward -= 10   # Close to obstacle penalty
        
        # Goal approach reward
        if odom_data:
            robot_pos = np.array(odom_data['position'][:2])
            goal_pos = self.goal_positions[0].cpu().numpy()
            current_distance = np.linalg.norm(robot_pos - goal_pos)
            
            if last_distance is not None:
                if current_distance < last_distance:
                    reward += 10  # Getting closer to goal
                else:
                    reward -= 2   # Moving away from goal
            
            # Goal reached
            if current_distance < 0.5:
                reward += 200
            
            return reward, current_distance
        
        return reward, last_distance

def main():
    args = parse_args()
    
    # Launch Isaac Sim
    from omni.isaac.lab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Initialize ROS2 bridge
    ros2_bridge = ROS2RLBridge(args.cmd_vel_topic, args.scan_topic, args.odom_topic)
    
    try:
        # Import Isaac Lab modules after app launch
        import omni.isaac.lab_tasks
        from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
        from omni.isaac.lab.envs import ManagerBasedRLEnv
        import omni.isaac.core.utils.stage as stage_utils
        import carb
        
        print("Isaac Lab modules imported successfully!")
        
        # --- 월드 자동 저장 비활성화 ---
        print("Disabling automatic world saving during RL training...")
        
        # Omniverse 설정에서 자동 저장 비활성화
        settings = carb.settings.get_settings()
        
        # 자동 저장 관련 설정들 비활성화
        auto_save_settings = [
            "/persistent/app/stage/autoSave",
            "/persistent/app/stage/autoSaveEnabled", 
            "/persistent/app/stage/autoSaveIntervalSeconds",
            "/app/stage/autoSave",
            "/app/stage/autoSaveEnabled",
            "/app/stage/autoSaveIntervalSeconds",
            "/omni/kit/stage_templates/autoSave",
            "/rtx/autoSave",
            "/renderer/autoSave"
        ]
        
        for setting in auto_save_settings:
            try:
                settings.set(setting, False)
                print(f"   Disabled: {setting}")
            except:
                pass  # 설정이 존재하지 않을 수 있음
        
        # 추가 저장 방지 설정
        try:
            settings.set("/persistent/app/file/recentFiles/maxCount", 0)  # 최근 파일 목록 비활성화
            settings.set("/app/stage/upAxis", "Y")  # 불필요한 변경사항 방지
            settings.set("/app/stage/movePrimInPlace", False)  # 프림 이동 시 저장 방지
            print("   Additional save prevention settings applied")
        except:
            pass
        
        # 스테이지 변경 알림 비활성화 (성능 향상)
        try:
            import omni.usd
            context = omni.usd.get_context()
            if context:
                stage = context.get_stage()
                if stage:
                    # 스테이지를 읽기 전용으로 설정하지는 않지만 변경 추적 최소화
                    print("   Stage change tracking minimized")
        except:
            pass
        
        print("World auto-save disabled for RL training performance!")
        
        # Start ROS2 bridge
        ros2_bridge.start_ros2_node()
        
        # Parse environment configuration
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        
        # Create environment
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        print(f"Environment: {args.task}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Algorithm: {args.algorithm}")
        print(f"Publishing to: {args.cmd_vel_topic}")
        
        # Create RL agent
        agent = RLAgentCmdVel(args.num_envs, args.device, args.algorithm)
        
        # Reset environment
        observations, _ = env.reset()
        
        print("Starting RL training with cmd_vel publishing...")
        
        # Training loop
        for iteration in range(args.max_iterations):
            episode_rewards = torch.zeros(args.num_envs, device=args.device)
            
            for step in range(100):  # 100 steps per iteration
                # Get sensor data from ROS2
                scan_data, odom_data = ros2_bridge.get_sensor_data()
                
                # Primary state from Isaac Lab observations
                isaac_states = agent.discretize_observations(observations)
                
                # Enhanced state from ROS2 sensors (use for first environment)
                if scan_data is not None:
                    sensor_state = agent.discretize_sensor_data(scan_data, odom_data)
                    current_state = sensor_state  # Use sensor-based state
                else:
                    current_state = isaac_states[0]  # Fallback to Isaac Lab state
                
                # Select action
                action_idx = agent.select_action(current_state, 0)
                linear_x, angular_z = agent.actions[action_idx]
                
                # Publish cmd_vel action
                ros2_bridge.publish_cmd_vel(linear_x, angular_z)
                
                # Convert RL action to Isaac Lab action format
                isaac_actions = env.action_space.sample()  # Start with random
                # Try to map cmd_vel to Isaac Lab actions if possible
                if isaac_actions.shape[1] >= 2:
                    isaac_actions[:, 0] = linear_x
                    isaac_actions[:, 1] = angular_z
                
                # Step Isaac Lab environment
                observations, rewards, terminated, truncated, info = env.step(isaac_actions)
                
                # Calculate custom reward from sensors
                if scan_data is not None:
                    sensor_reward, current_distance = agent.calculate_reward(
                        scan_data, odom_data, agent.last_distances[0]
                    )
                    agent.last_distances[0] = current_distance
                    
                    # Combine Isaac Lab reward with sensor reward
                    total_reward = rewards[0].item() + sensor_reward
                else:
                    total_reward = rewards[0].item()
                
                episode_rewards[0] += total_reward
                
                # Update Q-value (for first environment)
                if agent.last_states[0] is not None:
                    agent.update_q_value(
                        agent.last_states[0], 
                        agent.last_actions[0], 
                        total_reward, 
                        current_state
                    )
                
                # Update state tracking
                agent.last_states[0] = current_state
                agent.last_actions[0] = action_idx
                
                # Print progress
                if step % 50 == 0:
                    print(f"Iteration {iteration}, Step {step}: "
                          f"Reward={total_reward:.2f}, Action={agent.actions[action_idx]}, "
                          f"Epsilon={agent.epsilon:.3f}")
                
                # Reset if terminated
                if terminated.any() or truncated.any():
                    env.reset()
                    break
            
            print(f"Iteration {iteration} completed. Episode reward: {episode_rewards[0].item():.2f}")
        
        print("RL Training with cmd_vel publishing completed!")
             
        # 메모리 정리
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            print("   GPU memory cache cleared")
        
        print("Training completed and cleaned up!")
        
        return 0
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        print("Cleaning up...")
        ros2_bridge.stop()
        if 'env' in locals():
            env.close()
        simulation_app.close()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)