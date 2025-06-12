import torch
import numpy as np
import math
import sys
import argparse
from typing import Dict, Any

# Isaac Lab imports
sys.path.append('/opt/IsaacLab/source')

def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Lab Navigation RL")
    parser.add_argument("--task", type=str, default="Isaac-Navigation-v0", help="Task name")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--episode_length", type=int, default=1000, help="Episode length")
    
    # AppLauncher arguments
    from omni.isaac.lab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Launch Isaac Sim
    from omni.isaac.lab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    try:
        # Isaac Lab imports (after app launch)
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
        from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
        from omni.isaac.lab.managers import EventTermCfg as EventTerm
        from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
        from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
        from omni.isaac.lab.managers import RewardTermCfg as RewTerm
        from omni.isaac.lab.managers import SceneEntityCfg
        from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
        from omni.isaac.lab.scene import InteractiveSceneCfg
        from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
        from omni.isaac.lab.terrains import TerrainImporterCfg
        from omni.isaac.lab.utils import configclass
        from omni.isaac.lab.utils.math import subtract_frame_transforms
        
        print("Isaac Lab modules imported successfully!")
        
        # Custom Navigation Environment Configuration
        @configclass
        class NavigationEnvCfg(ManagerBasedEnvCfg):
            """Isaac Lab Navigation Environment Configuration"""
            
            def __post_init__(self):
                """Post initialization"""
                # simulation settings
                self.decimation = 2
                self.episode_length_s = 20.0  # 20 seconds episode
                
                # viewer settings
                self.viewer.eye = (8.0, 0.0, 5.0)
                self.viewer.lookat = (0.0, 0.0, 0.0)
                
            # Scene settings
            scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=4.0, replicate_physics=True)
            
            # Basic scene
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                terrain_generator=None,
                max_init_terrain_level=5,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                debug_vis=False,
            )
            
            # Robot configuration
            robot_cfg: ArticulationCfg = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/isaac-sim/standalone_examples/api/omni.isaac.core/robots/Jetbot/jetbot.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        retain_accelerations=False,
                        linear_damping=0.0,
                        angular_damping=0.0,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1000.0,
                        max_depenetration_velocity=1.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=4,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.05),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "left_wheel": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["left_wheel"],
                        effort_limit=10.0,
                        velocity_limit=10.0,
                        stiffness=0.0,
                        damping=10.0,
                    ),
                    "right_wheel": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["right_wheel"],
                        effort_limit=10.0,
                        velocity_limit=10.0,
                        stiffness=0.0,
                        damping=10.0,
                    ),
                },
            )
            
            # Lidar sensor configuration
            lidar: RayCasterCfg = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/chassis",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
                attach_yaw_only=True,
                pattern_cfg=patterns.LidarPatternCfg(
                    channels=1,
                    vertical_fov_range=(0.0, 0.0),
                    horizontal_fov_range=(-180.0, 180.0),
                    horizontal_res=1.0,
                ),
                max_distance=5.0,
                drift_range=(-0.0, 0.0),
            )
            
            # Observations
            observations: ObsGroup = ObsGroup()
            
            # Actions (left wheel velocity, right wheel velocity)
            actions: int = 2
            
            # Events (for environment reset)
            events: dict = {}
            
            # Rewards
            rewards: dict = {}
            
            # Terminations
            terminations: dict = {}
        
        # Q-Learning Agent for Isaac Lab
        class QLearningAgent:
            def __init__(self, num_envs, num_actions=3, device="cuda"):
                self.num_envs = num_envs
                self.num_actions = num_actions
                self.device = device
                
                # Q-Learning parameters
                self.learning_rate = 0.1
                self.discount_factor = 0.9
                self.epsilon = 0.9
                self.epsilon_decay = 0.995
                self.min_epsilon = 0.05
                
                # Q-tables for each environment (simplified discrete states)
                self.q_tables = {}
                
                # Actions: 0=forward, 1=turn_left, 2=turn_right
                self.action_map = {
                    0: torch.tensor([1.0, 1.0]),   # forward
                    1: torch.tensor([0.5, -0.5]), # turn left
                    2: torch.tensor([-0.5, 0.5])  # turn right
                }
                
                # State tracking
                self.last_states = [None] * num_envs
                self.last_actions = [None] * num_envs
                self.goal_positions = torch.tensor([[5.0, 0.0]] * num_envs, device=device)
                
            def discretize_state(self, lidar_data, robot_pos, robot_quat):
                """Convert continuous observations to discrete states"""
                batch_size = lidar_data.shape[0]
                states = []
                
                for i in range(batch_size):
                    # Discretize lidar (front, left, right sectors)
                    ranges = lidar_data[i, 0, :]  # Get first channel
                    num_rays = ranges.shape[0]
                    
                    # Divide into 3 sectors
                    sector_size = num_rays // 3
                    front_min = torch.min(ranges[sector_size:2*sector_size])
                    left_min = torch.min(ranges[2*sector_size:])
                    right_min = torch.min(ranges[:sector_size])
                    
                    # Discretize distances
                    front_state = 0 if front_min < 0.5 else (1 if front_min < 1.0 else 2)
                    left_state = 0 if left_min < 0.5 else (1 if left_min < 1.0 else 2)
                    right_state = 0 if right_min < 0.5 else (1 if right_min < 1.0 else 2)
                    
                    # Goal direction (simplified)
                    goal_dir = self.goal_positions[i] - robot_pos[i, :2]
                    goal_angle = torch.atan2(goal_dir[1], goal_dir[0])
                    
                    # Robot orientation (from quaternion to yaw)
                    robot_yaw = torch.atan2(
                        2 * (robot_quat[i, 3] * robot_quat[i, 2] + robot_quat[i, 0] * robot_quat[i, 1]),
                        1 - 2 * (robot_quat[i, 1]**2 + robot_quat[i, 2]**2)
                    )
                    
                    angle_diff = goal_angle - robot_yaw
                    # Normalize angle to [-pi, pi]
                    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                    
                    # Discretize angle
                    angle_state = 0 if angle_diff < -math.pi/4 else (1 if angle_diff < math.pi/4 else 2)
                    
                    state = (front_state, left_state, right_state, angle_state)
                    states.append(state)
                
                return states
            
            def select_actions(self, states):
                """Select actions using epsilon-greedy policy"""
                actions = []
                
                for i, state in enumerate(states):
                    # Initialize Q-table for new states
                    if state not in self.q_tables:
                        self.q_tables[state] = np.zeros(self.num_actions)
                    
                    # Epsilon-greedy action selection
                    if np.random.random() < self.epsilon:
                        action_idx = np.random.randint(self.num_actions)
                    else:
                        action_idx = np.argmax(self.q_tables[state])
                    
                    actions.append(action_idx)
                
                # Convert to wheel velocities
                wheel_velocities = torch.zeros((len(actions), 2), device=self.device)
                for i, action_idx in enumerate(actions):
                    wheel_velocities[i] = self.action_map[action_idx].to(self.device)
                
                return wheel_velocities, actions
            
            def update_q_table(self, states, actions, rewards, next_states):
                """Update Q-table using Q-learning update rule"""
                for i in range(len(states)):
                    if self.last_states[i] is not None:
                        last_state = self.last_states[i]
                        last_action = self.last_actions[i]
                        reward = rewards[i].item()
                        
                        # Initialize Q-table for new states
                        if next_states[i] not in self.q_tables:
                            self.q_tables[next_states[i]] = np.zeros(self.num_actions)
                        
                        # Q-learning update
                        old_q = self.q_tables[last_state][last_action]
                        next_max_q = np.max(self.q_tables[next_states[i]])
                        new_q = old_q + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q)
                        self.q_tables[last_state][last_action] = new_q
                
                # Update state tracking
                self.last_states = states.copy()
                self.last_actions = actions.copy()
                
                # Decay epsilon
                if self.epsilon > self.min_epsilon:
                    self.epsilon *= self.epsilon_decay
        
        print("Creating Navigation Environment...")
        
        # Create environment configuration
        env_cfg = NavigationEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        
        # Create environment
        env = ManagerBasedEnv(cfg=env_cfg)
        
        print(f"Environment created with {args.num_envs} parallel environments")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Create Q-Learning agent
        agent = QLearningAgent(args.num_envs, device=args.device)
        
        print("Starting Q-Learning Navigation Training...")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Training loop
        for episode in range(100):  # 100 episodes
            episode_rewards = torch.zeros(args.num_envs, device=args.device)
            
            for step in range(args.episode_length):
                # Get robot states
                robot_pos = env.scene["robot"].data.root_pos_w
                robot_quat = env.scene["robot"].data.root_quat_w
                
                # Get lidar data (if available)
                if "lidar" in env.scene:
                    lidar_data = env.scene["lidar"].data.ray_hits_w
                else:
                    # Create dummy lidar data for testing
                    lidar_data = torch.ones((args.num_envs, 1, 360), device=args.device) * 2.0
                
                # Discretize states
                states = agent.discretize_state(lidar_data, robot_pos, robot_quat)
                
                # Select actions
                wheel_velocities, action_indices = agent.select_actions(states)
                
                # Convert to Isaac Lab action format
                actions = wheel_velocities * 5.0  # Scale velocities
                
                # Step environment
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # Calculate custom rewards
                goal_distances = torch.norm(robot_pos[:, :2] - agent.goal_positions, dim=1)
                custom_rewards = -goal_distances * 0.1  # Distance-based reward
                
                # Add collision penalty (if too close to obstacles)
                min_lidar_dist = torch.min(lidar_data.reshape(args.num_envs, -1), dim=1)[0]
                collision_penalty = torch.where(min_lidar_dist < 0.3, -10.0, 0.0)
                custom_rewards += collision_penalty
                
                # Goal reached reward
                goal_reached = goal_distances < 0.5
                custom_rewards += torch.where(goal_reached, 100.0, 0.0)
                
                episode_rewards += custom_rewards
                
                # Update Q-table
                next_states = agent.discretize_state(lidar_data, robot_pos, robot_quat)
                agent.update_q_table(states, action_indices, custom_rewards, next_states)
                
                # Print progress
                if step % 100 == 0:
                    mean_reward = torch.mean(episode_rewards).item()
                    mean_distance = torch.mean(goal_distances).item()
                    print(f"Episode {episode}, Step {step}: Avg Reward = {mean_reward:.2f}, Avg Distance = {mean_distance:.2f}, Epsilon = {agent.epsilon:.3f}")
                
                # Reset environments that reached goal or terminated
                reset_mask = goal_reached | terminated | truncated
                if reset_mask.any():
                    reset_env_ids = torch.where(reset_mask)[0]
                    env.reset(reset_env_ids)
            
            print(f"✅ Episode {episode} completed. Average reward: {torch.mean(episode_rewards).item():.2f}")
        
        print("🎉 Training completed!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Close environment and simulation
        if 'env' in locals():
            env.close()
        simulation_app.close()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)