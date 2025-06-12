#!/usr/bin/env python3
"""
Isaac Lab ROS2 Navigation with Nucleus World Loading
Nucleus에서 월드를 불러오고 ROS2 토픽으로 로봇을 제어하는 스크립트
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
    parser = argparse.ArgumentParser(description="Isaac Lab ROS2 Navigation with Nucleus World")
    parser.add_argument("--nucleus_world", type=str, default="/World/warehouse", 
                       help="Path to Nucleus world")
    parser.add_argument("--num_envs", type=int, default=1, 
                       help="Number of parallel environments")
    # Note: --headless and --device are automatically added by AppLauncher.add_app_launcher_args()
    parser.add_argument("--robot_prim", type=str, default="/World/Robot", 
                       help="Robot prim path in the scene")
    
    # ROS2 topics
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel", 
                       help="Command velocity topic")
    parser.add_argument("--scan_topic", type=str, default="/scan", 
                       help="Laser scan topic")
    parser.add_argument("--odom_topic", type=str, default="/odom", 
                       help="Odometry topic")
    
    # AppLauncher arguments (includes --headless, --device and other Isaac Sim args)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()

class ROS2Bridge:
    """ROS2와 Isaac Lab 간의 브리지 클래스"""
    
    def __init__(self, cmd_vel_topic="/cmd_vel", scan_topic="/scan", odom_topic="/odom"):
        # ROS2 초기화
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            from sensor_msgs.msg import LaserScan
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
            from std_msgs.msg import Header
            
            self.rclpy = rclpy
            self.Node = Node
            self.Twist = Twist
            self.LaserScan = LaserScan
            self.Odometry = Odometry
            self.Point = Point
            self.Pose = Pose
            self.Quaternion = Quaternion
            self.Vector3 = Vector3
            self.Header = Header
            
            print("ROS2 modules imported successfully!")
            
        except ImportError as e:
            print(f"ROS2 import failed: {e}")
            print("Please make sure ROS2 is properly installed and sourced")
            return
        
        self.cmd_vel_topic = cmd_vel_topic
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        
        # Command velocity storage
        self.latest_cmd_vel = None
        self.cmd_vel_lock = threading.Lock()
        
        # ROS2 node
        self.node = None
        self.ros_thread = None
        self.running = False
        
    def start_ros2_node(self):
        """Start ROS2 node in separate thread"""
        if not hasattr(self, 'rclpy'):
            print("ROS2 not available")
            return
            
        def ros2_thread():
            self.rclpy.init()
            
            class IsaacLabROS2Node(self.Node):
                def __init__(self, bridge):
                    super().__init__('isaac_lab_ros2_bridge')
                    self.bridge = bridge
                    
                    # Subscribers
                    self.cmd_vel_sub = self.create_subscription(
                        self.bridge.Twist,
                        bridge.cmd_vel_topic,
                        self.cmd_vel_callback,
                        10
                    )
                    
                    # Publishers
                    self.scan_pub = self.create_publisher(
                        self.bridge.LaserScan,
                        bridge.scan_topic,
                        10
                    )
                    
                    self.odom_pub = self.create_publisher(
                        self.bridge.Odometry,
                        bridge.odom_topic,
                        10
                    )
                    
                    self.get_logger().info(f"Isaac Lab ROS2 Bridge started")
                    self.get_logger().info(f"Subscribing to: {bridge.cmd_vel_topic}")
                    self.get_logger().info(f"Publishing to: {bridge.scan_topic}, {bridge.odom_topic}")
                
                def cmd_vel_callback(self, msg):
                    with self.bridge.cmd_vel_lock:
                        self.bridge.latest_cmd_vel = {
                            'linear_x': msg.linear.x,
                            'linear_y': msg.linear.y,
                            'angular_z': msg.angular.z
                        }
                    
                    # Debug output
                    if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
                        self.get_logger().info(
                            f"Received cmd_vel: linear.x={msg.linear.x:.2f}, angular.z={msg.angular.z:.2f}",
                            throttle_duration_sec=0.5
                        )
            
            self.node = IsaacLabROS2Node(self)
            
            # Spin ROS2 node
            try:
                while self.running:
                    self.rclpy.spin_once(self.node, timeout_sec=0.1)
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
        print("ROS2 bridge thread started")
    
    def get_cmd_vel(self):
        """Get latest command velocity"""
        with self.cmd_vel_lock:
            return self.latest_cmd_vel.copy() if self.latest_cmd_vel else None
    
    def publish_scan(self, ranges, angle_min=-math.pi, angle_max=math.pi):
        """Publish laser scan data"""
        if not self.node:
            return
            
        scan_msg = self.LaserScan()
        scan_msg.header = self.Header()
        scan_msg.header.stamp = self.node.get_clock().now().to_msg()
        scan_msg.header.frame_id = "laser"
        
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = angle_max
        scan_msg.angle_increment = (angle_max - angle_min) / len(ranges)
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        scan_msg.ranges = ranges.tolist()
        
        self.node.scan_pub.publish(scan_msg)
    
    def publish_odom(self, position, orientation, linear_vel, angular_vel):
        """Publish odometry data"""
        if not self.node:
            return
            
        odom_msg = self.Odometry()
        odom_msg.header = self.Header()
        odom_msg.header.stamp = self.node.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        
        # Position
        odom_msg.pose.pose.position = self.Point(x=position[0], y=position[1], z=position[2])
        odom_msg.pose.pose.orientation = self.Quaternion(
            x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
        )
        
        # Velocity
        odom_msg.twist.twist.linear = self.Vector3(x=linear_vel[0], y=linear_vel[1], z=linear_vel[2])
        odom_msg.twist.twist.angular = self.Vector3(x=angular_vel[0], y=angular_vel[1], z=angular_vel[2])
        
        self.node.odom_pub.publish(odom_msg)
    
    def stop(self):
        """Stop ROS2 bridge"""
        self.running = False
        if self.ros_thread:
            self.ros_thread.join(timeout=2.0)

class NucleusWorldLoader:
    """Nucleus 월드 로딩 클래스"""
    
    @staticmethod
    def load_world(nucleus_path, local_path="/World"):
        """Nucleus에서 월드를 로드"""
        try:
            import omni.isaac.core.utils.stage as stage_utils
            import omni.isaac.core.utils.nucleus as nucleus_utils
            
            print(f"Loading world from Nucleus: {nucleus_path}")
            
            # Nucleus 연결 확인
            nucleus_server = nucleus_utils.get_assets_root_path()
            if nucleus_server:
                print(f"Connected to Nucleus: {nucleus_server}")
            else:
                print("No Nucleus connection, using local assets")
                return False
            
            # 월드 파일 경로 구성
            if nucleus_path.startswith("omniverse://"):
                world_path = nucleus_path
            else:
                world_path = f"{nucleus_server}{nucleus_path}"
            
            print(f"Loading USD file: {world_path}")
            
            # USD 스테이지에 월드 로드
            prim = stage_utils.add_reference_to_stage(world_path, local_path)
            
            if prim:
                print(f"World loaded successfully at {local_path}")
                return True
            else:
                print(f"Failed to load world from {world_path}")
                return False
                
        except Exception as e:
            print(f"Error loading Nucleus world: {e}")
            return False

def main():
    args = parse_args()
    
    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Initialize ROS2 bridge
    ros2_bridge = ROS2Bridge(args.cmd_vel_topic, args.scan_topic, args.odom_topic)
    
    try:
        # Import Isaac Lab modules after app launch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, RigidObjectCfg
        from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import RayCasterCfg, patterns
        from isaaclab.utils import configclass
        import omni.isaac.core.utils.prims as prim_utils
        import omni.isaac.core.utils.stage as stage_utils
        import carb
        
        print("Isaac Lab modules imported successfully!")
        
        # --- 월드 자동 저장 비활성화 ---
        print("Disabling automatic world saving during operation...")
        
        settings = carb.settings.get_settings()
        auto_save_settings = [
            "/persistent/app/stage/autoSave",
            "/persistent/app/stage/autoSaveEnabled", 
            "/persistent/app/stage/autoSaveIntervalSeconds",
            "/app/stage/autoSave",
            "/app/stage/autoSaveEnabled",
            "/app/stage/autoSaveIntervalSeconds"
        ]
        
        for setting in auto_save_settings:
            try:
                settings.set(setting, False)
                print(f"   Disabled: {setting}")
            except:
                pass
        
        print("Auto-save disabled!")
        
        # Start ROS2 bridge
        ros2_bridge.start_ros2_node()
        time.sleep(2)  # Wait for ROS2 to initialize
        
        # Load Nucleus world
        print(f"Loading Nucleus world: {args.nucleus_world}")
        world_loader = NucleusWorldLoader()
        world_loaded = world_loader.load_world(args.nucleus_world)
        
        if not world_loaded:
            print("Using default empty world")
        
        # Robot configuration for ROS2 control
        @configclass
        class ROS2RobotEnvCfg(ManagerBasedEnvCfg):
            def __post_init__(self):
                self.decimation = 4
                self.episode_length_s = 1000.0  # Long episode for continuous operation
                
                # Viewer settings
                self.viewer.eye = (5.0, 5.0, 3.0)
                self.viewer.lookat = (0.0, 0.0, 0.0)
            
            # Scene
            scene: InteractiveSceneCfg = InteractiveSceneCfg(
                num_envs=args.num_envs, 
                env_spacing=2.0, 
                replicate_physics=True
            )
            
            # Robot (find existing robot in scene or spawn default)
            robot: ArticulationCfg = ArticulationCfg(
                prim_path=args.robot_prim,
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/isaac-sim/standalone_examples/api/omni.isaac.core/robots/Jetbot/jetbot.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_linear_velocity=10.0,
                        max_angular_velocity=10.0,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.05),
                    joint_pos={".*": 0.0},
                ),
                actuators={
                    "left_wheel": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["left_wheel"],
                        effort_limit=50.0,
                        velocity_limit=10.0,
                        stiffness=0.0,
                        damping=10.0,
                    ),
                    "right_wheel": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["right_wheel"],
                        effort_limit=50.0,
                        velocity_limit=10.0,
                        stiffness=0.0,
                        damping=10.0,
                    ),
                },
            )
            
            # Lidar sensor
            lidar: RayCasterCfg = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}" + args.robot_prim + "/chassis",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
                attach_yaw_only=True,
                pattern_cfg=patterns.LidarPatternCfg(
                    channels=1,
                    vertical_fov_range=(0.0, 0.0),
                    horizontal_fov_range=(-180.0, 180.0),
                    horizontal_res=1.0,
                ),
                max_distance=10.0,
                drift_range=(-0.0, 0.0),
            )
        
        # Create environment
        print("Creating ROS2-controlled environment...")
        env_cfg = ROS2RobotEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        
        # Check if robot exists in scene
        robot_prim = prim_utils.get_prim_at_path(args.robot_prim)
        if not robot_prim.IsValid():
            print(f"Robot not found at {args.robot_prim}, will spawn default Jetbot")
        else:
            print(f"Found existing robot at {args.robot_prim}")
        
        env = ManagerBasedEnv(cfg=env_cfg)
        
        print(f"Environment created successfully!")
        print(f"Listening for ROS2 commands on: {args.cmd_vel_topic}")
        print(f"Publishing sensor data on: {args.scan_topic}, {args.odom_topic}")
        
        # Reset environment
        obs, _ = env.reset()
        
        print("Starting ROS2 control loop...")
        print("Use 'ros2 topic pub /cmd_vel geometry_msgs/msg/Twist ...' to control robot")
        
        step_count = 0
        last_cmd_time = time.time()
        
        # Main control loop
        while True:
            # Get ROS2 command
            cmd_vel = ros2_bridge.get_cmd_vel()
            
            if cmd_vel:
                last_cmd_time = time.time()
                
                # Convert ROS2 cmd_vel to differential drive
                linear_x = cmd_vel['linear_x']
                angular_z = cmd_vel['angular_z']
                
                # Differential drive kinematics
                # Assume wheel separation of 0.2m
                wheel_separation = 0.2
                wheel_radius = 0.05
                
                # Convert to wheel velocities
                left_wheel_vel = (linear_x - angular_z * wheel_separation / 2) / wheel_radius
                right_wheel_vel = (linear_x + angular_z * wheel_separation / 2) / wheel_radius
                
                # Create action tensor (device will be set by Isaac Lab automatically)
                actions = torch.zeros((args.num_envs, 2), device="cuda")
                actions[:, 0] = left_wheel_vel   # Left wheel
                actions[:, 1] = right_wheel_vel  # Right wheel
                
            else:
                # Stop robot if no recent commands (timeout after 1 second)
                if time.time() - last_cmd_time > 1.0:
                    actions = torch.zeros((args.num_envs, 2), device="cuda")
                else:
                    # Keep last action
                    continue
            
            # Step simulation
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Get robot state for ROS2 publishing
            if hasattr(env.scene, 'robot'):
                robot_pos = env.scene.robot.data.root_pos_w[0].cpu().numpy()
                robot_quat = env.scene.robot.data.root_quat_w[0].cpu().numpy()  # [x,y,z,w]
                robot_vel = env.scene.robot.data.root_lin_vel_w[0].cpu().numpy()
                robot_ang_vel = env.scene.robot.data.root_ang_vel_w[0].cpu().numpy()
                
                # Publish odometry
                ros2_bridge.publish_odom(robot_pos, robot_quat, robot_vel, robot_ang_vel)
            
            # Get lidar data and publish
            if hasattr(env.scene, 'lidar'):
                lidar_data = env.scene.lidar.data.ray_hits_w[0, 0, :].cpu().numpy()
                # Convert distances to ranges
                ranges = np.linalg.norm(lidar_data, axis=-1) if lidar_data.ndim > 1 else lidar_data
                ros2_bridge.publish_scan(ranges)
            
            # Print status periodically
            step_count += 1
            if step_count % 100 == 0:
                status = "ACTIVE" if cmd_vel else "IDLE"
                print(f"Step {step_count}: Status = {status}, Last cmd = {cmd_vel}")
            
            # Reset if needed
            if terminated.any() or truncated.any():
                env.reset()
        
    except KeyboardInterrupt:
        print("Stopping ROS2 control...")
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