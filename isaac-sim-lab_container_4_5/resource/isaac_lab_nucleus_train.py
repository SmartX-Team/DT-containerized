#!/usr/bin/env python3
"""
Isaac Lab ROS2 강화학습 스크립트
외부 ROS2 토픽을 구독하여 강화학습만 수행
"""

import torch
import numpy as np
import sys
import argparse
import threading
import time

# Isaac Lab path
sys.path.append('/opt/IsaacLab/source')

def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Lab ROS2 Reinforcement Learning")
    parser.add_argument("--num_envs", type=int, default=1, 
                       help="Number of parallel environments")
    
    # ROS2 topics
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel", 
                       help="Command velocity topic")
    parser.add_argument("--scan_topic", type=str, default="/scan", 
                       help="Laser scan topic")
    parser.add_argument("--odom_topic", type=str, default="/odom", 
                       help="Odometry topic")
    
    # AppLauncher arguments
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()

class ROS2DataCollector:
    """외부 ROS2 토픽에서 데이터를 수집하는 클래스"""
    
    def __init__(self, cmd_vel_topic="/cmd_vel", scan_topic="/scan", odom_topic="/odom"):
        self.cmd_vel_topic = cmd_vel_topic
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        
        # Data storage
        self.latest_scan = None
        self.latest_odom = None
        self.latest_cmd_vel = None
        self.data_lock = threading.Lock()
        
        # ROS2 node
        self.node = None
        self.ros_thread = None
        self.running = False
        
        # ROS2 초기화
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            from sensor_msgs.msg import LaserScan
            from nav_msgs.msg import Odometry
            
            self.rclpy = rclpy
            self.Node = Node
            self.Twist = Twist
            self.LaserScan = LaserScan
            self.Odometry = Odometry
            
            print("ROS2 modules imported successfully!")
            
        except ImportError as e:
            print(f"WARNING: ROS2 import failed: {e}")
            print("Will use dummy data for testing")
            self.rclpy = None
            return
    
    def start_ros2_node(self):
        """Start ROS2 node to collect external data"""
        if not self.rclpy:
            print("ROS2 not available, using dummy data")
            return
            
        def ros2_thread():
            self.rclpy.init()
            
            class DataCollectorNode(self.Node):
                def __init__(self, collector):
                    super().__init__('isaac_lab_data_collector')
                    self.collector = collector
                    
                    # Subscribers to external topics
                    self.scan_sub = self.create_subscription(
                        self.collector.LaserScan,
                        collector.scan_topic,
                        self.scan_callback,
                        10
                    )
                    
                    self.odom_sub = self.create_subscription(
                        self.collector.Odometry,
                        collector.odom_topic,
                        self.odom_callback,
                        10
                    )
                    
                    # Publisher for commands
                    self.cmd_pub = self.create_publisher(
                        self.collector.Twist,
                        collector.cmd_vel_topic,
                        10
                    )
                    
                    self.get_logger().info(f"Collecting data from external ROS2 topics:")
                    self.get_logger().info(f"  - Scan: {collector.scan_topic}")
                    self.get_logger().info(f"  - Odom: {collector.odom_topic}")
                    self.get_logger().info(f"  - Publishing commands to: {collector.cmd_vel_topic}")
                
                def scan_callback(self, msg):
                    with self.collector.data_lock:
                        # Convert to numpy array
                        self.collector.latest_scan = np.array(msg.ranges, dtype=np.float32)
                
                def odom_callback(self, msg):
                    with self.collector.data_lock:
                        # Extract pose and velocity
                        self.collector.latest_odom = {
                            'position': [
                                msg.pose.pose.position.x,
                                msg.pose.pose.position.y,
                                msg.pose.pose.position.z
                            ],
                            'orientation': [
                                msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w
                            ],
                            'linear_velocity': [
                                msg.twist.twist.linear.x,
                                msg.twist.twist.linear.y,
                                msg.twist.twist.linear.z
                            ],
                            'angular_velocity': [
                                msg.twist.twist.angular.x,
                                msg.twist.twist.angular.y,
                                msg.twist.twist.angular.z
                            ]
                        }
                
                def publish_cmd_vel(self, linear_x, angular_z):
                    """Publish command velocity to external robot"""
                    cmd_msg = self.collector.Twist()
                    cmd_msg.linear.x = float(linear_x)
                    cmd_msg.angular.z = float(angular_z)
                    self.cmd_pub.publish(cmd_msg)
            
            self.node = DataCollectorNode(self)
            
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
        print("ROS2 data collector started")
    
    def get_observation(self):
        """Get current observation from ROS2 data"""
        with self.data_lock:
            obs = {}
            
            # Laser scan data
            if self.latest_scan is not None:
                obs['scan'] = self.latest_scan.copy()
            else:
                obs['scan'] = np.zeros(360, dtype=np.float32)  # Dummy data
            
            # Odometry data
            if self.latest_odom is not None:
                obs['position'] = np.array(self.latest_odom['position'], dtype=np.float32)
                obs['orientation'] = np.array(self.latest_odom['orientation'], dtype=np.float32)
                obs['linear_velocity'] = np.array(self.latest_odom['linear_velocity'], dtype=np.float32)
                obs['angular_velocity'] = np.array(self.latest_odom['angular_velocity'], dtype=np.float32)
            else:
                # Dummy data
                obs['position'] = np.zeros(3, dtype=np.float32)
                obs['orientation'] = np.array([0, 0, 0, 1], dtype=np.float32)
                obs['linear_velocity'] = np.zeros(3, dtype=np.float32)
                obs['angular_velocity'] = np.zeros(3, dtype=np.float32)
            
            return obs
    
    def send_action(self, linear_x, angular_z):
        """Send action to external robot via ROS2"""
        if self.node and hasattr(self.node, 'publish_cmd_vel'):
            self.node.publish_cmd_vel(linear_x, angular_z)
    
    def stop(self):
        """Stop ROS2 data collector"""
        self.running = False
        if self.ros_thread:
            self.ros_thread.join(timeout=2.0)

def main():
    args = parse_args()
    
    # Launch Isaac Sim (minimal setup)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Initialize ROS2 data collector
    ros2_collector = ROS2DataCollector(args.cmd_vel_topic, args.scan_topic, args.odom_topic)
    
    try:
        # Start ROS2 data collection
        ros2_collector.start_ros2_node()
        time.sleep(2)  # Wait for ROS2 to initialize
        
        print("Isaac Lab ROS2 RL Environment Started!")
        print("Collecting data from external ROS2 topics...")
        print("Running reinforcement learning loop...")
        
        step_count = 0
        
        # Main RL loop
        while True:
            # Get observation from external ROS2 topics
            obs = ros2_collector.get_observation()
            
            # Simple RL policy (replace with your actual RL algorithm)
            # For now, just send random actions
            if step_count % 100 == 0:  # Send action every 100 steps
                linear_x = np.random.uniform(-0.5, 0.5)
                angular_z = np.random.uniform(-1.0, 1.0)
                ros2_collector.send_action(linear_x, angular_z)
                
                print(f"Step {step_count}: Sent action - linear: {linear_x:.2f}, angular: {angular_z:.2f}")
                print(f"  Scan data shape: {obs['scan'].shape}")
                print(f"  Position: {obs['position']}")
            
            step_count += 1
            time.sleep(0.01)  # 100Hz loop
        
    except KeyboardInterrupt:
        print("Stopping RL training...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        print("Cleaning up...")
        ros2_collector.stop()
        simulation_app.close()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)