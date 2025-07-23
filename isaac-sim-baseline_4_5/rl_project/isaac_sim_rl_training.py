#!/usr/bin/env python3
"""
Isaac Sim 실제 토픽 기반 강화학습 훈련 실제 테스트
/virtual/scan, /virtual/odom 구독 → /cmd_vel 발행
장애물 회피 네비게이션 학습

실제 기존에 개발해둔 아이작심 +ROS2 환경이랑 잘 호환됨

해당 코드는 클로드 딸각질로 만듬
"""

import numpy as np
import torch
import time
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from collections import deque
import os

class IsaacSimRLAgent(Node):
    """Isaac Sim 강화학습 에이전트"""
    
    def __init__(self):
        super().__init__('isaac_sim_rl_agent')
        
        # ROS2 설정
        self.setup_ros_interface()
        
        # 데이터 저장
        self.latest_scan = None
        self.latest_odom = None
        self.data_ready = False
        
        # 명령 지속 발행을 위한 변수
        self.current_action = [0.0, 0.0]  # [linear_x, angular_z]
        self.cmd_publish_rate = 20  # 20Hz로 발행
        self.cmd_timer = self.create_timer(1.0/self.cmd_publish_rate, self.publish_cmd_callback)
        
        # 학습 환경 설정 (동적 크기 계산을 위해 나중에 설정)
        self.action_space_size = 2  # [linear_x, angular_z]
        self.target_scan_size = 180  # 목표 LiDAR 크기
        self.observation_space_size = self.target_scan_size  # 속도는 나중에 추가
        
        # 에피소드 관리
        self.max_episode_steps = 500
        self.current_step = 0
        self.episode_reward = 0
        self.episode_count = 0
        
        # 안전 설정
        self.collision_threshold = 0.3  # 30cm
        self.min_obstacle_distance = 1.0  # 1m
        
        # 이전 위치 추적 (정체 방지)
        self.position_history = deque(maxlen=10)
        self.last_position = None
        
        self.get_logger().info("Isaac Sim RL Agent 초기화 완료")
    
    def setup_ros_interface(self):
        """ROS2 인터페이스 설정"""
        # 구독자
        self.scan_sub = self.create_subscription(
            LaserScan, '/virtual/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/virtual/odom', self.odom_callback, 10)
        
        # 발행자
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
    
    def scan_callback(self, msg):
        """LiDAR 스캔 콜백"""
        self.latest_scan = msg
        self.check_data_ready()
    
    def odom_callback(self, msg):
        """Odometry 콜백"""
        self.latest_odom = msg
        self.check_data_ready()
        
        # 위치 히스토리 업데이트
        current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.position_history.append(current_pos)
        self.last_position = current_pos
    
    def check_data_ready(self):
        """데이터 준비 상태 확인"""
        if self.latest_scan is not None and self.latest_odom is not None:
            self.data_ready = True
    
    def get_observation(self):
        """현재 관측값 반환"""
        if not self.data_ready:
            return np.zeros(self.observation_space_size, dtype=np.float32)
        
        # LiDAR 데이터 전처리
        scan_data = self.preprocess_scan()
        
        # Odometry 데이터 추가 (속도 정보)
        linear_vel = self.latest_odom.twist.twist.linear.x
        angular_vel = self.latest_odom.twist.twist.angular.z
        
        # 디버깅용 로그
        if hasattr(self, '_first_obs') is False:
            self._first_obs = True
            self.get_logger().info(f"스캔 데이터 크기: {scan_data.shape}")
            self.get_logger().info(f"전체 관측값 크기: {scan_data.shape[0] + 2}")
        
        # 관측값 결합 (LiDAR + 2 속도)
        observation = np.concatenate([
            scan_data,
            [linear_vel, angular_vel]
        ]).astype(np.float32)
        
        return observation
    
    def preprocess_scan(self):
        """LiDAR 스캔 데이터 전처리"""
        ranges = np.array(self.latest_scan.ranges)
        
        # 무한값과 NaN 처리
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        
        # 최대 거리 제한
        ranges = np.clip(ranges, 0.1, 10.0)
        
        # 실제 데이터 크기 확인 후 다운샘플링 조정
        original_size = len(ranges)
        target_size = 180  # 목표 크기
        
        if original_size >= target_size:
            # 다운샘플링
            step = original_size // target_size
            downsampled = ranges[::step][:target_size]
        else:
            # 업샘플링 (패딩)
            downsampled = np.pad(ranges, (0, target_size - original_size), mode='edge')
        
        # 정규화 (0-10m → 0-1)
        normalized = downsampled / 10.0
        
        return normalized
    
    def publish_cmd_callback(self):
        """타이머 콜백: 현재 액션을 지속적으로 발행"""
        cmd_msg = Twist()
        cmd_msg.linear.x = float(self.current_action[0])
        cmd_msg.angular.z = float(self.current_action[1])
        self.cmd_pub.publish(cmd_msg)
    
    def execute_action(self, action):
        """액션 실행 - 현재 액션을 업데이트 (타이머가 지속적으로 발행)"""
        linear_x, angular_z = action
        
        # 안전 제한
        linear_x = np.clip(linear_x, -1.0, 2.0)  # 후진 제한, 최대 2m/s
        angular_z = np.clip(angular_z, -1.0, 1.0)  # 최대 1rad/s
        
        # 현재 액션 업데이트
        self.current_action = [linear_x, angular_z]
        
        self.get_logger().debug(f"액션 업데이트: linear={linear_x:.2f}, angular={angular_z:.2f}")
    
    def calculate_reward(self, action, observation):
        """보상 계산"""
        reward = 0.0
        
        if not self.data_ready:
            return -1.0
        
        # 스캔 데이터 추출 (속도 정보 제외)
        scan_data = observation[:-2]
        linear_vel = observation[-2]
        angular_vel = observation[-1]
        
        # 1. 충돌 방지 보상
        min_distance = np.min(scan_data) * 10.0  # 정규화 해제
        
        if min_distance < self.collision_threshold:
            reward -= 100.0  # 큰 패널티
            self.get_logger().warn(f"충돌 위험! 거리: {min_distance:.2f}m")
        elif min_distance < self.min_obstacle_distance:
            reward -= 10.0 * (self.min_obstacle_distance - min_distance)
        
        # 2. 전진 장려 (장애물이 없을 때)
        if min_distance > 2.0 and linear_vel > 0:
            reward += 2.0 * linear_vel  # 전진 속도에 비례
        
        # 3. 과도한 회전 방지
        reward -= 0.5 * abs(angular_vel)
        
        # 4. 정면 장애물 회피 보상
        front_indices = range(85, 95)  # 정면 ±5도
        front_distances = scan_data[front_indices] * 10.0
        front_min = np.min(front_distances)
        
        if front_min < 1.5:  # 정면에 장애물
            if abs(angular_vel) > 0.1:  # 회전하고 있으면
                reward += 1.0  # 회피 행동 보상
        
        # 5. 정체 방지 (같은 자리에서 맴돌기 방지)
        if len(self.position_history) >= 5:
            recent_positions = list(self.position_history)[-5:]
            distances = []
            for i in range(len(recent_positions)-1):
                dist = np.sqrt(
                    (recent_positions[i+1][0] - recent_positions[i][0])**2 +
                    (recent_positions[i+1][1] - recent_positions[i][1])**2
                )
                distances.append(dist)
            
            avg_movement = np.mean(distances)
            if avg_movement < 0.1:  # 거의 움직이지 않음
                reward -= 2.0
        
        # 6. 생존 보상 (매 스텝마다)
        reward += 0.1
        
        return reward
    
    def is_done(self, observation):
        """에피소드 종료 조건"""
        if not self.data_ready:
            return False
        
        # 충돌 감지
        scan_data = observation[:-2]
        min_distance = np.min(scan_data) * 10.0
        
        if min_distance < self.collision_threshold:
            self.get_logger().info("에피소드 종료: 충돌")
            return True
        
        # 최대 스텝 도달
        if self.current_step >= self.max_episode_steps:
            self.get_logger().info("에피소드 종료: 최대 스텝 도달")
            return True
        
        return False
    
    def reset_episode(self):
        """에피소드 리셋"""
        self.current_step = 0
        self.episode_reward = 0
        self.position_history.clear()
        
        # 로봇 정지
        self.current_action = [0.0, 0.0]
        
        # 데이터 안정화 대기
        time.sleep(1.0)
        
        self.get_logger().info(f"에피소드 {self.episode_count} 시작")


class TrainingCallback(BaseCallback):
    """학습 콜백"""
    
    def __init__(self, agent_node, verbose=1):
        super().__init__(verbose)
        self.agent_node = agent_node
        self.episode_rewards = []
        self.best_reward = float('-inf')
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self):
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-10:]
            mean_reward = np.mean(recent_rewards)
            
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                # 최고 성능 모델 저장
                model_path = "/opt/autonomous_driving_rl/models/best_model"
                self.model.save(model_path)
                self.agent_node.get_logger().info(f"새로운 최고 성능! 평균 보상: {mean_reward:.2f}")
        
        return True


def ros_spin_thread(node):
    """ROS2 스핀을 별도 스레드에서 실행"""
    rclpy.spin(node)


def main():
    """메인 학습 루프"""
    # ROS2 초기화
    rclpy.init()
    
    # 에이전트 노드 생성
    agent = IsaacSimRLAgent()
    
    # ROS2 스핀 스레드 시작
    spin_thread = threading.Thread(target=ros_spin_thread, args=(agent,))
    spin_thread.daemon = True
    spin_thread.start()
    
    # 초기 데이터 대기
    agent.get_logger().info("초기 데이터 수신 대기 중...")
    while not agent.data_ready:
        time.sleep(0.1)
    
    agent.get_logger().info("데이터 수신 완료, 학습 시작!")
    
    try:
        # 학습 환경 설정
        observation_size = agent.observation_space_size + 2  # LiDAR + 속도
        
        # 가상 환경 (실제로는 Isaac Sim과 상호작용)
        import gymnasium as gym
        from gymnasium import spaces
        
        class IsaacSimEnv(gym.Env):
            """Gymnasium 호환 Isaac Sim 환경"""
            
            def __init__(self, agent_node):
                super().__init__()
                self.agent = agent_node
                
                # 액션 스페이스 정의 [linear_x, angular_z]
                self.action_space = spaces.Box(
                    low=np.array([-1.0, -1.0]),
                    high=np.array([2.0, 1.0]),
                    dtype=np.float32
                )
                
                # 관측 스페이스 정의 (실제 첫 관측값 크기 기반)
                # 초기 관측값으로 크기 확인
                dummy_obs = self.agent.get_observation()
                obs_size = len(dummy_obs)
                
                self.observation_space = spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(obs_size,),
                    dtype=np.float32
                )
                
                agent.get_logger().info(f"관측 공간 크기: {obs_size}")
                
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.agent.reset_episode()
                observation = self.agent.get_observation()
                info = {'episode': self.agent.episode_count}
                return observation, info
            
            def step(self, action):
                self.agent.execute_action(action)
                time.sleep(0.5)  # 액션 지속 시간 늘림 (0.1 → 0.5초)
                
                observation = self.agent.get_observation()
                reward = self.agent.calculate_reward(action, observation)
                terminated = self.agent.is_done(observation)
                truncated = False  # 시간 제한은 is_done에서 처리
                
                self.agent.current_step += 1
                self.agent.episode_reward += reward
                
                if terminated:
                    self.agent.episode_count += 1
                    agent.get_logger().info(
                        f"에피소드 {self.agent.episode_count} 완료: "
                        f"보상 {self.agent.episode_reward:.2f}, "
                        f"스텝 {self.agent.current_step}"
                    )
                
                info = {
                    'episode': self.agent.episode_count,
                    'step': self.agent.current_step,
                    'reward': reward
                }
                
                return observation, reward, terminated, truncated, info
        
        # 학습 환경 생성
        env = IsaacSimEnv(agent)
        
        # PPO 모델 생성
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu',  # ROS2와 안정성을 위해 CPU 사용
            tensorboard_log="/opt/autonomous_driving_rl/logs/isaac_sim_training"
        )
        
        # 콜백 설정
        callback = TrainingCallback(agent)
        
        agent.get_logger().info("PPO 학습 시작!")
        
        # 학습 시작
        model.learn(
            total_timesteps=50000,  # 5만 스텝
            callback=callback,
            progress_bar=False  # progress_bar 비활성화
        )
        
        # 최종 모델 저장
        final_model_path = "/opt/autonomous_driving_rl/models/isaac_sim_final"
        model.save(final_model_path)
        agent.get_logger().info(f"학습 완료! 모델 저장: {final_model_path}")
        
    except KeyboardInterrupt:
        agent.get_logger().info("학습 중단됨")
    
    finally:
        # 로봇 정지
        agent.current_action = [0.0, 0.0]
        time.sleep(0.5)  # 정지 명령이 전송되도록 대기
        
        # 정리
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()