#!/usr/bin/env python3
"""
기본 강화학습 테스트
ROS2 토픽을 사용한 간단한 강화학습 환경과 PPO 학습 테스트
"""

import numpy as np
import torch
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class SimpleNavEnvironment:
    """간단한 네비게이션 환경 (ROS2 없이 테스트용)"""
    
    def __init__(self):
        self.action_space_size = 2  # [linear_x, angular_z]
        self.observation_space_size = 360  # LiDAR 포인트 수
        
        # 환경 상태
        self.robot_position = np.array([0.0, 0.0])
        self.robot_heading = 0.0
        self.target_position = np.array([5.0, 5.0])
        
        # 에피소드 관리
        self.max_steps = 200
        self.current_step = 0
        
        self.reset()
    
    def reset(self):
        """환경 초기화"""
        self.robot_position = np.array([0.0, 0.0])
        self.robot_heading = 0.0
        self.current_step = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """관측값 생성 (가상 LiDAR 데이터)"""
        # 간단한 가상 환경 (사각형 방 + 몇 개 장애물)
        ranges = []
        
        for i in range(360):
            angle = i * np.pi / 180 + self.robot_heading
            
            # 기본 벽까지의 거리 (10x10 방)
            wall_distance = 10.0
            
            # 몇 개 장애물 추가
            obstacles = [
                {'pos': np.array([2.0, 3.0]), 'radius': 0.5},
                {'pos': np.array([4.0, 1.0]), 'radius': 0.3},
                {'pos': np.array([3.0, 4.0]), 'radius': 0.4}
            ]
            
            min_distance = wall_distance
            
            # 장애물과의 거리 계산
            for obs in obstacles:
                obs_vector = obs['pos'] - self.robot_position
                obs_distance = np.linalg.norm(obs_vector)
                
                if obs_distance > 0:
                    obs_angle = np.arctan2(obs_vector[1], obs_vector[0])
                    angle_diff = abs(angle - obs_angle)
                    
                    if angle_diff < 0.1:  # 각도 범위 내
                        distance_to_obstacle = max(0.1, obs_distance - obs['radius'])
                        min_distance = min(min_distance, distance_to_obstacle)
            
            # 노이즈 추가
            min_distance += np.random.normal(0, 0.05)
            ranges.append(max(0.1, min_distance))
        
        return np.array(ranges, dtype=np.float32)
    
    def step(self, action):
        """환경 스텝"""
        # 액션 적용
        linear_x, angular_z = action
        
        # 로봇 위치 업데이트
        dt = 0.1
        self.robot_heading += angular_z * dt
        
        dx = linear_x * np.cos(self.robot_heading) * dt
        dy = linear_x * np.sin(self.robot_heading) * dt
        self.robot_position += np.array([dx, dy])
        
        # 관측값
        observation = self._get_observation()
        
        # 보상 계산
        reward = self._calculate_reward(observation, action)
        
        # 종료 조건
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or self._is_collision(observation)
        
        info = {
            'position': self.robot_position.copy(),
            'heading': self.robot_heading,
            'step': self.current_step
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self, observation, action):
        """보상 계산"""
        reward = 0.0
        
        # 목표 지향 보상
        distance_to_target = np.linalg.norm(self.robot_position - self.target_position)
        reward += -0.01 * distance_to_target
        
        # 충돌 방지 보상
        min_obstacle_distance = np.min(observation)
        if min_obstacle_distance < 1.0:
            reward += -10.0 * (1.0 - min_obstacle_distance)
        
        # 충돌 패널티
        if min_obstacle_distance < 0.3:
            reward -= 50.0
        
        # 액션 부드러움 보상
        linear_x, angular_z = action
        reward -= 0.01 * abs(angular_z)  # 급격한 회전 방지
        
        # 전진 보상
        if linear_x > 0:
            reward += 0.1
        
        return reward
    
    def _is_collision(self, observation):
        """충돌 확인"""
        return np.min(observation) < 0.3

class TestCallback(BaseCallback):
    """테스트용 콜백"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """스텝마다 호출되는 필수 메소드"""
        return True  # 학습 계속 진행
        
    def _on_rollout_end(self):
        """롤아웃 종료 시 호출"""
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-10:])  # 최근 10 에피소드 평균
            print(f"스텝 {self.num_timesteps}: 평균 보상 = {mean_reward:.2f}")
        
        return True

def test_simple_environment():
    """간단한 환경 테스트"""
    print("=" * 50)
    print("간단한 환경 테스트")
    print("=" * 50)
    
    try:
        env = SimpleNavEnvironment()
        
        # 환경 리셋 테스트
        obs = env.reset()
        print(f"✅ 환경 리셋 성공: observation shape = {obs.shape}")
        
        # 랜덤 액션 테스트
        total_reward = 0
        for step in range(10):
            action = np.random.uniform(-1, 1, 2)  # [linear_x, angular_z]
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if step == 0:
                print(f"✅ 스텝 실행 성공: reward = {reward:.2f}")
        
        print(f"✅ 10스텝 총 보상: {total_reward:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ 환경 테스트 실패: {e}")
        return False

def test_ppo_training():
    """PPO 학습 테스트"""
    print("\n" + "=" * 50)
    print("PPO 학습 테스트")
    print("=" * 50)
    
    try:
        # 간단한 gym 환경으로 PPO 테스트
        import gymnasium as gym
        
        # CartPole 환경으로 빠른 테스트
        env = gym.make('CartPole-v1')
        
        # PPO 모델 생성
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1,
            learning_rate=0.001,
            n_steps=128,
            batch_size=32,
            n_epochs=4
        )
        
        print("✅ PPO 모델 생성 성공")
        
        # 콜백 설정
        callback = TestCallback()
        
        # 짧은 학습 실행
        print("간단한 학습 시작 (1000 스텝)...")
        model.learn(total_timesteps=1000, callback=callback)
        print("✅ PPO 학습 테스트 성공")
        
        # 모델 저장 테스트
        model_path = "/opt/autonomous_driving_rl/models/test_ppo"
        model.save(model_path)
        print(f"✅ 모델 저장 성공: {model_path}")
        
        # 모델 로드 테스트
        loaded_model = PPO.load(model_path, env=env)
        print("✅ 모델 로드 성공")
        
        # 추론 테스트
        obs, _ = env.reset()
        action, _ = loaded_model.predict(obs)
        print(f"✅ 추론 테스트 성공: action = {action}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ PPO 학습 테스트 실패: {e}")
        return False

def test_tensorboard_logging():
    """TensorBoard 로깅 테스트"""
    print("\n" + "=" * 50)
    print("TensorBoard 로깅 테스트")
    print("=" * 50)
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        import tempfile
        import shutil
        
        # 로그 디렉토리 생성
        log_dir = "/opt/autonomous_driving_rl/logs/test_tensorboard"
        os.makedirs(log_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir)
        
        # 가상 학습 데이터 로깅
        for step in range(100):
            # 가상 손실값
            loss = 1.0 * np.exp(-step/50) + 0.1 * np.random.random()
            writer.add_scalar('train/loss', loss, step)
            
            # 가상 보상값  
            reward = -100 + step * 2 + 10 * np.random.random()
            writer.add_scalar('train/reward', reward, step)
            
            # 가상 학습률
            lr = 0.001 * (0.99 ** (step // 10))
            writer.add_scalar('train/learning_rate', lr, step)
        
        writer.close()
        
        # 로그 파일 확인
        log_files = os.listdir(log_dir)
        if any('tfevents' in f for f in log_files):
            print(f"✅ TensorBoard 로그 생성 성공: {log_dir}")
            print(f"   로그 파일: {len(log_files)}개")
            print(f"   실행 명령: tensorboard --logdir {log_dir}")
            return True
        else:
            print("❌ TensorBoard 로그 파일이 생성되지 않음")
            return False
            
    except Exception as e:
        print(f"❌ TensorBoard 테스트 실패: {e}")
        return False

def test_model_save_load():
    """모델 저장/로드 테스트"""
    print("\n" + "=" * 50)
    print("모델 저장/로드 테스트") 
    print("=" * 50)
    
    try:
        import gymnasium as gym
        
        # 환경과 모델 생성
        env = gym.make('CartPole-v1')
        model = PPO('MlpPolicy', env, verbose=0)
        
        # 초기 예측값 저장
        obs, _ = env.reset()
        initial_action, _ = model.predict(obs, deterministic=True)
        
        # 짧은 학습
        model.learn(total_timesteps=500)
        
        # 학습 후 예측값
        trained_action, _ = model.predict(obs, deterministic=True)
        
        # 모델 저장
        model_dir = "/opt/autonomous_driving_rl/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "test_save_load")
        model.save(model_path)
        
        # 모델 로드
        loaded_model = PPO.load(model_path, env=env)
        loaded_action, _ = loaded_model.predict(obs, deterministic=True)
        
        # 검증
        if np.array_equal(trained_action, loaded_action):
            print("✅ 모델 저장/로드 성공 (예측값 일치)")
            print(f"   저장 경로: {model_path}")
            print(f"   초기 액션: {initial_action}")
            print(f"   학습 후 액션: {trained_action}")
            print(f"   로드 후 액션: {loaded_action}")
            return True
        else:
            print("❌ 저장/로드 후 예측값 불일치")
            return False
            
        env.close()
        
    except Exception as e:
        print(f"❌ 모델 저장/로드 테스트 실패: {e}")
        return False

def test_ros_integration_preparation():
    """ROS2 통합 준비 테스트"""
    print("\n" + "=" * 50)
    print("ROS2 통합 준비 테스트")
    print("=" * 50)
    
    try:
        # ROS2 관련 라이브러리 import 테스트
        import rclpy
        from sensor_msgs.msg import LaserScan
        from geometry_msgs.msg import Twist
        print("✅ ROS2 메시지 타입 import 성공")
        
        # 가상 스캔 데이터로 처리 테스트
        fake_scan_data = np.random.uniform(0.5, 10.0, 360)
        
        # RL에서 사용할 관측값 전처리
        processed_obs = preprocess_scan_data(fake_scan_data)
        print(f"✅ 스캔 데이터 전처리 성공: {processed_obs.shape}")
        
        # 액션을 Twist 메시지로 변환 테스트
        action = np.array([0.5, -0.3])  # [linear_x, angular_z]
        twist_msg = action_to_twist(action)
        print(f"✅ 액션 → Twist 변환 성공: linear={twist_msg.linear.x:.2f}, angular={twist_msg.angular.z:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ROS2 통합 준비 테스트 실패: {e}")
        return False

def preprocess_scan_data(scan_ranges):
    """스캔 데이터 전처리 함수"""
    # numpy 배열로 변환
    ranges = np.array(scan_ranges)
    
    # 무한값과 NaN 처리
    ranges = np.where(np.isfinite(ranges), ranges, 10.0)
    
    # 정규화 (0-10m → 0-1)
    normalized = np.clip(ranges / 10.0, 0.0, 1.0)
    
    # 다운샘플링 (360 → 180)
    downsampled = normalized[::2]
    
    return downsampled.astype(np.float32)

def action_to_twist(action):
    """RL 액션을 Twist 메시지로 변환"""
    from geometry_msgs.msg import Twist
    
    twist = Twist()
    twist.linear.x = float(np.clip(action[0], -2.0, 2.0))  # 최대 속도 2m/s
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0  
    twist.angular.z = float(np.clip(action[1], -1.0, 1.0))  # 최대 각속도 1rad/s
    
    return twist

def main():
    """전체 테스트 실행"""
    print("기본 강화학습 테스트 시작...\n")
    
    # 디렉토리 생성
    os.makedirs("/opt/autonomous_driving_rl/models", exist_ok=True)
    os.makedirs("/opt/autonomous_driving_rl/logs", exist_ok=True)
    
    tests = [
        ("간단한 환경", test_simple_environment),
        ("PPO 학습", test_ppo_training),
        ("TensorBoard 로깅", test_tensorboard_logging),
        ("모델 저장/로드", test_model_save_load),
        ("ROS2 통합 준비", test_ros_integration_preparation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} 테스트 {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("기본 강화학습 테스트 결과 요약")
    print("=" * 60)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n총 {success_count}/{len(tests)} 테스트 통과")
    
    if success_count == len(tests):
        print("\n🎉 모든 기본 RL 기능이 정상 동작합니다!")
        print("\n다음 단계:")
        print("1. Isaac Sim에서 실제 로봇 시뮬레이션 실행")
        print("2. ROS2 토픽 확인: ros2 topic list")
        print("3. 실제 학습 스크립트 실행")
        return 0
    else:
        print("\n⚠️ 일부 기능에 문제가 있습니다.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)