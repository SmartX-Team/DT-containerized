#!/usr/bin/env python3
"""
ROS2 토픽 통신 테스트
Isaac Sim에서 /virtual/scan 구독, /cmd_vel 발행 테스트
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import time
import threading

class RosTopicTester(Node):
    def __init__(self):
        super().__init__('ros_topic_tester')
        
        # 구독자 설정
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/virtual/scan',
            self.scan_callback,
            10
        )
        
        # 발행자 설정
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 데이터 저장
        self.latest_scan = None
        self.scan_received_count = 0
        self.cmd_sent_count = 0
        
        # 테스트 통계
        self.start_time = time.time()
        
        self.get_logger().info("ROS2 토픽 테스터 초기화 완료")
    
    def scan_callback(self, msg):
        """LiDAR scan 데이터 콜백"""
        self.latest_scan = msg
        self.scan_received_count += 1
        
        if self.scan_received_count % 10 == 0:  # 10개마다 로그
            scan_info = self.analyze_scan(msg)
            self.get_logger().info(
                f"Scan #{self.scan_received_count}: "
                f"ranges={len(msg.ranges)}, "
                f"min_dist={scan_info['min_dist']:.2f}m, "
                f"max_dist={scan_info['max_dist']:.2f}m, "
                f"obstacles={scan_info['obstacle_count']}"
            )
    
    def analyze_scan(self, scan_msg):
        """스캔 데이터 분석"""
        ranges = np.array(scan_msg.ranges)
        
        # 무한값과 NaN 필터링
        valid_ranges = ranges[np.isfinite(ranges)]
        
        if len(valid_ranges) == 0:
            return {
                'min_dist': float('inf'),
                'max_dist': 0.0,
                'mean_dist': 0.0,
                'obstacle_count': 0
            }
        
        # 장애물 감지 (2m 이내)
        obstacle_threshold = 2.0
        obstacles = valid_ranges[valid_ranges < obstacle_threshold]
        
        return {
            'min_dist': float(np.min(valid_ranges)),
            'max_dist': float(np.max(valid_ranges)),
            'mean_dist': float(np.mean(valid_ranges)),
            'obstacle_count': len(obstacles)
        }
    
    def send_test_commands(self):
        """테스트용 cmd_vel 명령 발행"""
        test_commands = [
            {'linear': 0.5, 'angular': 0.0, 'duration': 2.0, 'name': '전진'},
            {'linear': 0.0, 'angular': 0.5, 'duration': 2.0, 'name': '좌회전'},
            {'linear': 0.0, 'angular': -0.5, 'duration': 2.0, 'name': '우회전'},
            {'linear': -0.3, 'angular': 0.0, 'duration': 1.0, 'name': '후진'},
            {'linear': 0.0, 'angular': 0.0, 'duration': 1.0, 'name': '정지'}
        ]
        
        for cmd in test_commands:
            self.get_logger().info(f"명령 실행: {cmd['name']}")
            
            # 명령 발행
            start_time = time.time()
            while time.time() - start_time < cmd['duration']:
                twist_msg = Twist()
                twist_msg.linear.x = cmd['linear']
                twist_msg.angular.z = cmd['angular']
                
                self.cmd_publisher.publish(twist_msg)
                self.cmd_sent_count += 1
                
                time.sleep(0.1)  # 10Hz
            
            # 명령 간 잠시 대기
            time.sleep(0.5)
        
        # 최종 정지 명령
        stop_msg = Twist()
        self.cmd_publisher.publish(stop_msg)
        self.get_logger().info("테스트 완료 - 로봇 정지")
    
    def print_statistics(self):
        """테스트 통계 출력"""
        elapsed_time = time.time() - self.start_time
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("테스트 통계")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"실행 시간: {elapsed_time:.1f}초")
        self.get_logger().info(f"수신한 스캔 메시지: {self.scan_received_count}개")
        self.get_logger().info(f"발행한 cmd_vel 메시지: {self.cmd_sent_count}개")
        
        if self.latest_scan:
            scan_info = self.analyze_scan(self.latest_scan)
            self.get_logger().info(f"최신 스캔 정보:")
            self.get_logger().info(f"  - 범위 개수: {len(self.latest_scan.ranges)}")
            self.get_logger().info(f"  - 최소 거리: {scan_info['min_dist']:.2f}m")
            self.get_logger().info(f"  - 최대 거리: {scan_info['max_dist']:.2f}m")
            self.get_logger().info(f"  - 평균 거리: {scan_info['mean_dist']:.2f}m")
            self.get_logger().info(f"  - 장애물 수 (2m내): {scan_info['obstacle_count']}")

def test_scan_processing():
    """스캔 데이터 처리 테스트 (ROS 없이)"""
    print("=" * 50)
    print("스캔 데이터 처리 함수 테스트")
    print("=" * 50)
    
    # 가짜 스캔 데이터 생성
    fake_ranges = []
    for i in range(360):  # 360도 스캔
        angle = i * np.pi / 180
        
        # 간단한 환경 시뮬레이션 (정사각형 방 + 몇 개 장애물)
        if 45 <= i <= 135:  # 앞쪽에 벽 (3m)
            distance = 3.0
        elif 225 <= i <= 315:  # 뒤쪽에 벽 (4m)
            distance = 4.0
        elif i <= 45 or i >= 315:  # 우측 벽 (2m)
            distance = 2.0
        elif 135 <= i <= 225:  # 좌측 벽 (2.5m)
            distance = 2.5
        else:
            distance = 3.0
            
        # 일부 노이즈 추가
        distance += np.random.normal(0, 0.1)
        
        # 일부 장애물 추가
        if 80 <= i <= 100:  # 앞쪽 장애물
            distance = min(distance, 1.5)
        
        fake_ranges.append(max(0.1, distance))  # 최소 거리 제한
    
    # numpy 배열로 변환
    ranges_array = np.array(fake_ranges)
    
    # 처리 테스트
    print(f"✅ 스캔 포인트 수: {len(ranges_array)}")
    print(f"✅ 최소 거리: {np.min(ranges_array):.2f}m")
    print(f"✅ 최대 거리: {np.max(ranges_array):.2f}m")
    print(f"✅ 평균 거리: {np.mean(ranges_array):.2f}m")
    
    # 장애물 감지 테스트
    obstacle_threshold = 2.0
    obstacles = ranges_array[ranges_array < obstacle_threshold]
    print(f"✅ {obstacle_threshold}m 이내 장애물: {len(obstacles)}개")
    
    # 특정 방향 범위 추출 (정면 ±30도)
    front_indices = list(range(0, 31)) + list(range(330, 360))
    front_ranges = ranges_array[front_indices]
    print(f"✅ 정면 범위 최소 거리: {np.min(front_ranges):.2f}m")
    
    return True

def main():
    print("ROS2 토픽 테스트 시작...")
    
    # 1. 스캔 데이터 처리 함수 테스트
    if not test_scan_processing():
        print("❌ 스캔 데이터 처리 테스트 실패")
        return
    
    # 2. ROS2 초기화
    try:
        rclpy.init()
        
        # 테스터 노드 생성
        tester = RosTopicTester()
        
        # 명령 발행을 별도 스레드에서 실행
        def command_thread():
            time.sleep(2.0)  # 스캔 데이터 수신 대기
            tester.send_test_commands()
        
        cmd_thread = threading.Thread(target=command_thread)
        cmd_thread.start()
        
        # 노드 스핀 (20초간)
        start_time = time.time()
        while time.time() - start_time < 20.0:
            rclpy.spin_once(tester, timeout_sec=0.1)
        
        # 통계 출력
        tester.print_statistics()
        
        # 정리
        cmd_thread.join()
        tester.destroy_node()
        rclpy.shutdown()
        
        print("🎉 ROS2 토픽 테스트 완료!")
        
    except Exception as e:
        print(f"❌ ROS2 테스트 중 오류 발생: {e}")
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()