"""
라이브러리 설치 및 동작 테스트
"""

import sys
import numpy as np

def test_pytorch():
    """PyTorch 테스트"""
    print("=" * 50)
    print("PyTorch 테스트")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        
        # CUDA 사용 가능 확인
        if torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능: {torch.cuda.device_count()}개 GPU")
            print(f"   현재 GPU: {torch.cuda.current_device()}")
            print(f"   GPU 이름: {torch.cuda.get_device_name()}")
        else:
            print("❌ CUDA 사용 불가능")
            
        # 간단한 텐서 연산 테스트
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print(f"✅ 텐서 연산 테스트 성공: {z.shape}")
        
        # GPU 텐서 연산 테스트
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print(f"✅ GPU 텐서 연산 성공: {z_gpu.device}")
            
    except Exception as e:
        print(f"❌ PyTorch 테스트 실패: {e}")
        return False
    
    return True

def test_stable_baselines3():
    """Stable-Baselines3 테스트"""
    print("\n" + "=" * 50)
    print("Stable-Baselines3 테스트")
    print("=" * 50)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        print("✅ Stable-Baselines3 import 성공")
        
        # 간단한 환경으로 모델 생성 테스트
        import gymnasium as gym
        env = gym.make('CartPole-v1')
        model = PPO('MlpPolicy', env, verbose=0)
        print("✅ PPO 모델 생성 성공")
        
        # 간단한 학습 테스트
        model.learn(total_timesteps=100)
        print("✅ 기본 학습 테스트 성공")
        
    except Exception as e:
        print(f"❌ Stable-Baselines3 테스트 실패: {e}")
        return False
    
    return True

def test_opencv():
    """OpenCV 테스트"""
    print("\n" + "=" * 50)
    print("OpenCV 테스트")
    print("=" * 50)
    
    try:
        import cv2
        print(f"✅ OpenCV 버전: {cv2.__version__}")
        
        # 간단한 이미지 처리 테스트
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"✅ 이미지 처리 테스트 성공: {gray.shape}")
        
    except Exception as e:
        print(f"❌ OpenCV 테스트 실패: {e}")
        return False
    
    return True

def test_tensorboard():
    """TensorBoard 테스트"""
    print("\n" + "=" * 50)
    print("TensorBoard 테스트")
    print("=" * 50)
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        import tempfile
        
        # 임시 로그 디렉토리에 테스트 로그 작성
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)
            
            # 스칼라 로그 테스트
            for i in range(10):
                writer.add_scalar('test/loss', np.random.random(), i)
            
            writer.close()
            print("✅ TensorBoard 로그 작성 성공")
        
    except Exception as e:
        print(f"❌ TensorBoard 테스트 실패: {e}")
        return False
    
    return True

def test_yaml_matplotlib():
    """YAML, Matplotlib 테스트"""
    print("\n" + "=" * 50)
    print("YAML, Matplotlib 테스트")
    print("=" * 50)
    
    try:
        import yaml
        import matplotlib
        matplotlib.use('Agg')  # GUI 없이 사용
        import matplotlib.pyplot as plt
        
        print(f"✅ YAML 버전: {yaml.__version__}")
        print(f"✅ Matplotlib 버전: {matplotlib.__version__}")
        
        # YAML 테스트
        test_data = {'test': [1, 2, 3], 'config': {'lr': 0.001}}
        yaml_str = yaml.dump(test_data)
        loaded_data = yaml.safe_load(yaml_str)
        assert loaded_data == test_data
        print("✅ YAML 직렬화/역직렬화 성공")
        
        # Matplotlib 테스트
        fig, ax = plt.subplots()
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        ax.plot(x, y)
        plt.close()
        print("✅ Matplotlib 그래프 생성 성공")
        
    except Exception as e:
        print(f"❌ YAML/Matplotlib 테스트 실패: {e}")
        return False
    
    return True

def main():
    """전체 테스트 실행"""
    print("강화학습 라이브러리 테스트 시작...\n")
    
    tests = [
        test_pytorch,
        test_stable_baselines3,
        test_opencv,
        test_tensorboard,
        test_yaml_matplotlib
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)
    
    test_names = [
        "PyTorch",
        "Stable-Baselines3",
        "OpenCV",
        "TensorBoard", 
        "YAML/Matplotlib"
    ]
    
    success_count = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n총 {success_count}/{len(tests)} 테스트 통과")
    
    if success_count == len(tests):
        print("🎉 모든 라이브러리가 정상 동작합니다!")
        return 0
    else:
        print("⚠️  일부 라이브러리에 문제가 있습니다.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)