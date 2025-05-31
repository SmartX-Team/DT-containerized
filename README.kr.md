# DT-containerized

> **NVIDIA Omniverse & ROS 2용 디지털 트윈 컨테이너 이미지**
> *미리 빌드되고 버전이 고정된 Docker 이미지로 단 한 번의 `docker run` 명령으로 Isaac Sim이나 다른 Omniverse 앱을 실행할 수 있습니다*

---

## 🔔 Docker Hub에서 미리 빌드된 이미지

가장 최신의 프로덕션 테스트를 거친 스택으로 완전히 구성된 이미지가 이미 Docker Hub에 올라와 있습니다:

```bash
docker pull ttyy441/issac-sim:0.4.5.1
# (저장소가 비공개인 경우) 
docker login -u <당신의-docker-ID>
```

가장 최신의 프로덕션 테스트를 거친 스택으로 완전히 구성된 이미지가 이미 Docker Hub에 올라와 있습니다. 정기적으로 점진적인 개선사항을 푸시하므로, 풀하기 전에 Docker Hub의 Tags 탭에서 최신 버전을 확인해주세요.

| 태그                         | Isaac Sim | ROS 2  | 참고사항                                                     |
| --------------------------- | --------- | ------ | ----------------------------------------------------------- |
| `ttyy441/issac-sim:0.4.5.1` | 4.5.1     | Humble | 기본값; ROS 브리지 + Nucleus 자동 로그인 포함                    |
| `ttyy441/issac-sim:<next>`  | *미정*     | Humble | NVIDIA에서 새로운 마이너/패치 버전을 출시할 때마다 나타날 예정      |

> **다른 것이 필요하신가요?** 이 저장소를 포크하여 Dockerfile을 조정하고 원하는 태그로 자신만의 이미지를 푸시하세요. 모든 헬퍼 스크립트는 독립적입니다.

---

## 📦 이 저장소가 존재하는 이유는?

Omniverse 앱의 재현성을 유지하는 것은 매우 번거로운 일입니다. 설치 파일은 용량이 크고, GPU 드라이버는 변수가 많으며, Nucleus 인증 문제는 CI/CD 파이프라인을 중단시키기도 합니다. 저희는 클라우드 네이티브 환경에 적합하도록 주요 기능이 자동으로 설정된 컨테이너가 필요했기에, 새로 커스텀 버전 컨테이너를 사용합니다.
 `<app>_container_<major>_<minor>` 폴더에 커밋합니다. Git 히스토리 + Docker 태그 = 완벽한 롤백.

**Isaac Sim** 이미지의 주요 기능:

* **ROS 2 Humble** 미리 설치되어 있고 `ros2_isaac_bridge` 활성화됨
* **Nucleus 자동 로그인** `nucleus-login.py`를 통해 (`OMNI_SERVER`, `OMNI_USER`, `OMNI_PASS` 환경 변수 읽음)
* 헤드리스 & GUI 실행 지원 (VNC, X11, PulseAudio)
* CUDA-11/12 호환; `nvidia/cuda:12.*-base`를 기반으로 빌드됨

향후 앱들—Omniverse Code, USD Composer, OV XR 등—은 같은 명명 규칙을 따르는 형제 폴더에 위치할 것입니다.

---

## 🗂️ 저장소 구조

```
DT-containerized/
├── isaac-sim_container_4_5/   # Isaac Sim 4.5.x용 Dockerfile + 스크립트
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── scripts/
│       └── nucleus-login.py
├── isaac-sim_container_4_6/   # ← 예정된 버전 플레이스홀더
└── <app>_container_<ver>/     # Omniverse Code, USD Composer, …
```

명명 규칙: **`<애플리케이션>_container_<major>_<minor>`**
예시: `code_container_2025_1`, `usd-composer_container_2024_3`.

---

## 🚀 빠른 시작 (미리 빌드된 이미지 사용)

#### 중요: 특정 버전에 대한 실행 명령을 확인하세요

```bash
docker run -d \
  --name isaac-sim \
  --restart unless-stopped \
  --runtime=nvidia --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_SERVER="omniverse://<nucleus-host>/NVIDIA/Assets/Isaac/4.5" \
  -e OMNI_USER=admin -e OMNI_PASS=******** \
  --network host \  # ROS 2 브리지는 호스트 네트워크에서 가장 잘 작동함
  -v ~/docker/isaac/cache/kit:/isaac-sim/kit/cache:rw \
  ttyy441/issac-sim:0.4.5.1
```

`entrypoint.sh`는 자동 Nucleus 로그인을 수행한 후 Isaac Sim을 실행합니다.

### 컨테이너 내에서 VS Code + ROS 2 개발

```bash
docker exec -it isaac-sim bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

---

## 🛠️ 자신만의 이미지 빌드하기

1. **클론 & 복사** 목표 버전에 가장 가까운 폴더를 복사합니다.

   ```bash
   cp -r isaac-sim_container_4_5 isaac-sim_container_4_6
   ```
2. **Dockerfile 편집** → Isaac Sim 설치 프로그램 URL과 체크섬을 변경합니다.
3. **빌드 & 테스트**:

   ```bash
   docker build -t <당신의이름>/isaac-sim:4.6 ./isaac-sim_container_4_6
   docker run --rm <당신의이름>/isaac-sim:4.6 --version
   ```
4. 원하는 레지스트리에 **푸시**합니다.

---

## 🤝 기여하기

* 이슈와 PR을 환영합니다!
* 문제를 더 빠르게 재현하기 위해 최소한의 **Nucleus 에셋 경로**와 **ROS 2 워크스페이스**를 포함해주세요.

---

## 🌐 지역화

* **English (기본값)** – README.md 파일.
* **한국어** – 이 파일.

---

## 📜 라이선스

Isaac Sim / Omniverse 바이너리는 © NVIDIA Corporation이며 자체 라이선스의 적용을 받습니다.