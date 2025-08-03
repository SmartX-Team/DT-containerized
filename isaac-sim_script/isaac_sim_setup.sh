#!/bin/bash

# Isaac Sim + Colmap 간단 설치 스크립트 (웜업 제외)
# 작성자: SmartX Team
# 버전: 4.1 (Simple Version - sudo podman)
# 전제조건: CUDA, NVIDIA Toolkit, Driver 설치 완료
# MoblieX Station 에서 안정적인 배포를 목표로 만들어짐

set -e  # 에러 발생시 스크립트 중단

echo "========================================="
echo "Isaac Sim + Colmap 간단 설치 (sudo podman)"
echo "========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 설정 변수
IMAGE_NAME="docker.io/ttyy441/isaac-sim:0.4.5.5"
CONTAINER_NAME="isaac-sim"
CACHE_BASE="$HOME/podman/isaac-sim"

# ROS2 도메인 기본값 설정
DEFAULT_ROS_DOMAIN_ID=0
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$DEFAULT_ROS_DOMAIN_ID}"


# 기본 Nucleus Server 주소 
DEFAULT_OMNI_SERVER="omniverse://10.38.38.40/NVIDIA/Assets/Isaac/4.5"
DEFAULT_OMNI_USER="admin"
DEFAULT_OMNI_PASS="admin"

# 전역 변수 초기화
OMNI_SERVER="${OMNI_SERVER:-$DEFAULT_OMNI_SERVER}"
OMNI_USER="${OMNI_USER:-$DEFAULT_OMNI_USER}"
OMNI_PASS="${OMNI_PASS:-$DEFAULT_OMNI_PASS}"

# 조이스틱, rviz2 등 편의성 기능 모와둔
ROS2_IMAGE_NAME="docker.io/ttyy441/ros2-container:0.5.0"
ROS2_CONTAINER_NAME="ros2-backend"

# 시스템 요구사항 확인
check_requirements() {
    log_info "시스템 요구사항 확인 중..."
    
    # NVIDIA GPU 확인
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA GPU 드라이버가 설치되지 않았습니다."
        exit 1
    fi
    
    # CUDA 확인
    if command -v nvcc &> /dev/null; then
        local CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_success "CUDA 확인: $CUDA_VERSION"
    else
        log_info "CUDA가 PATH에 없습니다. (컨테이너에서 사용 가능)"
    fi
    
    # Podman 확인
    if ! command -v podman &> /dev/null; then
        log_error "Podman이 설치되지 않았습니다."
        exit 1
    fi
    
    # sudo 권한 확인
    if ! sudo -n true 2>/dev/null; then
        log_info "sudo 권한이 필요합니다. 비밀번호를 입력해주세요."
        if ! sudo true; then
            log_error "sudo 권한이 필요합니다."
            exit 1
        fi
    fi
    
    # GPU 정보 표시
    log_info "GPU 정보:"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv
    
    log_success "시스템 요구사항 확인 완료"
}

# Isaac Sim 이미지 다운로드
pull_isaac_image() {
    log_info "Isaac Sim 이미지 다운로드 중..."
    log_info "이미지: $IMAGE_NAME"
    
    if sudo podman pull $IMAGE_NAME; then
        log_success "이미지 다운로드 완료"
    else
        log_error "이미지 다운로드 실패"
        exit 1
    fi
    
    # 이미지 정보 확인
    log_info "다운로드된 이미지 정보:"
    sudo podman images | grep issac-sim
}

# 캐시 디렉토리 생성
create_cache_directories() {
   log_info "캐시 디렉토리 확인 및 생성 중..."
   
   # 현재 사용자 정보 가져오기
   local USER_ID=$(id -u)
   local GROUP_ID=$(id -g)
   local USER_NAME=$(whoami)
   
   # 1. 기본 캐시 디렉토리들 체크 및 생성
   if [ ! -d "$CACHE_BASE/cache/kit" ] || [ ! -d "$CACHE_BASE/cache/ov" ] || [ ! -d "$CACHE_BASE/cache/pip" ] || [ ! -d "$CACHE_BASE/cache/glcache" ] || [ ! -d "$CACHE_BASE/cache/computecache" ] || [ ! -d "$CACHE_BASE/cache/nvidia" ]; then
       log_info "기본 캐시 디렉토리 생성 중..."
       mkdir -p $CACHE_BASE/cache/{kit,ov,pip,glcache,computecache,nvidia}
   else
       log_info "기본 캐시 디렉토리 존재함"
   fi
   
   # 2. 로그/데이터 디렉토리들 체크 및 생성
   if [ ! -d "$CACHE_BASE/logs" ] || [ ! -d "$CACHE_BASE/data" ] || [ ! -d "$CACHE_BASE/documents" ] || [ ! -d "$CACHE_BASE/config" ]; then
       log_info "로그/데이터 디렉토리 생성 중..."
       mkdir -p $CACHE_BASE/{logs,data,documents,config}
   else
       log_info "로그/데이터 디렉토리 존재함"
   fi
   
   # 3. Omniverse 디렉토리 체크 및 생성
   if [ ! -d "$CACHE_BASE/nvidia-omniverse/config" ] || [ ! -d "$CACHE_BASE/nvidia-omniverse/logs" ]; then
       log_info "Omniverse 디렉토리 생성 중..."
       sudo mkdir -p $CACHE_BASE/nvidia-omniverse/{config,logs}
   else
       log_info "Omniverse 디렉토리 존재함"
   fi
   
   # 4. Isaac Sim 내부 캐시 디렉토리 체크 및 생성
   if [ ! -d "$CACHE_BASE/isaac-cache-ov" ] || [ ! -d "$CACHE_BASE/isaac-local-share" ] || [ ! -d "$CACHE_BASE/isaac-ros" ]; then
       log_info "Isaac Sim 내부 캐시 디렉토리 생성 중..."
       sudo mkdir -p $CACHE_BASE/{isaac-cache-ov,isaac-local-share,isaac-ros}
   else
       log_info "Isaac Sim 내부 캐시 디렉토리 존재함"
   fi
   
   # 권한 설정 (항상 실행)
   sudo chown -R $USER_ID:$GROUP_ID $CACHE_BASE
   chmod -R 755 $CACHE_BASE
   
   log_success "캐시 디렉토리 확인/생성 완료"
   log_info "캐시 경로: $CACHE_BASE"
   log_info "소유자: $USER_NAME ($USER_ID:$GROUP_ID)"

   # Omniverse 설정 파일 권한 특별 처리
   chmod -R u+rwX,go+rX $CACHE_BASE/nvidia-omniverse

   # 디렉토리 구조 표시 (새로 생성된 경우만)
   if [ ! -f "$CACHE_BASE/.created" ]; then
       echo "디렉토리 구조:"
       tree $CACHE_BASE 2>/dev/null || find $CACHE_BASE -type d | sed 's|[^/]*/|  |g;s|^  ||'
       touch $CACHE_BASE/.created
   fi
}

# Colmap 의존성 설치
install_colmap_dependencies() {
    log_info "Colmap 의존성 설치 중..."
    
    sudo apt-get update
    sudo apt-get install -y \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
           
    log_success "Colmap 의존성 설치 완료"
}

# Colmap 빌드 및 설치
install_colmap() {
    log_info "Colmap 빌드 및 설치 중..."
    
    local COLMAP_DIR="$HOME/colmap_build"
    local INSTALL_PREFIX="/usr/local"
    
    # 기존 디렉토리 정리
    if [ -d "$COLMAP_DIR" ]; then
        log_warning "기존 Colmap 빌드 디렉토리 제거 중..."
        rm -rf "$COLMAP_DIR"
    fi
    
    # CUDA 호환성을 위한 컴파일러 설정
    log_info "CUDA 호환 컴파일러GCC 10 설정 중..."
    sudo apt-get update
    sudo apt-get install -y gcc-10 g++-10

    #환경변수 설정 (Colmap  공식 가이드 준수)
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10
    export CUDAHOSTCXX=/usr/bin/g++-10

    #Anaconda  환경 충돌 방지  PATH 정리
    export PATH=$(echo $PATH | sed 's|anaconda|replaced|g')
    log_info "Anaconda 경로 제거 완료" 
	
    # 라이브러리 경로 설정 (안정적인 링킹을 위해)
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
    log_info "라이브러리 경로 설정 완료: $LD_LIBRARY_PATH"

    # Git clone
    git clone https://github.com/colmap/colmap.git "$COLMAP_DIR"
    cd "$COLMAP_DIR"

    # 빌드 디렉토리 생성
    mkdir build
    cd build
           
    # CUDA 지원 확인
    if command -v nvcc &> /dev/null; then
        log_info "CUDA 발견됨 - GPU 가속으로 빌드 (GCC 10 사용)"
        # CMake 설정 (Colmap 공식 가이드 준수)
        cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native
    else
        log_warning "CUDA 없음 - CPU 전용으로 빌드"
        cmake .. -GNinja -DCUDA_ENABLED=OFF
    fi
    #build
    ninja
   
    #install
    sudo ninja install

    cd "$HOME"
    rm -rf "$COLMAP_DIR"

    if command -v colmap &> /dev/null; then
	log_success "Colmap 설치 완료"
        colmap --version
    else
        log_error "Colmap 설치 실패"
	exit 1
    fi	

}

# ROS2 컨테이너 실행
run_ros2_backend_container() {

    # 기존 ROS2 컨테이너 정리
    sudo podman stop $ROS2_CONTAINER_NAME 2>/dev/null || true
    sleep 2
    sudo podman rm $ROS2_CONTAINER_NAME 2>/dev/null || true
    sleep 2

    sudo podman run -it --rm \
        --name $ROS2_CONTAINER_NAME \
        --network host \
        --ipc=host \
        --privileged \
        -e QT_X11_NO_MITSHM=1 \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e SDL_VIDEODRIVER=x11 \
        -e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
        --device=/dev/dri:/dev/dri \
        --gpus all \
        --security-opt label=disable \
        $ROS2_IMAGE_NAME \
        isaac_sim

    if [ $? -eq 0 ]; then
        log_success "ROS2 백엔드 컨테이너가 성공적으로 실행되었습니다!"
        log_info "컨테이너 이름: $ROS2_CONTAINER_NAME"
        log_info "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
    else
        log_error "ROS2 백엔드 컨테이너 실행 실패"
        exit 1
    fi

}

# Isaac Sim 컨테이너 실행
run_isaac_container() {
    log_info "Isaac Sim 컨테이너 실행 중..."
    
    # 기존 컨테이너 정리
    sudo podman stop $CONTAINER_NAME 2>/dev/null || true
    sudo podman rm $CONTAINER_NAME 2>/dev/null || true
    
    # 실행 모드 선택
    echo ""
    log_info "Isaac Sim 실행 모드를 선택하세요:"
    echo "1) 백그라운드 헤드리스 모드 (웹 스트리밍)"
    echo "2) X11 GUI 모드 (직접 GUI 창)"
    echo ""
    
    while true; do
        read -p "선택 (1-2): " mode_choice
        case $mode_choice in
            1)
                RUN_MODE="headless"
                break
                ;;
            2)
                RUN_MODE="gui"
                break
                ;;
            *)
                log_warning "올바른 번호를 입력하세요 (1-2)"
                ;;
        esac
    done

    echo ""
    log_info "ROS2 도메인 설정"
    log_info "현재 ROS_DOMAIN_ID: $ROS_DOMAIN_ID (0-101 사이 값, 다른 로봇과 분리하려면 다른 값 사용)"
    echo ""

    read -p "ROS_DOMAIN_ID (현재: $ROS_DOMAIN_ID, 엔터=유지): " input_domain

    if [ -n "$input_domain" ]; then
        # 입력값 유효성 검사
        if [[ "$input_domain" =~ ^[0-9]+$ ]] && [ "$input_domain" -ge 0 ] && [ "$input_domain" -le 101 ]; then
            ROS_DOMAIN_ID="$input_domain"
            log_info "ROS_DOMAIN_ID 변경: $ROS_DOMAIN_ID"
        else
            log_warning "유효하지 않은 도메인 ID입니다. 0-101 사이 값을 입력하세요. 기본값 사용: $ROS_DOMAIN_ID"
        fi
    else
        log_info "기본 ROS_DOMAIN_ID 사용: $ROS_DOMAIN_ID"
    fi


    # Omniverse 설정 입력
    echo ""
    log_info "Omniverse Nucleus 서버 설정"
    log_info "현재 기본값: 서버=$OMNI_SERVER, 사용자=$OMNI_USER"
    echo ""

    read -p "Nucleus 서버 URL (현재: $OMNI_SERVER, 엔터=유지): " input_server
    read -p "사용자명 (현재: $OMNI_USER, 엔터=유지): " input_user
    read -s -p "비밀번호 (현재: [숨김], 엔터=유지): " input_pass
    echo ""
    
    # 입력값이 있으면 사용, 없으면 전역 기본값 유지
    if [ -n "$input_server" ]; then
        OMNI_SERVER="$input_server"
        log_info "서버 URL 변경: $OMNI_SERVER"
    else
        log_info "기본 서버 URL 사용: $OMNI_SERVER"
    fi

    if [ -n "$input_user" ]; then
        OMNI_USER="$input_user"
        log_info "사용자명 변경: $OMNI_USER"
    else
        log_info "기본 사용자명 사용: $OMNI_USER"
    fi

    if [ -n "$input_pass" ]; then
        OMNI_PASS="$input_pass"
        log_info "비밀번호 변경됨"
    else
        log_info "기본 비밀번호 사용"
    fi

    #=== 실제 컨테이너 실행 ===#
    if [ "$RUN_MODE" = "headless" ]; then
        create_cache_directories
        run_headless_container
        sleep 3
        run_ros2_backend_container
        
        
    else
        create_cache_directories
        run_gui_container
        sleep 3
        run_ros2_backend_container
        
    fi

}

# 헤드리스 모드 컨테이너 실행
run_headless_container() {
    log_info "백그라운드 헤드리스 모드로 실행 중..."
    
    # 현재 사용자 정보 가져오기
    local USER_ID=$(id -u)
    local GROUP_ID=$(id -g)
    
    sudo podman run --name $CONTAINER_NAME \
        --detach \
        --restart unless-stopped \
        --device nvidia.com/gpu=all \
        --ulimit nofile=8192:16384 \
        --network=host \
        --ipc=host \
        --privileged \
        -e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=Y" \
        -e "OMNI_SERVER=$OMNI_SERVER" \
        -e "OMNI_USER=$OMNI_USER" \
        -e "OMNI_PASS=$OMNI_PASS" \
        -e "OMNI_KIT_ALLOW_ROOT=1" \
        -e "NVIDIA_DRIVER_CAPABILITIES=all" \
        -v $HOME/Documents:$HOME/Documents:rw \
        -v $CACHE_BASE/cache/kit:/isaac-sim/kit/cache:rw \
        -v $CACHE_BASE/cache/ov:/root/.cache/ov:rw \
        -v $CACHE_BASE/cache/pip:/root/.cache/pip:rw \
        -v $CACHE_BASE/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v $CACHE_BASE/cache/computecache:/root/.nv/ComputeCache:rw \
        -v $CACHE_BASE/logs:/root/.nvidia-omniverse/logs:rw \
        -v $CACHE_BASE/data:/root/.local/share/ov/data:rw \
        -v $CACHE_BASE/documents:/root/Documents:rw \
        -v $CACHE_BASE/nvidia-omniverse:/root/.nvidia-omniverse:rw \
        -v $CACHE_BASE/cache/nvidia:/root/.cache/nvidia:rw \
        -v $CACHE_BASE/isaac-cache-ov:/isaac-sim/.cache/ov:rw \
        -v $CACHE_BASE/isaac-local-share:/isaac-sim/.local/share:rw \
        -v $CACHE_BASE/isaac-ros:/isaac-sim/.ros:rw \
        $IMAGE_NAME
    
    if [ $? -eq 0 ]; then
        log_success "Isaac Sim 헤드리스 컨테이너가 성공적으로 실행되었습니다!"
        log_info "컨테이너 이름: $CONTAINER_NAME"
        
        # 컨테이너 상태 확인
        sleep 10
        if sudo podman ps | grep -q $CONTAINER_NAME; then
            log_success "컨테이너가 정상적으로 실행 중입니다."
            echo ""
            log_info "  웹 스트리밍 접속: http://localhost:8211/streaming/webrtc-client?server=localhost"
        else
            log_warning "컨테이너 상태를 확인하세요: sudo podman logs $CONTAINER_NAME"
        fi
    else
        log_error "컨테이너 실행 실패"
        exit 1
    fi
}

# GUI 모드 컨테이너 실행
run_gui_container() {
    log_info "X11 GUI 모드로 실행 중..."

    # X11 설정 확인
    if [ -z "$DISPLAY" ]; then
        log_error "DISPLAY 환경변수가 설정되지 않았습니다."
        log_info "GUI 모드를 사용하려면 X11이 실행 중이어야 합니다."
        exit 1
    fi
    
    # 현재 사용자 정보 가져오기
    local USER_ID=$(id -u)
    local GROUP_ID=$(id -g)

    # 기존 GUI 컨테이너 정리
    sudo podman stop isaac-sim-gui 2>/dev/null || true
    sudo podman rm isaac-sim-gui 2>/dev/null || true
    
    log_info "GUI 창이 열릴 때까지 기다려주세요..."
    log_info "GUI 컨테이너를 백그라운드에서 시작합니다..."    
    
    
    sudo podman run --name $CONTAINER_NAME \
        -d \
        --device nvidia.com/gpu=all \
        --ulimit nofile=8192:16384 \
        --network=host \
        --ipc=host \
        --privileged \
        -e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
        -e "DISPLAY=$DISPLAY" \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=Y" \
        -e "OMNI_SERVER=$OMNI_SERVER" \
        -e "OMNI_USER=$OMNI_USER" \
        -e "OMNI_PASS=$OMNI_PASS" \
        -e "OMNI_KIT_ALLOW_ROOT=1" \
        -e "NVIDIA_DRIVER_CAPABILITIES=all" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $HOME/Documents:$HOME/Documents:rw \
        -v $CACHE_BASE/cache/kit:/isaac-sim/kit/cache:rw \
        -v $CACHE_BASE/cache/ov:/root/.cache/ov:rw \
        -v $CACHE_BASE/cache/pip:/root/.cache/pip:rw \
        -v $CACHE_BASE/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v $CACHE_BASE/cache/computecache:/root/.nv/ComputeCache:rw \
        -v $CACHE_BASE/logs:/root/.nvidia-omniverse/logs:rw \
        -v $CACHE_BASE/data:/root/.local/share/ov/data:rw \
        -v $CACHE_BASE/documents:/root/Documents:rw \
        -v $CACHE_BASE/nvidia-omniverse:/root/.nvidia-omniverse:rw \
        -v $CACHE_BASE/cache/nvidia:/root/.cache/nvidia:rw \
        -v $CACHE_BASE/isaac-cache-ov:/isaac-sim/.cache/ov:rw \
        -v $CACHE_BASE/isaac-local-share:/isaac-sim/.local/share:rw \
        -v $CACHE_BASE/isaac-ros:/isaac-sim/.ros:rw \
        $IMAGE_NAME \
        bash -c "ulimit -n 8192 && cd /isaac-sim && ./isaac-sim.sh"
    
    # GUI 모드에서는 컨테이너가 포그라운드에서 실행되므로
    # 종료 후 정리 메시지 표시
    echo ""
    log_info "Isaac Sim GUI가 종료되었습니다."
    
}

# 설치 완료 정보 출력
show_completion_info() {
    echo ""
    echo "========================================="
    log_success "설치 완료!"
    echo "========================================="
    echo ""
    
    echo " Isaac Sim 4.5 (sudo podman):"
    if [ "$RUN_MODE" = "headless" ]; then
        echo "  실행 모드:           헤드리스 (백그라운드)"
        echo "  웹 스트리밍:         http://localhost:8211/streaming/webrtc-client?server=localhost"
        echo "  컨테이너 상태:        sudo podman ps"
        echo "  컨테이너 접속:        sudo podman exec -it $CONTAINER_NAME /bin/bash"
        echo "  로그 확인:           sudo podman logs $CONTAINER_NAME"
        echo "  컨테이너 중지:        sudo podman stop $CONTAINER_NAME"
        echo "  컨테이너 시작:        sudo podman start $CONTAINER_NAME"
    elif [ "$RUN_MODE" = "gui" ]; then
        echo "  실행 모드:           GUI (직접 창)"
        echo "  빠른 재실행:         ./run_isaac_gui.sh"
    fi
    echo "   캐시 디렉토리: $CACHE_BASE"
    echo ""
    
    if [ "$INSTALL_ISAAC" = true ] && [ -f "./run_isaac_gui.sh" ]; then
        echo "  편의 스크립트 (빠른 실행):"
        echo "  GUI 모드:           ./run_isaac_gui.sh"
        echo "  헤드리스 모드:       ./run_isaac_headless.sh"
        echo ""
    fi
    
    echo " Colmap (CUDA 지원):"
    echo "  GUI 실행:            colmap gui"
    echo "  CLI 사용:            colmap --help"
    echo "  버전 확인:           colmap --version"
    echo "  Isaac Sim과 데이터 호환 가능"
    echo ""
    
    echo "  유용한 명령어:"
    echo "  GPU 확인:           nvidia-smi"
    echo "  GPU 테스트:         sudo podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi"
    echo "  시스템 상태:         sudo podman system info"
    echo "  이미지 목록:         sudo podman images"
    echo "  컨테이너 목록:       sudo podman ps -a"
    echo ""
    
    echo "  관련 링크:"
    echo "  Isaac Sim 문서:     https://docs.isaacsim.omniverse.nvidia.com/"
    echo "  Colmap 문서:        https://colmap.github.io/"
    echo ""
    
    echo "   설치가 완료되었습니다!"
    echo "   Isaac Sim 웹 인터페이스: http://localhost:8211/streaming/webrtc-client?server=localhost"
    echo ""
    if [ "$RUN_MODE" = "headless" ]; then
        log_info "⚠️  첫 실행 시 셰이더 컴파일로 인해 시간이 소요될 수 있습니다."
        echo "     웹 브라우저에서 접속하여 확인하세요: http://localhost:8211/streaming/webrtc-client?server=localhost"
    else
        log_info "⚠️  첫 실행 시 셰이더 컴파일로 인해 GUI 창이 나타나는데 시간이 소요될 수 있습니다."
    fi
}

# 메인 함수
main() {
    log_info "Isaac Sim + Colmap 설치/실행 스크립트 (sudo podman)"
    
    # 기존 설치 확인
    check_existing_installation
    
    echo ""
    log_info "설치할 구성 요소를 선택하세요:"
    echo "1) Isaac Sim만"
    echo "2) Colmap만"
    echo "3) 모든 구성 요소 (추천)"
    echo ""
    
    while true; do
        read -p "선택 (1-3): " choice
        choice=${choice:-3}
        case $choice in
            1)
                INSTALL_ISAAC=true
                INSTALL_COLMAP=false
                break
                ;;
            2)
                INSTALL_ISAAC=false
                INSTALL_COLMAP=true
                break
                ;;
            3)
                INSTALL_ISAAC=true
                INSTALL_COLMAP=true
                break
                ;;
            *)
                log_warning "올바른 번호를 입력하세요 (1-3)"
                ;;
        esac
    done
    
    log_info "선택: Isaac Sim=$INSTALL_ISAAC, Colmap=$INSTALL_COLMAP"
    
    # 시스템 요구사항 확인
    check_requirements
   
    # Colmap 설치
    if [ "$INSTALL_COLMAP" = true ]; then
        if [ "$EXISTING_COLMAP" = true ]; then
            log_info "기존 Colmap 설치를 사용합니다."
        else
            install_colmap_dependencies
            install_colmap
        fi
    fi
    
    # Isaac Sim 설치/실행
    if [ "$INSTALL_ISAAC" = true ]; then
        if [ "$EXISTING_ISAAC" = true ]; then
            log_info "기존 Isaac Sim 설치를 사용합니다."
        else
            pull_isaac_image
        fi
        create_cache_directories
        run_isaac_container
    fi
   
       
    # 실행 완료 후 편의 스크립트 생성
    if [ "$INSTALL_ISAAC" = true ]; then
        create_convenience_scripts
    fi
    
    show_completion_info
    
    log_success "모든 과정이 완료되었습니다!"
}

# 기존 설치 확인
check_existing_installation() {
    EXISTING_ISAAC=false
    EXISTING_COLMAP=false
    
    # Isaac Sim 설치 확인
    if [ -d "$CACHE_BASE" ] && sudo podman images | grep -q "issac-sim"; then
        EXISTING_ISAAC=true
        log_success "기존 Isaac Sim 설치를 발견했습니다!"
        OMNI_SERVER="omniverse://10.38.38.40/NVIDIA/Assets/Isaac/4.5"
        OMNI_USER="admin"
        OMNI_PASS="admin"
        echo ""
        log_info "빠른 실행 옵션:"
        echo "1) 바로 Isaac Sim GUI 실행"
        echo "2) 바로 Isaac Sim 헤드리스 실행"
        echo "3) 새로 설치/설정"
        echo ""
        
        read -p "선택 (1-3): " quick_choice
        
        case $quick_choice in
            1)
                log_info "Isaac Sim GUI를 바로 실행합니다..."
                RUN_MODE="gui"
                create_cache_directories
                
                run_gui_container
                sleep 3
                run_ros2_backend_container
                exit 0
                ;;
            2)
                log_info "Isaac Sim 헤드리스 모드를 바로 실행합니다..."
                RUN_MODE="headless"
                create_cache_directories

                run_headless_container
                sleep 3
                run_ros2_backend_container
                
                exit 0
                ;;
            3)
                log_info "새로 설치/설정을 진행합니다."
                ;;
            *)
                log_warning "올바른 번호를 입력하세요. 새로 설치를 진행합니다."
                ;;
        esac
    fi
    
    # Colmap 설치 확인
    if command -v colmap &> /dev/null; then
        EXISTING_COLMAP=true
        log_success "기존 Colmap 설치를 발견했습니다!"
    fi
}

# 편의 스크립트 생성 (!현재 오류 수정 안함 -> 누군가 해주삼)
create_convenience_scripts() {
    log_info "편의 스크립트 생성 중..."
    
    # 현재 사용자 정보 가져오기
    local USER_ID=$(id -u)
    local GROUP_ID=$(id -g)
    
    # GUI 실행 스크립트
    cat > "./run_isaac_gui.sh" << EOF
#!/bin/bash

# Isaac Sim GUI 빠른 실행 스크립트 (sudo podman)

echo "Isaac Sim GUI 실행 중..."

# 변수 설정
CACHE_BASE="$HOME/podman/isaac-sim"
IMAGE_NAME="docker.io/ttyy441/issac-sim:0.4.5.1"
USER_ID=$USER_ID
GROUP_ID=$GROUP_ID

OMNI_SERVER="${OMNI_SERVER:-omniverse://10.38.38.40/NVIDIA/Assets/Isaac/4.5}"
OMNI_USER="${OMNI_USER:-admin}"  
OMNI_PASS="${OMNI_PASS:-admin}"

# X11 설정 확인
if [ -z "\$DISPLAY" ]; then
    echo "DISPLAY 환경변수가 설정되지 않았습니다."
    echo "GUI 모드를 사용하려면 X11이 실행 중이어야 합니다."
    exit 1
fi


echo "GUI 창이 열릴 때까지 기다려주세요..."

# 기존 GUI 컨테이너 정리
sudo podman stop isaac-sim-gui 2>/dev/null || true
sudo podman rm isaac-sim-gui 2>/dev/null || true

# GUI 모드 실행
sudo podman run --name isaac-sim-gui \\
    -it \\
    --rm \\
    --device nvidia.com/gpu=all \\
    --ulimit nofile=8192:16384 \\
    --network=host \\
    --ipc=host \\
    --privileged \\
    -e "DISPLAY=\$DISPLAY" \\
    -e "ACCEPT_EULA=Y" \\
    -e "PRIVACY_CONSENT=Y" \\
    -e "OMNI_SERVER=\$OMNI_SERVER" \\
    -e "OMNI_USER=\$OMNI_USER" \\
    -e "OMNI_PASS=\$OMNI_PASS" \\
    -e "OMNI_KIT_ALLOW_ROOT=1" \\
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \\
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \\
    -v \$HOME/Documents:\$HOME/Documents:rw \\
    -v \$CACHE_BASE/cache/kit:/isaac-sim/kit/cache:rw \\
    -v \$CACHE_BASE/cache/ov:/root/.cache/ov:rw \\
    -v \$CACHE_BASE/cache/pip:/root/.cache/pip:rw \\
    -v \$CACHE_BASE/cache/glcache:/root/.cache/nvidia/GLCache:rw \\
    -v \$CACHE_BASE/cache/computecache:/root/.nv/ComputeCache:rw \\
    -v \$CACHE_BASE/logs:/root/.nvidia-omniverse/logs:rw \\
    -v \$CACHE_BASE/data:/root/.local/share/ov/data:rw \\
    -v \$CACHE_BASE/documents:/root/Documents:rw \\
    -v \$CACHE_BASE/nvidia-omniverse:/root/.nvidia-omniverse:rw \\
    -v \$CACHE_BASE/cache/nvidia:/root/.cache/nvidia:rw \\
    -v \$CACHE_BASE/isaac-cache-ov:/isaac-sim/.cache/ov:rw \\
    -v \$CACHE_BASE/isaac-local-share:/isaac-sim/.local/share:rw \\
    -v \$CACHE_BASE/isaac-ros:/isaac-sim/.ros:rw \\
    \$IMAGE_NAME \\
    bash -c "ulimit -n 8192 && cd /isaac-sim && ./isaac-sim.sh"



echo ""
echo "Isaac Sim GUI가 종료되었습니다."
EOF

    # 헤드리스 실행 스크립트
    cat > "./run_isaac_headless.sh" << EOF
#!/bin/bash

# Isaac Sim 헤드리스 빠른 실행 스크립트 (sudo podman)

echo "Isaac Sim 헤드리스 모드 실행 중..."

# 변수 설정
CACHE_BASE="$HOME/podman/isaac-sim"
IMAGE_NAME="docker.io/ttyy441/issac-sim:0.4.5.1"
CONTAINER_NAME="isaac-sim"
USER_ID=$USER_ID
GROUP_ID=$GROUP_ID

# 기존 컨테이너 정리
sudo podman stop \$CONTAINER_NAME 2>/dev/null || true
sudo podman rm \$CONTAINER_NAME 2>/dev/null || true

# 헤드리스 모드 실행
sudo podman run --name \$CONTAINER_NAME \\
    --detach \\
    --restart unless-stopped \\
    --device nvidia.com/gpu=all \\
    --ulimit nofile=8192:16384 \\
    --network=host \\
    --ipc=host \\
    --privileged \\
    -e "ACCEPT_EULA=Y" \\
    -e "PRIVACY_CONSENT=Y" \\
    -e "OMNI_SERVER=\$OMNI_SERVER" \\
    -e "OMNI_USER=\$OMNI_USER" \\
    -e "OMNI_PASS=\$OMNI_PASS" \\
    -e "OMNI_KIT_ALLOW_ROOT=1" \\
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \\
    -v \$HOME/Documents:\$HOME/Documents:rw \\
    -v \$CACHE_BASE/cache/kit:/isaac-sim/kit/cache:rw \\
    -v \$CACHE_BASE/cache/ov:/root/.cache/ov:rw \\
    -v \$CACHE_BASE/cache/pip:/root/.cache/pip:rw \\
    -v \$CACHE_BASE/cache/glcache:/root/.cache/nvidia/GLCache:rw \\
    -v \$CACHE_BASE/cache/computecache:/root/.nv/ComputeCache:rw \\
    -v \$CACHE_BASE/logs:/root/.nvidia-omniverse/logs:rw \\
    -v \$CACHE_BASE/data:/root/.local/share/ov/data:rw \\
    -v \$CACHE_BASE/documents:/root/Documents:rw \\
    -v \$CACHE_BASE/nvidia-omniverse:/root/.nvidia-omniverse:rw \\
    -v \$CACHE_BASE/cache/nvidia:/root/.cache/nvidia:rw \\
    -v \$CACHE_BASE/isaac-cache-ov:/isaac-sim/.cache/ov:rw \\
    -v \$CACHE_BASE/isaac-local-share:/isaac-sim/.local/share:rw \\
    -v \$CACHE_BASE/isaac-ros:/isaac-sim/.ros:rw \\    
    \$IMAGE_NAME

if [ \$? -eq 0 ]; then
    echo " Isaac Sim 헤드리스 컨테이너가 성공적으로 실행되었습니다!"
    echo ""
    echo " 웹 스트리밍 접속: http://localhost:8211/streaming/webrtc-client?server=localhost"
    echo ""
    echo " 관리 명령어:"
    echo "  컨테이너 상태:    sudo podman ps"
    echo "  로그 확인:       sudo podman logs isaac-sim"
    echo "  컨테이너 중지:    sudo podman stop isaac-sim"
    echo "  컨테이너 시작:    sudo podman start isaac-sim"
else
    echo "컨테이너 실행 실패"
    exit 1
fi
EOF

    # 실행 권한 부여
    chmod +x "./run_isaac_gui.sh"
    chmod +x "./run_isaac_headless.sh"
    
    log_success "편의 스크립트 생성 완료!"
    echo "  - GUI 실행:      ./run_isaac_gui.sh"
    echo "  - 헤드리스 실행: ./run_isaac_headless.sh"
}

# 스크립트 실행
main "$@"
