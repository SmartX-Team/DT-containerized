# Isaac Sim 4.5 Docker base Visualization

GIST AI 대학원에서 Visualization Display Wall 에 시연용 세팅 (도커파일, entrypoint.sh , docker-compose 포함)

isaac-sim_container_4_5 랑 기능적으로 대부분 동일

compose 파일에 Nucleus PW 는 지웠으니; 실제 다운받을때 작성 후 사용

지원 기능

- Headless 가 아닌 GUI 모드로 실행
- [NetAI] Extension 자동 등록
- Nucleus 서버 자동 로그인
- 주소로 지정한 usd 파일 스테이지로 자동 여는 기능
- docker compsoe template 파일

---

# Isaac Sim 4.5 Docker base Visualization

Demo setup for Visualization Display Wall at GIST AI Graduate School (includes Dockerfile, entrypoint.sh, docker-compose)

Functionally mostly identical to isaac-sim_container_4_5

Nucleus PW has been removed from compose file; write it in before use when actually downloading

Supported Features

- Run in GUI mode (not headless)
- [NetAI] Extension automatic registration
- Nucleus server automatic login
- Automatically open usd files specified by address to stage
- docker compose template file