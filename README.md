# DT-containerized

> **Digital‑Twin Container Images for NVIDIA Omniverse & ROS 2**
> *Pre‑built & version‑locked Docker images so you can spin up Isaac Sim or any other Omniverse app with a single `docker run`*

---

## 🔔 Pre‑built image on Docker Hub

A fully configured image for the most recent, production‑tested stack is already pushed to Docker Hub:

```bash
docker pull ttyy441/issac-sim:0.4.5.1
# (If the repository is private) 
docker login -u <your‑docker‑ID>
```

A fully configured image for the most recent, production‑tested stack is already pushed to Docker Hub. We regularly push incremental improvements, so please check the Tags tab on Docker Hub for the latest version before pulling.


| Tag                         | Isaac Sim | ROS 2  | Notes                                                       |
| --------------------------- | --------- | ------ | ----------------------------------------------------------- |
| `ttyy441/issac-sim:0.4.5.1` | 4.5.1     | Humble | Default; includes ROS bridge + Nucleus auto‑login           |
| `ttyy441/issac-sim:<next>`  | *TBD*     | Humble | Will appear whenever NVIDIA ships a new minor/patch version |

> **Need something different?** Feel free to fork this repo, tweak the Dockerfiles, and push your own image under any tag. All helper scripts are self‑contained.

---

## 📦 Why does this repo exist?

Keeping Omniverse apps reproducible is a cumbersome task: large installers, GPU driver quirks, and Nucleus auth that breaks CI/CD. We needed a container with key features auto-configured for a cloud-native workflow, and that's why this repository exists.  Here we lock every dependency inside a Docker image **per version** and commit the exact build context under
`<app>_container_<major>_<minor>` folders. Git history + Docker tag = perfect roll‑back.

Key features of the **Isaac Sim** image:

* **ROS 2 Humble** pre‑installed with `ros2_isaac_bridge` enabled
* **Nucleus auto‑login** via `nucleus-login.py` (reads `OMNI_SERVER`, `OMNI_USER`, `OMNI_PASS` env vars)
* Headless & GUI launch support (VNC, X11, PulseAudio)
* CUDA‑11/12 compatible; built on `nvidia/cuda:12.*-base`

Future apps—Omniverse Code, USD Composer, OV XR, etc.—will live in sibling folders that follow the same naming convention.

---

## 🗂️ Repository layout

```
DT-containerized/
├── isaac-sim_container_4_5/   # Dockerfile + scripts for Isaac Sim 4.5.x
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── scripts/
│       └── nucleus-login.py
├── isaac-sim_container_4_6/   # ← upcoming version placeholder
└── <app>_container_<ver>/     # Omniverse Code, USD Composer, …
```

Naming rule: **`<application>_container_<major>_<minor>`**
Examples: `code_container_2025_1`, `usd-composer_container_2024_3`.

---

## 🚀 Quick start (using the pre‑built image)

#### Important: Check the run command for your specific version

```bash
docker run -d \
  --name isaac-sim \
  --restart unless-stopped \
  --runtime=nvidia --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_SERVER="omniverse://<nucleus-host>/NVIDIA/Assets/Isaac/4.5" \
  -e OMNI_USER=admin -e OMNI_PASS=******** \
  --network host \  # ROS 2 bridge works best on host network
  -v ~/docker/isaac/cache/kit:/isaac-sim/kit/cache:rw \
  ttyy441/issac-sim:0.4.5.1
```

`entrypoint.sh` launches Isaac Sim after performing a silent Nucleus sign‑in.

### VS Code + ROS 2 development inside the container

```bash
docker exec -it isaac-sim bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

---

## 🛠️ Build your own image

1. **Clone & copy** the folder closest to your target version.

   ```bash
   cp -r isaac-sim_container_4_5 isaac-sim_container_4_6
   ```
2. **Edit the Dockerfile** → change Isaac Sim installer URL & checksum.
3. **Build & test**:

   ```bash
   docker build -t <yourname>/isaac-sim:4.6 ./isaac-sim_container_4_6
   docker run --rm <yourname>/isaac-sim:4.6 --version
   ```
4. **Push** to any registry you like.

---

## 🤝 Contributing

* Issues & PRs are welcome!
* Please include a minimal **Nucleus asset path** and **ROS 2 workspace** to reproduce problems faster.

---

## 🌐 Localizations

* **English (default)** – this file.
* **한국어** – see `README_KR.md`.

---

## 📜 License

Isaac Sim / Omniverse binaries are © NVIDIA Corporation and subject to their own licenses.

