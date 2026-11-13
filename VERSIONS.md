# Versions

Recorded 2026-10-12 on the dev machine (native, no container).

## System

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.5 LTS |
| ROS 2 | Jazzy (`ros-jazzy-ros-base` 0.11.0) |
| Gazebo | Gazebo Sim 8.15.0 (Harmonic), via `ros-jazzy-gz-sim-vendor` 0.0.13 |
| Python | 3.12.3 |
| GPU | none (CPU only) |

## apt packages (ros-jazzy-*)

| Package | Version | Why |
|---|---|---|
| ros-gz | 1.0.24 | Gazebo ↔ ROS bridge, spawning |
| xacro | 2.1.1 | robot description |
| rviz2 | 14.1.24 | visualization |
| ros2-control / controller-manager | 4.48.1 | control framework |
| ros2-controllers | 4.42.1 | joint_trajectory_controller, broadcasters |
| gz-ros2-control | 1.2.20 | Gazebo ros2_control plugin (arm) |
| pinocchio | 4.1.0 | dependency of `basic_quadruped_controller`; reach/collision checks (Python, with coal) |
| joint-state-publisher(-gui) | 2.4.3 | description tools |
| realsense2-description | 4.58.4 | only for the CHAMP candidate (evaluation) |
| robot-localization | 3.8.3 | CHAMP candidate (evaluation); our localization in A4 |
| vision-msgs | 4.1.1 | `/sim/ground_truth/objects`, `/people` (A1b) |
| gz-transport-vendor (Python bindings `gz.transport13`, `gz.msgs10`) | 0.0.9 | sim_bridge_gazebo nodes talk to Gazebo directly (A1b); comes with ros-gz |
| navigation2 (Nav2), incl. `nav2_regulated_pure_pursuit_controller`, `nav2_navfn_planner`, `nav2_velocity_smoother`, `nav2_bt_navigator` | 1.3.13 | navigation (A2): `packages/navigation` |

## Python (system, Ubuntu 24.04 packages)

| Package | Version | Why |
|---|---|---|
| numpy | 1.26.4 | scene generator, terrain (A1) |
| Pillow | 10.2.0 | heightmap / texture export (A1) |
| PyYAML | 6.0.1 | scenes, configs |
| pytest | 7.4.4 | `make test` |

## C++ libraries (manual install)

| Library | Version | Why |
|---|---|---|
| libtorch (CPU, cxx11 ABI) | 2.5.0+cpu, `https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcpu.zip` (sha256 `848ae52a422bf0bf94fd2ba2b048bb6c3e416f385cef390640760bd0baf7a40e`), unzipped to `~/libtorch` | `rl_quadruped_controller` (`locomotion:=legubiao_robot_lab`, the default since B4; `locomotion:=legubiao_rl`, HIMLoco policy, A2b). Build with `Torch_DIR=~/libtorch`; at run time `gazebo.launch.py` adds `$Torch_DIR/lib` (default `~/libtorch/lib`) to Gazebo's `LD_LIBRARY_PATH` |

## Runtime downloads

| Asset | Source | Notes |
|---|---|---|
| Walking person | Gazebo Fuel `Mingfei/actor` (`walk.dae`), CC BY 4.0 | downloaded by Gazebo on the first run of a scene with a person, cached in `~/.gz/fuel/` |

## Third-party source repos (pinned in `deps.repos`)

| Repo | Commit | Branch | Notes |
|---|---|---|---|
| [legubiao/quadruped_ros2_control](https://github.com/legubiao/quadruped_ros2_control) | `4dfc15ea73e7e453019a685d2199fd8406c93896` (2026-02-22) | dev | Apache-2.0. Patches in `patches/quadruped_ros2_control/`: `0001` link the Pinocchio 4 imported target, `0002` kp/kd `initial_value`s in `gz_quadruped_hardware`. Not built: ocs2 controller, unitree_sdk2 hardware, unitree_joystick_input (`patches/setup_deps.sh` adds COLCON_IGNORE). `rl_quadruped_controller` is built since A2b (needs libtorch); its policies come from `go2_description/config/{himloco,legged_gym,robot_lab}`. |
| [Interbotix/interbotix_ros_manipulators](https://github.com/Interbotix/interbotix_ros_manipulators) | `01c557d65f781697c3a05bffebafbd3404e03eab` (2025-01-02) | jazzy | BSD-3-Clause. Only `interbotix_xsarm_descriptions` is built (WidowX 250 6DOF meshes); everything else gets COLCON_IGNORE. |

Evaluated but not used: [khaledgabr77/unitree_go2_ros2](https://github.com/khaledgabr77/unitree_go2_ros2) `55ff151632a0220bf3ae0c0772392615d86da7a8` (2026-09-08), kept in `~/ros2_ws/src` with a COLCON_IGNORE.

## D0 prep: Mac (Isaac scene builder, no ROS) and Isaac Sim

Recorded 2026-11-13 on the user's Mac (Apple Silicon, macOS 26.6.2). Venv `~/venvs/beach_isaac` built with uv
from `sim/isaac/requirements-mac.txt` (`notes/briefs/D0_prep.md` §1).

| Component | Version | Why |
|---|---|---|
| Python | 3.12.13 (uv-managed) | Isaac Sim 6.2 bundles 3.12; the laptop has 3.12.3 |
| numpy / Pillow / PyYAML / pytest | 1.26.4 / 10.2.0 / 6.0.1 / 7.4.4 | same as the laptop, so the scene generator gives the same numbers |
| usd-core | 26.11 (newest on 2026-11-13) | `sim/isaac/load_scene.py`; pin to Isaac 6.2's OpenUSD after GPU session 1 |
| Isaac Sim container | `nvcr.io/nvidia/isaac-sim:6.2.0` (Kit SDK 110.4.0) | not run yet; OpenUSD version and GPU host driver recorded after GPU session 1 |
| Blender | 5.3.0 (hash 02f0899d78e3, 2026-11-10; `brew install --cask blender`, the cask listed 5.2.3) | `sim/isaac/tools/blender_preview.py`: Cycles on Metal (Apple M1 Pro GPU) |
