# Gazebo test tools (A1)

Diagnostic and check scripts used while building the A1 beach world and the A1b
robot integration. Background and results: `notes/logs/A1.md`. They run headless
on their own Gazebo partition / ROS domain (`scene_env.sh`: `GZ_PARTITION=a1b`,
`ROS_DOMAIN_ID=43`) and write to `sim/gazebo/worlds/generated/tools/` (git-ignored).

## Scene with the robot (A1b)

| Script | What it does |
|---|---|
| `scene_up.sh <seed>` / `scene_down.sh` | start `make scene` headless in the background, wait for the stand-up / stop everything (incl. stale servers) |
| `source scene_env.sh` | environment for `ros2` / `gz` commands against that scene |
| `startup_all.sh [seeds]` | start every seed: items detached, stand-up, RTF, camera rate |
| `check_ground_truth.py <sdf>` | `/sim/ground_truth/pose` + `objects` vs Gazebo's pose stream; objects vs spawn poses |
| `check_sensors.py` | GNSS error vs ground truth + covariance, IMU, saves camera rgb/depth PNGs |
| `check_people.py <seed>` | `/people` vs the rendered actor (top-down segmentation camera; no robot needed) |
| `dump_objects.py out.yaml` / `--compare a b` | world state at a sim time, to compare two launches |
| `goto.py x,y ...` | walk through waypoints on ground truth, logs zone/slope/tilt; publishes `/cmd_vel_nav` (through the safety supervisor, A2+; `--topic /cmd_vel` only for A1-style tests without the supervisor) |
| `slope_test.sh <seed> <deg>` | climb a seed's steepest line with the dunes scaled to `deg` (0 = as generated), and back |
| `slope_variant.py` | builds those variants (used by `slope_test.sh`) |
| `arm_pose.py <pose>` / `--joints` | move the arm to a named pose from `arm_poses.yaml` or to joint angles |
| `attach_test.py <item>` | gripper down, `/sim/attach`, lift, over the basket, `/sim/detach` |

## A2b (full skeleton runs)

| Script | What it does |
|---|---|
| `stop_sim.sh` | stop every Gazebo / A2 stack process by PID (incl. `kill -9` for the Gazebo server); prints `remaining: 0` when clean. Run before and after every run |
| `watch_until.py <limit_s> <stall_s>` | follow a run: task / safety state changes, pose every 120 s; stops on DONE / FELL / a stall |
| `trace.py <duration_s>` | dense trace every 0.5 s: pose, tilt, roll / pitch, arm joints, `/cmd_vel_nav`, `/cmd_vel`, states (found the falls and the bin tip-over) |
| `zone_walk_test.sh <seed> <label> <x0> <x1> <ya> <yb> [lanes]` | lane walks in one terrain zone through the supervisor; env `ARM_POSE`, `SPEED`, `SMOOTH` |
| `rtf_measure.sh` | real-time factor of a running scene |
| `turn_walk_test.sh <seed> [args]` / `turn_walk_test.py` | turn-on-the-spot → walk-off cycles through the supervisor, max tilt; env `GOTO` (walk somewhere first), `LABEL`, `TOOL` (another test script, e.g. `cmd_replay.py`) |
| `nav_walk_test.sh <seed> <label> --minutes N` / `nav_walk_test.py` | full skeleton without task / perception / grasping; Nav2 loops the search waypoints until N sim minutes, a fall (fall rate) or stuck (within 0.5 m for `--stall-min` sim min, default 2: stops, `stuck=True`); env `RECORD=1` |
| `cmd_record.py <wall_s> <out.jsonl>` | record `/cmd_vel_nav`, `/cmd_vel`, controller input as JSON lines |
| `cmd_replay.py <rec.jsonl> --start --end [--repeat]` | replay recorded `/cmd_vel_nav` with the original timing, stop at a fall |
| `velocity_track_test.py [--sweep] [--step-s 6]` | velocity tracking: holds command steps on `/cmd_vel_nav`, measures body velocity from ground truth (second half of each step); `--sweep`: slow commands (found HIMLoco's deadband). Run as `TOOL=` of `turn_walk_test.sh` |
| `hold_test.py [--minutes 5]` | stand still with no command, log tilt / roll / pitch every 30 sim s, stop at a fall; with `GOTO=` of `turn_walk_test.sh` to stand at a given spot |

`LOCO=legubiao_rl` (environment) selects the HIMLoco controller for `scene_up.sh`, `rtf_measure.sh`, `nav_walk_test.sh` and `turn_walk_test.sh` (default `legubiao`, the basic controller).

Full-run recipe: `notes/logs/A2.md`, "A2b".

## A1c (custom robot)

| Script | What it does |
|---|---|
| `physics_grasp_run.sh [args]` / `physics_grasp_test.py` | physics-grasp trials per item (flat world, own partition) → `results/physics_grasp/` |
| `slope_test.sh <seed> <deg>` | env `LOAD=2.0`, `VARIANT=custom`, `SMOOTH="--accel 0.3 --ang-accel 0.5"` |
| `turn_test.py` | abrupt walk → turn-on-the-spot cycles, max tilt |
| `a1c_walk_campaign.sh`, `a1c_walk_campaign2.sh` | the A1c slope / turn campaigns |
| `stop_all_tests.sh` | stop every campaign and test scene started from here (by PID; other projects untouched) |

All of these use `scene_env.sh`: loopback only (`GZ_IP=127.0.0.1`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`).
Run long campaigns under `systemd-inhibit --what=sleep:idle:handle-lid-switch ...` so the laptop doesn't suspend.
Don't run them while someone else uses the machine for another simulation.

## World only (A1a)

| Script | Checks |
|---|---|
| `run_heightmap_probe.sh [detector] [seed]` | collision heightmap matches `terrain.height_at` (drops boxes on a grid) |
| `settle_check.sh <seed>` | real-time factor, and that every item settles where it spawned |
| `actor_collision_test.sh [seed]` | the walking person has no collision (a box on its path doesn't move) |
| `compare_poses.py`, `heightmap_probe.py`, `gz_poses.py`, `common.sh` | helpers |

For rendered images of the world, use `python3 sim/gazebo/load_scene.py --seed N --debug-camera`
and subscribe to the `/debug/...` topics while it runs. Gazebo only renders a camera that has a
subscriber.
