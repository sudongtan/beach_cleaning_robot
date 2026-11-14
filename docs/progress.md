# Progress details

The [README](../README.md#progress) has the short version. This page has the details per phase: what was built, what the benchmark rows showed and what went wrong along the way. All numbers come from the run logs and from [`results/results.md`](../results/results.md).

**Contents:** [Components](#components-implemented-vs-planned) · [Part A](#part-a-gazebo-pipeline--benchmark) · [Part B](#part-b-real-upgrades) · [Part C](#part-c-ground-truth--real) · [Part D](#part-d-isaac-sim-optional) · [Engineering notes](#engineering-notes)

## Components: implemented vs planned

In this project a *baseline* reads the simulator's ground truth. Each baseline is built first so the whole pipeline can be tested end to end. Each one is then replaced by a real implementation behind a launch argument, and the baseline is kept for comparison.

| Component | Implemented now | Planned upgrade |
|---|---|---|
| Locomotion | [legubiao/quadruped_ros2_control](https://github.com/legubiao/quadruped_ros2_control) (pinned, two small patches) running the pretrained `robot_lab` policy (B4, see [B4](#b4-walking-controller-done-row-b4_robot_lab_flip)); fall detection makes the legs go passive. The basic model-based controller stays selectable (`locomotion:=legubiao`) | Later: an own Isaac Lab RL policy, trained with slow commands and the arm's weight |
| Robot | Go2 + Interbotix WidowX 250 6DOF + custom wide jaws (2–15 cm) + 34 × 30 cm basket, own xacro | — |
| World | 10 seeded scenes (25 × 20 m, 8–15 trash items, 6–12 distractors, a walking person in 5 of them), textured procedural meshes (C2) | — |
| Localization | Ground-truth pose published as `/odom`, `/odom/global` and TFs | C1: dual EKF + `navsat_transform` over noisy GNSS, leg odometry and IMU |
| Mapping | Static traversability grid from the terrain (steep slopes and water are blocked) | — |
| Perception | Ground-truth detections from the simulator | C3: a pretrained open-vocabulary model prompted with text (YOLOE), with no training, plus depth for 3D position; compared with a classical segmentation of our own |
| Classifier | Ground-truth material + sorting rules from config | C4: zero-shot classifier on close-up crops; confusion matrix and contamination rate |
| World model | Removes duplicate detections, tracks the coverage grid and what is in each compartment | — |
| Navigation | Nav2 (NavFn planner + Regulated Pure Pursuit) plus a camera-relative final approach | — |
| Grasping | Own IK and reach-envelope module, scripted top-down pick, "attach on contact" in simulation | C5: picks by contact physics: segment the item, estimate its geometry, generate grasp candidates (`simple_grasping` and our own), plan the motion (MoveIt 2 / MTC and our own planner), guarded descent, check that the grasp worked |
| Safety | Rule-based supervisor, the only publisher of `/cmd_vel`: person slow/stop, e-stop, geofence, arm interlock | — |
| Task | State machine with retries, skips, pause on a safety state and fall handling; one event log per run | — |
| Benchmark | `make bench`: 10 seeded scenes headless, one JSON per run, metrics from the final ground truth, one row per configuration in `results/results.md` | A row per upgrade |

## Part A: Gazebo pipeline + benchmark

### A0 Setup (done 2026-10-12)

The Go2 walks on `/cmd_vel` in Gazebo Harmonic with the legubiao controller. A WidowX 250 6DOF arm and a two-compartment basket are mounted through an own xacro. Two robot variants: `variant:=custom` (default; wide jaws for 2–15 cm items, 34 × 30 cm basket) and `variant:=stock` (WidowX gripper, small basket). The robot walked as steadily with 6 kg on its back as without.

### A1 Beach world (done 2026-10-30)

A seeded scene generator builds a 25 × 20 m beach: dune terrain with dry and wet sand zones, 8–15 trash items, 6–12 distractors (shells, rocks, seaweed, driftwood), two bins and, in half the scenes, a scripted walking person. Camera, depth, IMU and GNSS are bridged to ROS 2.

**Physics-grasp feasibility study.** Can the custom gripper hold each trash item by friction alone, through a lift, a wrist turn and 1 m of walking? Every item stayed in the jaws in at least 6 of 10 valid trials. Cans, the glass jar and the cup succeeded 10 of 10 times; the food container (6/10) and the cigarette pack (7/9) are the weak cases. The study led to resizing two items and dropping the 1.5 L bottle from the item set. Data and method: [results/physics_grasp/](../results/physics_grasp/).

### A2 Walking skeleton (done 2026-11-03)

Every pipeline component was first built as a ground-truth baseline with unit and node tests (A2a), then run together in Gazebo, end to end and unattended (A2b).

Integration runs (debugging runs, not benchmark results):

- `a2b_dev1` (dev scene 101, 15 × 12 m): **first complete unattended run.** All 6 items were picked, sorted and ended up in the correct bins, with no fall. One glass bottle came out of the basket together with the item stored on top of it and landed in the right bin; the task now counts such an item as placed.
- Scene 3 (25 × 20 m), four runs: each fell, for a different reason each time:
  1. a held item swept into the basket wall during storing and the rigid simulated grasp pushed the robot;
  2. it stepped on a 2 cm seaweed clump;
  3. Nav2's turn on the spot at a goal reversed direction;
  4. it tipped over while standing still at a bin, with no command at all.

  The first three were fixed: a new arm path, every distractor marked as a map obstacle, and smoothed commands. The fourth run got furthest: 8 items picked and stored and all 4 recyclable items placed, before the fall. The basket was also reworked: items go to the lowest free spot, and items that fall out are looked for again.

### A3 Benchmark (done 2026-11-04)

`make bench` runs the 10 seeded scenes headless, unattended, one run each, and writes one row to `results/results.md`. The **baseline** row: every item that reached a bin was in the right one, and no run had a safety violation. Few items reached a bin, though (17 of 110): 7 of the 10 runs ended with the walking controller falling (three of them on the short walk between the two bins) and 3 with the robot stuck in navigation. These two failures are where Part B starts.

## Part B: Real upgrades

The robot itself gets better while every input is still ground truth. Each upgrade adds one row and is judged against the row before, including upgrades that make things worse.

### B1 Bin rim height (done, row `b1_rim`)

Puts each bin's rim at its real height: the bins stand on sand 5–23 cm above the level the task assumed, so the arm released items lower than intended. With the right rim, 12 items reached a bin: 10 placed and 2 correctly recognised as already inside (carried along with another item). One item could no longer be placed (out of the arm's reach at the correct height); B2 adds a lower release for that case. The row also shows how noisy the walking controller is: all 10 runs ended in a fall, 4 of them in the first 160 s and 7 of them away from the bins, before or without any placement. B1 can't change those runs, so this row's totals are lower than the baseline's from the falls alone.

### B2 Place in both bins without walking between them (done, row `b2_bins`)

Replaces the walk from one bin to the other (back out, turn, walk) with one straight sideways step of 1.2 m, and lets the arm release an item a little lower when the full height is out of reach. 39 items reached a bin, two runs finished the whole beach for the first time, all 5 sideways steps went through, and no run fell at the bins. Three runs still ended stuck in navigation (that's B3).

This row has the project's first safety violation: in one run the robot was still creeping at 5–6 cm/s for about a second after the supervisor had stopped it, while the person walked up to 1.1 m. The supervisor's stop distance assumes the robot brakes faster than it did here; the longer runs meet the person far more often.

### B3 Nav2 "collision ahead" stalls (done, row `b3_nav`)

Nav2's path follower refuses to move while the robot's outline touches an obstacle cell, including while it turns on the spot towards its path, and the task's own short moves (the last steps to an item or a bin) don't look at the map; next to a rock, a heap of seaweed or the edge of the beach the robot could end up where it can't turn. Before every navigation goal the task now checks the map and, if needed, walks back the way it came, or a short way straight, until it has room to turn. No run got stuck (3 in each row before), three runs finished the whole beach, and there was no safety violation. The total stays at 39 items: the runs last longer, so more of them end in a fall of the walking controller (7), one of them during such a backing-out move.

### B4 Walking controller (done, row `b4_robot_lab_flip`)

Falls ended 5 to 10 of the 10 runs in every row up to B3, so the walking controller was the biggest lever. Candidates went through a repeatable fall test first (gates, then a short screening on 5 seeds), and only the winner got a benchmark row.

| Candidate | Result |
|---|---|
| `robot_lab` policy (pretrained, same controller repo, CPU libtorch, `locomotion:=legubiao_robot_lab`) | **Winner, now the default.** Passed the gates with 0 falls in screening. It first ignored the task's few-centimetre moves (8 items given up as "no progress"); a minimum speed for the task's short moves and calibrated speed maps for backward and sideways moves fixed most of that |
| HIMLoco (pretrained policy, same repo, `locomotion:=legubiao_rl`) | Never fell, but doesn't walk slowly (see below) |
| Tuned gains of the basic controller | 2 of 5 screening runs fell, and 0.14 picks per simulated minute against a gate of 0.26; not benched |
| OCS2 MPC controller (same repo) | Not tried: a large build, and `robot_lab` passed first |

| Row | Done / stall / fell (of 10) | Collected | Safety violations | Successful picks |
|---|---|---|---|---|
| `b3_nav` (basic controller) | 3 / 0 / 7 | 39/110 | 0 | 81/87 |
| `b4_robot_lab` | 9 / 1 / 0 | 79/110 | 1 | 115/137 |
| `b4_robot_lab_flip` | 7 / 3 / 0 | 75/110 | 0 | 104/120 |

The first row had one safety violation: the arm kept moving briefly after a safety stop. The grasping node now holds the arm at the controller's reference when a stop arrives, and the pose that shows an item to the camera was flipped to avoid a 3 rad wrist swing. The row was rerun with that fix; the rerun is the B4 row.

Three of the ten runs in the final row end stuck about 0.6 m from a bin, where the robot doesn't back away. The cause is still open: at the same spot a plain back-up command works, and in the run examined the robot had already stopped moving right after its last placement. Contact between the arm or body and the bin is the leading guess, not yet tested.

Both learned policies share one gap: their training sets every command below 0.2 m/s to zero, so they never learned the slow steps the final approach to an item and the docking at a bin need. A policy of our own, trained with slow commands and the arm's mass, is a later option on a rented GPU.

**HIMLoco test** (Long et al., ICLR 2024), compared with the current controller on the same tests:

| Test | HIMLoco | Current controller |
|---|---|---|
| Fill the basket, then walk 2 m out and back with 180° turns (4 walks) | no fall; 19 of 20 items stayed in the basket | 2 falls in 4 walks; 10 of 10 items stayed in the run without a fall |
| Stand 5 min at the spot of the stand-still fall | no fall (max tilt 7°) | no fall (max tilt 1°) |
| Slow commands: 0.05–0.10 m/s forward, slow right turns | **doesn't move** | follows them |
| Full scene-3 run | no item picked: both approaches stalled at the slow final steps | (see the A2 runs above) |

It also stands 9 cm lower and about 6° tilted with the arm and basket, which it never saw in training. One integration detail: upstream's config lists the leg joints in a different order from the policy's (front right first instead of front left), and with that order the robot fell as soon as the policy took over.

## Part C: Ground truth → real

Each step swaps one ground-truth baseline for a real component. A row is expected to be equal or worse than the ground-truth row: it reports what making that component real costs, judged by a component check with a pass rule, by safety, and by the gap to the row before.

### C1 Localization (implemented, re-running its benchmark)

Dual EKF (`robot_localization` + `navsat_transform`) over noisy GNSS, leg odometry and IMU, plus an own EKF for comparison that also estimates the leg odometry's scale online. Tuned offline on a recorded run: map error RMS 0.22 m (`robot_localization`) and 0.17 m (own EKF). Findings on the way:

- Leg odometry under-reads speed by 6–13 %, more at higher speed.
- With the tutorial's noise values, the map pose jumped with each GNSS fix (speed error 1.2 m/s median), and the supervisor, which measures the robot's speed from that pose, kept it stopped. Fixed in the localization config, not in the supervisor.
- `navsat_transform`'s very first output was in UTM coordinates instead of local ones, a 190 km jump that took minutes to decay. Fixed with a Mahalanobis rejection gate.
- The geofence margin was widened from 0.3 to 0.5 m to cover the localization error.

First benchmark row, `c1_ekf`: 7 runs finished, 3 stalled, no falls, 87 of 110 items collected, mean localization error 0.157 m (max 0.611 m). It also broke the arm safety rule 5 times, so it does not count as clean. Part of the cause was the supervisor reading the robot's speed from the corrected map pose, which jumps with GNSS fixes. The speed now comes from leg odometry, and the benchmark is being re-run.

### C2 Realistic trash visuals (done, no row)

UV-mapped meshes with invented, brand-free labels, caps, lids and transparent PET and glass, so the detector in C3 has something realistic to see. The visual bounds match the collision boxes within 2 mm. The rendering cost in Gazebo is 1.45 % of the real-time factor, within the 5 % limit.

### C3 Perception (built, failing its checks)

YOLOE, a pretrained open-vocabulary detector prompted with text, exported to OpenVINO for the CPU, with no training. Its masks plus the depth image give each item's 3D centre, size and yaw. A classical segmentation of our own (sand plane fit + blobs above it) is the comparison.

Both checks fail so far, so there is no benchmark row. Running the `n` model next to the full stack costs 18.5 % of the real-time factor standing and 11.4 % walking, and the `s` model 28 % and 20 %, against a limit of 10 %. Within 4 m the models find 0.7 % (`n`), 1.7 % (`s`) and 2.7 % (`m`) of the trash, against a target of 70 %, and they miss plainly visible bottles. Other prompts and lower confidence thresholds are being tried.

### C4 Classifier (built, narrowly failing its check)

A zero-shot classifier (MobileCLIP-S2 through OpenCLIP) on close-up crops of the held item, compared with an own classifier on colour features and nearest centroids. On 48 held-out crops, 13.0 % (own) and 12.0 % (MobileCLIP) of the items reaching the recyclable bin were wrong, against a limit of 10 %, one item too many in each case. No benchmark row yet.

### C5 Perception + grasping pipeline (planned)

The last and hardest step: picks by contact physics instead of the simulated attach, in the industry pattern look → plan → move → verify. The A1 feasibility study says the gripper can hold every item.

## Part D: Isaac Sim (optional)

- **D0 Preparation:** everything before the pipeline can run in NVIDIA Isaac Sim. First the 11 benchmark scenes as USD files, built and tested on a Mac and checked in Isaac Sim 6.2 on a rented GPU (done 2026-11-13: every scene opens without errors and renders from the same views as Gazebo, for $0.14); then the robot standing and walking with the same policy as in Gazebo, and the arm picking an item in attach mode.
- **D1 Port:** run the same pipeline in Isaac with zero changes outside `sim/` and `sim_bridge_isaac`.
- **D2 Comparison:** the same benchmark in Gazebo and Isaac, side by side.
- **D3 Soft sand:** a one-scene demo of the walking controller on particle sand.

D0 and D1 run in parallel with Part C. D2 and D3 wait until Part C is done, so the comparison uses the final pipeline.

## Engineering notes

- **Cross-phase tests.** Unit tests check that the parts fit together physically: the gripper fits every item, the arm reaches every compartment position and both bins, and the arm's sweep clears the bin wall. They were added after a run where the folded arm hit a bin and levered the robot over.
- **Debugging in a slow simulation.** The simulation runs at about 0.4× real time on the laptop CPU. A full scene takes about 15 simulated minutes, or 40 wall-clock minutes. Problems found in Gazebo runs and fixed include wall-clock timeouts that fired too early at that speed, "phantom" items from a gripper TF that lagged behind the camera, falls caused by walking with the arm raised, and the robot being unable to back away from a bin under Nav2. Each fix is backed by a regression test.
- **439 unit and node tests** (`make test`) run against a fake simulator, without Gazebo.
