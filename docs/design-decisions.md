# Design decisions

The trade-offs that shaped the project, each with the evidence behind it and the alternatives that were rejected. Numbers come from Gazebo runs and tests in this repository; benchmark numbers come only from [`results/results.md`](../results/results.md).

## Safety

### One gate for every motion command
- **Choice:** the rule-based safety supervisor (`packages/safety`) is the only publisher of `/cmd_vel`. Navigation, the task's own final approach and teleop all publish `/cmd_vel_nav`. The arm doesn't pass through the supervisor, so the grasping node holds it still unless `/safety/state == ok`, and any arm planner added later may only plan; execution stays behind that gate.
- **Why:** a safety layer is only as strong as its bypasses. Keeping the rules in plain Python (`safety/supervisor.py`, thresholds in `config/safety.yaml`) makes them small enough to test exhaustively. The supervisor starts in `stop`, and a missing or stale input means stop, never "keep walking".
- **Rejected:** routing arm trajectories through the supervisor (a second command path inside the component that must stay simplest).

### The geofence is enforced by the supervisor, not only by the map
- **Evidence:** in an early integration run, the task's final approach (direct velocity commands, not Nav2) chased a misplaced item past the edge and over the waterline. Only Nav2's map had kept the robot inside, and these commands didn't go through Nav2.
- **Choice:** near or outside the geofence the supervisor removes only the *outward* component of `/cmd_vel`. Turning and moving inward stay allowed, so the robot can always come back. Property tests (2000 random commands each): the output is never faster than the input, and never further out near the edge.
- **Rejected:** stopping completely at the edge (the robot could never return); checks in the task only (the supervisor is the one place every command passes).

### Stop distance grows with speed
- **Evidence:** the first supervisor stopped at exactly 1.5 m and sent zero at once, but the gait coasted for 0.38 s while the person closed to 1.25 m. The problem was the stopping time, not the reaction time.
- **Choice:** speed and separation monitoring after ISO/TS 15066 (built on ISO 13855, S = K·T + C): the stop threshold includes the reaction time, the robot's measured braking (0.15 m/s²) and the person's approach speed (1.6 m/s), so the robot is at rest before the person reaches 1.5 m. Thresholds: 1.86 m standing, 4.86 m at full walking speed.
- **Known gap:** the `b2_bins` row has one violation. The robot kept creeping at 5–6 cm/s for about a second after the stop, so the braking it achieved was weaker than the model assumes. It is reported, not explained away.

### The benchmark measures the robot, not the person
- **Choice:** a person violation is the robot *moving* (commanded, or > 0.05 m/s measured) while a person is within the stop distance. The closest distance is still reported per run.
- **Why:** the simulated person is scripted and walks up to a robot that has already stopped. A "closest distance" rule would fail a correctly stopped robot, and no change to the robot could fix that.

## Sensing and localization

### No SLAM, no lidar; GNSS fusion instead
- **Choice:** RGB-D head camera, IMU, plain GPS (~2 m RMS, not RTK) and leg odometry. Localization upgrade: `robot_localization` dual EKF + `navsat_transform`, plus an EKF of our own for comparison.
- **Why:** scan matching degenerates in open, feature-poor scenes, and a beach is weak along the shoreline. Outdoor ROS robots usually fuse GNSS. Plain GPS keeps the localization problem real: the map gets the robot to ~0.5 m, and the camera finishes the job.
- **Cost, stated openly:** without lidar, only the forward head camera could see a person. Today `/people` comes from ground truth, which hides this ([limitations](limitations.md)).

## Manipulation

### A custom wide gripper, with its costs written down
- **Evidence:** the stock WidowX gripper opens 0.8–5.2 cm between the finger faces (measured from the meshes), and every beach item is 5.5–12 cm on its narrow side.
- **Choice:** long parallel jaws (2–15 cm) and a larger basket as a second robot variant. Cross-phase tests (`sim/common/tests/test_cross_phase.py`) check every item against the jaws, the basket, the payload, the mass budget and the arm's reach on every `make test`.
- **Cost:** no catalogue gripper has this stroke at this mass, and the arm then exceeds its payload rating ([limitations](limitations.md)).

### Simulated attach first, physics grasping as a measured upgrade
- **Choice:** grasping has `grasp_mode:=attach|physics`. Attach goes through simulator-agnostic services (`/sim/attach`, `/sim/detach`), so the pipeline stays identical for an Isaac Sim port.
- **Evidence:** a [physics-grasp study](../results/physics_grasp/) (10 trials per item: lift, wrist turn, 1 m walk) found two items failing for geometric reasons: the gripper bar hit the tall jar and the 1.5 L bottle. The jar was resized to a real 4 oz product and the bottle removed, rather than tuning the simulation until it passed.

### The bin stand-off is set by the arm's sweep, not the head
- **Evidence:** the robot tipped over twice at the bin while standing still. Forward kinematics showed the folded arm sweeping 0.52 m ahead of the body, past the bin wall; the arm pushed the static bin and levered the robot over. The earlier reach check had kept only the *head* clear.
- **Choice:** place distance 0.87 m, and a test that sweeps every arm move at the bin against the wall. It fails at the old distance, as Gazebo did.

## Locomotion

### An off-the-shelf blind controller, chosen by measurement
- **Choice:** [legubiao/quadruped_ros2_control](https://github.com/legubiao/quadruped_ros2_control) `basic_quadruped_controller`, pinned, with two small patches.
- **Evidence:** with the full payload it walked straight with 0.2° drift over 10 s. CHAMP drifted 12–32°, and another controller in the same repo rolled over after turns.
- **Second implementation:** the pretrained HIMLoco policy didn't fall in our tests, but doesn't move at the slow speeds needed to approach an item. Its training zeroes commands below 0.2 m/s. It stays selectable (`locomotion:=legubiao_rl`), not the default.
- **Cost:** the blind gait fell in 5 to 10 of the 10 runs of every row up to B3. The fixes before B4 removed the triggers (sideways step between the bins, every distractor an obstacle, no turn-in-place at goals).

### A pretrained policy as the default since B4
- **Choice:** the pretrained `robot_lab` Go2 policy from the same controller repo (`locomotion:=legubiao_robot_lab`, CPU libtorch) replaced the basic controller as the default. The basic controller stays selectable (`locomotion:=legubiao`).
- **Evidence:** 0 falls in the 20 runs of its two benchmark rows, against 7 of 10 in the row before, and 75 of 110 items collected instead of 39, with 0 safety violations in the final row.
- **Alternatives:** HIMLoco doesn't move at slow speeds. Tuned gains of the basic controller fell in 2 of 5 screening runs and picked too slowly. OCS2 wasn't tried (a large build, and `robot_lab` passed first).
- **Cost:** it wasn't trained with the arm's weight or very slow commands, so small moves go through speed maps calibrated on one flat, dry-sand scene. 3 of 10 runs in the final row end stuck beside a bin, cause still open.

## Simulation and benchmark

### The simulator is a component
- **Choice:** components talk only through frozen ROS 2 interfaces. Simulator-specific code lives only in `sim/` and `sim_bridge_gazebo`, and ground truth comes through the same topics a real component would publish. Porting to Isaac Sim should need one new bridge package and no other changes.

### Making a laptop CPU fast enough
- **Evidence:** with the whole stack running, the real-time factor was 0.23. Each Python node using sim time spent 25–40 % of a core just on Gazebo's 1 kHz `/clock`.
- **Choice:** a small C++ node republishes the clock at 100 Hz for our nodes; the leg controllers keep the 1 kHz clock. RTF 0.23 → 0.42 with the robot working; physics unchanged. Making distractors static added about +40 %.

### Results only from logs
- **Choice:** `make bench` runs 10 seeded scenes headless with fixed caps (3600 s sim, a stall detector). Metrics come from the final ground truth (is the item really in the bin?), never from the task's own claims. Rows are generated from the run JSONs, never edited by hand, and every upgrade adds a row, including the ones that made things worse (`b1_rim`).
- **Fall cut-off:** safety events from 2 s before a fall onward are listed apart, not counted as violations. A fall is already a result of its own, and counting its effects twice would hide what the supervisor actually did.

## Perception (planned)

### Grounding models instead of training
- **Choice:** detection with a pretrained, text-prompted open-vocabulary model (YOLOE) on the CPU, with no training data pipeline, compared with a classical depth segmentation of our own. Detection precision, recall and mAP go into the benchmark row.
- **Risks checked first:** CPU latency next to Gazebo, and whether a model trained on real photos finds procedural trash. That is why realistic trash visuals come before the detector.
