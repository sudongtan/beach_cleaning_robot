# Beach Cleanup Robot Dog

**An autonomous legged robot with an arm that cleans and sorts beach litter end to end in simulation. It is benchmarked over 10 scenes, has a safety layer, and its architecture lets you swap the simulator.**

> 🚧 Work in progress. Current phase: A0 (setup). No results yet.

<!-- TODO: hero video (60–90 s): full cleanup run + 2–3 honest failure clips -->

## What it does

A simulated Unitree Go2 with an arm and a two-compartment basket on its back works through this loop:

1. Sweeps a beach inside a geofence and spots litter among shells, rocks and seaweed
2. Walks to each item, stops, and picks it up
3. Holds it in front of its camera to classify the material (PET, metal, glass, residual)
4. Stores it in the matching compartment: **recyclable** or **residual**
5. Walks to the bins and empties each compartment into the right bin

It slows down and stops for people, and it never moves its arm while someone is close.

## Why it's interesting

- **Full autonomy on a legged robot with an arm.** Recent research robots in this space are teleoperated or collect only one item type (see [Related work](#related-work)). This one runs the whole loop by itself and sorts what it collects.
- **Measured, not demoed.** Every change is benchmarked on the same 10 seeded scenes, and each upgrade is one row in the results table, regressions included.
- **Errors weighted by consequence.** The classifier says *what* an item is; config rules say *where* it goes. Putting residual waste in the recycling bin counts as worse than the reverse, so an uncertain item goes to residual.
- **Safety as a separate layer.** A simple rule-based supervisor is the only node allowed to command the legs.
- **Simulator-agnostic.** Components talk through frozen ROS 2 interfaces, and Gazebo and Isaac Sim are swapped with one launch argument.

## Architecture

<!-- TODO: architecture diagram -->

Perception → world model → task state machine → navigation → **safety supervisor** → locomotion; the task also drives grasping and the classifier. Localization fuses GPS, leg odometry and IMU, with no SLAM, because an open beach has too few features for it. 

## Results

<!-- Filled only from results/results.md — never by hand. -->

| Configuration | Items collected | Sorting accuracy | Contamination | False pickups | Min. distance to person | Localization drift |
|---|---|---|---|---|---|---|
| Baseline (ground truth) | – | – | – | – | – | – |

## Limitations

- Sand is approximated with a rough heightmap and friction zones; real granular sand is not simulated.
- Trash is rigid only. Material classification only shows performance on *simulated* textures (real plants use near-infrared sensors).
- Grasping may use a simulated "attach on contact" fallback; results are marked when it does.
- Simulation only, no real hardware.

Full list: [docs/limitations.md](docs/limitations.md).

## Related work

- **BinWalker** (IIT, 2026): a quadruped with an arm and a litter bin; teleoperated, collects bottles only. [arXiv 2603.10529](https://arxiv.org/abs/2603.10529)
- **VERO** (IIT, Journal of Field Robotics 2024): a quadruped with foot-mounted vacuums that collects cigarette butts.
- **Visual Whole-Body Control** (UCSD, CoRL 2024): learned whole-body picking from the ground with a legged manipulator. [Paper](https://proceedings.mlr.press/v270/liu25b.html)
- **RaiBo** (KAIST, Science Robotics 2023): learned quadruped locomotion on real sand.

## Running it

<!-- TODO after A0: install, build (`colcon build` from ~/ros2_ws), `make scene`, `make bench` -->

Stack: Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic (· Isaac Sim, optional)

