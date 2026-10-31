# A1 raw evaluation data (archived 2026-10-31)

Copied from `sim/gazebo/worlds/generated/tools/`, which is git-ignored and deleted by
`make clean`. Those tools write **one file per name and overwrite it on every run**, so what
survived is the *last* run under each name, not necessarily the run a decision rests on. The
accepted numbers are in `notes/logs/A1.md` (A1b verification, A1c issue 30) and
`notes/decisions.md`; this folder is the evidence that still exists. Original file dates below
(the copies have new timestamps).

## `slope/`: walking tests (`sim/gazebo/tools/slope_test.sh`, `goto.py`)

`slope_<seed>_<deg>.yaml` = the scene variant (dunes scaled to that slope), `.walk.log` = the
walk with its `RESULT` line, `.launch.log` = the simulation log.

| File | Date | Run | Result line | Used in |
|---|---|---|---|---|
| `slope_6_0` | 10-12 | A1b, stock, seed 6 flat | ok, max tilt 1.9° | A1b baseline |
| `slope_6_6.7` | 10-12 | A1b, stock | FELL, tilt 36° | A1b decision "7°" (superseded) |
| `slope_9_8.0`, `slope_9_9.0` | 10-12 | A1b, stock, empty basket | FELL | A1b decision "7°" (superseded) |
| `slope_9_5.0` | 10-13 | A1c, last run under this name | FELL at the start (tilt 99°, 0 waypoints) | **discarded** (network-outage period, A1 log issue 29); the accepted 5° passes were overwritten |
| `slope_9_6.0` | 10-13 | A1c, last run under this name | crashed (traceback, no result) | **discarded**; the accepted 6° passes were overwritten |
| `slope_9_7.0` | 10-13 | A1c round 3, custom 2 kg, smoothed | FELL past the crest, tilt 35° | decision "Walkable slope 6°" |
| `walk3`, `walk9`, `walk9b` | 10-12 | A1b scene walks | walk3 ok (dry + wet); walk9/9b FELL on seed 9's dune | A1b |
| `turn_flat.log` | 10-13 | A1c turn test, flat | simulation log only (no RESULT line kept) | A1 log issue 30 "turn test ok" |

**Gap:** the passing A1c runs behind the 6° limit (5° 2/2, 6° 2/2 with smoothing) have no
surviving raw file; only the A1 log table records them. Re-run with
`sim/gazebo/tools/slope_test.sh` if the limit is questioned. The tools still write to fixed
names, so copy results here before re-running.

## `startup/`: `scene_<seed>.log` (`startup_all.sh`)

Startup of each benchmark scene with the robot (detach, stand-up). Dates 10-12 (A1b), except
`scene_0`, `scene_3`, `scene_7`: 10-30, overwritten by the RTF runs for static distractors.

## `checks/`

| File | What |
|---|---|
| `camera_rgb.png`, `camera_depth.png` | headless camera frames (A1b DoD 3) |
| `held_item_custom.png` | lifted cigarette pack seen by the head camera (A1c item 2; the grasping `show_pose`) |
| `settle_5.log`, `settle_poses_5.txt` | items settling after spawn, seed 5 |
| `probe_ode.*`, `probe_expected.txt` | heightmap collision probe drops (A1 log issues 3, 4) |
| `det1.yaml`, `det2.yaml` | determinism check: the same seed generated twice |
| `actor_test.log` | walking-person actor test |

## RTF, static distractors (2026-10-30)

Measured with `sim/gazebo/tools/rtf_measure.sh 0 3 7` (sim-time advance over 30 s of wall time,
robot standing, headless; laptop on battery). Output as printed:

```
before (dynamic distractors)          after (static distractors)
seed 0: RTF 0.272 load 3.54 3499 MHz  seed 0: RTF 0.583 load 4.82 3300 MHz
seed 3: RTF 0.528 load 3.98 2900 MHz  seed 3: RTF 0.581 load 4.53 2699 MHz
seed 7: RTF 0.473 load 4.41 2999 MHz  seed 7: RTF 0.6   load 4.75 2999 MHz
```
