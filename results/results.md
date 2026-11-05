# Benchmark results

Rows are appended by `make bench` (`bench/runner.py`, `bench/report.py`), only from the run JSONs in
`results/runs/<row>_s<seed>.json`; never edited by hand. Spec: 10 scenes (seeds 0–9, `bench/scenes.yaml`),
one run each, headless Gazebo. Compare every new row with the previous one.

Columns (per-run values; "mean (min–max)" over the 10 runs unless it says sum):
- **Row / date / commit / sim:** row name, date the row was written, git commit of the runs (`*` = uncommitted changes), simulator.
- **Components:** locomotion / localization / perception / classifier / grasp mode. `gt` = ground-truth baseline.
- **Caps:** sim-time cap, wall-time cap, stall limit per run (`bench/bench.yaml`).
- **Ends:** how the runs ended: done (task finished), fell, stall (no progress), sim_cap / wall_cap (timeout), dead_stack / startup_failed (infrastructure).
- **Collected:** trash items inside a bin at the end of the run (final ground truth: bin inner box, below the rim), sum over all runs / items in all scenes.
- **Sorting acc.:** items in the correct bin (`config/sorting_rules.yaml`) / items in any bin, over all runs.
- **Contam.:** residual items in the recyclable bin, sum (weighted ×2 in "weighted errors" in the run JSONs).
- **False pickups:** distractors in a bin or the basket, or picked (`pick` event on a world-model item whose ground-truth id is a distractor), sum.
- **Grasp:** successful `pick` events / all, over all runs; "simulated attach" while `grasp_mode:=attach`.
- **Items/min:** collected items per sim minute (from the recorder's start after the stand-up to the end).
- **Falls / stuck / timeouts:** runs with a fall (tilt > 30°); stall ends plus `no_progress` skips; runs ended by a cap. Sums.
- **Distance:** metres walked (ground truth, steps > 1 cm).
- **Dup.:** extra world-model entries for the same ground-truth item (last `/world_model`), sum.
- **Coverage:** share of geofenced coverage cells swept (last `/world_model`).
- **Safety viol.:** person rule (robot moves while a person is within the stop distance), arm rule (an arm joint changes > 0.03 rad within 0.5 s while `/safety/state` ≠ ok; the per-seed table also gives the largest change since the state left ok, slow creep included), geofence rule (`base_link` outside). Sum over all runs; must be 0. Events from 2 s before a fall on are listed apart in the run JSON (`safety.after_fall`).
- **Min person d.:** closest robot–person distance over all runs with a person (m).
- **Class. acc.:** classifier accuracy on the held items (`classify` events vs the true material).
- **Drift:** localization error `/odom/global` vs ground truth, mean of the runs' means (max) in m.
- **Detection:** P / R / mAP; `n/a` while the detector reads ground truth.

| Row | Date | Commit | Sim | Components | Caps | Ends | Collected | Sorting acc. | Contam. | False pickups | Grasp | Items/min | Falls / stuck / timeouts | Distance (m) | Dup. | Coverage | Safety viol. | Min person d. (m) | Class. acc. | Drift (m) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 2026-11-04 | 318758e | gazebo | loco legubiao / loc gt / perc gt / cls gt / grasp attach | 3600 s sim / 200 min wall / stall 600 s | fell 7, stall 3 | 17/110 | 1.00 (17/17) | 0 | 0 | 0.92 (69/75) simulated attach | 0.12 (0.00–0.40) | 7 / 3 / 0 | 48 (8–92) | 10 | 0.84 (0.50–1.00) | 0 | 0.22 | 1.00 (69/69) | 0.002 (0.029) | n/a (ground truth) |
| `b1_rim` | 2026-11-04 | f63ed1a | gazebo | loco legubiao / loc gt / perc gt / cls gt / grasp attach | 3600 s sim / 200 min wall / stall 600 s | fell 10 | 12/110 | 1.00 (12/12) | 0 | 0 | 1.00 (50/50) simulated attach | 0.12 (0.00–0.53) | 10 / 0 / 0 | 43 (2–145) | 8 | 0.73 (0.22–1.00) | 0 | 0.09 | 1.00 (50/50) | 0.003 (0.036) | n/a (ground truth) |
| `b2_bins` | 2026-11-04 | c5dfba7 | gazebo | loco legubiao / loc gt / perc gt / cls gt / grasp attach | 3600 s sim / 200 min wall / stall 600 s | done 2, fell 5, stall 3 | 39/110 | 1.00 (39/39) | 0 | 0 | 0.93 (85/91) simulated attach | 0.14 (0.00–0.37) | 5 / 3 / 0 | 135 (0–367) | 13 | 0.82 (0.25–1.00) | 1 | 0.04 | 1.00 (85/85) | 0.002 (0.037) | n/a (ground truth) |
<!-- summary rows above this line -->

### Row `baseline`: per seed

| Seed | End | Sim s | Wall min | Collected | Sorting acc. | Contam. | False pickups | Grasp | Fall (t, tilt) | Stuck | Distance (m) | Dup. | Coverage | Safety viol. (person / arm / geofence) | Min person d. (m) | Arm change since not ok (rad) | Infra retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | stall | 893 | 40 | 2/12 | 1.00 | 0 | 0 | 5/5 | – (max 18°) | 1 | 26 | 1 | 0.84 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 1 | fell | 670 | 30 | 4/13 | 1.00 | 0 | 0 | 7/7 | 669 s, 48° | 0 | 65 | 2 | 0.86 | 0 (0 / 0 / 0) | 1.04 | 0.008 | 0 |
| 2 | fell | 433 | 22 | 0/15 | – | 0 | 0 | 8/8 | 431 s, 102° | 0 | 39 | 0 | 0.98 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 3 | fell | 338 | 14 | 0/9 | – | 0 | 0 | 5/5 | 337 s, 68° | 0 | 27 | 0 | 0.71 | 0 (0 / 0 / 0) | 0.93 | 0.014 | 0 |
| 4 | stall | 1384 | 70 | 4/13 | 1.00 | 0 | 0 | 15/21 | – (max 15°) | 1 | 82 | 4 | 0.98 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 5 | fell | 95 | 4 | 0/8 | – | 0 | 0 | 1/1 | 93 s, 66° | 0 | 8 | 0 | 0.50 | 0 (0 / 0 / 0) | 0.25 | 0.033 | 0 |
| 6 | stall | 601 | 26 | 0/8 | – | 0 | 0 | 5/5 | – (max 7°) | 1 | 35 | 0 | 0.73 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 7 | fell | 411 | 19 | 0/13 | – | 0 | 0 | 7/7 | 410 s, 61° | 0 | 25 | 0 | 0.76 | 0 (0 / 0 / 0) | 0.28 | 0.016 | 0 |
| 8 | fell | 756 | 32 | 5/11 | 1.00 | 0 | 0 | 9/9 | 755 s, 104° | 0 | 79 | 2 | 1.00 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 9 | fell | 833 | 34 | 2/8 | 1.00 | 0 | 0 | 7/7 | 833 s, 103° | 0 | 92 | 1 | 1.00 | 0 (0 / 0 / 0) | 0.22 | 0.063 | 0 |

### Row `b1_rim`: per seed

| Seed | End | Sim s | Wall min | Collected | Sorting acc. | Contam. | False pickups | Grasp | Fall (t, tilt) | Stuck | Distance (m) | Dup. | Coverage | Safety viol. (person / arm / geofence) | Min person d. (m) | Arm change since not ok (rad) | Infra retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | fell | 653 | 28 | 4/12 | 1.00 | 0 | 0 | 9/9 | 651 s, 52° | 0 | 54 | 3 | 0.98 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 1 | fell | 25 | 1 | 0/13 | – | 0 | 0 | 0/0 | 24 s, 34° | 0 | 2 | 0 | 0.22 | 0 (0 / 0 / 0) | 7.27 | 0.000 | 0 |
| 2 | fell | 87 | 4 | 0/15 | – | 0 | 0 | 1/1 | 86 s, 39° | 0 | 12 | 0 | 0.46 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 3 | fell | 545 | 26 | 0/9 | – | 0 | 0 | 7/7 | 544 s, 104° | 0 | 51 | 0 | 0.99 | 0 (0 / 0 / 0) | 0.96 | 0.028 | 0 |
| 4 | fell | 462 | 21 | 2/13 | 1.00 | 0 | 0 | 7/7 | 461 s, 67° | 0 | 36 | 1 | 0.78 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 5 | fell | 1088 | 50 | 1/8 | 1.00 | 0 | 0 | 10/10 | 1086 s, 100° | 0 | 145 | 2 | 1.00 | 0 (0 / 0 / 0) | 0.09 | 0.067 | 0 |
| 6 | fell | 159 | 6 | 0/8 | – | 0 | 0 | 1/1 | 157 s, 103° | 0 | 12 | 0 | 0.44 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 7 | fell | 105 | 4 | 0/13 | – | 0 | 0 | 1/1 | 105 s, 61° | 0 | 8 | 0 | 0.47 | 0 (0 / 0 / 0) | 0.28 | 0.017 | 0 |
| 8 | fell | 582 | 26 | 5/11 | 1.00 | 0 | 0 | 8/8 | 581 s, 33° | 0 | 62 | 2 | 1.00 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 9 | fell | 426 | 18 | 0/8 | – | 0 | 0 | 6/6 | 425 s, 104° | 0 | 45 | 0 | 1.00 | 0 (0 / 0 / 0) | 0.37 | 0.052 | 0 |

### Row `b2_bins`: per seed

| Seed | End | Sim s | Wall min | Collected | Sorting acc. | Contam. | False pickups | Grasp | Fall (t, tilt) | Stuck | Distance (m) | Dup. | Coverage | Safety viol. (person / arm / geofence) | Min person d. (m) | Arm change since not ok (rad) | Infra retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | fell | 15 | 1 | 0/12 | – | 0 | 0 | 0/0 | 14 s, 104° | 0 | 0 | 0 | 0.25 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 1 | fell | 182 | 8 | 0/13 | – | 0 | 0 | 3/3 | 181 s, 74° | 0 | 14 | 0 | 0.64 | 0 (0 / 0 / 0) | 0.80 | 0.282 | 0 |
| 2 | done | 2289 | 111 | 13/15 | 1.00 | 0 | 0 | 17/23 | – (max 7°) | 0 | 367 | 3 | 1.00 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 3 | done | 1764 | 104 | 8/9 | 1.00 | 0 | 0 | 8/8 | – (max 5°) | 0 | 275 | 2 | 1.00 | 0 (0 / 0 / 0) | 0.16 | 0.021 | 0 |
| 4 | stall | 1252 | 57 | 5/13 | 1.00 | 0 | 0 | 12/12 | – (max 17°) | 1 | 60 | 2 | 0.92 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 5 | fell | 1342 | 62 | 1/8 | 1.00 | 0 | 0 | 10/10 | 1341 s, 98° | 0 | 206 | 2 | 1.00 | 0 (0 / 0 / 0) | 0.17 | 0.039 | 0 |
| 6 | stall | 746 | 35 | 0/8 | – | 0 | 0 | 8/8 | – (max 5°) | 1 | 35 | 0 | 0.70 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 7 | fell | 370 | 17 | 0/13 | – | 0 | 0 | 6/6 | 369 s, 97° | 0 | 22 | 0 | 0.74 | 0 (0 / 0 / 0) | 0.28 | 0.042 | 0 |
| 8 | stall | 1466 | 66 | 9/11 | 1.00 | 0 | 0 | 13/13 | – (max 4°) | 1 | 157 | 3 | 1.00 | 0 (0 / 0 / 0) | – | 0.000 | 0 |
| 9 | fell | 1384 | 59 | 3/8 | 1.00 | 0 | 0 | 8/8 | 1383 s, 104° | 0 | 218 | 1 | 1.00 | 1 (1 / 0 / 0) | 0.04 | 0.092 | 0 |
