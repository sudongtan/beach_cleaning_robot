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
