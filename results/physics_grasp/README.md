# Physics-grasp feasibility (A1c item 7)

Can the custom robot hold the A1 trash items by contact forces alone (`grasp_mode:=physics`)?
Tool: `sim/gazebo/tools/physics_grasp_test.py`, run with `physics_grasp_run.sh` (flat world,
custom robot, own Gazebo partition, loopback only). Per trial: IK for the jaw tips 5 mm above
the ground, item spawned with its narrow side across the jaws and a random offset (±1 cm,
±10°), open → pre-grasp 10 cm above → descend → close fully (5 N joint limit) → wait 0.5 s →
lift 20 cm → hold 5 s → wrist +90° / −90° / 0° → walk 1 m. Result from the item's true pose
relative to `arm_ee_gripper_link`: **held** (drift < 2 cm), **slipped** (more, still off the
ground), **dropped** (on the ground or > 15 cm away), **invalid** (the arm never got down: jaw tips above
the item's top after the descent; a rig fault, not counted). Jaw and item friction: Gazebo default (μ = 1).

**Status (2026-10-30): all 8 items measured** (`table.md`, generated from the JSON files by
`sim/gazebo/tools/physics_grasp_table.py`). Every item is carried in ≥ 6/10 valid trials; the
cigarette pack (7/9) and the food container (6/10) are the weak ones. Glass jar and poly cup were
resized (`resized.json`, `resized_cup.json`); the 1.5 L bottle was removed from the item set.
Earlier runs were cut short three times by the machine (suspend, a Wi-Fi drop that breaks
gz-transport discovery, and another simulation being started on the machine); trials from those
windows were discarded. Files: `baseline.json` (can, cigarette pack), `baseline_part2.json`
(the others, old jar/cup/bottle sizes), `part2_food_container_console.log` (4 food-container
trials before the interruption).
