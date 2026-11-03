# Limitations

What this project does **not** show, stated up front. This list grows as the project progresses.

- **Simulation only.** No real hardware; nothing here has been tested on a physical robot.
- **Sand is approximated.** The beach is a rough heightmap with friction zones (dry and wet sand). Real granular sand, which sinks and deforms under the feet, is not simulated.
- **Rigid trash only.** Bottles, cans and cups don't crush or bend.
- **Material classification only shows performance on simulated textures.** Real recycling plants tell clear PET from glass with near-infrared sensors, not ordinary cameras.
- **Simulated grasping.** In Gazebo, grasps may use an "attach on contact" shortcut instead of real finger contact. Results are marked when they do.
- **The beach is a few simple layers.** Dry sand is one heightmap with gentle dunes and centimetre-scale roughness; the ripples are only a texture. Wet sand is a separate flat surface with its own friction. Water is a flat visual plane with no physics: the robot is kept out of it by the map and the geofence, not by simulated water.
- **Half-buried items are a visual trick.** Only the item's visual mesh is sunk into the sand; its collision shape stays on the surface, so the gripper never has to dig.
- **Only trash that fits the gripper.** Every item lies at most 7.5 cm tall (the jaws are 8.5 cm long) and at most 13 cm wide across the jaws. Large items are left out: there is no 1.5 L bottle (9 cm tall lying down), and the jar and cup are small 4 oz sizes.
- **Rocks, shells, seaweed and driftwood are fixed in place.** They are static to keep the simulation fast, so the robot can't push or kick them; it treats them as obstacles or steps on them.
- **Items collide as boxes.** Bottles, cans and cups use box collision shapes (meshes are only for looks), so they don't roll. Real sand would hold them in place too.
- **The walking person has no collision.** Gazebo actors follow a scripted path and are not physical: the robot could walk through them. Safety relies on the person's ground-truth position (`/people`), not on contact.
- **The wide gripper is a custom design.** The stock WidowX 250 gripper opens only 5.2 cm, too narrow for most beach trash. The `custom` robot variant uses long jaws on a wider slide (2–15 cm opening, 0.34 kg). No catalogue gripper matches this: commercial grippers with a similar stroke (e.g. Robotiq 2F-140, about 1 kg) would break the robot's mass budget.
- **The arm is pushed past its rating.** With the wide gripper and a 0.2 kg glass item, the WidowX-class arm carries about 0.4 kg more than its stock gripper alone, above the vendor's 0.25 kg working-payload rating. In simulation the joint torques stay within a third of the joint limits, but a real arm of this class would be at or past its rated payload. A stronger arm (as used by IIT's BinWalker) would need a larger robot than the Go2.
- **The basket is oversized for the robot.** The `custom` variant's 34 × 30 cm basket overhangs the Go2's body by about 10 cm on each side and 11 cm at the back, on a 4.5 cm rack so the legs don't hit it.
- **The walking controller is blind and falls now and then.** It follows a fixed gait and doesn't correct its balance from what it senses, so small steps, collisions and some turns can tip it over. It has also fallen while standing still. A fall ends the run (the legs go limp, there is no stand-up). A learned policy is being tested as a replacement.
- **Gentle dunes only.** The legged controller rolled over on 7° sand slopes with the arm and a 2 kg load in the basket, so the cleaning areas are kept under 6° and walking commands are smoothed. Steeper dunes are scenery behind the fence.
- **GNSS noise is idealised.** The simulated GPS error is independent from one fix to the next (white noise, about 2 m RMS). Real GPS error drifts slowly and is correlated in time, so real localization would be harder.
- **The walking person is scripted.** It follows a fixed path at a fixed speed and does not react to the robot.
