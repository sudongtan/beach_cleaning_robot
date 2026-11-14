# sim/isaac: the beach scenes in USD (D0a)

`load_scene.py` builds one OpenUSD stage per seed of `bench/scenes.yaml`: the same scene as the
Gazebo world (`sim/gazebo/load_scene.py`), with no robot and no person. It's plain OpenUSD
(UsdGeom, UsdPreviewSurface, UsdLux, UsdPhysics), so it builds and is tested on any machine with
`usd-core`. Only `render_check.py` and `tools/gpu_session.sh` need Isaac Sim (`# REQUIRES_GPU`).

| File | What |
|---|---|
| `load_scene.py` | `python sim/isaac/load_scene.py --seed N [--out PATH]` → `scenes/generated/scene_N.usda` (git-ignored) |
| `render.yaml` | light intensities, camera views, build limits, render-check thresholds |
| `render_checks.py` | pure helpers for the render check (images, trash motion, Kit log scan, verdict) |
| `render_check.py` | GPU: captures every scene camera, 2 s physics rest check → `checks.json`, `timing.json` |
| `tools/gpu_session.sh` | GPU: the whole session, restartable, ends in a tarball |
| `tests/` | `pytest sim/isaac -q` (Mac / laptop, no GPU) |

## Set up (Mac or laptop, no ROS)

```bash
uv venv --python 3.12 ~/venvs/beach_isaac
source ~/venvs/beach_isaac/bin/activate
uv pip install -r sim/isaac/requirements-mac.txt
pytest sim/isaac -q
for s in 0 1 2 3 4 5 6 7 8 9 101; do python sim/isaac/load_scene.py --seed $s; done
```

## Stage layout

```
/World                      defaultPrim; metersPerUnit 1, kilogramsPerUnit 1, upAxis Z; map frame
  PhysicsScene              gravity -z 9.81, default solver
  Looks/                    UsdPreviewSurface materials (textures relative to the .usda)
  PhysicsMaterials/         dry_sand, wet_sand (friction from scenes.yaml)
  Environment/terrain       mesh from terrain.height_grid, exact triangle collider, dry sand
  Environment/wet_zone      box collider (wet sand) + textured visual
  Environment/water         visual only
  Bins/ Landmarks/ Distractors/   static colliders (primitives, purpose guide)
  Trash/                    rigid bodies, MassAPI, primitive colliders
  Lights/sun, Lights/sky    DistantLight + DomeLight
  Cameras/top, oblique, robot_eye
```

Every object prim has `customData` `beach:class` and `beach:class_id` (from `objects.yaml`).

## GPU session (vast.ai), see `notes/briefs/D0_prep.md` §6–§8

Never rent without the user's go-ahead. Before session 1 (the user): cap and billing alert set,
credit topped up to the session budget **plus ~$1** (at a zero balance vast.ai stops the instance,
and with no card on file destroys it with its data), a row started in `budget.md`, NGC key ready.
On the Mac: `uv pip install vastai`, `vastai set api-key KEY`, and register the ssh key **before**
any create: `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"`.

1. **Search** (fields from vast.ai's `search offers` reference). `-o 'dph'` sorts ascending.
   `driver_version` is a string field: **read the driver column** of each result rather than trusting `>=`.
   ```bash
   vastai search offers 'gpu_name=RTX_4090 num_gpus=1 gpu_ram>=20 cpu_ram>=32 disk_space>=100 reliability>0.98 inet_down>=500 verified=true rentable=true driver_version>=580.95.05' -o 'dph'
   vastai search offers 'gpu_name=L40S num_gpus=1 cpu_ram>=32 disk_space>=100 reliability>0.98 inet_down>=500 verified=true rentable=true driver_version>=580.95.05' -o 'dph'
   ```
   Pick the cheapest with the highest driver (595.58.03 is NVIDIA's listed driver for 6.2).
2. **Create** (the user approves this exact command first). Storage bills from creation, GPU from `running`.
   `--ssh` replaces the image's entrypoint, which Isaac doesn't need.
   ```bash
   vastai create instance OFFER_ID --image nvcr.io/nvidia/isaac-sim:6.2.0 --disk 100 --ssh --direct \
     --env '-e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all'
   vastai show instance INSTANCE_ID      # wait for running; exited / offline: destroy, next host
   vastai ssh-url INSTANCE_ID            # -> ssh://root@HOST:PORT
   ```
   `ACCEPT_EULA` / `PRIVACY_CONSENT`: only after the user has read and accepted NVIDIA's terms.
   Try without a registry login first. If the pull fails with an auth error, destroy and create
   again with the NGC key, kept out of shell history (`\$oauthtoken` is NGC's literal user name):
   ```bash
   read -s NGC_API_KEY
   vastai create instance OFFER_ID ...same as above... --login "-u \$oauthtoken -p $NGC_API_KEY nvcr.io"
   ```
   If ssh never comes up on the Isaac image, it's not the host: stop and ask, don't try a second host.
3. **Upload** the branch's `sim/` with the generated scenes (Mac, repo root). `/workspace` is
   only a folder name, so create it:
   ```bash
   ssh -p PORT root@HOST 'mkdir -p /workspace/beach'
   rsync -az -e "ssh -p PORT" --exclude '__pycache__' --exclude 'generated/scene_*/' sim root@HOST:/workspace/beach/
   ```
4. **Run** inside the instance, under `tmux` so a dropped ssh doesn't stop it:
   ```bash
   cd /workspace/beach && bash sim/isaac/tools/gpu_session.sh /workspace/d0_s1 0          # session 1
   cd /workspace/beach && bash sim/isaac/tools/gpu_session.sh /workspace/d0_s2 0 1 2 3 4 5 6 7 8 9 101   # session 2
   ```
   **Download after each step that writes files** (not only at the end), from the Mac:
   `rsync -az -e "ssh -p PORT" root@HOST:/workspace/d0_s1/ results/d0_prep/d0_s1/`.
   Session 2 compiles shaders again (the cache dies with session 1's instance).
5. **Download, then destroy** (not stop: a stopped instance still bills its disk):
   ```bash
   rsync -az -e "ssh -p PORT" root@HOST:/workspace/d0_s1.tar.gz root@HOST:/workspace/d0_s1/ results/d0_prep/
   vastai destroy instance INSTANCE_ID
   ```
   Then fill in `budget.md`.

## D0b / D0c results (2026-11-14)

Custom variant, one RTX 4090 instance, $0.93: walks on flat ground (0.179 m/s for 0.2, 0.314 rad/s for 0.3; Gazebo 0.216 / 0.290) and on seed 3 (0.156 / 0.316; Gazebo 0.210 / 0.302), no falls; 9/9 picks stored in the right compartment, max tilt 4°. Per-run results in `results/d0/final/`, videos in `results/d0/videos/`.

**Known issue: the pick video.** In `results/d0/videos/d0_pick_bottle.mp4` the carried bottle isn't rendered moving with the gripper (it stays on the sand while the arm lifts). The physics run itself stored the item (its end state is on the compartment floor), but the held item is a kinematic body teleported with the tensor API while physics is stepped manually, and the renderer doesn't see those poses. Fix before re-recording: author the held item's pose in USD (or try `SimulationManager.step(..., update_fabric=True)`, or attach with a runtime fixed joint), render a few frames around the lift and look at them, then capture. Note also that the carry's "lifted" height in `result.json` reads back the pose the code wrote; only the end state is independent evidence.

## D0b / D0c: the robot walks and picks (one GPU instance)

Brief: `notes/briefs/D0.md`. The laptop's files (robot URDF + meshes, the robot_lab policy, the Gazebo
reference walks, the pick plans) are in `~/d0_from_laptop` (private repo branch `data`,
`notes/briefs/D0_laptop.md`); they never go into this repo (third-party meshes).

| File | What |
|---|---|
| `robot.yaml` | every D0b/D0c setting: import options, joints, stand, walk sequence, pick timing, pass thresholds |
| `walk.py` | the policy as Gazebo runs it: numpy network, StateRL.cpp's observation / action convention, the adapter's command shaping |
| `robot_model.py` | URDF facts: fixed transforms, basket compartments (= `grasping.arm`), drive gains |
| `pick.py` | the planner's pick + store steps as a timed schedule, and the per-case checks |
| `d0_runner.py` | the stand / walk / pick loops against any simulator (`io`), and the walk verdict |
| `check_robot_usd.py` | imported USD vs URDF; writes the joint drives (per-degree angular gains) |
| `compose.py` | test stage: a scene + the robot at `robot_start` + pick items + view cameras, 200 Hz physics |
| `import_robot.py`, `run_d0.py` | `# REQUIRES_GPU`: Isaac's URDF importer; the Isaac side of the loops |
| `tools/gpu_session_d0.sh` | `# REQUIRES_GPU`: the whole D0b/D0c session on one instance, restartable |

Mac: `uv pip install -r sim/isaac/requirements-mac.txt` (adds Pinocchio for the planner checks), then
`pytest sim/isaac -q` (the D0 tests skip without `~/d0_from_laptop` or `$BEACH_D0_LAPTOP_DIR`).

**Session** (one instance; same host, image and create command as above):
```bash
# from the Mac, repo root: the code (no git history, no private notes, no results) and the laptop files
ssh -p PORT root@HOST 'mkdir -p /workspace/beach'
rsync -az -e "ssh -p PORT" --exclude .git --exclude .git-private --exclude notes --exclude CLAUDE.md \
  --exclude results --exclude __pycache__ --exclude 'generated/' ./ root@HOST:/workspace/beach/
rsync -az -e "ssh -p PORT" --exclude 'c7/*/bag' --exclude gazebo_ref ~/d0_from_laptop root@HOST:/workspace/
# on the host, in tmux
cd /workspace/beach && BEACH_D0_LAPTOP_DIR=/workspace/d0_from_laptop bash sim/isaac/tools/gpu_session_d0.sh /workspace/d0
# from the Mac, after each step that writes files and before destroying
rsync -az -e "ssh -p PORT" root@HOST:/workspace/d0/ results/d0/
```
Steps: host checks → laptop files' md5s → URDF import → USD check + drives → compose 3 stages → walk on
flat ground (if it falls with the API's angular velocity, again with the finite-difference gyro) →
walk on seed 3 → 9 picks on flat ground → frame sequences (not judged) → summary + tarball.
Each step's verdict is the `ok` in its JSON; `summary.txt` lists them.
