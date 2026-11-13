# sim/isaac: the beach scenes in USD (D0 prep)

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

Never rent without the user's go-ahead. Install the CLI on the Mac (`uv pip install vastai`), then
`vastai set api-key KEY`.

1. **Search** (fields from vast.ai's `search offers` reference; `driver_version` is a string, so
   check the listed drivers by eye):
   ```bash
   vastai search offers 'gpu_name=RTX_4090 num_gpus=1 gpu_ram>=20 cpu_ram>=32 disk_space>=100 reliability>0.98 inet_down>=500 verified=true rentable=true driver_version>=580.95.05' -t on-demand -o 'dph'
   vastai search offers 'gpu_name=L40S num_gpus=1 cpu_ram>=32 disk_space>=100 reliability>0.98 inet_down>=500 verified=true rentable=true driver_version>=580.95.05' -t on-demand -o 'dph'
   ```
   Pick the cheapest with the highest driver (595.58.03 is NVIDIA's listed driver for 6.2).
2. **Create** (the user approves this exact command first). Storage bills from creation, GPU from `running`:
   ```bash
   vastai create instance OFFER_ID --image nvcr.io/nvidia/isaac-sim:6.2.0 --disk 100 --ssh --direct \
     --env '-e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all'
   vastai show instance INSTANCE_ID      # wait for status running (exited / offline: destroy, next host)
   vastai ssh-url INSTANCE_ID
   ```
   `ACCEPT_EULA` / `PRIVACY_CONSENT`: set only after the user has read and accepted NVIDIA's terms.
   If the pull fails on authentication, add `--login '-u $oauthtoken -p NGC_KEY nvcr.io'` (NGC's
   documented docker login; vast.ai documents `--login` only generically).
3. **Upload** the branch's `sim/` with the generated scenes (from the Mac, repo root):
   ```bash
   rsync -az -e "ssh -p PORT" --exclude '__pycache__' --exclude 'generated/scene_*/' sim root@HOST:/workspace/beach/
   ```
4. **Run** (inside the instance, under `tmux` so a dropped ssh doesn't stop it):
   ```bash
   cd /workspace/beach && bash sim/isaac/tools/gpu_session.sh /workspace/d0_s1 0          # session 1
   cd /workspace/beach && bash sim/isaac/tools/gpu_session.sh /workspace/d0_s2 0 1 2 3 4 5 6 7 8 9 101   # session 2
   ```
5. **Download, then destroy** (not stop: a stopped instance still bills its disk):
   ```bash
   scp -P PORT root@HOST:/workspace/d0_s1.tar.gz results/d0_prep/
   vastai destroy instance INSTANCE_ID
   ```
   Then fill in `budget.md`.
