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
