# Beach Cleanup Robot Dog: top-level tasks. Run from the repo root.
#   make generate                  bench/scenes.yaml: benchmark seeds 0-9 + dev scene 101 (15 x 12 m)
#   make test                      unit tests for sim/common
#   make scene SIM=gazebo SEED=3   scene 3 with the robot, sensors, bridge, ground truth (A1b);
#                                  VARIANT=custom (default) | stock robot
#                                  HEADLESS=true: server only; RVIZ=true: RViz with camera images;
#                                  needs `colcon build` first
#   make run SEED=3                scene 3 + the A2 skeleton (the robot cleans the beach);
#                                  HEADLESS / RVIZ as for scene; RUN_ID names the event log
#                                  LOCO=legubiao (default) | legubiao_rl (HIMLoco policy) |
#                                  legubiao_robot_lab (robot_lab policy) for
#                                  scene and run
#   make world SEED=3              Gazebo GUI with scene 3, no robot
#   make world-headless SEED=3     same, server only, headless rendering
#   make assets                    regenerate meshes and textures in sim/common
#   make bench SIM=gazebo          benchmark: seeds 0-9 headless, one after another (bench/runner.py);
#                                  SEEDS=0-9 | 0,3 | 101 (101 = dev scene: no row), ROW=baseline;
#                                  run JSONs in results/runs/, row in results/results.md
#   make bench-detached            the same, detached and with sleep / lid blocked (bench/start_detached.sh)

SHELL := /bin/bash
PYTHON ?= python3
SEEDS ?= all
SEED ?= 0
SIM ?= gazebo
HEADLESS ?= false
VARIANT ?= custom
RVIZ ?= false
RUN_ID ?=
LOCO ?= legubiao_robot_lab
ROW ?=
# Optional own Gazebo transport partition (e.g. GZ_PARTITION=a1) to keep parallel
# simulations apart; empty = the normal shared partition.
GZ_PARTITION ?=
GZ_ENV = $(if $(GZ_PARTITION),GZ_PARTITION=$(GZ_PARTITION))
ROS_SETUP ?= /opt/ros/jazzy/setup.bash
WS_SETUP ?= $(HOME)/ros2_ws/install/setup.bash
export WS_SETUP
# repository root for the A2 nodes (they read bench/, sim/common/ and config/ from it)
export BEACH_REPO := $(CURDIR)
WORLD = sim/gazebo/worlds/generated/scene_$(SEED).sdf
MANIFEST = $(CURDIR)/sim/gazebo/worlds/generated/scene_$(SEED).manifest.yaml

.PHONY: generate test assets terrain scene run world world-headless clean bench bench-detached

generate:
	$(PYTHON) sim/common/generate_scene.py --seeds $(SEEDS)

test:
	source $(ROS_SETUP) && source $(WS_SETUP) && $(PYTHON) -m pytest sim/common packages bench -q

assets:
	$(PYTHON) sim/common/make_assets.py

terrain:
	$(PYTHON) sim/common/export_terrain.py --seed $(SEED)

$(WORLD): FORCE
	$(PYTHON) sim/gazebo/load_scene.py --seed $(SEED) --variant $(VARIANT)

scene: $(WORLD)
	@[ "$(SIM)" = gazebo ] || { echo "SIM=$(SIM): only gazebo exists so far (isaac comes in B0)"; exit 1; }
	source $(ROS_SETUP) && source $(WS_SETUP) && $(GZ_ENV) ros2 launch bringup sim.launch.py sim:=$(SIM) \
	  scene:=$(MANIFEST) headless:=$(HEADLESS) rviz:=$(RVIZ) locomotion:=$(LOCO)

run: $(WORLD)
	@[ "$(SIM)" = gazebo ] || { echo "SIM=$(SIM): only gazebo exists so far (isaac comes in B0)"; exit 1; }
	source $(ROS_SETUP) && source $(WS_SETUP) && $(GZ_ENV) ros2 launch bringup sim.launch.py sim:=$(SIM) \
	  scene:=$(MANIFEST) headless:=$(HEADLESS) rviz:=$(RVIZ) locomotion:=$(LOCO) skeleton:=true run_id:=$(RUN_ID)

world: $(WORLD)
	source $(ROS_SETUP) && $(GZ_ENV) gz sim -r $(WORLD)

world-headless: $(WORLD)
	source $(ROS_SETUP) && $(GZ_ENV) gz sim -s -r --headless-rendering $(WORLD)

bench:
	$(PYTHON) bench/runner.py --sim $(SIM) --seeds $(SEEDS) $(if $(ROW),--row $(ROW))

bench-detached:
	bench/start_detached.sh SIM=$(SIM) SEEDS=$(SEEDS) $(if $(ROW),ROW=$(ROW))

clean:
	rm -rf sim/gazebo/worlds/generated sim/common/generated

FORCE:
