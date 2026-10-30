# Beach Cleanup Robot Dog: top-level tasks. Run from the repo root.
#   make generate                  bench/scenes.yaml for seeds 0-9
#   make test                      unit tests for sim/common
#   make scene SIM=gazebo SEED=3   scene 3 with the robot, sensors, bridge, ground truth (A1b);
#                                  VARIANT=custom (default) | stock robot
#                                  HEADLESS=true: server only; RVIZ=true: RViz with camera images;
#                                  needs `colcon build` first
#   make world SEED=3              Gazebo GUI with scene 3, no robot
#   make world-headless SEED=3     same, server only, headless rendering
#   make assets                    regenerate meshes and textures in sim/common

SHELL := /bin/bash
PYTHON ?= python3
SEEDS ?= 0-9
SEED ?= 0
SIM ?= gazebo
HEADLESS ?= false
VARIANT ?= custom
RVIZ ?= false
# Optional own Gazebo transport partition (e.g. GZ_PARTITION=a1) to keep parallel
# simulations apart; empty = the normal shared partition.
GZ_PARTITION ?=
GZ_ENV = $(if $(GZ_PARTITION),GZ_PARTITION=$(GZ_PARTITION))
ROS_SETUP ?= /opt/ros/jazzy/setup.bash
WS_SETUP ?= $(HOME)/ros2_ws/install/setup.bash
WORLD = sim/gazebo/worlds/generated/scene_$(SEED).sdf
MANIFEST = $(CURDIR)/sim/gazebo/worlds/generated/scene_$(SEED).manifest.yaml

.PHONY: generate test assets terrain scene world world-headless clean

generate:
	$(PYTHON) sim/common/generate_scene.py --seeds $(SEEDS)

test:
	source $(ROS_SETUP) && source $(WS_SETUP) && $(PYTHON) -m pytest sim/common -q

assets:
	$(PYTHON) sim/common/make_assets.py

terrain:
	$(PYTHON) sim/common/export_terrain.py --seed $(SEED)

$(WORLD): FORCE
	$(PYTHON) sim/gazebo/load_scene.py --seed $(SEED) --variant $(VARIANT)

scene: $(WORLD)
	@[ "$(SIM)" = gazebo ] || { echo "SIM=$(SIM): only gazebo exists so far (isaac comes in B0)"; exit 1; }
	source $(ROS_SETUP) && source $(WS_SETUP) && $(GZ_ENV) ros2 launch bringup sim.launch.py sim:=$(SIM) \
	  scene:=$(MANIFEST) headless:=$(HEADLESS) rviz:=$(RVIZ)

world: $(WORLD)
	source $(ROS_SETUP) && $(GZ_ENV) gz sim -r $(WORLD)

world-headless: $(WORLD)
	source $(ROS_SETUP) && $(GZ_ENV) gz sim -s -r --headless-rendering $(WORLD)

clean:
	rm -rf sim/gazebo/worlds/generated sim/common/generated

FORCE:
