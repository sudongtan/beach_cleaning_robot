"""Shared fixtures for the Isaac scene tests (run on any machine with usd-core: pytest sim/isaac -q).

Each scene is built once per session into a temporary folder; the Gazebo SDF of the same
seed is the reference (sim/gazebo/load_scene.build_world, imported here only, never by the
Isaac builder).
"""

import importlib.util
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
ISAAC = HERE.parent
REPO = ISAAC.parents[1]
sys.path.insert(0, str(ISAAC))
sys.path.insert(0, str(REPO / 'sim' / 'common'))

import load_scene as isaac_scene  # noqa: E402  (sim/isaac/load_scene.py)


def _load_gazebo_builder():
    spec = importlib.util.spec_from_file_location('gazebo_load_scene', REPO / 'sim' / 'gazebo' / 'load_scene.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gazebo_scene = _load_gazebo_builder()

with open(REPO / 'bench' / 'scenes.yaml') as _f:
    ALL_SEEDS = [s['seed'] for s in yaml.safe_load(_f)['scenes']]


@pytest.fixture(scope='session')
def configs():
    return isaac_scene.load_configs()     # objects, render, sensors


@pytest.fixture(scope='session')
def built(tmp_path_factory):
    """seed -> (usda path, build seconds), built on first use."""
    out_dir = tmp_path_factory.mktemp('usd')
    cache = {}

    def get(seed):
        if seed not in cache:
            cache[seed] = isaac_scene.write_scene(seed, out_dir / f'scene_{seed}.usda')
        return cache[seed]
    return get


@pytest.fixture(scope='session')
def gazebo_sdf():
    """seed -> parsed Gazebo world (with the debug cameras)."""
    cache = {}

    def get(seed):
        if seed not in cache:
            scene = isaac_scene.export_terrain.load_scene(seed)
            objects, _, _ = isaac_scene.load_configs()
            xml = gazebo_scene.build_world(scene, objects, debug_camera=True)
            cache[seed] = ET.fromstring(xml.split('\n', 1)[1])
        return cache[seed]
    return get


@pytest.fixture(scope='session')
def scene_of():
    cache = {}

    def get(seed):
        if seed not in cache:
            cache[seed] = isaac_scene.export_terrain.load_scene(seed)
        return cache[seed]
    return get


def open_stage(path):
    from pxr import Usd
    t0 = time.monotonic()
    stage = Usd.Stage.Open(str(path))
    assert stage is not None, f'could not open {path}'
    return stage
