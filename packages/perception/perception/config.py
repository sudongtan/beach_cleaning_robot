"""Loads the repo config files the perception nodes need (parameter `repo`)."""

from pathlib import Path

import yaml


def load(repo):
    if not repo:
        raise RuntimeError('parameter repo (repository root) is required')
    repo = Path(repo)
    objects = yaml.safe_load(open(repo / 'sim' / 'common' / 'objects.yaml'))
    return {
        'camera': yaml.safe_load(open(repo / 'config' / 'sensors.yaml'))['camera'],
        'perception': yaml.safe_load(open(repo / 'config' / 'perception.yaml'))['perception'],
        'sorting': yaml.safe_load(open(repo / 'config' / 'sorting_rules.yaml'))['sorting'],
        'materials': {s['material'] for s in objects['trash'].values()},
    }
