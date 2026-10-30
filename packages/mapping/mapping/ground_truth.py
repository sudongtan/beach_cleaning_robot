"""Ground-truth mapping baseline (mapping:=ground_truth).

Publishes, latched, for scene `seed` of bench/scenes.yaml:
  /map       nav_msgs/OccupancyGrid (map frame): terrain traversability (water, slopes over
             config/terrain.yaml max_slope_deg) + static obstacles + outside the geofence
  /geofence  geometry_msgs/PolygonStamped (map frame): the scene's cleaning area

Reads only repo files (bench/scenes.yaml, sim/common/objects.yaml + terrain.py, config/),
so it is the same for every simulator. Parameters: repo (repository root), seed.
"""

import sys
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point32, PolygonStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from mapping.grid import build

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def load(repo, seed):
    repo = Path(repo)
    sys.path.insert(0, str(repo / 'sim' / 'common'))
    import terrain as terrain_mod
    scene = next((s for s in yaml.safe_load(open(repo / 'bench' / 'scenes.yaml'))['scenes'] if s['seed'] == seed),
                 None)
    if scene is None:
        raise RuntimeError(f'seed {seed} not in bench/scenes.yaml')
    objects = yaml.safe_load(open(repo / 'sim' / 'common' / 'objects.yaml'))
    trav = yaml.safe_load(open(repo / 'config' / 'terrain.yaml'))['traversability']
    cfg = yaml.safe_load(open(repo / 'config' / 'mapping.yaml'))['mapping']
    res = trav['grid_resolution_m']
    t = terrain_mod.traversability(scene, trav['max_slope_deg'], res)
    occ = build(scene, objects, t['occupancy'], res, cfg['obstacle_models'], cfg['obstacle_margin_m'],
                cfg['lethal_outside_geofence'])
    return scene, occ, res


class GroundTruthMapping(Node):

    def __init__(self, **kwargs):
        super().__init__('mapping', **kwargs)
        repo = self.declare_parameter('repo', '').value
        seed = self.declare_parameter('seed', 0).value
        frame = self.declare_parameter('map_frame', 'map').value
        if not repo:
            raise RuntimeError('parameter repo (repository root) is required')
        scene, occ, res = load(repo, seed)
        stamp = self.get_clock().now().to_msg()

        grid = OccupancyGrid()
        grid.header.stamp, grid.header.frame_id = stamp, frame
        grid.info.map_load_time = stamp
        grid.info.resolution = float(res)
        grid.info.height, grid.info.width = occ.shape
        grid.info.origin.orientation.w = 1.0             # cell (0, 0) at map (0, 0), row 0 = south
        grid.data = occ.flatten().tolist()
        fence = PolygonStamped()
        fence.header.stamp, fence.header.frame_id = stamp, frame
        fence.polygon.points = [Point32(x=float(x), y=float(y)) for x, y in scene['geofence']['polygon']]

        self.create_publisher(OccupancyGrid, '/map', LATCHED).publish(grid)
        self.create_publisher(PolygonStamped, '/geofence', LATCHED).publish(fence)
        lethal = sum(v == 100 for v in grid.data) / len(grid.data)
        self.get_logger().info(f'scene {seed}: /map {occ.shape[1]} x {occ.shape[0]} cells at {res} m '
                               f'({lethal:.0%} lethal), /geofence {len(fence.polygon.points)} points')


def main():
    rclpy.init()
    node = GroundTruthMapping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
