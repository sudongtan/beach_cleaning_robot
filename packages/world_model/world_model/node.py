"""World model node (notes/architecture.md, /world_model).

  /detections (camera frame) --TF--> map; de-duplicated into items (world_model.model)
  camera pose (TF map -> camera_link) -> coverage grid of the geofence
  /world_model (beach_interfaces/WorldModel, 2 Hz), /world_model/update_item (UpdateItem)

Detections within ignore_near_gripper_m of the gripper TCP are dropped: that is the item
held up to the camera. The TCP is taken in the camera frame at the message time, else the
latest one (the camera is fixed to the body, so only arm motion in the lag matters). Item masses for the basket load come from bench/scenes.yaml +
sim/common/objects.yaml when the detection carries a scene id (ground truth), else the
configured default. Parameters: repo, seed. Thresholds: config/world_model.yaml.
"""

import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from beach_interfaces.msg import TrashItem
from beach_interfaces.msg import WorldModel as WorldModelMsg
from beach_interfaces.srv import UpdateItem
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from mapping.grid import inside_polygon
from perception.logic import Fov
from world_model.model import COMPARTMENTS, Coverage, WorldModel


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class WorldModelNode(Node):

    def __init__(self, **kwargs):
        super().__init__('world_model', **kwargs)
        repo = self.declare_parameter('repo', '').value
        if not repo:
            raise RuntimeError('parameter repo (repository root) is required')
        repo = Path(repo)
        seed = self.declare_parameter('seed', 0).value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_link').value
        self.tcp_frame = self.declare_parameter('tcp_frame', 'arm_ee_gripper_link').value
        cfg = yaml.safe_load(open(repo / 'config' / 'world_model.yaml'))['world_model']
        limits = yaml.safe_load(open(repo / 'config' / 'robot_limits.yaml'))['robot_limits']
        objects = yaml.safe_load(open(repo / 'sim' / 'common' / 'objects.yaml'))
        scene = next(s for s in yaml.safe_load(open(repo / 'bench' / 'scenes.yaml'))['scenes'] if s['seed'] == seed)
        masses = {t['id']: objects['trash'][t['model']]['mass'] for t in scene['trash']}
        masses.update({d['id']: objects['distractors'][d['model']]['mass'] for d in scene['distractors']})
        self.wm = WorldModel(cfg['dedup_radius_m'], cfg['max_items_per_compartment'],
                             limits['basket_load_max_kg'], cfg['default_item_mass_kg'], masses)
        self.ignore_near_gripper = cfg['ignore_near_gripper_m']

        sys.path.insert(0, str(repo / 'sim' / 'common'))
        import terrain as terrain_mod
        w, h = scene['area']['size']
        res = cfg['coverage_resolution_m']
        xs = (np.arange(int(round(w / res))) + 0.5) * res
        ys = (np.arange(int(round(h / res))) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        poly = scene['geofence']['polygon']
        self.coverage = Coverage(w, h, res, inside_polygon(X, Y, poly), terrain_mod.height_at(scene['terrain'], X, Y))
        self.fence = None
        if cfg['ignore_outside_geofence']:
            self.fence = lambda x, y: bool(inside_polygon(np.array([[x]]), np.array([[y]]), poly)[0, 0])
        self.fov = Fov(yaml.safe_load(open(repo / 'config' / 'sensors.yaml'))['camera'])

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(WorldModelMsg, '/world_model', 10)
        self.create_subscription(Detection3DArray, '/detections', self.on_detections, 10)
        self.create_service(UpdateItem, '/world_model/update_item', self.on_update)
        self.create_timer(0.5, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def lookup(self, target, source, stamp=None, warn=True):
        """Transform at `stamp` (a builtin Time), or the latest one; None if not available."""
        try:
            t = Time() if stamp is None else Time.from_msg(stamp)
            return self.tf_buffer.lookup_transform(target, source, t).transform
        except TransformException as e:
            if warn:
                self.get_logger().warn(f'no transform {target} <- {source}: {e}', throttle_duration_sec=2.0)
            return None

    def on_detections(self, msg):
        if not msg.detections:
            return
        # the robot's pose when the camera saw them, not the latest (A2b run 3: phantom items)
        to_map = self.lookup(self.map_frame, msg.header.frame_id, msg.header.stamp)
        if to_map is None:
            return
        # the arm's TFs often lag the camera's: without a fallback the held item became a new item
        # (A2b: phantoms next to every picked item, then picks of the item already in the basket)
        tcp = (self.lookup(msg.header.frame_id, self.tcp_frame, msg.header.stamp, warn=False)
               or self.lookup(msg.header.frame_id, self.tcp_frame))
        tf = TransformStamped()
        tf.transform = to_map
        for d in msg.detections:
            c = d.bbox.center.position
            if tcp is not None and math.dist((c.x, c.y, c.z), (tcp.translation.x, tcp.translation.y,
                                                               tcp.translation.z)) < self.ignore_near_gripper:
                continue
            pose = do_transform_pose(d.bbox.center, tf)
            p, q = pose.position, pose.orientation
            if self.fence is not None and not self.fence(p.x, p.y):
                continue                                  # outside the geofence: not ours to collect
            hyp = d.results[0].hypothesis
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            size = (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z)
            self.wm.add_detection(self.now(), hyp.class_id, p.x, p.y, p.z, hyp.score, d.id, yaw, size)

    def on_update(self, req, res):
        res.success, res.message = self.wm.update(req.id, req.status, req.material, req.bin_id, req.skip_reason)
        if not res.success:
            self.get_logger().warn(f'update rejected: {res.message}')
        self.publish()
        return res

    def tick(self):
        cam = self.lookup(self.map_frame, self.camera_frame)
        if cam is not None:
            t = cam.translation
            self.coverage.mark(rotation(cam.rotation), (t.x, t.y, t.z), self.fov)
        self.publish()

    def publish(self):
        msg = WorldModelMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        for it in self.wm.items.values():
            m = TrashItem()
            m.id, m.scene_id, m.object_class, m.material, m.bin_id = (it.id, it.scene_id, it.object_class,
                                                                      it.material, it.bin_id)
            m.pose.header = msg.header
            m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = it.x, it.y, it.z
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(it.yaw / 2), math.cos(it.yaw / 2)
            m.size.x, m.size.y, m.size.z = (float(v) for v in it.size)
            m.confidence, m.status, m.skip_reason, m.times_seen = (float(it.confidence), it.status,
                                                                   it.skip_reason, it.times_seen)
            m.last_seen.sec = int(it.last_seen)
            m.last_seen.nanosec = int((it.last_seen % 1) * 1e9)
            msg.items.append(m)
        cov = self.coverage
        msg.coverage.header = msg.header
        msg.coverage.info.resolution = float(cov.res)
        msg.coverage.info.height, msg.coverage.info.width = cov.grid.shape
        msg.coverage.info.origin.orientation.w = 1.0
        msg.coverage.data = cov.grid.flatten().tolist()
        msg.compartment_recyclable = list(self.wm.compartments[COMPARTMENTS[0]])
        msg.compartment_residual = list(self.wm.compartments[COMPARTMENTS[1]])
        msg.basket_load_kg = float(self.wm.load_kg())
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = WorldModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
