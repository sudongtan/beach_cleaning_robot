"""Ground-truth detector baseline (perception:=ground_truth).

  /sim/ground_truth/objects (map frame) -> /detections (camera_link frame)

Publishes what the camera could see: objects whose centre is inside the field of view
and depth range (config/sensors.yaml), with poses relative to camera_link from the true
relative pose, class trash / distractor, confidence 1.0, no material. The world model
puts them into the map with the estimated pose, so localization errors show up as they
would with a real detector (notes/briefs/A2.md). Detection ids are the scene ids
(ground truth only; a learned detector leaves them empty).

The true relative pose comes from the true robot pose (/sim/ground_truth/pose) and the
robot's own TF (perception.true_pose), not from TF map -> camera_link: with an estimated
localization that TF is off by the estimate's error, which the world model would cancel
again (B4). The two bins are reported too, as classes bin_recyclable / bin_residual (B4: the
task's bin approach uses them), from bench/scenes.yaml and the terrain under them (static).
"""

import math
import sys
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from perception import config
from perception.logic import BIN_CLASSES, Fov, object_class, scene_bins
from perception.true_pose import TruePose


class GroundTruthDetector(Node):

    def __init__(self, **kwargs):
        super().__init__('perception', **kwargs)
        cfg = config.load(self.declare_parameter('repo', '').value)
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_link').value
        self.fov = Fov(cfg['camera'], cfg['perception']['ground_truth_detector']['fov_margin_rad'])
        self.materials = cfg['materials']
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.truth = TruePose(self, self.tf_buffer)
        self.bins = self.load_bins(self.get_parameter('repo').value, self.declare_parameter('seed', 0).value)
        self.pub = self.create_publisher(Detection3DArray, '/detections', 10)
        self.create_subscription(Detection3DArray, '/sim/ground_truth/objects', self.on_objects, 10)

    @staticmethod
    def load_bins(repo, seed):
        """The scene's bins as ground-truth detections in the map (they don't move)."""
        repo = Path(repo)
        sys.path.insert(0, str(repo / 'sim' / 'common'))
        from terrain import ground_z
        scene = next(s for s in yaml.safe_load(open(repo / 'bench' / 'scenes.yaml'))['scenes'] if s['seed'] == seed)
        objects = yaml.safe_load(open(repo / 'sim' / 'common' / 'objects.yaml'))
        radius = yaml.safe_load(open(repo / 'config' / 'task.yaml'))['task']['fsm']['bin_ground_radius_m']
        out = []
        for cls, (x, y, z), yaw, size in scene_bins(scene, objects,
                                                    lambda bx, by: ground_z(scene['terrain'], bx, by, radius, 'min'),
                                                    radius):
            d = Detection3D()
            d.id = cls
            d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = x, y, z
            d.bbox.center.orientation.z, d.bbox.center.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = size
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls
            d.results.append(hyp)
            out.append(d)
        return out

    def on_objects(self, msg):
        # camera pose at the time of the ground truth, not the latest one: at RTF ~0.25 the latest
        # pose can be far off while the robot turns (A2b run 3: phantom items)
        try:
            tf = self.truth.lookup(self.camera_frame, msg.header.stamp)
        except TransformException as e:
            self.get_logger().warn(f'no camera pose at the message time, skipped: {e}', throttle_duration_sec=5.0)
            return
        out = Detection3DArray()
        out.header.stamp, out.header.frame_id = msg.header.stamp, self.camera_frame
        for d in list(msg.detections) + self.bins:
            pose = do_transform_pose(d.bbox.center, tf)
            p = pose.position
            if not self.fov.sees(p.x, p.y, p.z):
                continue
            det = Detection3D()
            det.header = out.header
            det.id = d.id
            det.bbox.center = pose
            det.bbox.size = d.bbox.size
            hyp = ObjectHypothesisWithPose()
            cls = d.results[0].hypothesis.class_id
            hyp.hypothesis.class_id = cls if cls in BIN_CLASSES else object_class(cls, self.materials)
            hyp.hypothesis.score = 1.0
            hyp.pose.pose = pose
            det.results.append(hyp)
            out.detections.append(det)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = GroundTruthDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
