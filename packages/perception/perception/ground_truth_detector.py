"""Ground-truth detector baseline (perception:=ground_truth).

  /sim/ground_truth/objects (map frame) -> /detections (camera_link frame)

Publishes what the camera could see: objects whose centre is inside the field of view
and depth range (config/sensors.yaml), with poses relative to camera_link from the true
relative pose, class trash / distractor, confidence 1.0, no material. The world model
puts them into the map with the estimated pose, so localization errors show up as they
would with a real detector (notes/briefs/A2.md). Detection ids are the scene ids
(ground truth only; a learned detector leaves them empty).
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from perception import config
from perception.logic import Fov, object_class


class GroundTruthDetector(Node):

    def __init__(self, **kwargs):
        super().__init__('perception', **kwargs)
        cfg = config.load(self.declare_parameter('repo', '').value)
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_link').value
        self.fov = Fov(cfg['camera'], cfg['perception']['ground_truth_detector']['fov_margin_rad'])
        self.materials = cfg['materials']
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(Detection3DArray, '/detections', 10)
        self.create_subscription(Detection3DArray, '/sim/ground_truth/objects', self.on_objects, 10)

    def on_objects(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform(self.camera_frame, msg.header.frame_id, Time())
        except TransformException as e:
            self.get_logger().warn(f'no camera pose, nothing published: {e}', throttle_duration_sec=2.0)
            return
        out = Detection3DArray()
        out.header.stamp, out.header.frame_id = msg.header.stamp, self.camera_frame
        for d in msg.detections:
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
            hyp.hypothesis.class_id = object_class(d.results[0].hypothesis.class_id, self.materials)
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
