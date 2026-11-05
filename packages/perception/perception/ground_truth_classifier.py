"""Ground-truth classifier baseline (classifier:=ground_truth): serves /classify.

Ignores the image. The held item is the /sim/ground_truth/objects entry nearest the
gripper TCP (arm_ee_gripper_link: the true robot pose + the robot's own TF, perception.true_pose,
not the estimated TF map -> base_link: B4) within match_radius_m (config/perception.yaml);
its true material gives confidence 1.0 and the bin from config/sorting_rules.yaml.
Nothing near the gripper -> no material, confidence 0 -> fallback bin (residual).
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from beach_interfaces.srv import ClassifyItem
from perception import config
from perception.logic import nearest, sort
from perception.true_pose import TruePose


class GroundTruthClassifier(Node):

    def __init__(self, **kwargs):
        super().__init__('classifier', **kwargs)
        cfg = config.load(self.declare_parameter('repo', '').value)
        self.tcp_frame = self.declare_parameter('tcp_frame', 'arm_ee_gripper_link').value
        self.radius = cfg['perception']['ground_truth_classifier']['match_radius_m']
        self.sorting, self.materials = cfg['sorting'], cfg['materials']
        self.objects = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.truth = TruePose(self, self.tf_buffer)
        self.create_subscription(Detection3DArray, '/sim/ground_truth/objects', self.on_objects, 10)
        self.create_service(ClassifyItem, '/classify', self.on_classify)

    def on_objects(self, msg):
        self.objects = msg

    def held_item(self):
        if self.objects is None:
            return None
        try:
            tcp = tuple(self.truth.map_from(self.tcp_frame)[:3, 3])
        except TransformException as e:
            self.get_logger().warn(f'no gripper pose: {e}')
            return None
        objs = [(d.id, d.results[0].hypothesis.class_id,
                 (d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z))
                for d in self.objects.detections]
        return nearest(objs, tcp, self.radius)

    def on_classify(self, request, response):
        held = self.held_item()
        material, confidence = (held[1], 1.0) if held else ('', 0.0)
        response.material, response.recyclable, response.bin_id = sort(material, confidence, self.sorting,
                                                                         self.materials)
        response.confidence = confidence
        self.get_logger().info(f'classify: {held[0] if held else "nothing near the gripper"} -> '
                               f'{response.material or "unknown"} -> {response.bin_id or "no bin"}')
        return response


def main():
    rclpy.init()
    node = GroundTruthClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
