"""Perception baseline logic as plain Python (no ROS).

Detector: which ground-truth objects the camera could see (centre inside the field of
view and depth range, in camera_link: x forward, y left, z up), and their class
(trash / distractor). Classifier: the ground-truth object held by the gripper, and the
bin for a material under config/sorting_rules.yaml.
"""

import math

import numpy as np

TRASH, DISTRACTOR = 'trash', 'distractor'


def vertical_fov(horizontal_fov, width, height):
    """Vertical field of view of a pinhole camera with square pixels."""
    return 2.0 * math.atan(math.tan(horizontal_fov / 2.0) * height / width)


class Fov:

    def __init__(self, camera, margin=0.0):
        """camera: the `camera` block of config/sensors.yaml."""
        self.h = camera['horizontal_fov_rad'] / 2.0 + margin
        self.v = vertical_fov(camera['horizontal_fov_rad'], camera['width'], camera['height']) / 2.0 + margin
        self.near, self.far = camera['depth_min_m'], camera['depth_max_m']

    def sees(self, x, y, z):
        """Is a point in camera_link (x forward) inside the view? Depth is measured along x."""
        if not self.near <= x <= self.far:
            return False
        return abs(math.atan2(y, x)) <= self.h and abs(math.atan2(z, x)) <= self.v

    def sees_many(self, x, y, z):
        """Vectorised sees() over numpy arrays."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        return ((x >= self.near) & (x <= self.far) & (np.abs(np.arctan2(y, x)) <= self.h)
                & (np.abs(np.arctan2(z, x)) <= self.v))


def object_class(gt_class, trash_materials):
    """Ground truth labels trash by material and distractors by type."""
    return TRASH if gt_class in trash_materials else DISTRACTOR


def nearest(objects, point, radius):
    """objects: [(id, class, (x, y, z))]; the one closest to point within radius, else None."""
    best, best_d = None, radius
    for obj in objects:
        d = math.dist(obj[2], point)
        if d <= best_d:
            best, best_d = obj, d
    return best


def sort(material, confidence, sorting, trash_materials):
    """(material, recyclable, bin_id) for a classification, following config/sorting_rules.yaml.

    Not trash -> distractor, no bin. Unknown material or low confidence -> the fallback
    bin (residual): never risk contaminating the recycling bin.
    """
    if material and material not in trash_materials:
        return DISTRACTOR, False, ''
    if not material or confidence < sorting['confidence_threshold'] or material not in sorting['rules']:
        return material, False, sorting['fallback_bin']
    bin_id = sorting['rules'][material]
    return material, bin_id == 'recyclable', bin_id
