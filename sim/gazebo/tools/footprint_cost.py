#!/usr/bin/env python3
"""Cost of the local costmap under the robot's footprint outline (B3: Nav2 "collision ahead" stalls).

Nav2's RPP refuses every command when the footprint at the robot's current pose costs LETHAL
(nav2_costmap_2d footprintCost: the polygon's edges only, a cell off the map counts as lethal).
This samples /local_costmap/costmap along /local_costmap/published_footprint (both in the costmap's
frame) and prints, every --every sim seconds, the robot pose, the outline's max cost and how many
outline cells are lethal (OccupancyGrid 100 = LETHAL_OBSTACLE, 99 = INSCRIBED_INFLATED, -1 unknown).
Run it next to a running stack (same GZ_PARTITION / ROS_DOMAIN_ID):

    footprint_cost.py [--every 5] [--count 0]
"""
import argparse
import math

import rclpy
from geometry_msgs.msg import PolygonStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile


def outline_cells(grid, polygon, step):
    info = grid.info
    res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
    pts = [(p.x, p.y) for p in polygon.points]
    out = []
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        n = max(1, math.ceil(math.hypot(bx - ax, by - ay) / step))
        for k in range(n):
            x, y = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
            i, j = int((y - oy) // res), int((x - ox) // res)
            if 0 <= i < info.height and 0 <= j < info.width:
                out.append(grid.data[i * info.width + j])
            else:
                out.append(None)                              # off the map: lethal for Nav2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=float, default=5.0, help='sim seconds between lines')
    ap.add_argument('--count', type=int, default=0, help='lines to print (0 = until killed)')
    a = ap.parse_args()
    rclpy.init()
    n = Node('footprint_cost', parameter_overrides=[Parameter('use_sim_time', value=True)])
    st = {'grid': None, 'fp': None, 'pose': None}
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    n.create_subscription(OccupancyGrid, '/local_costmap/costmap', lambda m: st.update(grid=m), latched)
    n.create_subscription(PolygonStamped, '/local_costmap/published_footprint', lambda m: st.update(fp=m), 10)
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(pose=m), 10)
    last, lines = None, 0
    while rclpy.ok() and (a.count == 0 or lines < a.count):
        rclpy.spin_once(n, timeout_sec=0.1)
        t = n.get_clock().now().nanoseconds * 1e-9
        if st['grid'] is None or st['fp'] is None or st['pose'] is None or (last is not None and t - last < a.every):
            continue
        last = t
        cells = outline_cells(st['grid'], st['fp'].polygon, st['grid'].info.resolution / 2)
        lethal = sum(1 for c in cells if c is None or c >= 100)
        known = [c for c in cells if c is not None]
        p = st['pose'].pose.pose
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        print(f't {t:7.1f}  robot ({p.position.x:6.2f}, {p.position.y:6.2f}, {math.degrees(yaw):+6.1f} deg)  '
              f'outline max cost {max(known) if known else "-"}  lethal {lethal}/{len(cells)}  '
              f'off-map {sum(1 for c in cells if c is None)}', flush=True)
        lines += 1


if __name__ == '__main__':
    main()
