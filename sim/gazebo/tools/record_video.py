#!/usr/bin/env python3
"""Record a video of a running scene, without the GUI (A2 DoD b3 / Packaging video).

Spawns a camera model with Gazebo's CameraVideoRecorder system (gz-sim 8: renders in the
server, records in sim time, example `camera_video_record_dbl_pendulum.sdf`), starts it,
and in `chase` mode moves the camera behind the robot every 0.1 s wall (yaw smoothed, so
turns on the spot don't whip the view). Stop it with SIGINT/SIGTERM: the recording is
finalised and saved, and the camera removed. While recording, the plugin writes
`<entity id>.mp4` into the Gazebo server's working directory (the repo root with `make run`)
and moves it to OUT on stop.

    record_video.py OUT.mp4 [--mode chase|fixed] [--fixed x,y,z,pitch,yaw]
                    [--distance 2.6] [--height 1.5] [--yaw-tau 1.5] [--world beach]

Needs the same GZ_PARTITION / GZ_IP and ROS_DOMAIN_ID as the running simulation. The robot
pose comes from ROS (/sim/ground_truth/pose): with a gz-transport subscription to the pose
stream in the same process, the service requests time out. The extra camera slows
the simulation (one more rendered 1280 x 720 view at 25 fps sim time).
"""

import argparse
import math
import os
import signal
import sys
import time

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.video_record_pb2 import VideoRecord
from gz.transport13 import Node
from nav_msgs.msg import Odometry
import rclpy

CAM = f'video_cam_{os.getpid()}'     # unique: removing and respawning one name races (spawn rejected)
SERVICE = f'/{CAM}/record_video'
TIMEOUT_MS = 5000

SDF = f"""<sdf version="1.9"><model name="{CAM}"><static>true</static><pose>{{pose}}</pose>
<link name="link"><sensor name="camera" type="camera">
  <camera><horizontal_fov>1.2</horizontal_fov><image><width>1280</width><height>720</height></image>
    <clip><near>0.1</near><far>80</far></clip></camera>
  <always_on>1</always_on><update_rate>25</update_rate><topic>{CAM}/image</topic>
  <plugin filename="gz-sim-camera-video-recorder-system" name="gz::sim::systems::CameraVideoRecorder">
    <service>{SERVICE}</service><use_sim_time>true</use_sim_time><fps>25</fps><bitrate>4000000</bitrate>
  </plugin>
</sensor></link></model></sdf>"""


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def set_rpy(pose, roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    o = pose.orientation
    o.w = cr * cp * cy + sr * sp * sy
    o.x = sr * cp * cy - cr * sp * sy
    o.y = cr * sp * cy + sr * cp * sy
    o.z = cr * cp * sy - sr * sp * cy


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def chase_pose(x, y, z, yaw, dist, height):
    """Camera behind the robot, looking at a point 0.3 m above its base."""
    return (x - dist * math.cos(yaw), y - dist * math.sin(yaw), z + height,
            math.atan2(height - 0.3, dist), yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--mode', choices=['chase', 'fixed'], default='chase')
    ap.add_argument('--fixed', default='7.5,-4,7,0.75,1.5708', help='x,y,z,pitch,yaw')
    ap.add_argument('--distance', type=float, default=2.6)
    ap.add_argument('--height', type=float, default=1.5)
    ap.add_argument('--yaw-tau', type=float, default=1.5, help='yaw smoothing, wall s')
    ap.add_argument('--world', default='beach')
    a = ap.parse_args()

    node = Node()
    robot = {}
    rclpy.init()
    ros = rclpy.create_node('record_video')

    def on_pose(msg):
        p = msg.pose.pose
        robot['p'] = (p.position.x, p.position.y, p.position.z, yaw_of(p.orientation))
    ros.create_subscription(Odometry, '/sim/ground_truth/pose', on_pose, 10)

    if a.mode == 'chase':
        t0 = time.time()
        while 'p' not in robot:
            if time.time() - t0 > 30:
                sys.exit('no robot pose on /sim/ground_truth/pose (ROS_DOMAIN_ID?)')
            rclpy.spin_once(ros, timeout_sec=0.1)
        x, y, z, yaw = robot['p']
        cam = chase_pose(x, y, z, yaw, a.distance, a.height)
    else:
        cam = tuple(float(v) for v in a.fixed.split(','))
    smooth_yaw = cam[4]

    req = EntityFactory(sdf=SDF.format(pose=f'{cam[0]} {cam[1]} {cam[2]} 0 {cam[3]} {cam[4]}'))
    for _ in range(3):                             # the busy server can miss the first timeout
        ok, rep = node.request(f'/world/{a.world}/create', req, EntityFactory, Boolean, 3 * TIMEOUT_MS)
        if ok and rep.data:
            break
    else:
        sys.exit('could not spawn the camera')
    for _ in range(50):                            # the sensor's service appears once it renders
        ok, rep = node.request(SERVICE, VideoRecord(start=True, format='mp4', save_filename=a.out),
                               VideoRecord, Boolean, TIMEOUT_MS)
        if ok and rep.data:
            break
        time.sleep(0.2)
    else:
        sys.exit('could not start recording')
    print(f'recording to {a.out} ({a.mode})', flush=True)

    stop = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    last = time.time()
    while not stop['now']:
        rclpy.spin_once(ros, timeout_sec=0.1)
        now = time.time()
        if a.mode != 'chase' or 'p' not in robot or now - last < 0.1:
            continue
        x, y, z, yaw = robot['p']
        k = 1.0 - math.exp(-(now - last) / a.yaw_tau)
        last = now
        smooth_yaw = wrap(smooth_yaw + k * wrap(yaw - smooth_yaw))
        cx, cy, cz, pitch, cyaw = chase_pose(x, y, z, smooth_yaw, a.distance, a.height)
        pose = Pose(name=CAM)
        pose.position.x, pose.position.y, pose.position.z = cx, cy, cz
        set_rpy(pose, 0.0, pitch, cyaw)
        node.request(f'/world/{a.world}/set_pose', pose, Pose, Boolean, 1000)

    ok, rep = node.request(SERVICE, VideoRecord(stop=True), VideoRecord, Boolean, 30000)
    print(f'stopped: {"saved " + a.out if ok and rep.data else "stop request failed"}', flush=True)
    time.sleep(3)                                  # the file is moved on the next render after stop
    node.request(f'/world/{a.world}/remove', Entity(name=CAM, type=Entity.MODEL), Entity, Boolean, TIMEOUT_MS)


if __name__ == '__main__':
    main()
