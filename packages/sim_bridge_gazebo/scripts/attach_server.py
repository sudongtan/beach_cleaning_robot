#!/usr/bin/env python3
"""Attach-on-contact plumbing for Gazebo (grasp_mode:=attach).

Serves /sim/attach and /sim/detach (beach_interfaces/srv/SimAttach) with the
robot's per-item DetachableJoint plugins (see beach_robot_gazebo.urdf.xacro).

Gazebo Harmonic attaches every DetachableJoint on startup, at the item's current
pose relative to the gripper. So on start this node detaches all items, then
(if `unpause`) unpauses the world, which the launch file starts paused. Until then
the robot hasn't moved and no item has been dragged.

Parameters:
  manifest (str)   scene_<seed>.manifest.yaml from sim/gazebo/load_scene.py
  unpause (bool)   unpause /world/<world_name> once every item is detached (default true)
  timeout_s (float) give up waiting for the plugins after this long (default 60)
"""

import threading
import time

import rclpy
import yaml
from beach_interfaces.srv import SimAttach
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.stringmsg_pb2 import StringMsg
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node as GzNode
from rclpy.node import Node

CONFIRM_TIMEOUT_S = 3.0


class AttachServer(Node):
    def __init__(self):
        super().__init__('attach_server')
        manifest = self.declare_parameter('manifest', '').value
        self.unpause = self.declare_parameter('unpause', True).value
        self.timeout = self.declare_parameter('timeout_s', 60.0).value
        with open(manifest) as f:
            m = yaml.safe_load(f)
        self.world = m['world_name']
        self.items = list(m['attach_items'])

        self.gz = GzNode()
        self.state = {}                                  # id -> 'attached' / 'detached'
        self.changed = {i: threading.Event() for i in self.items}
        self.pub_attach, self.pub_detach = {}, {}
        for i in self.items:
            self.pub_attach[i] = self.gz.advertise(f'/sim/gz/attach/{i}', Empty)
            self.pub_detach[i] = self.gz.advertise(f'/sim/gz/detach/{i}', Empty)
            self.gz.subscribe(StringMsg, f'/sim/gz/attach_state/{i}', self._state_cb(i))

        self.create_service(SimAttach, '/sim/attach', self.on_attach)
        self.create_service(SimAttach, '/sim/detach', self.on_detach)
        threading.Thread(target=self.startup, daemon=True).start()

    def _state_cb(self, item):
        def cb(msg):
            self.state[item] = msg.data
            self.changed[item].set()
        return cb

    def startup(self):
        """Detach everything the plugins attached on startup, then unpause."""
        t0 = time.monotonic()
        pending = set(self.items)
        while pending and time.monotonic() - t0 < self.timeout:
            # A detach request is ignored until the plugin has attached (robot spawned,
            # child found), so keep asking until each item reports 'detached'.
            for i in list(pending):
                if self.state.get(i) == 'detached':
                    pending.discard(i)
                else:
                    self.pub_detach[i].publish(Empty())
            time.sleep(0.2)
        if pending:
            self.get_logger().error(f'items still attached after {self.timeout:.0f} s: {sorted(pending)}')
            return
        self.get_logger().info(f'detached all {len(self.items)} items after startup '
                               f'({time.monotonic() - t0:.1f} s)')
        if self.unpause:
            req = WorldControl()
            req.pause = False
            ok, rep = self.gz.request(f'/world/{self.world}/control', req, WorldControl, Boolean, 5000)
            if ok and rep.data:
                self.get_logger().info(f'world {self.world} unpaused')
            else:
                self.get_logger().error(f'could not unpause world {self.world}')

    def _switch(self, request, response, want, pubs):
        i = request.object_id
        if i not in pubs:
            response.success, response.message = False, f'unknown or not attachable: {i!r}'
            return response
        if self.state.get(i) == want:
            response.success, response.message = True, f'{i} already {want}'
            return response
        self.changed[i].clear()
        pubs[i].publish(Empty())
        deadline = time.monotonic() + CONFIRM_TIMEOUT_S
        while time.monotonic() < deadline and self.state.get(i) != want:
            self.changed[i].wait(0.1)
            self.changed[i].clear()
        response.success = self.state.get(i) == want
        response.message = f'{i} {want}' if response.success else f'{i}: no confirmation from Gazebo'
        return response

    def on_attach(self, request, response):
        return self._switch(request, response, 'attached', self.pub_attach)

    def on_detach(self, request, response):
        return self._switch(request, response, 'detached', self.pub_detach)


def main():
    rclpy.init()
    node = AttachServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
