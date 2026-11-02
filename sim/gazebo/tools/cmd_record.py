#!/usr/bin/env python3
"""Record the velocity command chain (A2b problem E): /cmd_vel_nav ('nav'), /cmd_vel ('out') and the
controller's control_input ('inputs'), one JSON line per message with wall and sim time. Read-only;
the input for cmd_replay.py.

    cmd_record.py <wall seconds> <out.jsonl>
"""
import json
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message


def main():
    rclpy.init()
    n = Node('cmd_record')
    st = {'sim': 0.0}
    n.create_subscription(Clock, '/clock', lambda m: st.update(sim=m.clock.sec + m.clock.nanosec * 1e-9),
                          qos_profile_sensor_data)
    out = open(sys.argv[2], 'w')

    def rec(name):
        def cb(m):
            out.write(json.dumps({'topic': name, 'wall': time.monotonic(), 'sim': st['sim'],
                                  'msg': message_to_ordereddict(m)}) + '\n')
        return cb

    n.create_subscription(Twist, '/cmd_vel_nav', rec('nav'), 50)
    n.create_subscription(Twist, '/cmd_vel', rec('out'), 50)
    found = False
    end = time.monotonic() + float(sys.argv[1])
    while time.monotonic() < end:
        if not found:
            for name, types in n.get_topic_names_and_types():
                if name.endswith('control_input'):
                    found = True
                    n.create_subscription(get_message(types[0]), name, rec('inputs'), 50)
        rclpy.spin_once(n, timeout_sec=0.01)
    out.close()


if __name__ == '__main__':
    main()
