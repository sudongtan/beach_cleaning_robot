"""The robot_lab walking policy as Gazebo runs it, in numpy (D0b). No Isaac, no ROS, no torch.

Mirrors, line by line where it matters:
- legubiao rl_quadruped_controller `StateRL.cpp` at 4dfc15e (d0_from_laptop/policy/
  rl_quadruped_controller_src): observation = [ang_vel * ang_vel_scale, quatRotateInverse(q, (0,0,-1)),
  commands * (lin_vel_scale, lin_vel_scale, ang_vel_scale), (dof_pos - default) * dof_pos_scale,
  dof_vel * dof_vel_scale, last actions], clamped to +-clip_obs; actions clamped, hip actions times
  hip_scale_reduction (and that reduced action is what the next observation sees), target =
  action * action_scale + default; PD kp/kd from the config. Default = the controller's stand_pos.
  Joint order = the controller's `joints` (FR, FL, RR, RL x hip, thigh, calf). The quaternion is
  (w, x, y, z) ("isaacsim" framework), the IMU frame = trunk.
- the Gazebo adapter (packages/locomotion/locomotion/legubiao_adapter.py, robot_lab config): the
  wanted twist goes through the speed maps and the yaw trim; the policy's command term is then that
  twist (the controller scales the sticks back: stick = v / max_v, times (2, 2, 0.25) = v).
"""

import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GAZEBO_CONTROLLERS = REPO / 'packages' / 'sim_bridge_gazebo' / 'config' / 'gazebo_controllers.yaml'
CONTROLLER = 'rl_robot_lab_controller'


# ---------------------------------------------------------------- the adapter's pure helpers

def adapter_module():
    """packages/locomotion/locomotion/legubiao_adapter.py, imported with stub ROS modules where
    ROS isn't installed (its helpers are plain math; the node class is never used here)."""
    stubs = ['rclpy', 'rclpy.node', 'control_input_msgs', 'control_input_msgs.msg', 'geometry_msgs',
             'geometry_msgs.msg', 'nav_msgs', 'nav_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg', 'tf2_ros']
    added = []
    for name in stubs:
        try:
            importlib.import_module(name)
        except ImportError:
            m = types.ModuleType(name)
            m.__path__ = []                                      # a package, so submodules import

            def _dummy(attr):                                    # any class name -> a dummy class
                if attr.startswith('__'):
                    raise AttributeError(attr)
                return type(attr, (), {})
            m.__getattr__ = _dummy
            sys.modules[name] = m
            added.append(name)
    sys.path.insert(0, str(REPO / 'packages' / 'locomotion'))
    try:
        return importlib.import_module('locomotion.legubiao_adapter')
    finally:
        sys.path.remove(str(REPO / 'packages' / 'locomotion'))
        for name in added:
            sys.modules.pop(name, None)


def load_adapter_params(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    (node,) = doc.values()
    return node['ros__parameters']


class CommandShaper:
    """Wanted twist -> the policy's command term, as the Gazebo adapter makes it: speed maps per
    direction (map_planar) and the integral yaw trim from the gyro (yaw_trim_step)."""

    def __init__(self, params, adapter=None):
        self.a = adapter or adapter_module()
        self.p = params
        self.maps = {k: self.a.speed_map(params.get(f'{k}_in', []), params.get(f'{k}_out', []), k)
                     for k in ('vx_map', 'vx_back_map', 'vy_map', 'vy_neg_map')}
        self.trim = 0.0
        self.started = False
        self.limit = float(params['stick_limit'])
        self.max_v = (float(params['max_vx']), float(params['max_vy']), float(params['max_wz']))

    def __call__(self, vx, vy, wz, gyro_z, dt):
        """One adapter tick (Gazebo: 50 Hz publish_rate; here: every policy step)."""
        vx, vy = self.a.map_planar(vx, vy, self.maps)
        ki = float(self.p.get('yaw_trim_ki', 0.0))
        if ki > 0:
            if self.started:            # the adapter integrates from its second tick on
                self.trim = self.a.yaw_trim_step(self.trim, wz, gyro_z, dt, ki, float(self.p['yaw_trim_max']))
            self.started = True
            wz = wz + self.trim
        # sticks clamp at stick_limit; the policy then sees stick * max (StateRL scales back)
        (ly, lx, rx), _ = self.a.sticks(vx, vy, wz, self.max_v, self.limit)
        return np.array([ly * self.max_v[0], -lx * self.max_v[1], -rx * self.max_v[2]])


# ---------------------------------------------------------------- the policy network

class Mlp:
    """The exported robot_lab network: policy_weights.npz from the laptop."""

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        n = len([k for k in d.files if k.startswith('w') and k[1:].isdigit()])
        if n == 0:
            raise ValueError(f'{npz_path}: no w0.. layers')
        self.layers = [(d[f'w{i}'].astype(np.float32), d[f'b{i}'].astype(np.float32)) for i in range(n)]
        # e.g. "elu(alpha=1.0) between layers, none after the last; normalizer = Identity"
        desc = str(d['activation']) if 'activation' in d.files else 'elu'
        if 'identity' not in desc.lower().split('normalizer')[-1] and 'normalizer' in desc.lower():
            raise ValueError(f'unsupported observation normalizer: {desc}')
        self.activation = desc.split('(')[0].split()[0]

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for k, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if k < len(self.layers) - 1:
                x = _act(x, self.activation)
        return x


def _act(x, name):
    name = name.lower()
    if name == 'elu':
        return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    if name == 'relu':
        return np.maximum(x, 0)
    if name == 'tanh':
        return np.tanh(x)
    raise ValueError(f'unknown activation {name}')


# ---------------------------------------------------------------- observation, action

def quat_rotate_inverse(q, v):
    """StateRL.cpp quatRotateInverse, framework "isaacsim": q = (w, x, y, z); v in the world frame
    -> the same vector in the body frame."""
    q = np.asarray(q, dtype=np.float64)
    w, qv = q[0], q[1:4]
    v = np.asarray(v, dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * np.dot(qv, v) * 2.0
    return a - b + c


def load_controller(path=GAZEBO_CONTROLLERS, name=CONTROLLER):
    with open(path) as f:
        doc = yaml.safe_load(f)
    p = doc[name]['ros__parameters']
    return {'joints': list(p['joints']), 'stand_pos': [float(v) for v in p['stand_pos']],
            'update_rate': float(p['update_rate'])}


class Policy:
    """One StateRL controller: observation, network, actions -> joint position targets."""

    def __init__(self, policy_cfg, mlp, default_pos, joints):
        c = policy_cfg
        self.c = c
        self.mlp = mlp
        self.joints = list(joints)
        self.default = np.asarray(default_pos, dtype=np.float32)
        n = int(c['num_of_dofs'])
        assert len(self.default) == n == len(self.joints)
        if c['observations'] != ['ang_vel', 'gravity_vec', 'commands', 'dof_pos', 'dof_vel', 'actions']:
            raise ValueError(f"unexpected observation list {c['observations']}")
        if c.get('framework') != 'isaacsim':
            raise ValueError('quaternion order assumes framework "isaacsim" (w, x, y, z)')
        # StateRL ignores the yaml's commands_scale and uses (lin_vel_scale, lin_vel_scale, ang_vel_scale)
        self.commands_scale = np.array([c['lin_vel_scale'], c['lin_vel_scale'], c['ang_vel_scale']], dtype=np.float32)
        self.kp = np.asarray(c['rl_kp'], dtype=np.float32)
        self.kd = np.asarray(c['rl_kd'], dtype=np.float32)
        self.lo = np.asarray(c['clip_actions_lower'], dtype=np.float32)
        self.hi = np.asarray(c['clip_actions_upper'], dtype=np.float32)
        self.hip_idx = list(c['hip_scale_reduction_indices'])
        self.decimation = int(c['decimation'])
        self.reset()

    def reset(self):
        # StateRL::enter
        self.actions = np.zeros(len(self.joints), dtype=np.float32)
        self.targets = self.default.copy()

    def observation(self, quat_wxyz, gyro_body, command, dof_pos, dof_vel):
        c = self.c
        obs = np.concatenate([
            np.asarray(gyro_body, dtype=np.float32) * c['ang_vel_scale'],
            quat_rotate_inverse(quat_wxyz, (0.0, 0.0, -1.0)).astype(np.float32),
            np.asarray(command, dtype=np.float32) * self.commands_scale,
            (np.asarray(dof_pos, dtype=np.float32) - self.default) * c['dof_pos_scale'],
            np.asarray(dof_vel, dtype=np.float32) * c['dof_vel_scale'],
            self.actions,
        ])
        assert obs.shape == (int(c['num_observations']),), obs.shape
        return np.clip(obs, -c['clip_obs'], c['clip_obs'])

    def act(self, obs):
        """StateRL::runModel after forward(): returns the joint position targets."""
        a = np.clip(self.mlp(obs[None, :])[0], self.lo, self.hi)
        a = a.copy()
        a[self.hip_idx] *= self.c['hip_scale_reduction']
        self.actions = a.astype(np.float32)               # the reduced action is fed back (quirk)
        self.targets = a * self.c['action_scale'] + self.default
        return self.targets

    def step(self, quat_wxyz, gyro_body, command, dof_pos, dof_vel):
        return self.act(self.observation(quat_wxyz, gyro_body, command, dof_pos, dof_vel))


class Scheduler:
    """Physics at `physics_hz`, the policy every `decimation` physics steps (Gazebo: 200 Hz / 4)."""

    def __init__(self, decimation):
        self.decimation = int(decimation)
        self.n = 0

    def due(self):
        """Call once per physics step; True when the policy runs on this step (the first one too)."""
        run = self.n % self.decimation == 0
        self.n += 1
        return run


# ---------------------------------------------------------------- state helpers

def tilt_deg(quat_wxyz):
    """Body tilt from vertical, degrees."""
    g = quat_rotate_inverse(quat_wxyz, (0.0, 0.0, -1.0))
    return math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def world_to_body(quat_wxyz, v_world):
    return quat_rotate_inverse(quat_wxyz, v_world)


def speeds(trace, t0, t1):
    """Mean planar speed (m/s) and yaw rate (rad/s) over [t0, t1] of a trace of
    (t, x, y, yaw) rows, as c7_summary does (displacement / time, unwrapped yaw)."""
    rows = [r for r in trace if t0 <= r[0] <= t1]
    if len(rows) < 2:
        return float('nan'), float('nan')
    (ta, xa, ya, _), (tb, xb, yb, _) = rows[0], rows[-1]
    yaws = np.unwrap([r[3] for r in rows])
    dt = tb - ta
    return math.hypot(xb - xa, yb - ya) / dt, (yaws[-1] - yaws[0]) / dt


def build_policy(laptop, robot_cfg):
    """Policy + CommandShaper from the laptop files and the repo configs."""
    pc = robot_cfg['policy']
    with open(laptop / pc['config']) as f:
        policy_cfg = yaml.safe_load(f)
    ctrl = load_controller()
    if ctrl['joints'] != robot_cfg['robot']['leg_joints']:
        raise ValueError('robot.yaml leg_joints != the Gazebo controller joints')
    if abs(ctrl['update_rate'] - pc['physics_hz']) > 1e-9:
        raise ValueError('physics_hz != the Gazebo controller update_rate')
    policy = Policy(policy_cfg, Mlp(laptop / pc['weights']), ctrl['stand_pos'], ctrl['joints'])
    shaper = CommandShaper(load_adapter_params(REPO / pc['adapter_config']))
    return policy, shaper, policy_cfg
