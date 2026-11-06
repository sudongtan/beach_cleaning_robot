import math

import pytest

from locomotion.legubiao_adapter import clamp, rotate_to_body, sticks, tilt_deg, yaw_trim_step, map_speed, speed_map, map_planar


def test_rotate_identity():
    assert rotate_to_body((0, 0, 0, 1), (1.0, 2.0, 3.0)) == [1.0, 2.0, 3.0]


def test_rotate_yaw_90():
    # body yawed +90 deg: world +y is body +x
    s = math.sin(math.pi / 4)
    v = rotate_to_body((0, 0, s, s), (0.0, 1.0, 0.0))
    assert abs(v[0] - 1.0) < 1e-9 and abs(v[1]) < 1e-9


def test_clamp():
    assert clamp(2.0) == 1.0 and clamp(-3.0) == -1.0 and clamp(0.25) == 0.25


def test_tilt():
    assert tilt_deg((0, 0, 0, 1)) == 0.0
    assert tilt_deg((0, 0, math.sin(1.0), math.cos(1.0))) < 1e-6          # yaw only: upright
    s = math.sin(math.radians(45) / 2)
    assert abs(tilt_deg((s, 0, 0, math.cos(math.radians(45) / 2))) - 45.0) < 1e-6   # rolled 45 deg
    assert abs(tilt_deg((1, 0, 0, 0)) - 180.0) < 1e-6                    # upside down


def test_sticks_basic_limits():
    (ly, lx, rx), clamped = sticks(0.2, 0.15, 0.25, (0.4, 0.3, 0.5))
    assert (ly, lx, rx) == (0.5, -0.5, -0.5) and not clamped
    (ly, _, _), clamped = sticks(0.8, 0.0, 0.0, (0.4, 0.3, 0.5))
    assert ly == 1.0 and clamped


def test_sticks_beyond_one_with_a_limit():
    # robot_lab: sticks = v / 2 and 4 * wz (StateRL scales them by 2, 2, 0.25)
    (ly, lx, rx), clamped = sticks(0.1, 0.0, 0.5, (2.0, 2.0, 0.25), limit=4.0)
    assert (ly, lx, rx) == (0.05, 0.0, -2.0) and not clamped
    (_, _, rx), clamped = sticks(0.0, 0.0, 1.5, (2.0, 2.0, 0.25), limit=4.0)
    assert rx == -4.0 and clamped


def test_yaw_trim_step_integrates_and_limits():
    assert math.isclose(yaw_trim_step(0.0, 0.0, 0.2, 0.1, 1.0, 0.4), -0.02)
    assert math.isclose(yaw_trim_step(-0.39, 0.0, 0.2, 1.0, 1.0, 0.4), -0.4)
    assert math.isclose(yaw_trim_step(0.1, 0.3, 0.3, 0.02, 1.0, 0.4), 0.1)


def test_yaw_trim_cancels_a_constant_bias():
    # plant fitted to the bare-Go2 robot_lab runs (B4 log): measured = 0.15 + 0.73 * policy command
    trim, dt = 0.0, 0.02
    for wz in (0.0, 0.1, -0.1):
        for _ in range(int(15 / dt)):
            meas = 0.15 + 0.73 * (wz + trim)
            trim = yaw_trim_step(trim, wz, meas, dt, 1.0, 0.4)
        assert abs(0.15 + 0.73 * (wz + trim) - wz) < 0.01


def test_map_speed():
    tin, tout = [0.0, 0.02, 0.10, 0.25], [0.0, 0.08, 0.15, 0.25]
    assert map_speed(0.0, tin, tout) == 0.0
    assert math.isclose(map_speed(0.01, tin, tout), 0.04)
    assert math.isclose(map_speed(0.10, tin, tout), 0.15)
    assert math.isclose(map_speed(-0.06, tin, tout), -0.115)
    assert math.isclose(map_speed(0.3, tin, tout), 0.3)   # beyond the table: last offset, here 0
    assert math.isclose(map_speed(-0.3, [0.0, 0.1], [0.0, 0.2]), -0.4)   # last offset +0.1 kept
    assert map_speed(0.07, [], []) == 0.07           # off


def test_speed_map():
    assert speed_map([0.0, 0.1], [0.0, 0.2], 'vy_map') == ([0.0, 0.1], [0.0, 0.2])
    assert speed_map([0.0], [0.0], 'vy_map') == ([], [])       # parameter default: off
    with pytest.raises(ValueError):
        speed_map([0.0, 0.1], [0.0], 'vy_map')


def test_map_planar():
    off = ([], [])
    m = {'vx_map': ([0.0, 0.1, 0.25], [0.0, 0.15, 0.25]), 'vx_back_map': ([0.0, 0.05, 0.1], [0.0, 0.21, 0.25]),
         'vy_map': ([0.0, 0.05], [0.0, 0.21]), 'vy_neg_map': off}
    assert map_planar(0.0, 0.0, m) == (0.0, 0.0)
    assert map_planar(0.2, 0.05, {k: off for k in m}) == (0.2, 0.05)           # all off
    assert all(math.isclose(a, b) for a, b in zip(map_planar(-0.05, 0.0, m), (-0.21, 0.0)))
    vx, vy = map_planar(0.0, -0.05, m)                         # right falls back to left
    assert math.isclose(vx, 0.0, abs_tol=1e-12) and math.isclose(vy, -0.21)
    vx, vy = map_planar(-0.05, 0.05, m)                        # diagonal: both maps, direction kept
    assert math.isclose(vx, -vy) and math.hypot(vx, vy) > 0.21
    # walking forward fast, a small sideways correction is not lifted
    vx, vy = map_planar(0.3, 0.03, m)
    assert math.isclose(vy / vx, 0.1) and math.hypot(vx, vy) < 1.02 * math.hypot(0.3, 0.03)
