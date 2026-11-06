import math

from locomotion.legubiao_adapter import clamp, rotate_to_body, sticks, tilt_deg, yaw_trim_step


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
