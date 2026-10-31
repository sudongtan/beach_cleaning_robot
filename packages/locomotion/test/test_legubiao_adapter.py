import math

from locomotion.legubiao_adapter import clamp, rotate_to_body, tilt_deg


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
