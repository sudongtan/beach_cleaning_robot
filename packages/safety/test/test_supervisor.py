"""Every safety rule (the evidence for "zero safety violations"). Thresholds from config/safety.yaml."""

from pathlib import Path

import pytest
import yaml

from safety.supervisor import ESTOP, OK, SLOW, STOP, ZERO, Params, Supervisor, geofence_filter

CFG = Path(__file__).resolve().parents[3] / 'config' / 'safety.yaml'
P = Params.from_dict(yaml.safe_load(open(CFG))['safety_supervisor']['ros__parameters'])
CMD = (0.3, 0.1, 0.4)


FENCE = [(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (0.0, 30.0)]


def ready(t=0.0, people=()):
    """Supervisor with fresh inputs at time t, already released to ok if nobody is near; the
    robot stands in the middle of a 40 x 30 m geofence."""
    s = Supervisor(P)
    s.on_geofence(FENCE)
    s.on_pose((20.0, 15.0, 0.0))
    s.on_people(t, list(people))
    s.on_cmd(t, *CMD)
    return s


def run(s, t0, t1, people=(), dt=0.05, cmd=True):
    """Feed fresh inputs every dt from t0 to t1; return the last output."""
    t, out = t0, None
    while t <= t1 + 1e-9:
        s.on_people(t, list(people))
        if cmd:
            s.on_cmd(t, *CMD)
        out = s.step(t)
        t += dt
    return out


def test_no_people_message_yet_is_stop():
    s = Supervisor(P)
    s.on_cmd(0.0, *CMD)
    assert s.step(0.0) == (STOP, ZERO)


def test_starts_in_stop_then_releases_to_ok():
    s = ready()
    assert s.step(0.0) == (STOP, ZERO)                   # startup: hold until inputs are trusted
    assert run(s, 0.05, P.release_hold_s - 0.1)[0] == STOP
    assert run(s, P.release_hold_s, P.release_hold_s + 0.2) == (OK, CMD)


def released(people=()):
    s = ready()
    run(s, 0.0, P.release_hold_s + 0.2, people)
    return s, P.release_hold_s + 0.25


def test_ok_passes_command_through():
    s, t = released()
    assert run(s, t, t + 0.5) == (OK, CMD)


def test_person_in_slow_zone_scales_all_velocities():
    s, t = released()
    state, cmd = run(s, t, t + 0.1, people=[P.slow_distance_m - 0.5])
    assert state == SLOW
    assert cmd == pytest.approx(tuple(v * P.slow_factor for v in CMD))


def test_person_in_stop_zone_is_zero():
    s, t = released()
    assert run(s, t, t + 0.1, people=[P.stop_distance_m - 0.2]) == (STOP, ZERO)


def test_nearest_person_counts():
    s, t = released()
    assert run(s, t, t + 0.1, people=[10.0, P.stop_distance_m - 0.1, 5.0])[0] == STOP


def test_escalation_is_immediate():
    s, t = released()
    s.on_people(t, [P.stop_distance_m - 0.1])
    assert s.step(t) == (STOP, ZERO)


def test_stop_hysteresis_margin_and_hold():
    s, t = released()
    run(s, t, t + 0.1, people=[1.0])
    t += 0.2
    # just beyond the threshold but inside the margin: stays stopped however long
    assert run(s, t, t + 3.0, people=[P.stop_distance_m + P.release_margin_m / 2])[0] == STOP
    t += 3.05
    # beyond threshold + margin, but not for long enough yet
    assert run(s, t, t + P.release_hold_s - 0.2, people=[2.0])[0] == STOP
    t += P.release_hold_s
    # held long enough: steps down to slow (2 m is still inside the slow zone)
    assert run(s, t, t + 0.3, people=[2.0])[0] == SLOW


def test_slow_hysteresis():
    s, t = released()
    run(s, t, t + 0.1, people=[2.5])
    t += 0.2
    assert run(s, t, t + 2.0, people=[P.slow_distance_m + P.release_margin_m / 2])[0] == SLOW
    t += 2.05
    assert run(s, t, t + P.release_hold_s + 0.2, people=[P.slow_distance_m + P.release_margin_m + 0.1])[0] == OK


def test_flicker_resets_the_hold_timer():
    s, t = released()
    run(s, t, t + 0.1, people=[1.0])
    t += 0.2
    run(s, t, t + P.release_hold_s * 0.8, people=[5.0])
    t += P.release_hold_s * 0.8 + 0.05
    run(s, t, t + 0.05, people=[1.0])                     # back inside for one sample
    t += 0.1
    assert run(s, t, t + P.release_hold_s * 0.8, people=[5.0])[0] == STOP


def test_people_stale_is_stop():
    s, t = released()
    s.on_cmd(t + P.people_timeout_s + 0.1, *CMD)
    assert s.step(t + P.people_timeout_s + 0.1) == (STOP, ZERO)


def test_people_stale_never_releases():
    s, t = released()
    run(s, t, t + 0.1, people=[1.0])
    last = t + 0.1
    for k in range(1, 60):                                # 3 s with commands but no /people
        s.on_cmd(last + 0.05 * k, *CMD)
        assert s.step(last + 0.05 * k) == (STOP, ZERO)


def test_cmd_stale_is_zero_but_state_unchanged():
    s, t = released()
    s.on_people(t + P.cmd_timeout_s + 0.1, [])
    assert s.step(t + P.cmd_timeout_s + 0.1) == (OK, ZERO)


def test_estop_latches_until_released():
    s, t = released()
    s.on_estop(True)
    assert run(s, t, t + 0.2) == (ESTOP, ZERO)
    assert run(s, t + 0.25, t + 5.0) == (ESTOP, ZERO)     # no new /estop messages: still latched
    s.on_estop(False)
    assert run(s, t + 5.05, t + 5.1) == (OK, CMD)


def test_estop_has_priority_over_everything():
    s, t = released()
    s.on_estop(True)
    assert run(s, t, t + 0.1, people=[0.5])[0] == ESTOP
    assert run(s, t + 0.15, t + 0.2, people=[20.0])[0] == ESTOP


def test_estop_release_with_person_close_is_stop():
    s, t = released()
    s.on_estop(True)
    run(s, t, t + 0.1, people=[1.0])
    s.on_estop(False)
    assert run(s, t + 0.15, t + 0.2, people=[1.0]) == (STOP, ZERO)


def test_slow_never_exceeds_scaled_command():
    s, t = released()
    for d in (2.9, 2.0, 1.6):
        state, cmd = run(s, t, t + 0.05, people=[d])
        assert state == SLOW and all(abs(c) <= abs(v) * P.slow_factor + 1e-12 for c, v in zip(cmd, CMD))
        t += 0.1


# --- geofence ---------------------------------------------------------------------------------
M = P.geofence_margin_m


def fence(pose, v):
    return geofence_filter(v, pose, FENCE, M)


def test_geofence_no_effect_away_from_the_edge():
    assert fence((20.0, 15.0, 0.0), (0.3, 0.1)) == pytest.approx((0.3, 0.1))


def test_geofence_blocks_outward_near_the_edge():
    assert fence((40.0 - M / 2, 15.0, 0.0), (0.3, 0.0)) == pytest.approx((0.0, 0.0))   # east, facing east


def test_geofence_allows_along_and_inward():
    assert fence((40.0 - M / 2, 15.0, 0.0), (0.0, 0.2)) == pytest.approx((0.0, 0.2))   # along the edge
    assert fence((40.0 - M / 2, 15.0, 0.0), (-0.3, 0.0)) == pytest.approx((-0.3, 0.0))  # back inside


def test_geofence_diagonal_keeps_the_parallel_part():
    vx, vy = fence((40.0 - M / 2, 15.0, 0.0), (0.3, 0.2))
    assert (vx, vy) == pytest.approx((0.0, 0.2))


def test_geofence_rotated_robot():
    """Facing north (yaw 90 deg) at the east edge: body -y points east (outward)."""
    import math
    vx, vy = fence((40.0 - M / 2, 15.0, math.pi / 2), (0.2, -0.3))
    assert (vx, vy) == pytest.approx((0.2, 0.0), abs=1e-9)


def test_geofence_corner_blocks_both_outward_directions():
    vx, vy = fence((40.0 - M / 2, 30.0 - M / 2, 0.0), (0.3, 0.3))
    assert (vx, vy) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_geofence_outside_only_inward():
    assert fence((41.0, 15.0, 0.0), (0.3, 0.0)) == pytest.approx((0.0, 0.0))
    assert fence((41.0, 15.0, 0.0), (-0.3, 0.0)) == pytest.approx((-0.3, 0.0))
    assert fence((20.0, -1.0, 0.0), (0.2, -0.2)) == pytest.approx((0.2, 0.0))       # south of the fence


def test_geofence_never_increases_speed():
    import random
    rnd = random.Random(0)
    for _ in range(2000):
        pose = (rnd.uniform(-2, 42), rnd.uniform(-2, 32), rnd.uniform(-3.2, 3.2))
        v = (rnd.uniform(-0.4, 0.4), rnd.uniform(-0.3, 0.3))
        out = fence(pose, v)
        assert out[0] ** 2 + out[1] ** 2 <= v[0] ** 2 + v[1] ** 2 + 1e-12


def test_geofence_never_moves_outward_near_the_edge():
    """Property: near or outside the fence, the filtered velocity never increases the distance outside
    / decreases the distance to the edge from inside (checked with a small step)."""
    import math
    import random
    from safety.supervisor import closest_on_segment, inside_polygon
    rnd = random.Random(1)

    def signed_depth(x, y):
        d = min(math.dist((x, y), closest_on_segment(x, y, a, b)) for a, b in zip(FENCE, FENCE[1:] + FENCE[:1]))
        return d if inside_polygon(x, y, FENCE) else -d
    for _ in range(2000):
        x, y, yaw = rnd.uniform(-1, 41), rnd.choice([rnd.uniform(-1, 1), rnd.uniform(29, 31)]), rnd.uniform(-3.2, 3.2)
        if rnd.random() < 0.5:
            x, y = rnd.choice([rnd.uniform(-1, 1), rnd.uniform(39, 41)]), rnd.uniform(-1, 31)
        before = signed_depth(x, y)
        if before >= M:
            continue
        vx, vy = fence((x, y, yaw), (rnd.uniform(-0.4, 0.4), rnd.uniform(-0.3, 0.3)))
        c, s = math.cos(yaw), math.sin(yaw)
        dt = 1e-3
        after = signed_depth(x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt)
        assert after >= before - 1e-6          # second order: sliding past a corner (v^2 dt^2 / 2d)


def test_missing_fence_or_pose_stops_linear_motion_keeps_turning():
    s, t = released()
    s.on_pose(None)
    state, cmd = run(s, t, t + 0.1)
    assert state == OK and cmd == (0.0, 0.0, CMD[2])
    s.on_pose((20.0, 15.0, 0.0))
    s.on_geofence([])
    assert run(s, t + 0.15, t + 0.2)[1] == (0.0, 0.0, CMD[2])


def test_geofence_does_not_change_the_state():
    s, t = released()
    s.on_pose((40.0 - M / 2, 15.0, 0.0))
    state, cmd = run(s, t, t + 0.1)
    assert state == OK and cmd[0] == pytest.approx(0.0) and cmd[1] == pytest.approx(CMD[1])
