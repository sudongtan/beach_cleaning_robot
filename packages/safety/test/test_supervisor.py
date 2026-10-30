"""Every safety rule (the evidence for "zero safety violations"). Thresholds from config/safety.yaml."""

from pathlib import Path

import pytest
import yaml

from safety.supervisor import ESTOP, OK, SLOW, STOP, ZERO, Params, Supervisor

CFG = Path(__file__).resolve().parents[3] / 'config' / 'safety.yaml'
P = Params.from_dict(yaml.safe_load(open(CFG))['safety_supervisor']['ros__parameters'])
CMD = (0.3, 0.1, 0.4)


def ready(t=0.0, people=()):
    """Supervisor with fresh inputs at time t, already released to ok if nobody is near."""
    s = Supervisor(P)
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
