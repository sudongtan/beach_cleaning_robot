"""Pick/place sequences and the arm safety rule, with a fake arm (no ROS)."""

import pytest

from grasping.sequence import Runner, bin_steps, ground_steps, pick_steps, store_steps

POSES = {'up': {'arm': ['up']}, 'rest': {'arm': ['rest']}, 'over_recyclable': {'arm': ['over_rec']}, 'over_residual': {'arm': ['over_res']}}
PLAN = {'above': ['above'], 'hover': ['hover'], 'lift': ['lift']}


class FakeIo:
    """Records what happened; `unsafe` is a set of times (ticks) when the safety state is not ok."""

    def __init__(self, unsafe=(), interrupt_at=(), fail=()):
        self.t = 0.0
        self.log = []
        self.unsafe = set(unsafe)
        self.interrupt_at = list(interrupt_at)      # arm targets that get interrupted once
        self.fail = set(fail)
        self.moving_while_unsafe = 0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t = round(self.t + s, 6)

    def safety_ok(self):
        return int(self.t) not in self.unsafe

    def _act(self, what):
        if not self.safety_ok():
            self.moving_while_unsafe += 1
        self.log.append(what)

    def move_arm(self, q, secs):
        if q[0] in self.interrupt_at:
            self.interrupt_at.remove(q[0])
            self.log.append(('interrupted', q[0]))
            self.t += 1.0                           # the stop happens; state goes bad for this tick
            self.unsafe.add(int(self.t))
            return 'interrupted'
        if q[0] in self.fail:
            return 'failed'
        self._act(('arm', q[0]))
        self.t += 0.1
        return 'ok'

    def fingers(self, v, secs):
        self._act(('fingers', v))
        return True

    def attach(self, oid):
        self._act(('attach', oid))
        return 'attach' not in self.fail

    def detach(self, oid):
        self._act(('detach', oid))
        return True


def runner(io, hold=30.0):
    return Runner(io, move_time=2.5, finger_time=1.0, hold_timeout=hold)


def test_pick_order():
    io = FakeIo()
    ok, msg = runner(io).run(pick_steps(PLAN, POSES, ['show'], 0.075, 0.01, 'wm_001'))
    assert (ok, msg) == (True, 'ok')
    assert io.log == [('arm', 'up'), ('fingers', 0.075), ('arm', 'above'), ('arm', 'hover'),
                      ('attach', 'wm_001'), ('fingers', 0.01), ('arm', 'lift'), ('arm', 'show')]


def test_attach_happens_at_hover_before_closing():
    """Gazebo can't attach while touching: attach must come before the jaws close."""
    io = FakeIo()
    runner(io).run(pick_steps(PLAN, POSES, ['show'], 0.075, 0.01, 'x'))
    assert io.log.index(('attach', 'x')) == io.log.index(('arm', 'hover')) + 1
    assert io.log.index(('attach', 'x')) < io.log.index(('fingers', 0.01))


def test_store_and_bin_and_ground_sequences():
    io = FakeIo()
    assert runner(io).run(store_steps(POSES, 'residual', 0.075, 'a'))[0]
    assert ('arm', 'over_res') in io.log and io.log[-3:] == [('detach', 'a'), ('arm', 'up'), ('arm', 'rest')]
    io = FakeIo()
    assert runner(io).run(bin_steps(POSES, 'recyclable', PLAN, ['bin'], 0.075, 0.01, 'b'))[0]
    order = [e for e in io.log if e[0] in ('arm', 'attach', 'detach')]
    # into the compartment above the item, down, attach, straight up, only then swing
    assert order[order.index(('arm', 'over_rec')):order.index(('arm', 'bin')) + 2] == [
        ('arm', 'over_rec'), ('arm', 'above'), ('arm', 'hover'), ('attach', 'b'), ('arm', 'above'),
        ('arm', 'up'), ('arm', 'bin'), ('detach', 'b')]
    io = FakeIo()
    assert runner(io).run(ground_steps(PLAN, POSES, 0.075, 'c'))[0]
    assert io.log.index(('detach', 'c')) > io.log.index(('arm', 'hover'))


def test_no_step_starts_while_unsafe():
    io = FakeIo(unsafe={0, 1, 2})                      # someone near for the first 3 s
    ok, _ = runner(io).run(pick_steps(PLAN, POSES, ['show'], 0.075, 0.01, 'x'))
    assert ok and io.moving_while_unsafe == 0
    assert io.t >= 3.0


def test_interrupted_move_is_repeated_after_the_hold():
    io = FakeIo(interrupt_at=['hover'])
    ok, _ = runner(io).run(pick_steps(PLAN, POSES, ['show'], 0.075, 0.01, 'x'))
    assert ok and io.moving_while_unsafe == 0
    i = io.log.index(('interrupted', 'hover'))
    assert io.log[i + 1] == ('arm', 'hover')           # resumed at the same step
    assert io.log.count(('attach', 'x')) == 1


def test_hold_timeout_fails():
    io = FakeIo(unsafe=set(range(0, 100)))
    ok, msg = runner(io, hold=5.0).run(pick_steps(PLAN, POSES, ['show'], 0.075, 0.01, 'x'))
    assert (ok, msg) == (False, 'safety_hold_timeout') and io.log == []


@pytest.mark.parametrize('fail,msg', [({'hover'}, 'arm_failed (descend)'), ({'attach'}, 'attach_failed')])
def test_failures_are_reported(fail, msg):
    io = FakeIo(fail=fail)
    assert runner(io).run(pick_steps(PLAN, POSES, ['show'], 0.075, 0.01, 'x')) == (False, msg)


def test_cancel():
    io = FakeIo()
    calls = iter([False, False, True])
    assert runner(io).run(pick_steps(PLAN, POSES, ['show'], 0.075, 0.01, 'x'),
                          cancelled=lambda: next(calls, True)) == (False, 'cancelled')


def test_failure_after_release_still_counts_as_placed():
    """The item is in the compartment once detached; failing the move back to `up` doesn't undo that."""
    class FailLastUp(FakeIo):
        def move_arm(self, q, secs):
            if q[0] == 'up' and ('detach', 'a') in self.log:
                return 'failed'
            return super().move_arm(q, secs)
    io = FailLastUp()
    ok, msg = runner(io).run(store_steps(POSES, 'residual', 0.075, 'a'))
    assert ok and msg == 'released; then arm_failed (up)'


def test_every_walking_sequence_ends_in_rest():
    """The robot walks after store, put-back and bin: the arm must be in the walking pose."""
    for steps in (store_steps(POSES, 'residual', 0.075, 'a'), ground_steps(PLAN, POSES, 0.075, 'a'),
                  bin_steps(POSES, 'recyclable', PLAN, ['bin'], 0.075, 0.01, 'a')):
        assert steps[-1].value == ['rest'] and steps[-2].value == ['up']


def test_runner_tracks_the_held_item():
    """The node releases a held item after a failure (A2b: an item stayed attached)."""
    io = FakeIo(fail={'up'})
    r = runner(io)
    ok, _ = r.run(bin_steps(POSES, 'recyclable', PLAN, ['bin'], 0.075, 0.01, 'b')[3:])   # from above_item
    assert not ok and r.held == 'b'
    io = FakeIo()
    r = runner(io)
    assert r.run(bin_steps(POSES, 'recyclable', PLAN, ['bin'], 0.075, 0.01, 'b'))[0] and r.held is None


def test_store_with_a_planned_spot_lowers_the_item_with_the_jaws_closed():
    io = FakeIo()
    plan = {'above': ['above'], 'release': ['low']}
    assert runner(io).run(store_steps(POSES, 'recyclable', 0.075, 'a', plan))[0]
    order = [e for e in io.log if e[0] in ('arm', 'detach', 'fingers')]
    assert order == [('arm', 'up'), ('arm', 'above'), ('arm', 'low'), ('detach', 'a'),
                     ('arm', 'above'), ('arm', 'up'), ('arm', 'rest')]
