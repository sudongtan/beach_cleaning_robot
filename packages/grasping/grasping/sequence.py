"""Pick / place step sequences and the runner that executes them under the arm safety rule,
as plain Python (no ROS). The node supplies an `io` object; tests use a fake one.

Every sequence after which the robot walks ends in `rest`, the walking pose from A0: walking
with the arm at `up` fell in the wet-zone test (A2b, 2026-10-31). Pick ends at the camera-view
pose (the robot stands still to classify, then stores).

io interface:
  move_arm(q, secs) -> 'ok' | 'interrupted' | 'failed'   ('interrupted': stopped for safety)
  fingers(value, secs) -> bool
  attach(object_id) -> bool, detach(object_id) -> bool   (attach mode: /sim/attach, /sim/detach)
  safety_ok() -> bool                                    (/safety/state == ok)
  sleep(seconds), now() -> seconds
"""

from dataclasses import dataclass


@dataclass
class Step:
    kind: str           # arm, fingers, attach, detach
    value: object       # joint list, finger value or object id
    phase: str          # feedback text


def pick_steps(plan, poses, show_pose, finger_open, finger_closed, object_id):
    """Top-down attach-mode pick ending with the item held in front of the camera."""
    return [Step('arm', poses['up']['arm'], 'up'),
            Step('fingers', finger_open, 'open'),
            Step('arm', plan['above'], 'above'),
            Step('arm', plan['hover'], 'descend'),
            Step('attach', object_id, 'attach'),
            Step('fingers', finger_closed, 'close'),
            Step('arm', plan['lift'], 'lift'),
            Step('arm', show_pose, 'show')]


def store_steps(poses, bin_id, finger_open, object_id, plan=None):
    """Release the held item into compartment bin_id: lowered to a planned spot (plan from
    arm.plan_store; the jaws stay closed, the item hangs below them), or without a plan from the
    fixed over-compartment pose."""
    if plan:
        return [Step('arm', poses['up']['arm'], 'up'),
                Step('arm', poses[f'over_{bin_id}']['arm'], f'over_{bin_id}'),
                Step('arm', plan['above'], 'above_spot'),
                Step('arm', plan['release'], 'lower'),
                Step('detach', object_id, 'release'),
                Step('arm', plan['above'], 'above_spot'),
                Step('arm', poses['up']['arm'], 'up'),
                Step('arm', poses['rest']['arm'], 'rest')]
    return [Step('arm', poses['up']['arm'], 'up'),
            Step('arm', poses[f'over_{bin_id}']['arm'], f'over_{bin_id}'),
            Step('fingers', finger_open, 'open'),
            Step('detach', object_id, 'release'),
            Step('arm', poses['up']['arm'], 'up'),
            Step('arm', poses['rest']['arm'], 'rest')]


def ground_steps(plan, poses, finger_open, object_id):
    """Put the held item back where it was picked (the reverse of the pick)."""
    return [Step('arm', poses['up']['arm'], 'up'),
            Step('arm', plan['above'], 'above'),
            Step('arm', plan['hover'], 'descend'),
            Step('fingers', finger_open, 'open'),
            Step('detach', object_id, 'release'),
            Step('arm', plan['above'], 'above'),
            Step('arm', poses['up']['arm'], 'up'),
            Step('arm', poses['rest']['arm'], 'rest')]


def bin_steps(poses, bin_id, plan, release_q, finger_open, finger_closed, object_id):
    """Take item object_id out of compartment bin_id and release it over the bin (one item).
    plan (grasping.arm.plan_unload): jaws hover just above the item, then lift straight up until
    it clears the rim. Attaching from above the rim instead (the item ~23 cm below the jaws)
    dragged it through the basket wall (A2b a2b_bin2)."""
    return [Step('arm', poses['up']['arm'], 'up'),
            Step('fingers', finger_closed, 'close'),          # jaws closed inside the compartment
            Step('arm', poses[f'over_{bin_id}']['arm'], f'over_{bin_id}'),
            Step('arm', plan['above'], 'above_item'),
            Step('arm', plan['hover'], 'descend'),
            Step('attach', object_id, 'attach'),
            Step('arm', plan['above'], 'lift'),
            Step('arm', poses['up']['arm'], 'up'),
            Step('arm', release_q, 'over_bin'),
            Step('fingers', finger_open, 'open'),
            Step('detach', object_id, 'release'),
            Step('fingers', finger_closed, 'close'),
            Step('arm', poses['up']['arm'], 'up'),
            Step('arm', poses['rest']['arm'], 'rest')]


def recovery_steps(poses):
    """Back to the walking pose after a failed sequence (via `up`, like every named move)."""
    return [Step('arm', poses['up']['arm'], 'up'), Step('arm', poses['rest']['arm'], 'rest')]


class Runner:

    def __init__(self, io, move_time, finger_time, hold_timeout, poll=0.1):
        self.io, self.move_time, self.finger_time = io, move_time, finger_time
        self.hold_timeout, self.poll = hold_timeout, poll
        self.held = None                # object attached by the last attach step, until its detach

    def wait_safe(self, feedback, phase):
        """Hold until /safety/state is ok. False if that takes longer than the hold timeout."""
        if self.io.safety_ok():
            return True
        feedback(f'hold ({phase})')
        start = self.io.now()
        while not self.io.safety_ok():
            if self.io.now() - start > self.hold_timeout:
                return False
            self.io.sleep(self.poll)
        return True

    def run(self, steps, feedback=lambda phase: None, cancelled=lambda: False):
        """Execute steps; returns (success, message). A move stopped for safety is repeated
        once the state is ok again. Every step starts only while the state is ok. Once the item
        has been released (detach succeeded) it is where it was meant to go, so a later failure
        (e.g. the move back to `up`) is reported as success with a note."""
        released = False
        for step in steps:
            while True:
                if cancelled():
                    return False, 'cancelled'
                if not self.wait_safe(feedback, step.phase):
                    return False, 'safety_hold_timeout'
                feedback(step.phase)
                if step.kind == 'arm':
                    r = self.io.move_arm(step.value, self.move_time)
                    if r == 'interrupted':
                        continue
                    if r != 'ok':
                        if released:
                            return True, f'released; then arm_failed ({step.phase})'
                        return False, f'arm_failed ({step.phase})'
                elif step.kind == 'fingers':
                    if not self.io.fingers(step.value, self.finger_time):
                        return False, f'fingers_failed ({step.phase})'
                elif step.kind == 'attach':
                    if not self.io.attach(step.value):
                        return False, 'attach_failed'
                    self.held = step.value
                elif step.kind == 'detach':
                    if not self.io.detach(step.value):
                        return False, 'detach_failed'
                    self.held, released = None, True
                else:
                    raise ValueError(step.kind)
                break
        return True, 'ok'
