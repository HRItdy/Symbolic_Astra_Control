"""The skill library Jev chooses from.

Each skill is a short closed-loop behaviour with an explicit success test. The
planner never sees coordinates -- it names a skill and an object, and the code
here resolves that to motion. Keeping the vocabulary small and each skill
independently testable is what makes the decision layer's job tractable.
"""

from __future__ import annotations

import numpy as np

from jev_policy.perception.scene_state import SceneState
from jev_policy.skills.base import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    Skill,
    SkillResult,
    SkillStatus,
    hold_action,
)
from jev_policy.skills.controller import position_action, reached

# Height above an object to hover before descending, metres.
APPROACH_CLEARANCE = 0.10
# Measured on the LIBERO Panda: the finger pads separate along world +y and are
# 0.084m apart when open, and the eef site sits ~3mm below the pads.
GRIPPER_SPAN = 0.084
GRASP_AXIS = np.array([0.0, 1.0, 0.0])
# Height above an object's body origin to place the pads before closing.
GRASP_Z_OFFSET = 0.015
# How far above the table to lift a grasped object.
LIFT_HEIGHT = 0.12
# Position tolerance for "arrived", metres.
REACH_TOL = 0.015
# Control steps to hold a gripper command so the fingers actually close/open.
GRIPPER_SETTLE_STEPS = 12


def grasp_point(obj) -> np.ndarray:
    """Where the eef must be to close on `obj`.

    An object narrower than the gripper's opening is grasped at its centre. A
    wider one -- a bowl or plate, whose outer diameter exceeds the Panda's
    0.084m opening -- cannot be enclosed at all, so we aim half a gripper-span
    off centre along the finger axis. That puts one finger inside the bowl and
    one outside, and closing pinches the rim. Aiming at the centre instead just
    rests both fingers on the rim and stalls the descent, which is exactly what
    the untuned version did.
    """
    xy = obj.pos[:2].astype(float).copy()
    if 2.0 * obj.radius > GRIPPER_SPAN * 0.95:
        xy = xy + GRASP_AXIS[:2] * (GRIPPER_SPAN / 2.0)
    return np.array([xy[0], xy[1], obj.pos[2] + GRASP_Z_OFFSET])


class MoveAbove(Skill):
    """Hover over an object, keeping the current gripper state.

    Always the first half of a pick: descending from directly above avoids
    knocking the target over on approach.
    """

    max_steps = 90

    def _step(self, state: SceneState) -> SkillResult:
        if self.target not in state.objects:
            return SkillResult(hold_action(), SkillStatus.FAILED, f"no object {self.target}")

        obj = state.get(self.target)
        gripper = GRIPPER_CLOSE if state.held_object else GRIPPER_OPEN
        # Hover over the grasp point, not the centre, so the descent is straight
        # down and does not clip the rim on the way in.
        point = grasp_point(obj)
        goal = np.array([point[0], point[1], obj.pos[2] + APPROACH_CLEARANCE])

        if reached(state.eef_pos, goal, REACH_TOL):
            return SkillResult(hold_action(gripper), SkillStatus.SUCCESS, "above target")
        return SkillResult(position_action(state.eef_pos, goal, gripper), SkillStatus.RUNNING)


class Descend(Skill):
    """Lower onto the target until the gripper straddles it."""

    max_steps = 60

    def _step(self, state: SceneState) -> SkillResult:
        if self.target not in state.objects:
            return SkillResult(hold_action(), SkillStatus.FAILED, f"no object {self.target}")

        obj = state.get(self.target)
        goal = grasp_point(obj)

        # Bail out if we drifted off the grasp point horizontally -- descending
        # now would push the object away instead of grasping it.
        if float(np.linalg.norm(goal[:2] - state.eef_pos[:2])) > 0.05:
            return SkillResult(hold_action(GRIPPER_OPEN), SkillStatus.FAILED, "drifted off target")

        if reached(state.eef_pos, goal, REACH_TOL):
            return SkillResult(hold_action(GRIPPER_OPEN), SkillStatus.SUCCESS, "at grasp height")
        return SkillResult(
            position_action(state.eef_pos, goal, GRIPPER_OPEN, kp=0.8), SkillStatus.RUNNING
        )


class CloseGripper(Skill):
    """Close the fingers.

    Deliberately does *not* assert a grasp formed. robosuite's `_check_grasp`
    needs the contact to bear load, so it still reads False in the moments after
    the fingers close on a rim and only flips true once the object is lifted.
    Failing here would abort every correct grasp; `Lift` is where a missed grasp
    is actually detectable, so the check lives there.
    """

    max_steps = GRIPPER_SETTLE_STEPS + 4

    def _step(self, state: SceneState) -> SkillResult:
        if self.elapsed < GRIPPER_SETTLE_STEPS:
            return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.RUNNING)
        return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.SUCCESS, "gripper closed")


class OpenGripper(Skill):
    max_steps = GRIPPER_SETTLE_STEPS + 4

    def _step(self, state: SceneState) -> SkillResult:
        if self.elapsed < GRIPPER_SETTLE_STEPS:
            return SkillResult(hold_action(GRIPPER_OPEN), SkillStatus.RUNNING)
        return SkillResult(hold_action(GRIPPER_OPEN), SkillStatus.SUCCESS, "released")


class Lift(Skill):
    """Raise straight up, and verify the object came with us.

    This is the grasp verifier: the object's own z must rise, not just the
    gripper's. That distinguishes a real grasp from fingers that closed on air.
    """

    max_steps = 70
    #: The target must rise at least this far to count as grasped.
    MIN_OBJECT_RISE = 0.04

    def __init__(self, target: str | None = None, height: float = LIFT_HEIGHT) -> None:
        super().__init__(target)
        self.height = height
        self._goal_z: float | None = None
        self._start_obj_z: float | None = None

    def _step(self, state: SceneState) -> SkillResult:
        if self._goal_z is None:
            self._goal_z = state.eef_pos[2] + self.height
            if self.target in state.objects:
                self._start_obj_z = float(state.get(self.target).pos[2])

        goal = np.array([state.eef_pos[0], state.eef_pos[1], self._goal_z])

        if not reached(state.eef_pos, goal, REACH_TOL):
            return SkillResult(
                position_action(state.eef_pos, goal, GRIPPER_CLOSE), SkillStatus.RUNNING
            )

        # At height: did anything come up with us?
        if self._start_obj_z is not None:
            rise = float(state.get(self.target).pos[2]) - self._start_obj_z
            if rise < self.MIN_OBJECT_RISE:
                return SkillResult(
                    hold_action(GRIPPER_CLOSE), SkillStatus.FAILED, f"object rose only {rise:.3f}m"
                )
            return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.SUCCESS, f"lifted {rise:.3f}m")

        if state.held_object is None:
            return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.FAILED, "nothing held")
        return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.SUCCESS, "lifted")


class PlaceOn(Skill):
    """Carry the held object over the target and lower it for release.

    The gripper is not the payload. A bowl is held by its rim, roughly half a
    gripper-span off its own centre, so driving the *eef* to the target leaves
    the bowl offset by that much -- enough to miss a plate entirely. We measure
    the object's current offset from the eef and subtract it, so the object,
    not the hand, ends up over the target.
    """

    max_steps = 140
    #: Clearance above the destination surface at release.
    RELEASE_CLEARANCE = 0.04

    def _step(self, state: SceneState) -> SkillResult:
        held = state.held_object
        if held is None:
            return SkillResult(hold_action(GRIPPER_OPEN), SkillStatus.FAILED, "nothing held")
        if self.target not in state.objects:
            return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.FAILED, f"no {self.target}")

        if self.target == held:
            return SkillResult(
                hold_action(GRIPPER_CLOSE), SkillStatus.FAILED, "cannot place an object onto itself"
            )

        target = state.get(self.target)
        payload = state.get(held)

        # Where the eef must be for the payload to sit over the target.
        grasp_offset = payload.pos[:2] - state.eef_pos[:2]
        goal_xy = target.pos[:2] - grasp_offset
        # Clear the destination by the payload's own height plus a margin.
        goal_z = target.pos[2] + self.RELEASE_CLEARANCE + (state.eef_pos[2] - payload.pos[2])
        goal = np.array([goal_xy[0], goal_xy[1], goal_z])

        if reached(state.eef_pos, goal, 0.02):
            return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.SUCCESS, "over target")
        return SkillResult(
            position_action(state.eef_pos, goal, GRIPPER_CLOSE), SkillStatus.RUNNING
        )


class PullOpen(Skill):
    """Open a drawer/door: grasp the handle and pull toward the robot (-y)."""

    max_steps = 110
    PULL_DISTANCE = 0.18

    def __init__(self, target: str | None = None) -> None:
        super().__init__(target)
        self._anchor: np.ndarray | None = None

    def _step(self, state: SceneState) -> SkillResult:
        if self.target not in state.objects:
            return SkillResult(hold_action(), SkillStatus.FAILED, f"no {self.target}")

        obj = state.get(self.target)
        if obj.is_open:
            return SkillResult(hold_action(GRIPPER_OPEN), SkillStatus.SUCCESS, "already open")

        if self._anchor is None:
            self._anchor = state.eef_pos.copy()

        goal = self._anchor - np.array([0.0, self.PULL_DISTANCE, 0.0])
        if reached(state.eef_pos, goal, 0.02):
            status = SkillStatus.SUCCESS if obj.is_open else SkillStatus.FAILED
            return SkillResult(hold_action(GRIPPER_CLOSE), status, f"pulled, open={obj.is_open}")
        return SkillResult(
            position_action(state.eef_pos, goal, GRIPPER_CLOSE, kp=0.9), SkillStatus.RUNNING
        )


class PushClosed(Skill):
    """Close a drawer/door: push away from the robot (+y)."""

    max_steps = 110
    PUSH_DISTANCE = 0.18

    def __init__(self, target: str | None = None) -> None:
        super().__init__(target)
        self._anchor: np.ndarray | None = None

    def _step(self, state: SceneState) -> SkillResult:
        if self.target not in state.objects:
            return SkillResult(hold_action(), SkillStatus.FAILED, f"no {self.target}")

        obj = state.get(self.target)
        if obj.is_open is False:
            return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.SUCCESS, "already closed")

        if self._anchor is None:
            self._anchor = state.eef_pos.copy()

        goal = self._anchor + np.array([0.0, self.PUSH_DISTANCE, 0.0])
        if reached(state.eef_pos, goal, 0.02):
            return SkillResult(hold_action(GRIPPER_CLOSE), SkillStatus.SUCCESS, "pushed")
        return SkillResult(
            position_action(state.eef_pos, goal, GRIPPER_CLOSE, kp=0.9), SkillStatus.RUNNING
        )


class Retreat(Skill):
    """Back off upward. The recovery behaviour when a decision is low-confidence
    or a skill failed -- it restores a clean, reachable state to re-plan from."""

    max_steps = 40

    def _step(self, state: SceneState) -> SkillResult:
        gripper = GRIPPER_CLOSE if state.held_object else GRIPPER_OPEN
        goal = state.eef_pos + np.array([0.0, 0.0, 0.08])
        if self.elapsed >= 20:
            return SkillResult(hold_action(gripper), SkillStatus.SUCCESS, "retreated")
        return SkillResult(position_action(state.eef_pos, goal, gripper), SkillStatus.RUNNING)


#: The decision vocabulary handed to Jev. Descriptions are sent verbatim as
#: `Choice` criteria, so they are prompt surface -- edit them with that in mind.
SKILL_LIBRARY: dict[str, tuple[type[Skill], str]] = {
    "move_above": (MoveAbove, "Move the empty gripper to hover directly above the target object."),
    "descend": (Descend, "Lower the open gripper onto the target object, ready to grasp it."),
    "close_gripper": (CloseGripper, "Close the gripper to grasp whatever it is currently around."),
    "lift": (Lift, "Lift the currently grasped object straight up off the surface."),
    "place_on": (PlaceOn, "Carry the grasped object above the target and position it for release."),
    "open_gripper": (OpenGripper, "Open the gripper to release the object it is holding."),
    "pull_open": (PullOpen, "Grasp the target's handle and pull it open toward the robot."),
    "push_closed": (PushClosed, "Push the target closed, away from the robot."),
    "retreat": (Retreat, "Back the gripper upward and away to recover from a failed attempt."),
}


def make_skill(name: str, target: str | None) -> Skill:
    if name not in SKILL_LIBRARY:
        raise KeyError(f"unknown skill {name!r}; expected one of {sorted(SKILL_LIBRARY)}")
    return SKILL_LIBRARY[name][0](target)
