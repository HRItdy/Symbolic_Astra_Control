import dataclasses

import numpy as np
import pytest

from jev_policy.config import ACTION_DIM
from jev_policy.skills.base import GRIPPER_CLOSE, GRIPPER_OPEN, SkillStatus
from jev_policy.skills.primitives import (
    APPROACH_CLEARANCE,
    CloseGripper,
    Descend,
    Lift,
    MoveAbove,
    PlaceOn,
    make_skill,
)
from tests.fakes import make_state


def test_actions_are_well_formed_and_bounded():
    skill = MoveAbove("bowl_1")
    result = skill.step(make_state())
    assert result.action.shape == (ACTION_DIM,)
    assert np.all(np.abs(result.action) <= 1.0)


def test_move_above_drives_toward_the_object():
    state = make_state(eef_pos=(0.0, 0.0, 1.10))
    result = MoveAbove("bowl_1").step(state)
    # bowl is at x=+0.05, y=-0.10 relative to an eef at the origin.
    assert result.action[0] > 0
    assert result.action[1] < 0
    assert result.status is SkillStatus.RUNNING


def test_move_above_succeeds_at_the_hover_point():
    bowl = (0.05, -0.10, 0.92)
    state = make_state(eef_pos=(bowl[0], bowl[1], bowl[2] + APPROACH_CLEARANCE))
    assert MoveAbove("bowl_1").step(state).status is SkillStatus.SUCCESS


def test_descend_refuses_when_not_aligned():
    state = make_state(eef_pos=(0.5, 0.5, 1.10))
    result = Descend("bowl_1").step(state)
    assert result.status is SkillStatus.FAILED
    assert "drifted" in result.note


def test_close_gripper_closes_and_defers_the_grasp_check():
    """A fresh grasp does not register until it bears load, so CloseGripper
    must not fail on it -- Lift is the verifier."""
    skill = CloseGripper()
    state = make_state()
    for _ in range(skill.max_steps - 1):
        result = skill.step(state)
    assert result.status is SkillStatus.SUCCESS
    assert result.action[6] == GRIPPER_CLOSE


def test_lift_fails_when_the_object_does_not_rise():
    skill = Lift("bowl_1")
    # Already at the lift height, but the bowl stayed on the table.
    at_height = make_state(eef_pos=(0.0, 0.0, 1.10))
    skill.step(at_height)
    skill._goal_z = at_height.eef_pos[2]
    result = skill.step(at_height)
    assert result.status is SkillStatus.FAILED
    assert "rose only" in result.note


def test_lift_succeeds_when_the_object_comes_up_with_the_gripper():
    skill = Lift("bowl_1")
    start = make_state(eef_pos=(0.0, 0.0, 1.10), held="bowl_1", gripper_qpos=(0.005, -0.005))
    skill.step(start)
    skill._goal_z = start.eef_pos[2]
    risen = make_state(
        eef_pos=(0.0, 0.0, 1.10),
        objects={"bowl_1": (0.05, -0.10, 1.02), "plate_1": (-0.12, -0.05, 0.91)},
        held="bowl_1",
        gripper_qpos=(0.005, -0.005),
    )
    result = skill.step(risen)
    assert result.status is SkillStatus.SUCCESS


def test_place_on_requires_something_held():
    assert PlaceOn("plate_1").step(make_state()).status is SkillStatus.FAILED


def test_skills_time_out_instead_of_hanging():
    skill = MoveAbove("bowl_1")
    state = make_state()
    for _ in range(skill.max_steps + 1):
        result = skill.step(state)
    assert result.status is SkillStatus.FAILED
    assert "timed out" in result.note


def test_missing_target_fails_cleanly():
    result = MoveAbove("nonexistent").step(make_state())
    assert result.status is SkillStatus.FAILED


def test_unknown_skill_name_raises():
    with pytest.raises(KeyError):
        make_skill("teleport", "bowl_1")


def test_narrow_object_is_grasped_at_its_centre():
    from jev_policy.skills.primitives import GRIPPER_SPAN, grasp_point
    from tests.fakes import make_state as _ms

    obj = _ms().get("bowl_1")
    narrow = dataclasses.replace(obj, radius=0.015)
    point = grasp_point(narrow)
    assert np.allclose(point[:2], narrow.pos[:2])


def test_wide_object_is_grasped_at_its_rim():
    """A bowl wider than the gripper opening cannot be enclosed; the grasp must
    be offset by half a span so one finger goes inside it."""
    from jev_policy.skills.primitives import GRASP_AXIS, GRIPPER_SPAN, grasp_point
    from tests.fakes import make_state as _ms

    obj = _ms().get("bowl_1")
    wide = dataclasses.replace(obj, radius=0.047)
    point = grasp_point(wide)
    offset = point[:2] - wide.pos[:2]
    assert np.allclose(offset, GRASP_AXIS[:2] * (GRIPPER_SPAN / 2.0))
