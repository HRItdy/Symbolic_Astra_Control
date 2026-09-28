"""Jev-Omni planner tests, against a fake server -- no GPU, no weights."""

import numpy as np
import pytest

from jev_policy.config import PlannerConfig
from jev_policy.plugins.jev.omni import NO_TARGET, JevOmniPlanner, _compose_views
from tests.fakes import FakeOmniServer, make_state


def build(server, **kwargs):
    planner = JevOmniPlanner(PlannerConfig(backend="jev-omni", **kwargs))
    planner._predict = lambda s, q, o, img: (server({"state": s, "question": q, "options": o,
                                                     "has_image": img is not None}), 0.05)
    return planner


def state_with_views(**kwargs):
    state = make_state(**kwargs)
    views = {
        "agentview": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand": np.zeros((128, 128, 3), dtype=np.uint8),
    }
    return type(state)(**{**state.__dict__, "rgb": views})


def test_both_camera_views_are_sent_as_one_image():
    """predict() takes a single media item, so the two cameras must be composed."""
    png = _compose_views({
        "agentview": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand": np.zeros((128, 128, 3), dtype=np.uint8),
    })
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(png))
    assert img.width == 256 and img.height == 128


def test_composed_image_is_downscaled_past_the_width_cap():
    png = _compose_views({"agentview": np.zeros((600, 600, 3), dtype=np.uint8)}, max_width=512)
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(png)).width == 512


def test_the_image_reaches_the_model():
    server = FakeOmniServer()
    build(server).decide(state_with_views(), history=[])
    assert all(r["has_image"] for r in server.requests)


def test_prompt_omits_object_coordinates():
    """The model can see; feeding it poses would re-introduce privileged state."""
    server = FakeOmniServer()
    build(server).decide(state_with_views(), history=[])

    text = server.requests[0]["state"]
    assert "holding: nothing" in text
    assert "distance_from_gripper" not in text
    assert "offset(" not in text


def test_two_calls_per_decision_by_default():
    server = FakeOmniServer()
    build(server).decide(state_with_views(), history=[])
    assert len(server.requests) == 2


def test_done_probe_is_opt_in_and_adds_a_third_call():
    server = FakeOmniServer()
    decision = build(server, ask_done=True).decide(state_with_views(), history=[])
    assert len(server.requests) == 3
    assert server.requests[2]["options"] == ["yes", "no"]
    assert decision.task_complete_p > 0


def test_target_options_are_truncated_to_the_models_usable_width():
    """LIBERO registers 23 objects in one scene; the card only claims <= 20."""
    objects = {f"obj_{i}": (0.01 * i, 0.0, 0.9) for i in range(30)}
    server = FakeOmniServer()
    planner = build(server, max_options=20)
    planner.decide(state_with_views(objects=objects, interest=("obj_0",)), history=[])

    target_options = server.requests[1]["options"]
    assert len(target_options) <= 20
    assert target_options[-1] == NO_TARGET
    assert target_options[0] == "obj_0"  # goal object ranked first


def test_none_target_becomes_python_none():
    server = FakeOmniServer(answers=[(0, 0.9), (0, 0.9)])
    planner = build(server)
    planner._target_options = lambda state, exclude=None: [NO_TARGET, "bowl_1"]
    assert planner.decide(state_with_views(), history=[]).target is None


def test_low_confidence_is_flagged():
    decision = build(FakeOmniServer(answers=[(0, 0.2)]), min_confidence=0.45).decide(
        state_with_views(), history=[]
    )
    assert "low confidence" in decision.note


def test_cache_keys_on_poses_not_just_text():
    """Poses are absent from a vision prompt, so identical text must not alias
    two different scenes."""
    a = state_with_views()
    b = state_with_views(objects={"bowl_1": (0.30, -0.10, 0.92), "plate_1": (-0.12, -0.05, 0.91)})
    assert a.to_proprioception_text() == b.to_proprioception_text()
    assert a.cache_key() != b.cache_key()


def test_identical_scenes_hit_the_cache():
    server = FakeOmniServer()
    planner = build(server, cache_decisions=True)
    planner.decide(state_with_views(), history=[])
    second = planner.decide(state_with_views(), history=[])
    assert len(server.requests) == 2  # not 4
    assert second.cached


def test_unreachable_server_raises_a_pointed_error():
    planner = JevOmniPlanner(PlannerConfig(backend="jev-omni", port=1, request_timeout_s=0.5))
    with pytest.raises(RuntimeError, match="serve_jev_omni"):
        planner.decide(state_with_views(), history=[])


def test_targetless_skills_skip_the_target_question():
    """close_gripper/open_gripper/retreat act where the hand already is;
    asking costs a round trip and invites a wrong answer."""
    server = FakeOmniServer(answers=[(2, 0.9)])  # index 2 == close_gripper
    decision = build(server).decide(state_with_views(), history=[])
    assert decision.skill == "close_gripper"
    assert decision.target is None
    assert len(server.requests) == 1


def test_place_on_excludes_the_held_object():
    """Otherwise the model answers with the object it is holding and the place
    becomes a no-op onto itself -- observed on task 0 before this fix."""
    server = FakeOmniServer(answers=[(4, 0.9), (0, 0.9)])  # index 4 == place_on
    held = state_with_views(held="bowl_1", gripper_qpos=(0.005, -0.005))
    decision = build(server).decide(held, history=[])

    assert decision.skill == "place_on"
    options = server.requests[1]["options"]
    assert "bowl_1" not in options
    assert "plate_1" in options
    assert decision.target == "plate_1"


def test_place_onto_the_held_object_fails_loudly():
    from jev_policy.skills.base import SkillStatus
    from jev_policy.skills.primitives import PlaceOn

    state = make_state(held="bowl_1", gripper_qpos=(0.005, -0.005))
    result = PlaceOn("bowl_1").step(state)
    assert result.status is SkillStatus.FAILED
    assert "onto itself" in result.note


def test_cache_is_not_consulted_across_differing_histories():
    """Same scene, different history must be a cache miss -- otherwise a
    no-progress decision is re-served forever."""
    from tests.fakes import FakeOmniServer

    server = FakeOmniServer()
    planner = build(server, cache_decisions=True)
    state = state_with_views()

    planner.decide(state, history=["PlaceOn(plate_1) -> ok"])
    calls_after_first = len(server.requests)
    planner.decide(state, history=["PlaceOn(plate_1) -> already done 3x, it changes nothing now"])
    assert len(server.requests) > calls_after_first
