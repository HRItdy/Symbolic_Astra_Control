"""Hosted Jev planner tests.

The plugin is off by default; these import it directly, so the code stays
guarded even while the backend is unregistered.
"""

from jev_policy.config import PlannerConfig
from jev_policy.plugins.jev.planner import NO_TARGET, JevPlanner
from tests.fakes import FakeTypeSafeClient, make_state


def build(client, **kwargs):
    planner = JevPlanner(PlannerConfig(backend="jev", **kwargs))
    planner._client = client
    return planner


def test_target_options_come_from_the_live_scene():
    """The point of typed output: an invalid target is unrepresentable."""
    client = FakeTypeSafeClient()
    planner = build(client)
    planner.decide(make_state(), history=[])

    criteria = client.calls[0]["questions"]["target"].criteria
    assert set(criteria) == {"bowl_1", "plate_1", NO_TARGET}
    assert criteria["bowl_1"] == "named by the task goal"


def test_state_payload_carries_task_scene_and_history():
    client = FakeTypeSafeClient()
    planner = build(client)
    planner.decide(make_state(), history=["MoveAbove(bowl_1) -> ok"])

    payload = client.calls[0]["state"]
    assert "pick up the bowl" in payload["task"]
    assert "holding: nothing" in payload["scene"]
    assert "MoveAbove(bowl_1) -> ok" in payload["completed_so_far"]


def test_history_is_truncated_to_the_recent_tail():
    client = FakeTypeSafeClient()
    planner = build(client)
    planner.decide(make_state(), history=[f"step {i}" for i in range(50)])

    assert client.calls[0]["state"]["completed_so_far"].count("\n") == 7


def test_none_target_becomes_python_none():
    planner = build(FakeTypeSafeClient(skill="close_gripper", target=NO_TARGET))
    assert planner.decide(make_state(), history=[]).target is None


def test_confidence_and_done_probability_are_surfaced():
    planner = build(FakeTypeSafeClient(confidence=0.31, done_p=0.77))
    decision = planner.decide(make_state(), history=[])
    assert decision.confidence == 0.31
    assert decision.task_complete_p == 0.77
    assert "low confidence" in decision.note


def test_identical_states_hit_the_cache():
    client = FakeTypeSafeClient()
    planner = build(client, cache_decisions=True)

    planner.decide(make_state(), history=[])
    second = planner.decide(make_state(), history=[])

    assert len(client.calls) == 1
    assert second.cached
    assert planner.stats["cache_hits"] == 1


def test_changed_scene_misses_the_cache():
    client = FakeTypeSafeClient()
    planner = build(client, cache_decisions=True)

    planner.decide(make_state(), history=[])
    planner.decide(make_state(held="bowl_1", gripper_qpos=(0.005, -0.005)), history=[])

    assert len(client.calls) == 2


def test_stats_track_latency_and_confidence():
    planner = build(FakeTypeSafeClient(confidence=0.8), cache_decisions=False)
    planner.decide(make_state(), history=[])
    planner.decide(make_state(), history=[])

    stats = planner.stats
    assert stats["calls"] == 2
    assert stats["mean_confidence"] == 0.8
    assert stats["mean_latency_s"] >= 0.0


