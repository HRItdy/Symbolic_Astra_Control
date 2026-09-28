"""Astra direct-control tests, against a fake OpenAI client -- no API calls."""

import json

import numpy as np
import pytest

from jev_policy.config import ACTION_DIM, PlannerConfig
from jev_policy.control.astra import AstraController
from tests.fakes import make_state


class FakeResponses:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        payload = self.payloads[min(len(self.requests) - 1, len(self.payloads) - 1)]

        class Usage:
            input_tokens, output_tokens = 1000, 500

        class Response:
            output_text = json.dumps(payload)
            usage = Usage()

        return Response()


class FakeClient:
    def __init__(self, payloads):
        self.responses = FakeResponses(payloads)


def build(payloads, **kwargs):
    controller = AstraController(PlannerConfig(backend="astra", **kwargs))
    controller._client = FakeClient(payloads)
    return controller


def state_with_views(**kwargs):
    state = make_state(**kwargs)
    views = {
        "agentview": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand": np.zeros((128, 128, 3), dtype=np.uint8),
    }
    return type(state)(**{**state.__dict__, "rgb": views})


def test_delta_mode_passes_actions_through_to_the_simulator():
    payload = {"reasoning": "descend", "actions": [[0.1, 0.0, -0.3, 0, 0, 0, -1.0]] * 4}
    chunk = build([payload]).act(state_with_views(), [])

    assert len(chunk.actions) == 4
    assert chunk.actions[0].shape == (ACTION_DIM,)
    assert chunk.note == "descend"


def test_out_of_range_values_are_clipped_not_trusted():
    """The model was never trained to emit these, so they are validated."""
    payload = {"reasoning": "", "actions": [[99.0, -99.0, 0, 0, 0, 0, 5.0]]}
    action = build([payload]).act(state_with_views(), []).actions[0]
    assert np.all(action <= 1.0) and np.all(action >= -1.0)


def test_malformed_rows_are_dropped():
    payload = {"reasoning": "", "actions": [[0.1, 0.2], [0.0] * 7, "garbage", [float("nan")] * 7]}
    actions = build([payload]).act(state_with_views(), []).actions
    assert len(actions) == 1


def test_chunk_is_truncated_to_the_configured_length():
    payload = {"reasoning": "", "actions": [[0.0] * 7] * 100}
    assert len(build([payload], chunk_steps=10).act(state_with_views(), []).actions) == 10


def test_unparseable_response_yields_an_empty_chunk_not_a_crash():
    controller = build([{}])
    controller._client.responses.create = lambda **k: type(
        "R", (), {"output_text": "not json", "usage": None}
    )()
    chunk = controller.act(state_with_views(), [])
    assert chunk.actions == []
    assert "unparseable" in chunk.note


def test_waypoint_mode_expands_one_target_into_a_servoed_burst():
    payload = {"reasoning": "hover", "target_xyz": [0.05, -0.10, 1.02],
               "gripper": "open", "steps": 12}
    chunk = build([payload], control_mode="waypoint").act(state_with_views(), [])

    assert len(chunk.actions) == 12
    # Start is (0,0,1.10); the target is right, forward-negative and below.
    assert chunk.actions[0][0] > 0
    assert chunk.actions[0][2] < 0
    assert chunk.actions[0][6] == -1.0  # open


def test_waypoint_gripper_closed_is_encoded():
    payload = {"reasoning": "", "target_xyz": [0.0, 0.0, 1.0], "gripper": "closed", "steps": 3}
    chunk = build([payload], control_mode="waypoint").act(state_with_views(), [])
    assert all(a[6] == 1.0 for a in chunk.actions)


def test_waypoint_with_a_bad_target_returns_nothing():
    payload = {"reasoning": "", "target_xyz": [0.0, 0.0], "gripper": "open", "steps": 5}
    assert build([payload], control_mode="waypoint").act(state_with_views(), []).actions == []


def test_the_image_and_robot_pose_are_both_sent():
    controller = build([{"reasoning": "", "actions": [[0.0] * 7]}])
    controller.act(state_with_views(), ["step 10: commanded 20 actions"])

    content = controller._client.responses.requests[0]["input"][1]["content"]
    kinds = [c["type"] for c in content]
    assert "input_image" in kinds
    text = " ".join(c["text"] for c in content if c["type"] == "input_text")
    assert "gripper_position" in text
    assert "WHAT YOU DID SO FAR" in text


def test_cost_is_tracked():
    controller = build([{"reasoning": "", "actions": [[0.0] * 7]}])
    controller.act(state_with_views(), [])
    controller.act(state_with_views(), [])

    stats = controller.stats
    assert stats["calls"] == 2
    assert stats["input_tokens"] == 2000
    assert stats["output_tokens"] == 1000
    # 2000 in @ $10/M + 1000 out @ $50/M = $0.02 + $0.05
    assert stats["estimated_cost_usd"] == pytest.approx(0.07, abs=1e-6)


def test_reasoning_effort_is_forwarded():
    controller = build([{"reasoning": "", "actions": [[0.0] * 7]}], reasoning_effort="medium")
    controller.act(state_with_views(), [])
    assert controller._client.responses.requests[0]["reasoning"]["effort"] == "medium"


def test_object_poses_are_withheld_by_default():
    """Privileged information: off unless explicitly asked for."""
    controller = build([{"reasoning": "", "actions": [[0.0] * 7]}])
    controller.act(state_with_views(), [])
    text = " ".join(
        c["text"]
        for c in controller._client.responses.requests[0]["input"][1]["content"]
        if c["type"] == "input_text"
    )
    assert "OBJECT POSITIONS" not in text


def test_object_poses_can_be_supplied_as_an_ablation():
    controller = build([{"reasoning": "", "actions": [[0.0] * 7]}], include_object_poses=True)
    controller.act(state_with_views(), [])
    text = " ".join(
        c["text"]
        for c in controller._client.responses.requests[0]["input"][1]["content"]
        if c["type"] == "input_text"
    )
    assert "OBJECT POSITIONS" in text
    assert "bowl_1" in text
    assert "named by the task" in text
