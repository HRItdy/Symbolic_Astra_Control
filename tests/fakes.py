"""Test doubles: a scene builder and a fake Jev client.

These let the whole stack -- scene text, skills, runner, planner -- be tested
without MuJoCo, without LIBERO, and without spending API calls.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from jev_policy.perception.scene_state import ObjectObs, SceneState


def make_state(
    eef_pos=(0.0, 0.0, 1.10),
    gripper_qpos=(0.04, -0.04),
    objects=None,
    held=None,
    task="pick up the bowl and place it on the plate",
    interest=("bowl_1", "plate_1"),
    step=0,
) -> SceneState:
    if objects is None:
        objects = {
            "bowl_1": (0.05, -0.10, 0.92),
            "plate_1": (-0.12, -0.05, 0.91),
        }
    obs = {}
    for name, pos in objects.items():
        obs[name] = ObjectObs(
            name=name,
            pos=np.array(pos, dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            is_fixture=False,
            grasped=(name == held),
        )
    return SceneState(
        task=task,
        eef_pos=np.array(eef_pos, dtype=float),
        eef_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_qpos=np.array(gripper_qpos, dtype=float),
        objects=obs,
        objects_of_interest=interest,
        step=step,
    )


@dataclasses.dataclass
class FakeAnswer:
    choice: str = ""
    probabilities: dict = dataclasses.field(default_factory=dict)
    confidence: float = 1.0
    noul: float = 0.0


@dataclasses.dataclass
class FakeResponse:
    choices: dict
    nouls: dict
    model: str = "jev-1.13.0"


class FakeTypeSafeClient:
    """Stands in for TypeSafeClient, and records what was asked."""

    def __init__(self, skill="move_above", target="bowl_1", confidence=0.9, done_p=0.02):
        self.skill = skill
        self.target = target
        self.confidence = confidence
        self.done_p = done_p
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        return FakeResponse(
            choices={
                "skill": FakeAnswer(choice=self.skill, confidence=self.confidence),
                "target": FakeAnswer(choice=self.target, confidence=self.confidence),
            },
            nouls={"task_complete": FakeAnswer(noul=self.done_p)},
        )

    def close(self):
        pass


class FakeOmniServer:
    """Stands in for the Jev-Omni HTTP server, recording each request."""

    def __init__(self, answers=None):
        # answers: list of (prediction_index, confidence) consumed in order,
        # cycling on the last entry.
        self.answers = answers or [(0, 0.9)]
        self.requests = []

    def __call__(self, payload):
        self.requests.append(payload)
        idx, conf = self.answers[min(len(self.requests) - 1, len(self.answers) - 1)]
        options = payload["options"]
        idx = min(idx, len(options) - 1)
        probs = {o: (conf if i == idx else (1 - conf) / max(1, len(options) - 1))
                 for i, o in enumerate(options)}
        return {
            "prediction": options[idx],
            "prediction_index": idx,
            "confidence": conf,
            "probabilities": probs,
        }
