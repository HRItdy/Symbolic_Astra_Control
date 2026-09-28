"""Controller interface: the model drives the robot itself.

A `Planner` picks a named skill and hand-written code produces the motion. A
`Controller` skips that layer -- the model emits the robot commands directly,
and `skills/` is not involved at all.
"""

from __future__ import annotations

import abc
import dataclasses

import numpy as np

from jev_policy.perception.scene_state import SceneState


@dataclasses.dataclass
class ControlChunk:
    """A short burst of actions to execute before observing again.

    Chunking is what makes direct control possible at all: LIBERO steps at 20Hz
    and a frontier model answers in seconds, so one call has to cover many
    control steps. Every action is a LIBERO OSC_POSE command --
    [dx, dy, dz, drx, dry, drz, gripper], each in [-1, 1].
    """

    actions: list[np.ndarray]
    latency_s: float = 0.0
    note: str = ""
    #: Token accounting, so cost per episode is visible rather than a surprise.
    input_tokens: int = 0
    output_tokens: int = 0


class Controller(abc.ABC):
    @abc.abstractmethod
    def act(self, state: SceneState, history: list[str]) -> ControlChunk: ...

    def reset(self) -> None: ...

    @property
    def stats(self) -> dict:
        return {}
