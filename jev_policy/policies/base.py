"""Planner interface: scene state in, next skill out."""

from __future__ import annotations

import abc
import dataclasses

from jev_policy.perception.scene_state import SceneState


@dataclasses.dataclass(frozen=True)
class Decision:
    skill: str
    target: str | None
    #: Calibrated confidence in the skill choice, 0-1. Heuristic planners report
    #: 1.0; only Jev produces a meaningful value here.
    confidence: float = 1.0
    #: P(task already complete), for diagnostics. Success is always judged by
    #: the simulator, never by this.
    task_complete_p: float = 0.0
    #: Wall-clock latency of the decision, seconds.
    latency_s: float = 0.0
    cached: bool = False
    note: str = ""


class Planner(abc.ABC):
    """Chooses the next skill. Called only at skill boundaries, never inside
    the 20Hz control loop."""

    @abc.abstractmethod
    def decide(self, state: SceneState, history: list[str]) -> Decision: ...

    def reset(self) -> None:
        """Clear any per-episode state."""

    @property
    def stats(self) -> dict:
        return {}
