"""Skill interface: the layer between Jev's decisions and the robot's joints."""

from __future__ import annotations

import abc
import dataclasses
import enum

import numpy as np

from jev_policy.config import ACTION_DIM
from jev_policy.perception.scene_state import SceneState

GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


class SkillStatus(enum.Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclasses.dataclass
class SkillResult:
    action: np.ndarray
    status: SkillStatus
    note: str = ""


class Skill(abc.ABC):
    """A closed-loop motion primitive.

    Skills own the control loop between decisions: `step` is called at the
    simulator's 20Hz and must return in microseconds. No model call may happen
    inside a skill -- that is the planner's job, and it runs at skill
    boundaries only.
    """

    #: Give up after this many control steps; prevents a wedged skill from
    #: burning the whole episode budget.
    max_steps: int = 80

    def __init__(self, target: str | None = None) -> None:
        self.target = target
        self.elapsed = 0

    @property
    def name(self) -> str:
        return type(self).__name__

    def __str__(self) -> str:
        return f"{self.name}({self.target})" if self.target else self.name

    def step(self, state: SceneState) -> SkillResult:
        self.elapsed += 1
        if self.elapsed > self.max_steps:
            return SkillResult(hold_action(), SkillStatus.FAILED, "timed out")
        return self._step(state)

    @abc.abstractmethod
    def _step(self, state: SceneState) -> SkillResult: ...


def hold_action(gripper: float = GRIPPER_OPEN) -> np.ndarray:
    action = np.zeros(ACTION_DIM, dtype=np.float64)
    action[6] = gripper
    return action
