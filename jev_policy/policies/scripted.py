"""Model-free planners.

These exist so the skill layer can be debugged independently of Jev. If the
heuristic planner cannot solve a pick-and-place task, the fault is in
`skills/`, not in the decision layer -- and no API budget was spent finding
that out.
"""

from __future__ import annotations

import random

from jev_policy.perception.scene_state import SceneState
from jev_policy.policies.base import Decision, Planner
from jev_policy.skills.primitives import SKILL_LIBRARY


class HeuristicPlanner(Planner):
    """A fixed pick-and-place sequence over the task's objects of interest.

    LIBERO goals are overwhelmingly "pick up A, put it on/in B", and the BDDL
    `obj_of_interest` list is ordered accordingly. It solves the easy half of
    the benchmark and is the control condition to measure Jev's decisions
    against.

    The sequence is driven by an index, not by reading `held_object` back out of
    the simulator: robosuite only reports a grasp once it bears load, so a
    state-based branch sits in the pick phase forever and never reaches the
    lift. Progress is judged from the runner's history instead, which records
    whether each skill succeeded.
    """

    #: (skill, which object of interest it acts on)
    SEQUENCE = (
        ("move_above", 0),
        ("descend", 0),
        ("close_gripper", None),
        ("lift", 0),
        ("place_on", 1),
        ("open_gripper", None),
    )

    def __init__(self, config=None) -> None:
        self.config = config
        self._stage = 0

    def reset(self) -> None:
        self._stage = 0

    def decide(self, state: SceneState, history: list[str]) -> Decision:
        # Restart the approach when the previous skill failed; a failed grasp
        # has to be re-attempted from the hover pose, not continued from.
        if history and "failed" in history[-1]:
            self._stage = 0

        interest = [n for n in state.objects_of_interest if n in state.objects]
        skill, slot = self.SEQUENCE[min(self._stage, len(self.SEQUENCE) - 1)]
        self._stage = (self._stage + 1) % len(self.SEQUENCE)

        target = None
        if slot is not None and slot < len(interest):
            target = interest[slot]

        return Decision(skill=skill, target=target, confidence=1.0, note="heuristic")


class RandomPlanner(Planner):
    """Uniformly random legal skills. Smoke-tests the loop; solves nothing."""

    def __init__(self, config=None, seed: int = 0) -> None:
        self.config = config
        self._rng = random.Random(seed)

    def decide(self, state: SceneState, history: list[str]) -> Decision:
        skill = self._rng.choice(sorted(SKILL_LIBRARY))
        target = self._rng.choice(sorted(state.objects)) if state.objects else None
        return Decision(skill=skill, target=target, confidence=0.0, note="random")
