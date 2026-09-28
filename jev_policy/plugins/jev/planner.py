"""Jev as the decision layer.

Jev is a System One model: unstructured state in, typed decisions out, 70-500ms
per call, text only. That shapes the whole design:

* **It cannot see.** LIBERO's cameras are useless to it, so the scene reaches it
  as the symbolic text built in `perception/scene_state.py`.
* **It cannot servo.** At 70-500ms it is nowhere near LIBERO's 20Hz control
  rate, so it is queried only at skill boundaries -- roughly 5-15 calls per
  episode -- while `skills/` runs the intervening closed loop.
* **It cannot emit an invalid answer.** Targets are enumerated as `Choice`
  criteria built from the objects actually present in this scene, so a
  hallucinated object name is impossible by construction rather than by
  validation. This is the property we are actually buying.
* **Its confidence is calibrated.** Rather than executing every decision, a
  choice below `min_confidence` triggers a retreat-and-re-ask instead of a
  likely-wrong grasp. Tune the threshold on a dev suite.
"""

from __future__ import annotations

import dataclasses
import logging
import time

from jev_policy.config import PlannerConfig
from jev_policy.perception.scene_state import SceneState
from jev_policy.policies.base import Decision, Planner
from jev_policy.skills.primitives import SKILL_LIBRARY

logger = logging.getLogger(__name__)

#: Choice option meaning "this decision point needs no object".
NO_TARGET = "none"

#: Jev input pricing, USD per million tokens. Output tokens are free.
INPUT_USD_PER_MTOK = 0.042

SKILL_INSTRUCTIONS = (
    "A Panda robot arm is performing the task described in the state. "
    "Given the current scene, choose the single next motion primitive to execute. "
    "Respect preconditions: do not grasp while already holding something, do not "
    "place or lift while holding nothing, and hover above an object before descending onto it."
)

TARGET_INSTRUCTIONS = (
    "Which object should the chosen primitive act on? Choose the object the task "
    "requires next. Choose 'none' only for primitives that take no object, such as "
    "closing or opening the gripper."
)

DONE_INSTRUCTIONS = (
    "Based on the scene state alone, has the task described in the state been fully completed?"
)

DONE_CRITERIA = {
    "true": "The goal configuration described by the task is already satisfied in the scene.",
    "false": "The task still requires at least one more robot action.",
}


class JevPlanner(Planner):
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self._client = None
        self._cache: dict[tuple, Decision] = {}
        self._calls = 0
        self._cache_hits = 0
        self._input_tokens = 0
        self._latencies: list[float] = []
        self._confidences: list[float] = []
        self._low_confidence = 0

    # ------------------------------------------------------------- client ---

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from typesafe_sdk import RetryPolicy, TypeSafeClient
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "The Jev planner needs the TypeSafe SDK: pip install typesafe-sdk"
            ) from exc

        # Retries are the SDK's job, not ours: a transient 5xx mid-episode would
        # otherwise abort a rollout and silently depress the success rate.
        self._client = TypeSafeClient(
            model=self.config.model,
            timeout=self.config.request_timeout_s,
            retry=RetryPolicy(max_retries=self.config.max_retries),
        )
        return self._client

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
            self._client = None

    # ------------------------------------------------------------ prompts ---

    def _target_criteria(self, state: SceneState) -> dict[str, str]:
        """Enumerate legal targets from the live scene.

        Objects named by the task's BDDL goal are listed first and labelled, so
        the model has a reason to prefer them, but every present object stays
        selectable -- some tasks need an intermediate object the goal omits.
        """
        criteria: dict[str, str] = {}
        for name in state.objects_of_interest:
            if name in state.objects:
                criteria[name] = "named by the task goal"
        for name in sorted(state.objects):
            criteria.setdefault(name, "present in the scene")
        criteria[NO_TARGET] = "this primitive does not act on an object"
        return criteria

    def _build_state(self, state: SceneState, history: list[str]) -> dict:
        payload = {
            "task": state.task,
            "scene": state.to_text(),
        }
        if history:
            # Only the recent tail: older steps are already reflected in the
            # scene, and the full trace would just burn context.
            payload["completed_so_far"] = "\n".join(history[-8:])
        return payload

    def _questions(self, state: SceneState) -> dict:
        from typesafe_sdk import Choice, Noul

        return {
            "skill": Choice(
                instructions=SKILL_INSTRUCTIONS,
                criteria={name: desc for name, (_, desc) in SKILL_LIBRARY.items()},
            ),
            "target": Choice(
                instructions=TARGET_INSTRUCTIONS,
                criteria=self._target_criteria(state),
            ),
            "task_complete": Noul(instructions=DONE_INSTRUCTIONS, criteria=DONE_CRITERIA),
        }

    # ------------------------------------------------------------- decide ---

    def decide(self, state: SceneState, history: list[str]) -> Decision:
        # The history is part of the key: a skill that completes without moving
        # anything (a place when already over the target) leaves the scene
        # identical, so keying on the scene alone re-serves the same decision
        # forever and the episode livelocks.
        key = (state.cache_key(), tuple(history[-3:]))
        if self.config.cache_decisions and key in self._cache:
            self._cache_hits += 1
            return dataclasses.replace(self._cache[key], cached=True, latency_s=0.0)

        decision = self._query(state, history)

        if self.config.cache_decisions:
            self._cache[key] = decision
        return decision

    def _query(self, state: SceneState, history: list[str]) -> Decision:
        client = self._ensure_client()

        started = time.perf_counter()
        response = client.system_one(
            state=self._build_state(state, history),
            questions=self._questions(state),
        )
        latency = time.perf_counter() - started

        self._calls += 1
        self._latencies.append(latency)
        usage = getattr(response, "usage", None)
        if usage is not None and getattr(usage, "input_tokens", None):
            self._input_tokens += usage.input_tokens

        skill_answer = response.choices["skill"]
        target_answer = response.choices["target"]
        confidence = float(skill_answer.confidence)
        self._confidences.append(confidence)

        target = target_answer.choice
        if target == NO_TARGET:
            target = None

        note = ""
        if confidence < self.config.min_confidence:
            self._low_confidence += 1
            note = f"low confidence {confidence:.2f} < {self.config.min_confidence}"
            logger.info("Jev unsure (%s); retreating instead of acting", note)

        return Decision(
            skill=skill_answer.choice,
            target=target,
            confidence=confidence,
            task_complete_p=float(response.nouls["task_complete"].noul),
            latency_s=latency,
            note=note,
        )

    # -------------------------------------------------------------- admin ---

    def reset(self) -> None:
        # The cache is keyed by full scene text, so it stays valid across
        # episodes of the same task and is deliberately not cleared here.
        pass

    @property
    def stats(self) -> dict:
        n = len(self._latencies)
        return {
            "model": self.config.model,
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "low_confidence_decisions": self._low_confidence,
            "mean_latency_s": (sum(self._latencies) / n) if n else 0.0,
            "p95_latency_s": (sorted(self._latencies)[int(0.95 * (n - 1))] if n else 0.0),
            "mean_confidence": (
                sum(self._confidences) / len(self._confidences) if self._confidences else 0.0
            ),
            "input_tokens": self._input_tokens,
            "estimated_cost_usd": self._input_tokens / 1e6 * INPUT_USD_PER_MTOK,
        }
