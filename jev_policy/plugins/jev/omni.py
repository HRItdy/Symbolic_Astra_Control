"""Jev-Omni as the decision layer -- a multimodal planner that can see.

Jev-Omni (`akhilaaa3/Jev-Omni`) is an **independent community model**, not
TypeSafe's Jev: a Gemma-4-12B-IT fine-tune that implements the same
typed-decision idea. It is not affiliated with or endorsed by TypeSafe AI, and
its numbers are its own. Three differences drive this file:

1. **It sees.** Text, image, audio and video. The planner is handed the
   agentview and wrist cameras, so the prompt no longer has to list object
   coordinates -- the model reads the scene off the pixels. What stays in text
   is proprioception (gripper open/closed, what is held), which a real robot
   knows about itself.
2. **One question per call.** There is no batched `system_one`; `predict()`
   takes a single question and a flat list of options. A decision that was one
   hosted call is now two local ones, so the `task_complete` probe is opt-in.
3. **It runs locally on a GPU**, and its dependency set (torch >= 2.10,
   transformers 5.17) is incompatible with LIBERO's (numpy < 2, mujoco 2.3.7).
   So it lives behind `scripts/serve_jev_omni.py` in its own environment and is
   reached over HTTP -- the same split openpi uses for its policy server.

Note the model card's guidance of **<= 20 options**: the head accepts 256 but
quality above 20 is unestablished, so the target list is ranked and truncated.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import time
import urllib.error
import urllib.request

import numpy as np

from jev_policy.config import PlannerConfig
from jev_policy.perception.scene_state import SceneState
from jev_policy.policies.base import Decision, Planner
from jev_policy.skills.primitives import SKILL_LIBRARY

logger = logging.getLogger(__name__)

NO_TARGET = "none"

SKILL_QUESTION = (
    "A Panda robot arm is doing the task below. Looking at the camera views and the "
    "gripper state, which single motion primitive should it execute next? Do not grasp "
    "while already holding something, do not lift or place while holding nothing, and "
    "hover above an object before descending onto it."
)

TARGET_QUESTION = (
    "Which object should that primitive act on? Pick the object the task needs next."
)

#: For these the target is a *destination*, not the thing being manipulated, so
#: the held object must be excluded -- otherwise the model reliably answers with
#: the object it is already holding and the place is a no-op onto itself.
DESTINATION_QUESTION = (
    "The robot is holding an object. Where should it put that object down? "
    "Pick the destination surface or container the task requires."
)
DESTINATION_SKILLS = {"place_on"}

#: These take no object at all. Asking costs a round trip and invites a wrong
#: answer, so the question is skipped entirely.
TARGETLESS_SKILLS = {"close_gripper", "open_gripper", "retreat"}

DONE_QUESTION = "Has the task described above been fully completed in the current scene?"


def _compose_views(rgb: dict[str, np.ndarray], max_width: int = 512) -> bytes:
    """Stack the camera views into one PNG.

    `predict()` accepts a single media item per call, so the two LIBERO cameras
    have to travel as one image. Side by side keeps both at full resolution and
    keeps the wrist view -- which is what disambiguates "about to grasp" from
    "already holding" -- legible.
    """
    from PIL import Image

    order = [name for name in ("agentview", "robot0_eye_in_hand") if name in rgb]
    frames = [Image.fromarray(np.asarray(rgb[name], dtype=np.uint8)) for name in order]
    if not frames:
        raise ValueError("no camera views to send")

    height = max(f.height for f in frames)
    total_width = sum(f.width for f in frames)
    canvas = Image.new("RGB", (total_width, height))
    x = 0
    for frame in frames:
        canvas.paste(frame, (x, 0))
        x += frame.width

    if canvas.width > max_width:
        scale = max_width / canvas.width
        canvas = canvas.resize((max_width, int(canvas.height * scale)), Image.BILINEAR)

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


class JevOmniPlanner(Planner):
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self._cache: dict[tuple, Decision] = {}
        self._calls = 0
        self._cache_hits = 0
        self._latencies: list[float] = []
        self._confidences: list[float] = []
        self._low_confidence = 0

    # --------------------------------------------------------------- http ---

    @property
    def _url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/predict"

    def _predict(self, state_text: str, question: str, options: list[str], image: bytes | None):
        payload = {"state": state_text, "question": question, "options": options}
        if image is not None:
            payload["image_png_b64"] = base64.b64encode(image).decode("ascii")

        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Jev-Omni server unreachable at {self._url} -- start it with "
                f"scripts/serve_jev_omni.py ({exc})"
            ) from exc
        latency = time.perf_counter() - started

        if "error" in result:
            raise RuntimeError(f"Jev-Omni server error: {result['error']}")

        self._calls += 1
        self._latencies.append(latency)
        return result, latency

    # ------------------------------------------------------------ options ---

    def _target_options(self, state: SceneState, exclude: str | None = None) -> list[str]:
        """Rank the scene's objects and truncate to the model's usable width.

        Objects named by the task goal come first, then the rest by distance
        from the gripper. The model card only claims quality up to 20 options,
        and LIBERO registers 23 objects and placement regions in a single scene,
        so this truncation is load-bearing, not cosmetic.
        """
        goal_objects = [
            n for n in state.objects_of_interest if n in state.objects and n != exclude
        ]
        others = sorted(
            (n for n in state.objects if n not in goal_objects and n != exclude),
            key=lambda n: float(np.linalg.norm(state.objects[n].pos - state.eef_pos)),
        )
        budget = max(2, self.config.max_options - 1)  # leave room for "none"
        return [*goal_objects, *others][:budget] + [NO_TARGET]

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
        image = _compose_views(state.rgb) if state.rgb else None
        if image is None:
            logger.warning("no camera views available; Jev-Omni is deciding blind")

        lines = [f"TASK: {state.task}", "", state.to_proprioception_text()]
        if history:
            lines += ["", "STEPS SO FAR:", *history[-8:]]
        state_text = "\n".join(lines)

        skill_names = list(SKILL_LIBRARY)
        skill_options = [f"{name}: {desc}" for name, (_, desc) in SKILL_LIBRARY.items()]
        skill_result, latency = self._predict(state_text, SKILL_QUESTION, skill_options, image)
        skill = skill_names[skill_result["prediction_index"]]
        confidence = float(skill_result["confidence"])
        total_latency = latency

        if skill in TARGETLESS_SKILLS:
            # No question asked: these primitives act on whatever the gripper is
            # already at, and a spurious target only muddies the trace.
            target = None
        else:
            destination = skill in DESTINATION_SKILLS
            target_options = self._target_options(
                state, exclude=state.held_object if destination else None
            )
            target_result, latency = self._predict(
                state_text + f"\n\nCHOSEN PRIMITIVE: {skill}",
                DESTINATION_QUESTION if destination else TARGET_QUESTION,
                target_options,
                image,
            )
            total_latency += latency
            target = target_result["prediction"]
            if target == NO_TARGET:
                target = None

        done_p = 0.0
        if self.config.ask_done:
            done_result, latency = self._predict(state_text, DONE_QUESTION, ["yes", "no"], image)
            total_latency += latency
            done_p = float(done_result["probabilities"].get("yes", 0.0))

        self._confidences.append(confidence)
        note = ""
        if confidence < self.config.min_confidence:
            self._low_confidence += 1
            note = f"low confidence {confidence:.2f} < {self.config.min_confidence}"

        return Decision(
            skill=skill,
            target=target,
            confidence=confidence,
            task_complete_p=done_p,
            latency_s=total_latency,
            note=note,
        )

    @property
    def stats(self) -> dict:
        n = len(self._latencies)
        return {
            "model": "Jev-Omni (community, Gemma-4-12B-IT finetune)",
            "vision": True,
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "low_confidence_decisions": self._low_confidence,
            "mean_latency_s": (sum(self._latencies) / n) if n else 0.0,
            "p95_latency_s": (sorted(self._latencies)[int(0.95 * (n - 1))] if n else 0.0),
            "mean_confidence": (
                sum(self._confidences) / len(self._confidences) if self._confidences else 0.0
            ),
        }
