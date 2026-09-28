"""GPT-6-Astra driving the LIBERO Panda directly.

No skill library: the model is handed the cameras and the robot's state and
returns robot commands itself. Two things shape this file.

**It cannot run at 20Hz.** LIBERO steps 20 times a second; Astra answers in
seconds. So the model returns an *action chunk* -- many control steps per call,
executed open loop before the next observation. This is the same trick the
action-chunking VLAs use (pi-0 and friends emit ~50-step chunks), except those
models were trained to produce actions and this one was not.

**Tokens cost money.** At $10/M in and $50/M out, with an image on every call,
an episode is cents and a full suite is tens of dollars. Reasoning tokens bill
as output, so `reasoning_effort` defaults to low, usage is tracked on every
call, and `stats` reports the running spend.

Two control modes:

* `delta` -- the literal reading of "control directly": the model emits raw
  OSC_POSE deltas, exactly what `env.step` consumes. Nothing interprets them.
* `waypoint` -- the model emits a target end-effector position and gripper
  state, and the same P-controller the skills use servos there. Still the model
  choosing the motion, but in a space it can reason about (centimetres in the
  world) rather than one it has never been trained on (normalised deltas per
  control tick).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time

import numpy as np

from jev_policy.config import ACTION_DIM, OSC_POS_LIMIT, PlannerConfig
from jev_policy.control.base import ControlChunk, Controller
from jev_policy.perception.scene_state import SceneState
from jev_policy.skills.controller import position_action

logger = logging.getLogger(__name__)

USD_PER_INPUT_TOKEN = 10.0 / 1e6
USD_PER_OUTPUT_TOKEN = 50.0 / 1e6

SYSTEM_PROMPT = """You are controlling a Franka Panda arm in the LIBERO simulator.

The arm is position-controlled at 20Hz. You see two cameras: the left image is a \
third-person view of the workspace, the right image is the wrist camera looking \
out from between the gripper fingers.

World axes, from the robot's point of view:
  +x is to the right, +y is away from the robot (forward), +z is up.
The table surface is at about z=0.90. Objects sit on it at z=0.90-0.92.

Work carefully and in stages: hover above a target before descending onto it, \
close the gripper only when the fingers straddle the object, lift before \
carrying, and release only once the object is over its destination."""

DELTA_INSTRUCTIONS = """Return the next {n} control steps.

Each step is [dx, dy, dz, drx, dry, drz, gripper]:
  dx, dy, dz    translation for this step, each in [-1, 1], where 1.0 moves {limit_cm:.0f}cm
  drx, dry, drz rotation, each in [-1, 1]; use 0.0 unless you must reorient
  gripper       -1.0 to open, +1.0 to close

The steps execute back-to-back without further input from you, so plan the whole \
burst: a typical move is many small steps in a consistent direction rather than \
one large one."""

WAYPOINT_INSTRUCTIONS = """Return the next end-effector waypoint.

  target_xyz  absolute world position to move the gripper to, in metres
  gripper     "open" or "closed"
  steps       how many 20Hz control steps to spend getting there (5-40)

A controller servos the arm to target_xyz, so give the position you want \
reached, not an increment."""

DELTA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reasoning", "actions"],
    "properties": {
        "reasoning": {"type": "string", "description": "One sentence on the current sub-goal."},
        "actions": {
            "type": "array",
            "description": "Control steps, each 7 numbers in [-1, 1].",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 7,
                "maxItems": 7,
            },
        },
    },
}

WAYPOINT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reasoning", "target_xyz", "gripper", "steps"],
    "properties": {
        "reasoning": {"type": "string"},
        "target_xyz": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
        "gripper": {"type": "string", "enum": ["open", "closed"]},
        "steps": {"type": "integer"},
    },
}


def _compose_views(rgb: dict[str, np.ndarray], max_width: int = 768) -> bytes:
    from PIL import Image

    order = [n for n in ("agentview", "robot0_eye_in_hand") if n in rgb]
    frames = [Image.fromarray(np.asarray(rgb[n], dtype=np.uint8)) for n in order]
    if not frames:
        raise ValueError("no camera views")

    height = max(f.height for f in frames)
    canvas = Image.new("RGB", (sum(f.width for f in frames), height))
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


class AstraController(Controller):
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self._client = None
        self._calls = 0
        self._latencies: list[float] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._parse_failures = 0

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(timeout=self.config.request_timeout_s)
        return self._client

    # ------------------------------------------------------------- prompt ---

    def _observation(self, state: SceneState, history: list[str]) -> str:
        eef = state.eef_pos
        lines = [
            f"TASK: {state.task}",
            "",
            f"gripper_position: x={eef[0]:+.3f} y={eef[1]:+.3f} z={eef[2]:+.3f}  (metres, world)",
            f"gripper: {'open' if state.gripper_open else 'closed'}",
            f"holding: {state.held_object or 'nothing'}",
            f"control_step: {state.step}",
        ]
        if self.config.include_object_poses and state.objects:
            lines += ["", "OBJECT POSITIONS (world metres):"]
            for name in sorted(state.objects):
                obj = state.objects[name]
                marker = " <- named by the task" if name in state.objects_of_interest else ""
                lines.append(
                    f"  {name}: x={obj.pos[0]:+.3f} y={obj.pos[1]:+.3f} "
                    f"z={obj.pos[2]:+.3f}{marker}"
                )

        if history:
            lines += ["", "WHAT YOU DID SO FAR:", *history[-6:]]
        return "\n".join(lines)

    def _instructions(self) -> str:
        if self.config.control_mode == "waypoint":
            return WAYPOINT_INSTRUCTIONS
        return DELTA_INSTRUCTIONS.format(
            n=self.config.chunk_steps, limit_cm=OSC_POS_LIMIT * 100
        )

    # --------------------------------------------------------------- call ---

    def act(self, state: SceneState, history: list[str]) -> ControlChunk:
        client = self._ensure_client()
        waypoint = self.config.control_mode == "waypoint"

        content = [{"type": "input_text", "text": self._observation(state, history)}]
        if state.rgb:
            image = base64.b64encode(_compose_views(state.rgb)).decode("ascii")
            content.append(
                {"type": "input_image", "image_url": f"data:image/png;base64,{image}"}
            )
        content.append({"type": "input_text", "text": self._instructions()})

        started = time.perf_counter()
        response = client.responses.create(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "robot_command",
                    "strict": True,
                    "schema": WAYPOINT_SCHEMA if waypoint else DELTA_SCHEMA,
                }
            },
        )
        latency = time.perf_counter() - started

        self._calls += 1
        self._latencies.append(latency)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._input_tokens += getattr(usage, "input_tokens", 0) or 0
            self._output_tokens += getattr(usage, "output_tokens", 0) or 0

        try:
            payload = json.loads(response.output_text)
        except (json.JSONDecodeError, AttributeError) as exc:
            self._parse_failures += 1
            logger.warning("could not parse Astra response: %s", exc)
            return ControlChunk([], latency_s=latency, note=f"unparseable response: {exc}")

        actions = (
            self._waypoint_actions(payload, state) if waypoint else self._delta_actions(payload)
        )
        return ControlChunk(
            actions=actions,
            latency_s=latency,
            note=str(payload.get("reasoning", ""))[:120],
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

    # ------------------------------------------------------------ decode ---

    def _delta_actions(self, payload: dict) -> list[np.ndarray]:
        """Clamp and shape whatever came back into legal LIBERO actions.

        The model is not trained to emit these, so the values are validated
        rather than trusted: wrong-length rows are dropped and everything is
        clipped to [-1, 1] so a bad number cannot reach the simulator.
        """
        actions = []
        for row in payload.get("actions", [])[: self.config.chunk_steps]:
            try:
                vector = np.asarray(row, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if vector.shape != (ACTION_DIM,) or not np.all(np.isfinite(vector)):
                continue
            actions.append(np.clip(vector, -1.0, 1.0))
        return actions

    def _waypoint_actions(self, payload: dict, state: SceneState) -> list[np.ndarray]:
        """Expand one waypoint into a servoed burst of control steps."""
        try:
            target = np.asarray(payload["target_xyz"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return []
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return []

        gripper = 1.0 if payload.get("gripper") == "closed" else -1.0
        steps = int(np.clip(payload.get("steps", 20), 1, self.config.chunk_steps))

        # Open-loop expansion from the pose we last observed: the chunk runs
        # without new observations, so the servo integrates its own prediction.
        actions, position = [], state.eef_pos.astype(float).copy()
        for _ in range(steps):
            action = position_action(position, target, gripper)
            actions.append(action)
            position = position + action[:3] * OSC_POS_LIMIT
        return actions

    # -------------------------------------------------------------- admin ---

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self._input_tokens * USD_PER_INPUT_TOKEN
            + self._output_tokens * USD_PER_OUTPUT_TOKEN
        )

    @property
    def stats(self) -> dict:
        n = len(self._latencies)
        return {
            "model": self.config.model,
            "control_mode": self.config.control_mode,
            "reasoning_effort": self.config.reasoning_effort,
            "chunk_steps": self.config.chunk_steps,
            "calls": self._calls,
            "parse_failures": self._parse_failures,
            "mean_latency_s": (sum(self._latencies) / n) if n else 0.0,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }
