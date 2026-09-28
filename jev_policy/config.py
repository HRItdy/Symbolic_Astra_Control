"""Configuration for a LIBERO evaluation run.

Parsed from the CLI by tyro (see scripts/eval_libero.py), so every field here is
a `--flag`; nested dataclasses become `--planner.xxx`.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

TaskSuite = Literal["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

# Per-suite episode budgets: the longest training demo, rounded up. This is the
# convention the published LIBERO numbers use -- changing it makes results
# incomparable to the literature.
MAX_STEPS_BY_SUITE: dict[str, int] = {
    "libero_spatial": 220,  # longest training demo: 193
    "libero_object": 280,  # longest training demo: 254
    "libero_goal": 300,  # longest training demo: 270
    "libero_10": 520,  # longest training demo: 505
    "libero_90": 400,  # longest training demo: 373
}

# LIBERO action space: 6-DoF end-effector delta (OSC_POSE) + 1 gripper command.
ACTION_DIM = 7
# No-op with the gripper held open, used while the sim settles.
DUMMY_ACTION = [0.0] * 6 + [-1.0]
# Resolution LIBERO training data was rendered at (video only -- Jev is text-only).
ENV_RESOLUTION = 256
# robosuite OSC_POSE default output limits. A position command of 1.0 means
# "move 5cm this control step"; we normalise metre-space targets by this.
OSC_POS_LIMIT = 0.05
OSC_ROT_LIMIT = 0.5


@dataclasses.dataclass(frozen=True)
class PlannerConfig:
    """Which decision layer drives skill selection."""

    # Registered planner. See jev_policy.policies.registry.available().
    #   astra     - GPT-6-Astra driving the robot directly (no skill library)
    #   heuristic - scripted planner; no API key, used to validate the skill layer
    #   random    - picks legal skills at random; harness smoke test only
    #   jev / jev-omni - in the `jev` plugin, off by default; see
    #                    jev_policy/plugins/jev/README.md
    backend: str = "heuristic"

    # Pin the model: prompts and thresholds are tuned per-version and would
    # silently shift under an alias that moves.
    model: str = "gpt-6-astra"

    # Below this confidence we do not execute the chosen skill; we run a
    # recovery behaviour and re-ask with the failure appended to the history.
    # This is the whole point of a calibrated model -- tune it on a dev suite.
    min_confidence: float = 0.45

    # Probability above which the `task_complete` Noul is believed. Only used
    # for logging/early-stop diagnostics; ground-truth success always comes
    # from the simulator, never from the model.
    done_threshold: float = 0.9

    # Re-ask at most this many times per decision point before giving up.
    max_retries: int = 2

    # --- locally served backends (jev-omni) ---
    # Jev-Omni runs behind scripts/serve_jev_omni.py, because its dependency set
    # (torch >= 2.10, transformers 5.17) cannot share an interpreter with
    # LIBERO's (numpy < 2, mujoco 2.3.7).
    host: str = "127.0.0.1"
    port: int = 8000

    # Jev-Omni answers one question per call, so probing "is the task done"
    # costs a third round trip per decision. Off by default: the simulator is
    # the authority on success, and this is only ever diagnostic.
    ask_done: bool = False

    # Cap on options per question. The Jev-Omni card claims quality only up to
    # 20 options; LIBERO registers 23 objects and regions in a single scene, so
    # the target list is ranked by relevance and truncated to this.
    max_options: int = 20

    # Skip the API call when the discretised scene state is unchanged from the
    # last decision. Scene state changes slowly, so this cuts call volume a lot.
    cache_decisions: bool = True

    request_timeout_s: float = 120.0

    # --- direct control (astra) ---
    # How the model commands the robot.
    #   delta    - it emits raw OSC_POSE deltas; nothing interprets them
    #   waypoint - it emits a target end-effector pose and the skill layer's
    #              P-controller servos there
    control_mode: Literal["delta", "waypoint"] = "delta"

    # Control steps per model call. LIBERO steps at 20Hz and a frontier model
    # answers in seconds, so one call must cover many steps. Larger chunks mean
    # fewer calls and less money, but longer open-loop stretches with no
    # feedback.
    chunk_steps: int = 20

    # Reasoning tokens bill at the output rate, so this is the main cost dial.
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"

    # Also give the model each object's world position, read from the simulator.
    # This is privileged information a real robot would not have, so it is off
    # by default -- but it is the ablation that separates "cannot perceive" from
    # "cannot control": with exact coordinates, failure can no longer be blamed
    # on visual grounding.
    include_object_poses: bool = False


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)

    task_suite_name: TaskSuite = "libero_spatial"
    # Evaluate a single task id instead of the whole suite (debugging aid).
    task_id: int | None = None
    # Rollouts per task. The standard protocol is 50; drop it while iterating.
    num_trials_per_task: int = 50

    # Steps to idle at episode start while dropped objects come to rest. LIBERO
    # spawns objects in the air; acting before they land wastes the episode.
    num_steps_wait: int = 10

    # Where results.json and failure videos are written.
    out_dir: str = "results"
    save_video: Literal["all", "failure", "none"] = "failure"
    seed: int = 7

    def max_steps(self) -> int:
        return MAX_STEPS_BY_SUITE[self.task_suite_name]
