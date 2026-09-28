"""The rollout loop.

The whole architecture is visible in `run_episode`: a fast inner loop that steps
a skill against the simulator at 20Hz, and a slow outer loop that asks the
planner what to do next only when a skill finishes. Jev sits in the outer loop,
where 70-500ms of latency costs a few seconds per episode instead of making
real-time control impossible.
"""

from __future__ import annotations

import logging

import numpy as np

from jev_policy.config import DUMMY_ACTION, EvalConfig
from jev_policy.control.base import Controller
from jev_policy.metrics import EpisodeResult
from jev_policy.perception.scene_state import read_scene
from jev_policy.policies.base import Planner
from jev_policy.skills.base import SkillStatus
from jev_policy.skills.primitives import Retreat, make_skill

logger = logging.getLogger(__name__)

#: Consecutive identical successful skills before the history says so outright.
REPEAT_LIMIT = 2


def run_episode(
    env,
    planner,
    config: EvalConfig,
    task_id: int,
    task_description: str,
    episode_idx: int,
    initial_state,
    collect_frames: bool = True,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    if isinstance(planner, Controller):
        return _run_direct_episode(
            env, planner, config, task_id, task_description, episode_idx,
            initial_state, collect_frames,
        )

    env.reset()
    obs = env.set_init_state(initial_state)
    planner.reset()

    frames: list[np.ndarray] = []
    history: list[str] = []
    confidences: list[float] = []
    latencies: list[float] = []
    skills_executed: list[str] = []

    skill = None
    decisions = 0
    last_choice = None
    repeats = 0
    success = False
    failure_reason = ""
    t = 0
    max_t = config.max_steps() + config.num_steps_wait

    while t < max_t:
        # LIBERO drops objects at reset; acting before they settle wastes the
        # episode and corrupts the scene state the planner reads.
        if t < config.num_steps_wait:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            t += 1
            continue

        # LIBERO renders both cameras upside down.
        views = {
            name.replace("_image", ""): np.ascontiguousarray(obs[name][::-1, ::-1])
            for name in ("agentview_image", "robot0_eye_in_hand_image")
            if name in obs
        }
        if collect_frames and "agentview" in views:
            frames.append(views["agentview"])

        try:
            state = read_scene(env, task_description, t, rgb=views or None)
        except Exception as exc:
            failure_reason = f"scene read failed: {exc}"
            logger.exception("scene read failed at t=%d", t)
            break

        # --- outer loop: ask the planner only at skill boundaries ---
        if skill is None:
            try:
                decision = planner.decide(state, history)
            except Exception as exc:
                failure_reason = f"planner error: {exc}"
                logger.exception("planner failed at t=%d", t)
                break

            decisions += 1
            if not decision.cached:
                latencies.append(decision.latency_s)
            confidences.append(decision.confidence)

            min_conf = config.planner.min_confidence
            if decision.confidence < min_conf:
                # Calibration earning its keep: back off and re-plan rather
                # than commit to a grasp the model does not believe in.
                history.append(
                    f"skipped {decision.skill}({decision.target}) "
                    f"- confidence {decision.confidence:.2f} below {min_conf}"
                )
                skill = Retreat()
            else:
                try:
                    skill = make_skill(decision.skill, decision.target)
                except KeyError as exc:
                    failure_reason = f"invalid skill: {exc}"
                    break
            skills_executed.append(str(skill))
            logger.debug("t=%d decision=%s conf=%.2f", t, skill, decision.confidence)

        # --- inner loop: pure code, no model call ---
        result = skill.step(state)
        obs, _, done, _ = env.step(result.action.tolist())
        t += 1

        if result.status is SkillStatus.SUCCESS:
            # A primitive that reports success without advancing the scene is a
            # livelock, not progress: the planner sees the same world and picks
            # it again. Say so explicitly in the history so the next decision
            # has a reason to choose differently.
            choice = str(skill)
            repeats = repeats + 1 if choice == last_choice else 0
            last_choice = choice
            if repeats >= REPEAT_LIMIT:
                history.append(f"{skill} -> already done {repeats + 1}x, it changes nothing now")
                logger.debug("skill %s repeated %d times without progress", skill, repeats + 1)
            else:
                history.append(f"{skill} -> ok")
            skill = None
        elif result.status is SkillStatus.FAILED:
            history.append(f"{skill} -> failed: {result.note}")
            logger.debug("skill %s failed: %s", skill, result.note)
            skill = None

        # Ground truth comes from the simulator, never from the model.
        if done or env.check_success():
            success = True
            break

    if not success and not failure_reason:
        failure_reason = "step budget exhausted"

    result = EpisodeResult(
        task_id=task_id,
        task_description=task_description,
        episode=episode_idx,
        success=success,
        steps=t,
        decisions=decisions,
        skills_executed=skills_executed,
        mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
        mean_latency_s=float(np.mean(latencies)) if latencies else 0.0,
        failure_reason="" if success else failure_reason,
    )
    return result, frames


def _run_direct_episode(
    env,
    controller: Controller,
    config: EvalConfig,
    task_id: int,
    task_description: str,
    episode_idx: int,
    initial_state,
    collect_frames: bool,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    """Rollout with no skill layer: the model's own actions reach the simulator.

    The structure is the same two-rate loop, but the inner loop replays a chunk
    the model produced instead of running a hand-written primitive. An empty
    chunk is a dead end -- with nothing to execute the state cannot change, so
    the next call would see the same thing -- so the episode ends rather than
    spending the budget asking again.
    """
    env.reset()
    obs = env.set_init_state(initial_state)
    controller.reset()

    frames: list[np.ndarray] = []
    history: list[str] = []
    latencies: list[float] = []
    notes: list[str] = []

    calls = 0
    success = False
    failure_reason = ""
    t = 0
    max_t = config.max_steps() + config.num_steps_wait

    while t < max_t:
        if t < config.num_steps_wait:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            t += 1
            continue

        views = {
            name.replace("_image", ""): np.ascontiguousarray(obs[name][::-1, ::-1])
            for name in ("agentview_image", "robot0_eye_in_hand_image")
            if name in obs
        }
        if collect_frames and "agentview" in views:
            frames.append(views["agentview"])

        try:
            state = read_scene(env, task_description, t, rgb=views or None)
        except Exception as exc:
            failure_reason = f"scene read failed: {exc}"
            logger.exception("scene read failed at t=%d", t)
            break

        try:
            chunk = controller.act(state, history)
        except Exception as exc:
            failure_reason = f"controller error: {exc}"
            logger.exception("controller failed at t=%d", t)
            break

        calls += 1
        latencies.append(chunk.latency_s)
        if chunk.note:
            notes.append(chunk.note)

        if not chunk.actions:
            failure_reason = chunk.note or "controller returned no actions"
            break

        history.append(
            f"step {t}: commanded {len(chunk.actions)} actions"
            + (f" -- {chunk.note}" if chunk.note else "")
        )

        for action in chunk.actions:
            obs, _, done, _ = env.step(np.asarray(action, dtype=np.float64).tolist())
            t += 1
            if done or env.check_success():
                success = True
                break
            if t >= max_t:
                break
        if success:
            break

    if not success and not failure_reason:
        failure_reason = "step budget exhausted"

    result = EpisodeResult(
        task_id=task_id,
        task_description=task_description,
        episode=episode_idx,
        success=success,
        steps=t,
        decisions=calls,
        skills_executed=notes[:20],
        mean_confidence=0.0,
        mean_latency_s=float(np.mean(latencies)) if latencies else 0.0,
        failure_reason="" if success else failure_reason,
    )
    return result, frames
