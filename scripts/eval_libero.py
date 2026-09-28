#!/usr/bin/env python3
"""Evaluate a planner on a LIBERO task suite.

    python scripts/eval_libero.py --planner.backend heuristic --num-trials-per-task 5
    python scripts/eval_libero.py --planner.backend jev --task-suite-name libero_object
"""

from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jev_policy.config import ENV_RESOLUTION, EvalConfig  # noqa: E402
from jev_policy.envs.libero_env import get_task_init_states, load_task_suite, make_env  # noqa: E402
from jev_policy.metrics import RunMetrics  # noqa: E402
from jev_policy.policies.registry import build_planner  # noqa: E402
from jev_policy.runner import run_episode  # noqa: E402

logger = logging.getLogger("eval")


def save_video(frames, path: pathlib.Path) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=20)


def main(config: EvalConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(config.seed)

    suite = load_task_suite(config.task_suite_name)
    planner = build_planner(config.planner)
    metrics = RunMetrics(config.task_suite_name)

    out_dir = pathlib.Path(config.out_dir) / config.task_suite_name / config.planner.backend
    task_ids = [config.task_id] if config.task_id is not None else list(range(suite.n_tasks))

    logger.info(
        "suite=%s tasks=%d trials/task=%d planner=%s",
        config.task_suite_name,
        len(task_ids),
        config.num_trials_per_task,
        config.planner.backend,
    )

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = get_task_init_states(suite, task_id)
        env, description = make_env(task, config.seed, ENV_RESOLUTION)

        try:
            for episode_idx in range(config.num_trials_per_task):
                result, frames = run_episode(
                    env=env,
                    planner=planner,
                    config=config,
                    task_id=task_id,
                    task_description=description,
                    episode_idx=episode_idx,
                    initial_state=initial_states[episode_idx % len(initial_states)],
                    collect_frames=config.save_video != "none",
                )
                metrics.add(result)

                want_video = config.save_video == "all" or (
                    config.save_video == "failure" and not result.success
                )
                if want_video:
                    tag = "success" if result.success else "failure"
                    slug = description.replace(" ", "_")[:60]
                    save_video(frames, out_dir / "videos" / f"task{task_id}_ep{episode_idx}_{slug}_{tag}.mp4")

                logger.info(
                    "task %d ep %d: %s (%d steps, %d decisions) %s",
                    task_id,
                    episode_idx,
                    "SUCCESS" if result.success else "fail",
                    result.steps,
                    result.decisions,
                    result.failure_reason,
                )
        finally:
            env.close()

        logger.info("running success rate: %.1f%%", 100 * metrics.success_rate)

    path = metrics.save(out_dir / "results.json", planner.stats)
    logger.info("=" * 60)
    logger.info(
        "%s | %s | %d/%d = %.1f%%",
        config.task_suite_name,
        config.planner.backend,
        metrics.successes,
        metrics.total,
        100 * metrics.success_rate,
    )
    if planner.stats:
        logger.info("planner stats: %s", planner.stats)
    logger.info("results written to %s", path)


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
