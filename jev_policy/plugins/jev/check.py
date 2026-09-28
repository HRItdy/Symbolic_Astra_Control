#!/usr/bin/env python3
"""Verify the Jev connection and measure decision latency -- no simulator needed.

    export TYPESAFE_API_KEY=sk-...
    python scripts/check_jev.py
"""

from __future__ import annotations

import os
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jev_policy.config import PlannerConfig  # noqa: E402
from jev_policy.perception.scene_state import ObjectObs, SceneState  # noqa: E402
from jev_policy.policies.jev_planner import JevPlanner  # noqa: E402

N_TRIALS = 5


def fake_state() -> SceneState:
    """A plausible mid-episode LIBERO scene, hand-built so this runs without MuJoCo."""
    return SceneState(
        task="pick up the black bowl between the plate and the ramekin and place it on the plate",
        eef_pos=np.array([0.0, 0.0, 1.05]),
        eef_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_qpos=np.array([0.04, -0.04]),
        objects={
            "akita_black_bowl_1": ObjectObs(
                "akita_black_bowl_1", np.array([0.05, -0.10, 0.92]), np.zeros(4), False
            ),
            "plate_1": ObjectObs("plate_1", np.array([-0.12, -0.05, 0.91]), np.zeros(4), False),
            "wooden_cabinet_1": ObjectObs(
                "wooden_cabinet_1", np.array([0.20, 0.15, 0.95]), np.zeros(4), True, is_open=False
            ),
        },
        objects_of_interest=("akita_black_bowl_1", "plate_1"),
        step=42,
    )


def main() -> int:
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set. Export it and retry.")
        return 1

    state = fake_state()
    print("State sent to Jev:\n")
    print(state.to_text())
    print()

    planner = JevPlanner(PlannerConfig(backend="jev", cache_decisions=False))
    latencies = []
    try:
        for i in range(N_TRIALS):
            started = time.perf_counter()
            decision = planner.decide(state, history=[])
            latencies.append(time.perf_counter() - started)
            print(
                f"[{i + 1}/{N_TRIALS}] skill={decision.skill:<14} "
                f"target={str(decision.target):<22} "
                f"confidence={decision.confidence:.3f} "
                f"P(done)={decision.task_complete_p:.3f} "
                f"{latencies[-1] * 1000:.0f}ms"
            )
    finally:
        planner.close()

    print(
        f"\nlatency: median {statistics.median(latencies) * 1000:.0f}ms, "
        f"max {max(latencies) * 1000:.0f}ms"
    )
    print("Decisions must be stable across identical states; if they are not, lower the")
    print("temperature of the task by tightening the skill descriptions in skills/primitives.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
