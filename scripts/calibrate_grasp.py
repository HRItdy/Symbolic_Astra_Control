#!/usr/bin/env python3
"""Find the vertical offset between the eef site and a successful grasp.

The Panda's `robot0_eef` site sits above the fingertips, so descending to an
object's body z puts the fingers through the table. This sweeps candidate
offsets and reports which ones actually produce a grasp that survives a lift --
run it once per gripper/object class and put the winner in
`skills/primitives.py::GRASP_Z_OFFSET`.

    MUJOCO_GL=egl python scripts/calibrate_grasp.py --task-id 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jev_policy.config import DUMMY_ACTION  # noqa: E402
from jev_policy.envs.libero_env import (  # noqa: E402
    get_task_init_states,
    load_task_suite,
    make_env,
)
from jev_policy.perception.scene_state import read_scene  # noqa: E402
from jev_policy.skills.base import GRIPPER_CLOSE, GRIPPER_OPEN  # noqa: E402
from jev_policy.skills.controller import position_action  # noqa: E402
from jev_policy.skills.primitives import grasp_point  # noqa: E402

OFFSETS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]


def servo(env, target_fn, gripper, steps, task):
    """Drive the eef toward a target for `steps` control steps."""
    state = None
    for _ in range(steps):
        state = read_scene(env, task, 0)
        action = position_action(state.eef_pos, target_fn(state), gripper)
        env.step(action.tolist())
    return read_scene(env, task, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()

    suite = load_task_suite(args.suite)
    task = suite.get_task(args.task_id)
    init_states = get_task_init_states(suite, args.task_id)
    env, desc = make_env(task, seed=7, resolution=128)
    print(f"task: {desc}\n")

    print(f"{'offset':>8}  {'grasped':>8}  {'lifted':>7}  {'obj z rise':>10}")
    print("-" * 42)

    for offset in OFFSETS:
        env.reset()
        env.set_init_state(init_states[0])
        for _ in range(10):
            env.step(DUMMY_ACTION)

        state = read_scene(env, desc, 0)
        target_name = state.objects_of_interest[0]
        start_z = state.get(target_name).pos[2]

        # Hover over the calibrated grasp point, descend to obj.z + offset,
        # close, then lift. The xy comes from grasp_point() so wide objects are
        # approached at the rim, as the real skill does.
        servo(env, lambda s: np.array([*grasp_point(s.get(target_name))[:2], start_z + 0.10]),
              GRIPPER_OPEN, 60, desc)
        servo(env, lambda s: np.array([*grasp_point(s.get(target_name))[:2], start_z + offset]),
              GRIPPER_OPEN, 60, desc)
        for _ in range(15):
            env.step([0.0] * 6 + [GRIPPER_CLOSE])
        after_close = read_scene(env, desc, 0)
        grasped = after_close.get(target_name).grasped

        lifted_state = servo(
            env, lambda s: np.array([*s.eef_pos[:2], start_z + 0.25]), GRIPPER_CLOSE, 40, desc
        )
        rise = lifted_state.get(target_name).pos[2] - start_z
        lifted = rise > 0.05

        print(f"{offset:8.2f}  {str(grasped):>8}  {str(lifted):>7}  {rise:10.3f}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
