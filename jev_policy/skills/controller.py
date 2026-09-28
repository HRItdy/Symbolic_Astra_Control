"""Cartesian servoing for LIBERO's OSC_POSE controller.

LIBERO drives a Panda through robosuite's OSC_POSE controller, so an action is
a *delta* in end-effector space, normalised to [-1, 1]: 1.0 means "move by the
controller's per-step limit" (5cm in position, 0.5rad in rotation by default).

Inverse kinematics and the impedance loop live below this, inside robosuite --
which is exactly the split TypeSafe describes for Jev: the model decides, the
physics and IK stay in code.
"""

from __future__ import annotations

import numpy as np

from jev_policy.config import ACTION_DIM, OSC_POS_LIMIT

# Proportional gain on Cartesian position error. Above ~2.0 the Panda overshoots
# and oscillates around the target; below ~0.5 skills time out.
DEFAULT_KP = 1.2
# Never command more than this fraction of the per-step limit, so a large error
# produces a smooth approach instead of a saturated lunge.
MAX_STEP_FRACTION = 0.6


def position_action(
    eef_pos: np.ndarray,
    target_pos: np.ndarray,
    gripper: float,
    kp: float = DEFAULT_KP,
    max_fraction: float = MAX_STEP_FRACTION,
) -> np.ndarray:
    """P-control toward `target_pos`, holding orientation."""
    error = np.asarray(target_pos, dtype=np.float64) - np.asarray(eef_pos, dtype=np.float64)
    command = kp * error / OSC_POS_LIMIT
    command = np.clip(command, -max_fraction, max_fraction)

    action = np.zeros(ACTION_DIM, dtype=np.float64)
    action[:3] = command
    action[6] = gripper
    return action


def reached(eef_pos: np.ndarray, target_pos: np.ndarray, tol: float) -> bool:
    return bool(np.linalg.norm(np.asarray(target_pos) - np.asarray(eef_pos)) < tol)
