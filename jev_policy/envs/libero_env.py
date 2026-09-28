"""LIBERO environment construction, plus the compatibility shims it needs."""

from __future__ import annotations

import functools
import pathlib

from jev_policy.config import ENV_RESOLUTION


@functools.lru_cache(maxsize=1)
def _patch_torch_load_for_init_states() -> None:
    """Let LIBERO's `torch.load` read its own init-state files on torch >= 2.6.

    LIBERO stores per-task initial simulator states as pickled numpy arrays and
    reads them with a bare `torch.load`. Torch 2.6 flipped `weights_only` to
    True, which rejects the numpy reconstructor and breaks
    `get_task_init_states` on any recent torch.

    These files ship inside the LIBERO checkout -- local, trusted benchmark
    data, not downloaded weights -- so we allowlist exactly the numpy globals
    they contain rather than disabling `weights_only` wholesale.
    """
    import numpy as np
    import torch

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:  # torch < 2.6: nothing to do
        return

    allowed = [np.ndarray, np.dtype]
    for name in ("_reconstruct", "scalar"):
        fn = getattr(np.core.multiarray, name, None)
        if fn is not None:
            allowed.append(fn)
    # numpy >= 1.25 exposes concrete dtype classes that pickle by reference.
    for dtype_name in ("Float64DType", "Float32DType", "Int64DType", "UInt8DType"):
        dtype_cls = getattr(getattr(np, "dtypes", None), dtype_name, None)
        if dtype_cls is not None:
            allowed.append(dtype_cls)

    add_safe_globals(allowed)


def load_task_suite(task_suite_name: str):
    from libero.libero import benchmark

    _patch_torch_load_for_init_states()
    return benchmark.get_benchmark_dict()[task_suite_name]()


def get_task_init_states(task_suite, task_id: int):
    _patch_torch_load_for_init_states()
    return task_suite.get_task_init_states(task_id)


def make_env(task, seed: int, resolution: int = ENV_RESOLUTION):
    """Build an offscreen LIBERO env for one task.

    Returns (env, task_description).
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    # The seed affects object placement even when a fixed initial state is set
    # afterwards, so it must be applied before reset.
    env.seed(seed)
    return env, task.language
