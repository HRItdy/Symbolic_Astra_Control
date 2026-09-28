"""Jev backends: TypeSafe's hosted System One model, and the community Jev-Omni.

Disabled by default -- see `jev_policy/plugins/__init__.py` and the README in
this directory. The loaders below are deliberately lazy: importing this package
must not import torch, the TypeSafe SDK, or anything else the core harness does
not need.
"""

from __future__ import annotations


def _hosted():
    from jev_policy.plugins.jev.planner import JevPlanner

    return JevPlanner


def _omni():
    from jev_policy.plugins.jev.omni import JevOmniPlanner

    return JevOmniPlanner


BACKENDS = {"jev": _hosted, "jev-omni": _omni}
