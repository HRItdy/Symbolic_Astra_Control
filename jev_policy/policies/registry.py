"""Backend lookup.

Core backends are always present. Optional ones come from
`jev_policy.plugins`, which is off unless switched on -- so a backend that needs
a GPU or a paid API is never imported, let alone required, by a default run.
"""

from __future__ import annotations

from jev_policy import plugins
from jev_policy.config import PlannerConfig
from jev_policy.policies.base import Planner


def _heuristic():
    from jev_policy.policies.scripted import HeuristicPlanner

    return HeuristicPlanner


def _random():
    from jev_policy.policies.scripted import RandomPlanner

    return RandomPlanner


def _astra():
    from jev_policy.control.astra import AstraController

    return AstraController


#: name -> lazy loader. Values are imported only when that backend is selected.
CORE_BACKENDS = {
    "heuristic": _heuristic,
    "random": _random,
    "astra": _astra,
}


def _all() -> dict:
    return {**CORE_BACKENDS, **plugins.backends()}


def available() -> tuple[str, ...]:
    return tuple(_all())


def build_planner(config: PlannerConfig):
    """Build the selected backend.

    Returns a `Planner` (chooses skills) or a `Controller` (emits robot actions
    itself); the runner dispatches on which it got.
    """
    backends = _all()
    if config.backend not in backends:
        known = ", ".join(sorted(backends))
        extra = ""
        if config.backend in {"jev", "jev-omni"}:
            extra = (
                " -- the Jev backends live in a plugin that is currently off; "
                "enable it with JEV_POLICY_PLUGINS=jev "
                "(see jev_policy/plugins/jev/README.md)"
            )
        raise ValueError(f"unknown backend {config.backend!r}; available: {known}{extra}")

    cls = backends[config.backend]()
    return cls(config)


# Backwards-compatible alias: the thing built is no longer always a planner.
build_agent = build_planner
