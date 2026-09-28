"""Optional backends, off unless switched on.

A plugin is a package under this one exposing `BACKENDS`: a mapping of backend
name to a zero-argument import hook returning the class. Nothing in a disabled
plugin is imported, so a plugin may depend on packages the core does not have.

Enable one either by editing `ENABLED_BY_DEFAULT` below, or without touching the
source at all:

    JEV_POLICY_PLUGINS=jev python scripts/eval_libero.py --planner.backend jev-omni
"""

from __future__ import annotations

import importlib
import os

#: Plugins active without any environment variable. Add "jev" here to make the
#: Jev and Jev-Omni backends permanently available again.
ENABLED_BY_DEFAULT: tuple[str, ...] = ()

#: Every plugin that ships in this tree, whether enabled or not.
KNOWN = ("jev",)


def enabled() -> tuple[str, ...]:
    override = os.environ.get("JEV_POLICY_PLUGINS")
    if override is None:
        return ENABLED_BY_DEFAULT
    names = tuple(n.strip() for n in override.split(",") if n.strip())
    unknown = [n for n in names if n not in KNOWN]
    if unknown:
        raise ValueError(f"unknown plugin(s) {unknown}; known plugins: {list(KNOWN)}")
    return names


def backends() -> dict:
    """Backend name -> loader, for every enabled plugin."""
    found: dict = {}
    for name in enabled():
        module = importlib.import_module(f"jev_policy.plugins.{name}")
        found.update(module.BACKENDS)
    return found


def describe() -> str:
    """One line for the CLI, so a disabled backend is discoverable, not mysterious."""
    active = enabled()
    off = [n for n in KNOWN if n not in active]
    parts = []
    if active:
        parts.append(f"enabled: {', '.join(active)}")
    if off:
        parts.append(f"available but off: {', '.join(off)} (set JEV_POLICY_PLUGINS)")
    return "; ".join(parts) or "none"
