"""The Jev backends must be genuinely off, and genuinely easy to switch on."""

import importlib

import pytest

from jev_policy import plugins
from jev_policy.config import PlannerConfig
from jev_policy.policies import registry


def reload_all():
    importlib.reload(plugins)
    importlib.reload(registry)


def test_jev_backends_are_not_registered_by_default(monkeypatch):
    monkeypatch.delenv("JEV_POLICY_PLUGINS", raising=False)
    reload_all()
    assert "jev" not in registry.available()
    assert "jev-omni" not in registry.available()
    assert "heuristic" in registry.available()
    assert "astra" in registry.available()


def test_the_env_var_switches_them_on(monkeypatch):
    monkeypatch.setenv("JEV_POLICY_PLUGINS", "jev")
    reload_all()
    assert "jev" in registry.available()
    assert "jev-omni" in registry.available()


def test_asking_for_a_disabled_backend_says_how_to_enable_it(monkeypatch):
    monkeypatch.delenv("JEV_POLICY_PLUGINS", raising=False)
    reload_all()
    with pytest.raises(ValueError, match="JEV_POLICY_PLUGINS=jev"):
        registry.build_planner(PlannerConfig(backend="jev-omni"))


def test_an_unknown_plugin_is_rejected(monkeypatch):
    monkeypatch.setenv("JEV_POLICY_PLUGINS", "nope")
    reload_all()
    with pytest.raises(ValueError, match="unknown plugin"):
        registry.available()


def test_core_import_does_not_pull_in_plugin_dependencies():
    """A default run must not need the TypeSafe SDK, torch, or a GPU.

    Run in a subprocess: sibling tests import the plugin directly, so measuring
    `sys.modules` in-process would only observe their imports.
    """
    import subprocess
    import sys
    import pathlib as _pathlib

    script = (
        "import sys;"
        "from jev_policy.config import PlannerConfig;"
        "from jev_policy.policies.registry import build_planner;"
        "build_planner(PlannerConfig(backend='heuristic'));"
        "leaked=[m for m in ('typesafe_sdk','torch','jev_policy.plugins.jev.omni',"
        "'jev_policy.plugins.jev.planner') if m in sys.modules];"
        "print(','.join(leaked))"
    )
    env = {"PYTHONPATH": str(_pathlib.Path(__file__).resolve().parents[1]), "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "", f"default run imported: {out.stdout.strip()}"


def teardown_module():
    reload_all()
