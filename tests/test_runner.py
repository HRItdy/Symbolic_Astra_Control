"""Runner tests against a stub simulator -- no MuJoCo required."""

import numpy as np

import jev_policy.runner as runner_mod
from jev_policy.config import EvalConfig, PlannerConfig
from jev_policy.policies.base import Decision, Planner
from jev_policy.runner import run_episode
from tests.fakes import make_state


class StubEnv:
    """Accepts actions, and reports success after `succeed_after` steps."""

    def __init__(self, succeed_after=None):
        self.succeed_after = succeed_after
        self.steps = 0
        self.actions = []

    def reset(self):
        self.steps = 0

    def set_init_state(self, state):
        return {}

    def step(self, action):
        self.steps += 1
        self.actions.append(action)
        return {}, 0.0, False, {}

    def check_success(self):
        return self.succeed_after is not None and self.steps >= self.succeed_after

    def close(self):
        pass


class ScriptedPlanner(Planner):
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.asked = 0

    def decide(self, state, history):
        self.asked += 1
        return self.decisions[min(self.asked - 1, len(self.decisions) - 1)]


def config(**kwargs):
    return EvalConfig(
        task_suite_name="libero_spatial",
        num_steps_wait=2,
        planner=PlannerConfig(backend="heuristic", **kwargs),
    )


def patch_scene(monkeypatch, state=None):
    monkeypatch.setattr(
        runner_mod, "read_scene", lambda env, task, t, rgb=None: state or make_state()
    )


def test_success_is_taken_from_the_simulator(monkeypatch):
    patch_scene(monkeypatch)
    env = StubEnv(succeed_after=20)
    planner = ScriptedPlanner([Decision(skill="move_above", target="bowl_1", confidence=0.9)])

    result, _ = run_episode(env, planner, config(), 0, "task", 0, None, collect_frames=False)

    assert result.success
    assert result.steps <= 25


def test_failure_when_budget_is_exhausted(monkeypatch):
    patch_scene(monkeypatch)
    env = StubEnv(succeed_after=None)
    planner = ScriptedPlanner([Decision(skill="move_above", target="bowl_1", confidence=0.9)])

    result, _ = run_episode(env, planner, config(), 0, "task", 0, None, collect_frames=False)

    assert not result.success
    assert result.failure_reason == "step budget exhausted"


def test_settle_steps_run_before_any_decision(monkeypatch):
    patch_scene(monkeypatch)
    env = StubEnv(succeed_after=3)
    planner = ScriptedPlanner([Decision(skill="move_above", target="bowl_1", confidence=0.9)])

    run_episode(env, planner, config(), 0, "task", 0, None, collect_frames=False)

    # First two actions are the settle no-op: zero motion, gripper open.
    assert planner.asked == 0 or env.actions[0][:6] == [0.0] * 6


def test_low_confidence_triggers_retreat_not_the_chosen_skill(monkeypatch):
    patch_scene(monkeypatch)
    env = StubEnv(succeed_after=None)
    planner = ScriptedPlanner([Decision(skill="descend", target="bowl_1", confidence=0.05)])

    result, _ = run_episode(
        env, planner, config(min_confidence=0.45), 0, "task", 0, None, collect_frames=False
    )

    assert all("Retreat" in s for s in result.skills_executed)
    assert "Descend" not in result.skills_executed


def test_planner_errors_end_the_episode_cleanly(monkeypatch):
    patch_scene(monkeypatch)

    class Boom(Planner):
        def decide(self, state, history):
            raise RuntimeError("api down")

    result, _ = run_episode(
        StubEnv(), Boom(), config(), 0, "task", 0, None, collect_frames=False
    )

    assert not result.success
    assert "api down" in result.failure_reason


def test_actions_sent_to_the_env_are_valid(monkeypatch):
    patch_scene(monkeypatch)
    env = StubEnv(succeed_after=30)
    planner = ScriptedPlanner([Decision(skill="move_above", target="bowl_1", confidence=0.9)])

    run_episode(env, planner, config(), 0, "task", 0, None, collect_frames=False)

    for action in env.actions:
        assert len(action) == 7
        assert all(-1.0 <= float(a) <= 1.0 for a in action)


def test_repeated_no_progress_skill_is_called_out_in_the_history(monkeypatch):
    """A primitive that succeeds without changing the scene is a livelock; the
    history must say so or the planner picks it again forever."""
    patch_scene(monkeypatch)
    seen = []

    class Recorder(Planner):
        def decide(self, state, history):
            seen.append(list(history))
            return Decision(skill="open_gripper", target=None, confidence=0.9)

    run_episode(
        StubEnv(succeed_after=None), Recorder(), config(), 0, "t", 0, None, collect_frames=False
    )

    flat = [line for h in seen for line in h]
    assert any("already done" in line and "changes nothing" in line for line in flat)


# --- direct control (no skill layer) ------------------------------------------


def test_controller_actions_reach_the_simulator_unmediated(monkeypatch):
    """Direct control must bypass skills entirely: what the model emits is what
    the simulator gets."""
    patch_scene(monkeypatch)
    from jev_policy.control.base import ControlChunk, Controller

    commanded = [np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, -1.0])] * 5

    class Fixed(Controller):
        def act(self, state, history):
            return ControlChunk(actions=list(commanded), latency_s=0.01, note="go")

    env = StubEnv(succeed_after=None)
    result, _ = run_episode(env, Fixed(), config(), 0, "t", 0, None, collect_frames=False)

    assert not result.success
    post_settle = env.actions[config().num_steps_wait:]
    assert post_settle[0] == commanded[0].tolist()


def test_empty_chunk_ends_the_episode_rather_than_spinning(monkeypatch):
    """With nothing to execute the state cannot change, so re-asking would burn
    the budget on an identical observation."""
    patch_scene(monkeypatch)
    from jev_policy.control.base import ControlChunk, Controller

    class Empty(Controller):
        def __init__(self):
            self.calls = 0

        def act(self, state, history):
            self.calls += 1
            return ControlChunk(actions=[], note="unparseable response")

    controller = Empty()
    result, _ = run_episode(
        StubEnv(), controller, config(), 0, "t", 0, None, collect_frames=False
    )

    assert controller.calls == 1
    assert "unparseable" in result.failure_reason


def test_controller_success_is_still_the_simulator_s_call(monkeypatch):
    patch_scene(monkeypatch)
    from jev_policy.control.base import ControlChunk, Controller

    class Mover(Controller):
        def act(self, state, history):
            return ControlChunk(actions=[np.zeros(7)] * 5)

    result, _ = run_episode(
        StubEnv(succeed_after=14), Mover(), config(), 0, "t", 0, None, collect_frames=False
    )
    assert result.success
