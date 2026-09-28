"""Core planner tests."""

from jev_policy.policies.scripted import HeuristicPlanner
from tests.fakes import make_state


def test_heuristic_planner_runs_the_full_pick_and_place_sequence():
    planner = HeuristicPlanner()
    state = make_state()
    decisions = [planner.decide(state, []) for _ in range(6)]

    assert [d.skill for d in decisions] == [
        "move_above",
        "descend",
        "close_gripper",
        "lift",
        "place_on",
        "open_gripper",
    ]
    assert decisions[0].target == "bowl_1"
    assert decisions[4].target == "plate_1"
    assert decisions[2].target is None


def test_heuristic_planner_restarts_the_approach_after_a_failure():
    planner = HeuristicPlanner()
    state = make_state()
    planner.decide(state, [])
    planner.decide(state, ["MoveAbove(bowl_1) -> ok"])
    retry = planner.decide(state, ["Descend(bowl_1) -> failed: drifted off target"])
    assert retry.skill == "move_above"
