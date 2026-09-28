import numpy as np

from tests.fakes import make_state


def test_text_is_relative_and_discretised():
    text = make_state().to_text()
    assert "robot_gripper: open" in text
    assert "holding: nothing" in text
    assert "bowl_1" in text and "plate_1" in text
    # Positions must be whole centimetres, never raw floats.
    assert "0.05" not in text


def test_held_object_is_reported():
    state = make_state(held="bowl_1", gripper_qpos=(0.005, -0.005))
    assert state.held_object == "bowl_1"
    assert not state.gripper_open
    assert "holding: bowl_1" in state.to_text()


def test_cache_key_is_stable_under_float_jitter():
    a = make_state(eef_pos=(0.0, 0.0, 1.10))
    b = make_state(eef_pos=(0.0001, -0.0002, 1.1001))
    assert a.cache_key() == b.cache_key()


def test_cache_key_changes_when_scene_changes():
    a = make_state()
    b = make_state(held="bowl_1")
    assert a.cache_key() != b.cache_key()
