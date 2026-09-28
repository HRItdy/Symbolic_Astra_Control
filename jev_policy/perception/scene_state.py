"""Symbolic scene state: MuJoCo floats in, text Jev can read out.

Jev is text-only, so the pixels LIBERO renders are useless to it. Instead we
read the simulator's object registry and turn it into a compact symbolic
description: where things are relative to the gripper, what is held, what is
open, what rests on what.

Two design rules here matter:

1. **Discretise.** Positions are reported in centimetres, rounded. Raw floats
   jitter every control step, which would defeat decision caching and add token
   noise for no information gain.
2. **Relative, not absolute.** The planner picks skills, not coordinates, so it
   needs "mug is 12cm in front of the gripper", not a world-frame pose. Absolute
   poses stay in `objects` for the skill layer, which does use them.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Gripper finger separation (sum of both qpos) below which we call it closed.
GRIPPER_CLOSED_QPOS = 0.05
# An object within this distance of the eef, with the gripper closed, is held.
HELD_DISTANCE_M = 0.06


@dataclasses.dataclass(frozen=True)
class ObjectObs:
    name: str
    pos: np.ndarray  # world xyz, metres
    quat: np.ndarray  # wxyz, MuJoCo convention
    is_fixture: bool
    #: Largest horizontal distance from the body centre to any contact geom.
    #: The skill layer uses this to decide whether the gripper can close around
    #: the object at all, or must grasp its rim instead.
    radius: float = 0.02
    # None when the object has no articulation (most pick-and-place objects).
    is_open: bool | None = None
    # True when robosuite reports a successful two-finger grasp.
    grasped: bool = False

    def xy_distance_to(self, other: np.ndarray) -> float:
        return float(np.linalg.norm(self.pos[:2] - np.asarray(other)[:2]))


@dataclasses.dataclass(frozen=True)
class SceneState:
    task: str
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_qpos: np.ndarray
    objects: dict[str, ObjectObs]
    # Names the BDDL goal actually refers to. The planner's target vocabulary is
    # built from this, so a chosen target is always a real, relevant object.
    objects_of_interest: tuple[str, ...]
    step: int
    #: Camera frames, keyed by view name. Populated only for planners that can
    #: see; text-only planners ignore this and pay nothing for it.
    rgb: dict[str, np.ndarray] | None = None

    @property
    def gripper_open(self) -> bool:
        return float(np.sum(np.abs(self.gripper_qpos))) > GRIPPER_CLOSED_QPOS

    @property
    def held_object(self) -> str | None:
        for name, obj in self.objects.items():
            if obj.grasped:
                return name
        return None

    def get(self, name: str) -> ObjectObs:
        return self.objects[name]

    # ---------------------------------------------------------------- text ---

    def to_text(self) -> str:
        """Render the state for Jev. Stable across jitter, ~150-300 tokens."""
        lines = [f"robot_gripper: {'open' if self.gripper_open else 'closed'}"]

        held = self.held_object
        lines.append(f"holding: {held if held else 'nothing'}")

        lines.append("objects:")
        for name in sorted(self.objects):
            obj = self.objects[name]
            rel = (obj.pos - self.eef_pos) * 100.0
            dist = float(np.linalg.norm(rel))
            parts = [
                f"  - {name}:",
                f"distance_from_gripper={dist:.0f}cm",
                f"offset(right={rel[0]:+.0f},forward={rel[1]:+.0f},up={rel[2]:+.0f})cm",
            ]
            if obj.is_open is not None:
                parts.append(f"state={'open' if obj.is_open else 'closed'}")
            if obj.grasped:
                parts.append("grasped=true")
            if obj.is_fixture:
                parts.append("fixture=true")
            lines.append(" ".join(parts))

        return "\n".join(lines)

    def pose_digest(self) -> tuple:
        """Coarse object poses, to the centimetre.

        Caching must key on this and not only on `to_text()`. A vision planner's
        prompt deliberately omits object coordinates -- it can see them -- so two
        genuinely different scenes can share identical text. Keying on text
        alone would then serve a stale decision for a moved object.
        """
        return tuple(
            (name, *np.round(self.objects[name].pos * 100).astype(int).tolist())
            for name in sorted(self.objects)
        )

    def cache_key(self) -> tuple:
        """Identity for decision caching."""
        return (self.task, self.to_text(), self.pose_digest())

    def to_proprioception_text(self) -> str:
        """Gripper state only -- no object coordinates.

        What a real robot knows about itself without perceiving the scene. This
        is the state a vision planner gets; it reads the objects off the image.
        """
        held = self.held_object
        return "\n".join(
            [
                f"robot_gripper: {'open' if self.gripper_open else 'closed'}",
                f"holding: {held if held else 'nothing'}",
            ]
        )


def _quat_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat, dtype=np.float64)


def _horizontal_radius(inner, handle, center: np.ndarray, default: float = 0.02) -> float:
    """Half-width of an object in the xy plane, from its collision geoms.

    A bowl's outer diameter can exceed the gripper's maximum opening, in which
    case a grasp aimed at its centre just presses the fingers onto the rim. The
    skill layer needs this number to notice that and aim for the rim instead.
    """
    geoms = getattr(handle, "contact_geoms", None) if handle is not None else None
    if not geoms:
        return default

    sim = inner.sim
    radii = []
    for geom in geoms:
        try:
            gid = sim.model.geom_name2id(geom)
        except (KeyError, ValueError):
            continue
        radii.append(float(np.linalg.norm(sim.data.geom_xpos[gid][:2] - center[:2])))
    return max(radii) if radii else default


def read_scene(env, task: str, step: int, rgb: dict | None = None) -> SceneState:
    """Build a SceneState from a live LIBERO env.

    `env` is a libero ControlEnv (e.g. OffScreenRenderEnv); `env.env` is the
    underlying robosuite task, which owns the object registry.
    """
    inner = env.env
    robot = inner.robots[0]
    eef_pos = np.asarray(inner.sim.data.site_xpos[robot.eef_site_id], dtype=np.float64)
    eef_quat = np.asarray(
        inner.sim.data.body_xquat[inner.sim.model.body_name2id(robot.robot_model.eef_name)],
        dtype=np.float64,
    )
    gripper_qpos = np.asarray(
        [inner.sim.data.qpos[x] for x in robot._ref_gripper_joint_pos_indexes],
        dtype=np.float64,
    )

    objects: dict[str, ObjectObs] = {}
    for name, state in inner.object_states_dict.items():
        try:
            geom = state.get_geom_state()
        except (KeyError, AttributeError):
            # Site-backed regions have no body; the skill layer resolves those
            # through the region table instead.
            continue

        # Articulated only: a cabinet/drawer exposes an `is_open` affordance on
        # the object itself. Ordinary objects have a free joint too, and probing
        # that raises -- a free joint's qpos address is a (start, end) range, not
        # a scalar index.
        is_open: bool | None = None
        handle = None
        try:
            handle = inner.get_object(name)
        except (KeyError, ValueError):
            handle = None
        if handle is not None and hasattr(handle, "is_open"):
            try:
                is_open = bool(state.is_open())
            except (
                AttributeError,
                KeyError,
                ValueError,
                IndexError,
                TypeError,
                # LIBERO's ArticulatedObject base declares is_open() but leaves
                # it unimplemented for several subclasses.
                NotImplementedError,
            ):
                is_open = None

        grasped = False
        if handle is not None and getattr(handle, "contact_geoms", None):
            try:
                grasped = bool(
                    inner._check_grasp(gripper=robot.gripper, object_geoms=handle.contact_geoms)
                )
            except (AttributeError, KeyError, ValueError, IndexError, TypeError):
                grasped = False

        pos = np.asarray(geom["pos"], dtype=np.float64).copy()
        objects[name] = ObjectObs(
            name=name,
            pos=pos,
            quat=_quat_to_wxyz(geom["quat"]).copy(),
            is_fixture=name in inner.fixtures_dict,
            radius=_horizontal_radius(inner, handle, pos),
            is_open=is_open,
            grasped=grasped,
        )

    return SceneState(
        task=task,
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        gripper_qpos=gripper_qpos,
        objects=objects,
        objects_of_interest=tuple(inner.obj_of_interest),
        rgb=rgb,
        step=step,
    )
