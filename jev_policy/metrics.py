"""Result accounting for an evaluation run."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections import defaultdict


@dataclasses.dataclass
class EpisodeResult:
    task_id: int
    task_description: str
    episode: int
    success: bool
    steps: int
    decisions: int
    skills_executed: list[str] = dataclasses.field(default_factory=list)
    mean_confidence: float = 0.0
    mean_latency_s: float = 0.0
    failure_reason: str = ""


class RunMetrics:
    def __init__(self, suite: str) -> None:
        self.suite = suite
        self.episodes: list[EpisodeResult] = []

    def add(self, result: EpisodeResult) -> None:
        self.episodes.append(result)

    @property
    def total(self) -> int:
        return len(self.episodes)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def per_task(self) -> dict[int, dict]:
        grouped: dict[int, list[EpisodeResult]] = defaultdict(list)
        for e in self.episodes:
            grouped[e.task_id].append(e)

        out = {}
        for task_id, eps in sorted(grouped.items()):
            out[task_id] = {
                "task": eps[0].task_description,
                "episodes": len(eps),
                "successes": sum(1 for e in eps if e.success),
                "success_rate": sum(1 for e in eps if e.success) / len(eps),
                "mean_steps": sum(e.steps for e in eps) / len(eps),
                "mean_decisions": sum(e.decisions for e in eps) / len(eps),
            }
        return out

    def summary(self, planner_stats: dict | None = None) -> dict:
        return {
            "suite": self.suite,
            "episodes": self.total,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "per_task": self.per_task(),
            "planner": planner_stats or {},
        }

    def save(self, path: str | pathlib.Path, planner_stats: dict | None = None) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary(planner_stats)
        payload["episodes_detail"] = [dataclasses.asdict(e) for e in self.episodes]
        path.write_text(json.dumps(payload, indent=2))
        return path
