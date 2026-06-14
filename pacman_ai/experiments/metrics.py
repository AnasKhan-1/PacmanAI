"""
Metrics collection and CSV export.

FR-32: Record per episode: win/loss, score, steps, pellets, runtime.
FR-34: Export results to CSV.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class EpisodeResult:
    """Statistics for a single game episode."""

    episode: int
    agent: str
    maze: str
    seed: int
    won: bool
    score: int
    steps: int
    pellets_eaten: int
    runtime_ms: float


class MetricsCollector:
    """Accumulate episode results and export to CSV."""

    def __init__(self):
        self.results: List[EpisodeResult] = []

    def add(self, result: EpisodeResult) -> None:
        self.results.append(result)

    def to_csv(self, path: str | Path) -> None:
        """Write all results to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            return
        fieldnames = list(asdict(self.results[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))

    # ── Summary statistics ───────────────────────────────

    def win_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.won) / len(self.results)

    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def avg_steps(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.steps for r in self.results) / len(self.results)

    def avg_runtime_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.runtime_ms for r in self.results) / len(self.results)

    def summary_dict(self) -> dict:
        return {
            "episodes": len(self.results),
            "win_rate": f"{self.win_rate():.2%}",
            "avg_score": f"{self.avg_score():.1f}",
            "avg_steps": f"{self.avg_steps():.1f}",
            "avg_runtime_ms": f"{self.avg_runtime_ms():.2f}",
        }
