"""
Experiment configuration — JSON / CLI-based.

FR-37: Configuration through file or CLI arguments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    maze_file: str = "pacman_ai/mazes/medium.txt"
    agent: str = "random"
    agent_kwargs: Dict = field(default_factory=dict)
    num_episodes: int = 100
    seed: int = 42
    num_ghosts: int = 2
    ghost_behavior: str = "chase"
    training: bool = False
    output_csv: str = "results.csv"
    output_dir: str = "output"
    step_limit: int = 3000
    lives: int = 3

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
