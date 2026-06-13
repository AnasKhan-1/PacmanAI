"""
Random agent — baseline (FR-22).
"""

from __future__ import annotations

import random

from pacman_ai.engine.state import Direction, GameState
from .base import Agent


class RandomAgent(Agent):
    """Pick a legal action uniformly at random."""

    name = "Random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def get_action(self, state: GameState) -> Direction:
        legal = state.get_legal_actions(0)
        return self.rng.choice(legal)
