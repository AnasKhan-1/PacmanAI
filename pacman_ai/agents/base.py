"""
Base agent interface.

FR-20: Agent interface — takes a game state, returns an action.
FR-21: Agents are interchangeable through configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pacman_ai.engine.state import Direction, GameState


class Agent(ABC):
    """Abstract base class for all Pacman agents."""

    name: str = "BaseAgent"

    @abstractmethod
    def get_action(self, state: GameState) -> Direction:
        """Choose the next action given the current game state."""
        ...

    # ── Optional RL hooks (no-ops by default) ────────────

    def observe_transition(
        self,
        state: GameState,
        action: Direction,
        next_state: GameState,
        reward: float,
    ) -> None:
        """Called after each step so RL agents can learn."""
        pass

    def start_episode(self) -> None:
        """Called at the start of each episode."""
        pass

    def end_episode(self) -> None:
        """Called at the end of each episode."""
        pass

    def set_training(self, training: bool) -> None:
        """Toggle training mode (RL agents respect this)."""
        pass
