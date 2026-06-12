"""
Game state — immutable, copyable state object and direction enum.

FR-2: Immutable/copyable state with positions, pellets, score, lives, ghost modes.
FR-3: Legal-actions function.
FR-4: Successor function that returns a new state without modifying the original.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple

from .maze import Maze


class Direction(Enum):
    """Possible movement directions (and stop)."""

    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    STOP = (0, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    def opposite(self) -> "Direction":
        opposites = {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST,
            Direction.STOP: Direction.STOP,
        }
        return opposites[self]


class GhostMode(Enum):
    """Ghost behaviour mode."""

    CHASE = "chase"
    SCATTER = "scatter"
    FRIGHTENED = "frightened"  # power-pellet scared
    EATEN = "eaten"  # eyes returning to ghost house


@dataclass(frozen=True)
class GhostState:
    """State of a single ghost."""

    position: Tuple[int, int]
    mode: GhostMode = GhostMode.CHASE
    start_position: Tuple[int, int] = (0, 0)
    frightened_timer: int = 0  # steps remaining in frightened mode


@dataclass(frozen=True)
class GameState:
    """Complete, immutable snapshot of a Pacman game.

    All mutating operations return a *new* GameState; this one is never changed.
    """

    maze: Maze
    pacman_position: Tuple[int, int]
    ghosts: Tuple[GhostState, ...]
    pellets: FrozenSet[Tuple[int, int]]
    power_pellets: FrozenSet[Tuple[int, int]]
    pacman_direction: Direction = Direction.EAST
    score: int = 0
    lives: int = 3
    step_count: int = 0
    is_win: bool = False
    is_lose: bool = False
    pellets_eaten: int = 0

    # ── Configuration (scoring / timers) ─────────────────

    pellet_score: int = 10
    power_pellet_score: int = 50
    ghost_eat_score: int = 200
    win_bonus: int = 500
    death_penalty: int = -500
    step_penalty: int = -1
    frightened_duration: int = 20
    step_limit: int = 3000

    # ── Queries ──────────────────────────────────────────

    def get_legal_actions(self, agent_index: int = 0) -> List[Direction]:
        """Return legal actions for an agent.

        agent_index 0 = Pacman; 1..N = ghosts.
        Pacman may also STOP; ghosts may not (keeps the game moving).
        """
        if agent_index == 0:
            pos = self.pacman_position
            actions = [Direction.STOP]
        else:
            ghost = self.ghosts[agent_index - 1]
            pos = ghost.position
            actions = []

        for d in (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST):
            nx, ny = pos[0] + d.dx, pos[1] + d.dy
            if self.maze.is_valid_position(nx, ny):
                actions.append(d)

        # Ghosts must have at least one option
        if not actions:
            actions = [Direction.STOP]
        return actions

    def is_terminal(self) -> bool:
        """Return True if the game has ended (win or loss)."""
        return self.is_win or self.is_lose

    @property
    def num_ghosts(self) -> int:
        return len(self.ghosts)

    # ── Factory ──────────────────────────────────────────

    @classmethod
    def from_maze(
        cls,
        maze: Maze,
        num_ghosts: int | None = None,
        lives: int = 3,
        **config,
    ) -> "GameState":
        """Create the initial GameState from a Maze object."""
        if num_ghosts is None:
            num_ghosts = len(maze.ghost_starts)
        num_ghosts = min(num_ghosts, len(maze.ghost_starts))

        ghost_states = tuple(
            GhostState(position=maze.ghost_starts[i], start_position=maze.ghost_starts[i])
            for i in range(num_ghosts)
        )

        return cls(
            maze=maze,
            pacman_position=maze.pacman_start,
            ghosts=ghost_states,
            pellets=maze.pellets,
            power_pellets=maze.power_pellets,
            lives=lives,
            **config,
        )
