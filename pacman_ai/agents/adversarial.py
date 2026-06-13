"""
Adversarial agents — Minimax (alpha-beta) and Expectimax.

FR-24: Minimax with alpha-beta pruning and configurable depth.
FR-25: Expectimax modelling ghosts as random.
FR-26: Custom evaluation function.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Tuple

from pacman_ai.engine.state import Direction, GameState
from pacman_ai.engine.rules import GameRules
from .base import Agent


# ── Evaluation function (FR-26) ─────────────────────────

def _bfs_nearest(
    state: GameState, start: Tuple[int, int], targets: frozenset[Tuple[int, int]]
) -> int:
    """Return the BFS distance from *start* to the nearest target.

    Returns a large number if no target is reachable.
    """
    if not targets:
        return 0
    if start in targets:
        return 0
    visited: set[Tuple[int, int]] = {start}
    queue: deque[Tuple[Tuple[int, int], int]] = deque()
    for d in (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST):
        nx, ny = start[0] + d.dx, start[1] + d.dy
        if state.maze.is_valid_position(nx, ny):
            if (nx, ny) in targets:
                return 1
            visited.add((nx, ny))
            queue.append(((nx, ny), 1))
    while queue:
        pos, dist = queue.popleft()
        for d in (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST):
            nx, ny = pos[0] + d.dx, pos[1] + d.dy
            if (nx, ny) not in visited and state.maze.is_valid_position(nx, ny):
                if (nx, ny) in targets:
                    return dist + 1
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return 999


def evaluation_function(state: GameState) -> float:
    """Evaluate a non-terminal state for the adversarial agents.

    Features:
      - Current score
      - Distance to nearest pellet (negative: closer is better)
      - Distance to nearest ghost (positive: farther is better, unless frightened)
      - Number of remaining pellets (fewer is better)
    """
    if state.is_win:
        return 100_000
    if state.is_lose:
        return -100_000

    score = float(state.score)
    all_pellets = state.pellets | state.power_pellets

    # Distance to nearest food
    nearest_food = _bfs_nearest(state, state.pacman_position, all_pellets)
    score -= 2.0 * nearest_food

    # Ghost distances
    for ghost in state.ghosts:
        dist = abs(state.pacman_position[0] - ghost.position[0]) + abs(
            state.pacman_position[1] - ghost.position[1]
        )
        if ghost.mode.value == "frightened":
            # Closer is better when ghost is frightened
            score += max(0, 10 - dist) * 20
        else:
            if dist <= 1:
                score -= 500
            elif dist <= 3:
                score -= 50

    # Penalise remaining pellets
    score -= 4.0 * len(all_pellets)

    return score


# ── Helper: one-step successor with a single ghost ──────

def _get_successor_for_agent(
    state: GameState, agent_index: int, action: Direction
) -> GameState:
    """Produce successor after one agent moves.

    agent_index 0 = Pacman; ≥1 = ghost.
    """
    if agent_index == 0:
        return GameRules.get_successor(state, action, ghost_actions=None)
    else:
        ghost_actions = [Direction.STOP] * state.num_ghosts
        ghost_actions[agent_index - 1] = action
        return GameRules.get_successor(state, Direction.STOP, ghost_actions)


# ── Minimax with alpha-beta (FR-24) ─────────────────────

class MinimaxAgent(Agent):
    """Minimax agent with alpha-beta pruning."""

    name = "Minimax"

    def __init__(self, depth: int = 2):
        self.depth = depth

    def get_action(self, state: GameState) -> Direction:
        best_action = Direction.STOP
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in state.get_legal_actions(0):
            successor = GameRules.get_successor(
                state, action, [Direction.STOP] * state.num_ghosts
            )
            value = self._min_value(successor, 0, 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

    def _max_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        if state.is_terminal() or depth >= self.depth:
            return evaluation_function(state)

        value = -math.inf
        for action in state.get_legal_actions(0):
            successor = GameRules.get_successor(
                state, action, [Direction.STOP] * state.num_ghosts
            )
            value = max(value, self._min_value(successor, depth, 1, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(
        self,
        state: GameState,
        depth: int,
        ghost_index: int,
        alpha: float,
        beta: float,
    ) -> float:
        if state.is_terminal():
            return evaluation_function(state)

        value = math.inf
        for action in state.get_legal_actions(ghost_index):
            ghost_actions = [Direction.STOP] * state.num_ghosts
            ghost_actions[ghost_index - 1] = action

            if ghost_index < state.num_ghosts:
                # More ghosts to move at this depth
                successor = GameRules.get_successor(
                    state, Direction.STOP, ghost_actions
                )
                value = min(
                    value,
                    self._min_value(
                        successor, depth, ghost_index + 1, alpha, beta
                    ),
                )
            else:
                # All ghosts moved — next Pacman layer
                successor = GameRules.get_successor(
                    state, Direction.STOP, ghost_actions
                )
                value = min(
                    value, self._max_value(successor, depth + 1, alpha, beta)
                )

            if value < alpha:
                return value
            beta = min(beta, value)
        return value


# ── Expectimax (FR-25) ───────────────────────────────────

class ExpectimaxAgent(Agent):
    """Expectimax agent — ghosts modelled as uniformly random."""

    name = "Expectimax"

    def __init__(self, depth: int = 2):
        self.depth = depth

    def get_action(self, state: GameState) -> Direction:
        best_action = Direction.STOP
        best_value = -math.inf

        for action in state.get_legal_actions(0):
            successor = GameRules.get_successor(
                state, action, [Direction.STOP] * state.num_ghosts
            )
            value = self._exp_value(successor, 0, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _max_value(self, state: GameState, depth: int) -> float:
        if state.is_terminal() or depth >= self.depth:
            return evaluation_function(state)

        value = -math.inf
        for action in state.get_legal_actions(0):
            successor = GameRules.get_successor(
                state, action, [Direction.STOP] * state.num_ghosts
            )
            value = max(value, self._exp_value(successor, depth, 1))
        return value

    def _exp_value(
        self, state: GameState, depth: int, ghost_index: int
    ) -> float:
        if state.is_terminal():
            return evaluation_function(state)

        actions = state.get_legal_actions(ghost_index)
        prob = 1.0 / len(actions)
        total = 0.0

        for action in actions:
            ghost_actions = [Direction.STOP] * state.num_ghosts
            ghost_actions[ghost_index - 1] = action

            if ghost_index < state.num_ghosts:
                successor = GameRules.get_successor(
                    state, Direction.STOP, ghost_actions
                )
                total += prob * self._exp_value(successor, depth, ghost_index + 1)
            else:
                successor = GameRules.get_successor(
                    state, Direction.STOP, ghost_actions
                )
                total += prob * self._max_value(successor, depth + 1)

        return total
