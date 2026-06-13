"""
Search agents — BFS and A* toward the nearest pellet.

FR-23: BFS and A* agents using Manhattan / maze distance heuristic.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import List, Tuple

from pacman_ai.engine.state import Direction, GameState
from .base import Agent


def _get_targets(state: GameState) -> frozenset[Tuple[int, int]]:
    """Return the set of all uneaten pellets and power pellets."""
    return state.pellets | state.power_pellets


def _reconstruct_first_action(
    came_from: dict, start: Tuple[int, int], goal: Tuple[int, int]
) -> Direction:
    """Walk backwards from goal to start and return the first action."""
    current = goal
    while came_from[current][0] != start:
        current = came_from[current][0]
    # came_from[current] == (start, direction)
    return came_from[current][1]


def _neighbors(state: GameState, pos: Tuple[int, int]):
    """Yield (new_pos, direction) for valid moves from *pos*."""
    for d in (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST):
        nx, ny = pos[0] + d.dx, pos[1] + d.dy
        if state.maze.is_valid_position(nx, ny):
            yield (nx, ny), d


# ── BFS ──────────────────────────────────────────────────

class BFSAgent(Agent):
    """BFS to the nearest pellet (FR-23)."""

    name = "BFS"

    def get_action(self, state: GameState) -> Direction:
        targets = _get_targets(state)
        if not targets:
            return Direction.STOP

        start = state.pacman_position
        if start in targets:
            # already on a pellet — any move is fine, pick first non-stop
            legal = state.get_legal_actions(0)
            for d in legal:
                if d != Direction.STOP:
                    return d
            return Direction.STOP

        visited: set[Tuple[int, int]] = {start}
        # (position, first_direction)
        queue: deque[Tuple[Tuple[int, int], Direction]] = deque()

        for npos, d in _neighbors(state, start):
            if npos in targets:
                return d
            visited.add(npos)
            queue.append((npos, d))

        while queue:
            pos, first_dir = queue.popleft()
            for npos, _ in _neighbors(state, pos):
                if npos not in visited:
                    if npos in targets:
                        return first_dir
                    visited.add(npos)
                    queue.append((npos, first_dir))

        return Direction.STOP  # no reachable pellets


# ── A* ───────────────────────────────────────────────────

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _min_manhattan_to_targets(
    pos: Tuple[int, int], targets: frozenset[Tuple[int, int]]
) -> int:
    if not targets:
        return 0
    return min(_manhattan(pos, t) for t in targets)


class AStarAgent(Agent):
    """A* to the nearest pellet with Manhattan distance heuristic (FR-23)."""

    name = "A*"

    def get_action(self, state: GameState) -> Direction:
        targets = _get_targets(state)
        if not targets:
            return Direction.STOP

        start = state.pacman_position
        if start in targets:
            legal = state.get_legal_actions(0)
            for d in legal:
                if d != Direction.STOP:
                    return d
            return Direction.STOP

        # Priority queue: (f, g, position, first_direction)
        counter = 0
        h = _min_manhattan_to_targets(start, targets)
        open_set: list[Tuple[int, int, int, Tuple[int, int], Direction]] = []
        g_scores: dict[Tuple[int, int], int] = {start: 0}

        for npos, d in _neighbors(state, start):
            if npos in targets:
                return d
            g = 1
            h_n = _min_manhattan_to_targets(npos, targets)
            heapq.heappush(open_set, (g + h_n, g, counter, npos, d))
            counter += 1
            g_scores[npos] = g

        while open_set:
            f, g, _, pos, first_dir = heapq.heappop(open_set)
            if g > g_scores.get(pos, float("inf")):
                continue
            for npos, _ in _neighbors(state, pos):
                new_g = g + 1
                if new_g < g_scores.get(npos, float("inf")):
                    g_scores[npos] = new_g
                    if npos in targets:
                        return first_dir
                    h_n = _min_manhattan_to_targets(npos, targets)
                    heapq.heappush(
                        open_set, (new_g + h_n, new_g, counter, npos, first_dir)
                    )
                    counter += 1

        return Direction.STOP
