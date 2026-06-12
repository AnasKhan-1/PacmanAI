"""
Ghost behaviour strategies with authentic arcade logic:
  - 🔴 Blinky (Red)    – "Shadow": relentless chase + Cruise Elroy speed boost.
  - 💗 Pinky (Pink)    – "Speedy": ambushes 4 tiles ahead with the authentic upward bug.
  - 🔷 Inky (Cyan)     – "Bashful": vector doubling between Blinky and Pacman +2 tiles ahead.
  - 🍊 Clyde (Orange)  – "Pokey": chases if > 8 tiles away; panics to bottom-left corner if <= 8 tiles.
  - Staggered Ghost House Exit Routine: ghosts exit through the doorway within seconds of spawning.

FR-11: Random ghost baseline.
FR-12: Chase ghosts with authentic targeted pathfinding.
FR-14: Configurable ghost behaviors.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Tuple

from .maze import Maze
from .state import Direction, GameState, GhostMode


def _nearest_walkable_tile(maze: Maze, target: Tuple[int, int]) -> Tuple[int, int]:
    """If target is a wall or outside, find the nearest walkable tile to it."""
    tx, ty = maze.wrap_position(target[0], target[1])
    if maze.is_valid_position(tx, ty):
        return (tx, ty)

    for r in range(1, 12):
        candidates = []
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                nx, ny = maze.wrap_position(tx + dx, ty + dy)
                if maze.is_valid_position(nx, ny):
                    candidates.append((nx, ny))
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    nx, ny = maze.wrap_position(tx + dx, ty + dy)
                    if maze.is_valid_position(nx, ny):
                        candidates.append((nx, ny))
        if candidates:
            return min(candidates, key=lambda p: (p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2)

    return maze.pacman_start


def _bfs_first_step(
    state: GameState,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    legal_moves: List[Direction],
) -> Direction:
    """BFS shortest path to goal tile."""
    if not legal_moves:
        return Direction.STOP

    goal = _nearest_walkable_tile(state.maze, goal)
    if start == goal:
        return Direction.STOP

    visited = {start}
    queue: deque[Tuple[Tuple[int, int], Direction]] = deque()

    for d in legal_moves:
        nx, ny = state.maze.wrap_position(start[0] + d.dx, start[1] + d.dy)
        if (nx, ny) not in visited and state.maze.is_valid_position(nx, ny):
            if (nx, ny) == goal:
                return d
            visited.add((nx, ny))
            queue.append(((nx, ny), d))

    while queue:
        pos, first_d = queue.popleft()
        for d in (Direction.NORTH, Direction.WEST, Direction.SOUTH, Direction.EAST):
            nx, ny = state.maze.wrap_position(pos[0] + d.dx, pos[1] + d.dy)
            if (nx, ny) not in visited and state.maze.is_valid_position(nx, ny):
                if (nx, ny) == goal:
                    return first_d
                visited.add((nx, ny))
                queue.append(((nx, ny), first_d))

    return legal_moves[0]


def _frightened_action(
    state: GameState,
    ghost_pos: Tuple[int, int],
    pacman_pos: Tuple[int, int],
    legal: List[Direction],
    rng: random.Random,
) -> Direction:
    """In frightened mode, maximize distance from Pac-Man or move randomly."""
    moves = [d for d in legal if d != Direction.STOP]
    if not moves:
        return Direction.STOP

    best_moves: list[Direction] = []
    max_dist = -1.0
    for d in moves:
        nx, ny = state.maze.wrap_position(ghost_pos[0] + d.dx, ghost_pos[1] + d.dy)
        dist = (nx - pacman_pos[0]) ** 2 + (ny - pacman_pos[1]) ** 2
        if dist > max_dist:
            max_dist = dist
            best_moves = [d]
        elif dist == max_dist:
            best_moves.append(d)

    return rng.choice(best_moves) if best_moves else rng.choice(moves)


class GhostBehavior(ABC):
    """Abstract ghost behaviour strategy."""

    @abstractmethod
    def choose_action(
        self, state: GameState, ghost_index: int, rng: random.Random, legal: List[Direction]
    ) -> Direction:
        ...


class RandomGhost(GhostBehavior):
    """Uniform random legal actions baseline (FR-11)."""

    def choose_action(
        self, state: GameState, ghost_index: int, rng: random.Random, legal: List[Direction]
    ) -> Direction:
        moves = [d for d in legal if d != Direction.STOP]
        return rng.choice(moves) if moves else Direction.STOP


class ChaseGhost(GhostBehavior):
    """Direct shortest-path chase toward Pacman."""

    def choose_action(
        self, state: GameState, ghost_index: int, rng: random.Random, legal: List[Direction]
    ) -> Direction:
        ghost = state.ghosts[ghost_index]
        if ghost.mode == GhostMode.FRIGHTENED:
            return _frightened_action(state, ghost.position, state.pacman_position, legal, rng)
        moves = [d for d in legal if d != Direction.STOP]
        return _bfs_first_step(state, ghost.position, state.pacman_position, moves)


# ── 🔴 Blinky (Red) – "Shadow" ───────────────────────────

class BlinkyGhost(GhostBehavior):
    """Blinky: relentlessly pursues Pac-Man's exact tile.

    Special Mechanic: Cruise Elroy speed boost is handled in GhostController.
    """

    def choose_action(
        self, state: GameState, ghost_index: int, rng: random.Random, legal: List[Direction]
    ) -> Direction:
        ghost = state.ghosts[ghost_index]
        if ghost.mode == GhostMode.FRIGHTENED:
            return _frightened_action(state, ghost.position, state.pacman_position, legal, rng)

        moves = [d for d in legal if d != Direction.STOP]
        target = state.pacman_position
        return _bfs_first_step(state, ghost.position, target, moves)


# ── 💗 Pinky (Pink) – "Speedy" ───────────────────────────

class PinkyGhost(GhostBehavior):
    """Pinky: ambushes 4 tiles ahead of Pac-Man.

    Original Upward Bug: If Pac-Man is facing Up, targets 4 tiles up AND 4 tiles left.
    """

    def choose_action(
        self, state: GameState, ghost_index: int, rng: random.Random, legal: List[Direction]
    ) -> Direction:
        ghost = state.ghosts[ghost_index]
        if ghost.mode == GhostMode.FRIGHTENED:
            return _frightened_action(state, ghost.position, state.pacman_position, legal, rng)

        px, py = state.pacman_position
        p_dir = getattr(state, "pacman_direction", Direction.EAST)

        if p_dir == Direction.NORTH:
            # The classic arcade upward bug: 4 tiles up and 4 tiles left
            target = (px - 4, py - 4)
        else:
            target = (px + p_dir.dx * 4, py + p_dir.dy * 4)

        moves = [d for d in legal if d != Direction.STOP]
        return _bfs_first_step(state, ghost.position, target, moves)


# ── 🔷 Inky (Cyan) – "Bashful" ───────────────────────────

class InkyGhost(GhostBehavior):
    """Inky: targets twice the vector from Blinky to 2 tiles ahead of Pac-Man.

    Includes the classic upward bug on the intermediate tile.
    """

    def choose_action(
        self, state: GameState, ghost_index: int, rng: random.Random, legal: List[Direction]
    ) -> Direction:
        ghost = state.ghosts[ghost_index]
        if ghost.mode == GhostMode.FRIGHTENED:
            return _frightened_action(state, ghost.position, state.pacman_position, legal, rng)

        px, py = state.pacman_position
        p_dir = getattr(state, "pacman_direction", Direction.EAST)

        # Intermediate tile: 2 tiles ahead of Pacman (with upward bug)
        if p_dir == Direction.NORTH:
            inter = (px - 2, py - 2)
        else:
            inter = (px + p_dir.dx * 2, py + p_dir.dy * 2)

        # Blinky position (ghost 0, or fallback to center)
        blinky_pos = state.ghosts[0].position if len(state.ghosts) > 0 else (13, 14)

        # Vector from Blinky to intermediate point, doubled
        target = (2 * inter[0] - blinky_pos[0], 2 * inter[1] - blinky_pos[1])
        moves = [d for d in legal if d != Direction.STOP]
        return _bfs_first_step(state, ghost.position, target, moves)


# ── 🍊 Clyde (Orange) – "Pokey" ──────────────────────────

class ClydeGhost(GhostBehavior):
    """Clyde: chases Pac-Man if > 8 tiles away; retreats to bottom-left corner if <= 8 tiles."""

    def choose_action(
        self, state: GameState, ghost_index: int, rng: random.Random, legal: List[Direction]
    ) -> Direction:
        ghost = state.ghosts[ghost_index]
        if ghost.mode == GhostMode.FRIGHTENED:
            return _frightened_action(state, ghost.position, state.pacman_position, legal, rng)

        gx, gy = ghost.position
        px, py = state.pacman_position

        # Euclidean distance to Pac-Man
        dist = math.hypot(gx - px, gy - py)

        if dist > 8.0:
            target = (px, py)
        else:
            # Within 8 tiles: retreat to bottom-left scatter corner
            target = (1, state.maze.height - 2)

        moves = [d for d in legal if d != Direction.STOP]
        return _bfs_first_step(state, ghost.position, target, moves)


# ── Ghost Controller ─────────────────────────────────────

class GhostController:
    """Manages ghost behaviours for a whole game with Cruise Elroy & ghost house exit routine."""

    def __init__(
        self,
        behaviors: GhostBehavior | List[GhostBehavior] | str = "classic",
        num_ghosts: int = 4,
        rng: random.Random | None = None,
        speed: float = 0.75,
    ):
        if isinstance(behaviors, str):
            if behaviors == "random":
                self._behaviors = [RandomGhost() for _ in range(num_ghosts)]
            elif behaviors == "classic":
                # 4 distinct authentic ghosts
                ghost_classes = [BlinkyGhost, PinkyGhost, InkyGhost, ClydeGhost]
                self._behaviors = [ghost_classes[i % 4]() for i in range(num_ghosts)]
            else:
                self._behaviors = [ChaseGhost() for _ in range(num_ghosts)]
        elif isinstance(behaviors, GhostBehavior):
            self._behaviors = [behaviors] * num_ghosts
        else:
            self._behaviors = list(behaviors)
            while len(self._behaviors) < num_ghosts:
                self._behaviors.append(ChaseGhost())

        self.rng = rng or random.Random()
        self.speed = max(0.1, min(1.2, float(speed)))
        self.speed_scale = self.speed / 0.75 if abs(self.speed - 0.75) > 1e-4 else 1.0
        self._accumulators = [1.0] * num_ghosts

        # Staggered ghost house release delays (logical steps):
        # 🔴 Blinky: leaves immediately (0s)
        # 💗 Pinky: leaves after ~1.2s (12 steps)
        # 🔷 Inky: leaves after ~2.8s (26 steps)
        # 🍊 Clyde: leaves after ~4.4s (42 steps)
        self._spawn_timers = [0, 12, 26, 42]
        while len(self._spawn_timers) < num_ghosts:
            self._spawn_timers.append(len(self._spawn_timers) * 14)

        self._has_exited = [False] * num_ghosts
        self._door_x: int = 13
        self._door_y: int = 11
        self._door_entries: set[Tuple[int, int]] = set()

    def _init_house_geometry(self, state: GameState) -> None:
        """Determine ghost house doorway exit tile and one-way entrance barrier."""
        if not state.maze.ghost_starts:
            self._door_x = state.maze.width // 2
            self._door_y = max(0, state.maze.height // 2 - 2)
            return

        gx_vals = [p[0] for p in state.maze.ghost_starts]
        gy_vals = [p[1] for p in state.maze.ghost_starts]
        min_gx, max_gx = min(gx_vals), max(gx_vals)
        min_gy = min(gy_vals)

        # In 28x30 map, door is at y=12, x in (13, 14); exit is y=11
        self._door_x = (min_gx + max_gx) // 2
        self._door_y = min_gy - 2

        # Record doorway tiles that ghosts outside cannot enter
        doorway_y = min_gy - 1
        self._door_entries = {
            (x, doorway_y)
            for x in range(min_gx - 1, max_gx + 2)
            if state.maze.is_valid_position(x, doorway_y)
        }

    def get_effective_speed(self, i: int, state: GameState) -> float:
        """Return the current continuous speed factor for ghost i.

        Calibrated to arcade baseline:
        - Pac-Man Normal: 0.80 max base speed (handled in GameUI)
        - Pac-Man Frightened: 0.90 max base speed
        - Ghost Normal: 0.75 max base speed
        - Ghost Frightened: 0.50 max base speed
        - Ghost Eyes (Eaten): 2.00 max base speed (returns quickly to house)
        - Cruise Elroy: Blinky speeds up as dots dwindle
        """
        ghost = state.ghosts[i] if i < len(state.ghosts) else None
        scale = getattr(self, "speed_scale", 1.0)

        # 1. Eaten ghost eyes move equal to or slightly slower than normal ghost speed (no 2.0x or 1.5x)
        if ghost is not None and ghost.mode == GhostMode.EATEN:
            return 0.85 * 0.75 * scale  # 0.85 * ghost_normal_speed

        # 2. Frightened ghosts slow down to 50% speed
        if ghost is not None and ghost.mode == GhostMode.FRIGHTENED:
            return 0.50 * scale

        # 3. Normal ghost: 75% base speed
        eff_speed = 0.75

        # 🔴 Blinky Cruise Elroy: speeds up as dots dwindle
        if i == 0 and len(self._behaviors) > 0 and isinstance(self._behaviors[0], BlinkyGhost):
            remaining_dots = len(state.pellets) + len(state.power_pellets)
            if remaining_dots <= 20:
                eff_speed = 0.85
            elif remaining_dots <= 60:
                eff_speed = 0.80

        return eff_speed * scale

    def is_inside_house(self, gx: int, gy: int, state: GameState) -> bool:
        """Check if tile (gx, gy) is inside the central ghost house."""
        if not state.maze.ghost_starts:
            return gy >= self._door_y + 2
        gx_vals = [p[0] for p in state.maze.ghost_starts]
        min_gx, max_gx = min(gx_vals), max(gx_vals)
        return gy >= self._door_y + 2 and (min_gx - 1 <= gx <= max_gx + 1)

    def get_single_ghost_action(self, i: int, state: GameState) -> Direction:
        """Compute the next directional move for ghost i without accumulator skipping."""
        if not self._door_entries:
            self._init_house_geometry(state)

        while len(self._behaviors) < state.num_ghosts:
            self._behaviors.append(ChaseGhost())
        while len(self._has_exited) < state.num_ghosts:
            self._has_exited.append(False)
        while len(self._spawn_timers) < state.num_ghosts:
            self._spawn_timers.append(0)

        ghost = state.ghosts[i]
        gx, gy = ghost.position
        raw_legal = state.get_legal_actions(i + 1)

        # ── Eaten Ghost (Eyes Only) Return Navigation ──
        if ghost.mode == GhostMode.EATEN:
            door_x = self._door_x
            door_y = self._door_y
            doorway_y = self._door_y + 1

            # Check if eyes have safely entered inside the ghost house
            if self.is_inside_house(gx, gy, state):
                # House reached! Will be respawned in GameUI
                return Direction.STOP

            # At door entrance (above house) -> step SOUTH into doorway
            if (gx == door_x or gx == door_x + 1) and gy == door_y:
                if Direction.SOUTH in raw_legal:
                    return Direction.SOUTH

            # Inside doorway -> step SOUTH into house
            if gy == doorway_y:
                if Direction.SOUTH in raw_legal:
                    return Direction.SOUTH

            # In open maze: shortest-path BFS to outside door tile
            valid_moves = [d for d in raw_legal if d != Direction.STOP]
            return _bfs_first_step(state, (gx, gy), (door_x, door_y), valid_moves)

        # Check if ghost has reached outside the ghost house (y <= door_y)
        if gy <= self._door_y:
            self._has_exited[i] = True

        # Once outside, ghosts cannot re-enter the ghost house doorway!
        if self._has_exited[i]:
            legal = [
                d for d in raw_legal
                if state.maze.wrap_position(gx + d.dx, gy + d.dy) not in self._door_entries
            ]
            if not legal:
                legal = raw_legal
        else:
            legal = raw_legal

        # ── Ghost House Exit Routine ──
        if not self._has_exited[i]:
            # Waiting inside box for release timer
            if self._spawn_timers[i] > 0:
                self._spawn_timers[i] -= 1
                bob_dir = Direction.NORTH if (self._spawn_timers[i] // 3) % 2 == 0 else Direction.SOUTH
                if bob_dir in legal:
                    return bob_dir
                valid_moves = [d for d in legal if d != Direction.STOP]
                return self.rng.choice(valid_moves) if valid_moves else Direction.STOP

            # Timer expired: Leave the ghost house!
            if gx < self._door_x and Direction.EAST in legal:
                return Direction.EAST
            elif gx > self._door_x + 1 and Direction.WEST in legal:
                return Direction.WEST
            elif Direction.NORTH in legal:
                return Direction.NORTH
            else:
                return _bfs_first_step(state, (gx, gy), (self._door_x, self._door_y), [d for d in legal if d != Direction.STOP])

        # ── Normal Roaming / Chase / Ambush AI ──
        return self._behaviors[i].choose_action(state, i, self.rng, legal)

    def get_actions(self, state: GameState) -> List[Direction]:
        """Return a list of ghost actions for turn-based execution."""
        actions: list[Direction] = []
        for i in range(state.num_ghosts):
            eff_speed = self.get_effective_speed(i, state)
            while len(self._accumulators) <= i:
                self._accumulators.append(1.0)
            self._accumulators[i] += eff_speed
            if self._accumulators[i] < 1.0:
                actions.append(Direction.STOP)
                continue
            self._accumulators[i] -= 1.0
            actions.append(self.get_single_ghost_action(i, state))
        return actions
