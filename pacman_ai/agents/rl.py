"""
Reinforcement-learning agents — tabular Q-learning and approximate Q-learning.

FR-27: Tabular Q-learning with configurable LR, discount, exploration.
FR-28: Approximate Q-learning with feature-based linear function approximator.
FR-30: Save / load learned weights / tables.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

from pacman_ai.engine.state import Direction, GameState, GhostMode
from .base import Agent


# ── Feature extraction (FR-28 / data requirements) ──────

def _bfs_food_distance(state: GameState, start_pos: Tuple[int, int]) -> int:
    """Return exact shortest walkable corridor distance from start_pos to nearest food pellet."""
    all_food = state.pellets | state.power_pellets
    if not all_food:
        return 0
    if start_pos in all_food:
        return 0

    visited = {start_pos}
    queue = deque([(start_pos[0], start_pos[1], 0)])
    while queue:
        cx, cy, d = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = state.maze.wrap_position(cx + dx, cy + dy)
            if (nx, ny) not in visited and not state.maze.is_wall(nx, ny):
                if (nx, ny) in all_food:
                    return d + 1
                visited.add((nx, ny))
                queue.append((nx, ny, d + 1))
    return 999


def _bfs_nearest_food_direction(state: GameState, pos: Tuple[int, int]) -> Direction:
    """Return the first corridor step direction that leads toward the nearest food pellet."""
    all_food = state.pellets | state.power_pellets
    if not all_food:
        return Direction.STOP

    queue = deque()
    visited = {pos}
    for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
        nx, ny = state.maze.wrap_position(pos[0] + d.dx, pos[1] + d.dy)
        if not state.maze.is_wall(nx, ny):
            if (nx, ny) in all_food:
                return d
            visited.add((nx, ny))
            queue.append((nx, ny, d))

    while queue:
        cx, cy, first_d = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = state.maze.wrap_position(cx + dx, cy + dy)
            if (nx, ny) not in visited and not state.maze.is_wall(nx, ny):
                if (nx, ny) in all_food:
                    return first_d
                visited.add((nx, ny))
                queue.append((nx, ny, first_d))
    return Direction.STOP


def extract_features(state: GameState, action: Direction) -> Dict[str, float]:
    """Return a feature dictionary for approximate Q-learning.

    Features (from §3.4):
      - bias: always 1.0
      - nearest_food: 1/(shortest corridor distance to nearest pellet via BFS)
      - nearest_ghost: 1/(distance to nearest ghost)
      - ghosts_nearby: # of ghosts within radius 3
      - eats_food: 1 if action leads to a cell with a pellet
    """
    raw_x = state.pacman_position[0] + action.dx
    raw_y = state.pacman_position[1] + action.dy
    px, py = state.maze.wrap_position(raw_x, raw_y)
    new_pos = (px, py)

    # If the action leads into a wall, use old position
    if state.maze.is_wall(px, py):
        new_pos = state.pacman_position

    features: Dict[str, float] = {"bias": 1.0}

    # Nearest food via BFS walkable corridor distance (eliminates wall entrapment)
    all_food = state.pellets | state.power_pellets
    if all_food:
        dist = _bfs_food_distance(state, new_pos)
        features["nearest_food"] = 1.0 / (dist + 1)
    else:
        features["nearest_food"] = 0.0

    # Nearest ghost
    if state.ghosts:
        ghost_dists = [
            abs(new_pos[0] - g.position[0]) + abs(new_pos[1] - g.position[1])
            for g in state.ghosts
            if g.mode != GhostMode.FRIGHTENED
        ]
        if ghost_dists:
            features["nearest_ghost"] = 1.0 / (min(ghost_dists) + 1)
        else:
            features["nearest_ghost"] = 0.0
    else:
        features["nearest_ghost"] = 0.0

    # Ghosts within radius 3
    nearby = sum(
        1
        for g in state.ghosts
        if abs(new_pos[0] - g.position[0]) + abs(new_pos[1] - g.position[1]) <= 3
        and g.mode != GhostMode.FRIGHTENED
    )
    features["ghosts_nearby"] = float(nearby) / max(len(state.ghosts), 1)

    # Eats food?
    features["eats_food"] = 1.0 if new_pos in all_food else 0.0

    return features


# ── Tabular Q-learning (FR-27) ───────────────────────────

class QLearningAgent(Agent):
    """Tabular Q-learning agent with a compact, learnable corridor-based state abstraction."""

    name = "Q-Learning"

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        self._training = True

    # ── State key ────────────────────────────────────────

    @staticmethod
    def _state_key(state: GameState) -> Tuple:
        """Produce a compact, learnable state abstraction for tabular Q-learning."""
        food_dir = _bfs_nearest_food_direction(state, state.pacman_position)
        px, py = state.pacman_position

        # Threat direction: nearest non-frightened ghost within radius 4
        threat_dir = Direction.STOP
        min_ghost_dist = 999
        for g in state.ghosts:
            if g.mode != GhostMode.FRIGHTENED:
                d = abs(px - g.position[0]) + abs(py - g.position[1])
                if d < min_ghost_dist:
                    min_ghost_dist = d
                    if d <= 4:
                        dx = 1 if g.position[0] > px else (-1 if g.position[0] < px else 0)
                        dy = 1 if g.position[1] > py else (-1 if g.position[1] < py else 0)
                        threat_dir = (dx, dy)

        # Scared ghosts nearby?
        scared_near = any(
            g.mode == GhostMode.FRIGHTENED and (abs(px - g.position[0]) + abs(py - g.position[1])) <= 4
            for g in state.ghosts
        )

        return (
            state.pacman_position,
            food_dir,
            threat_dir,
            scared_near,
        )

    def _q(self, state: GameState, action: Direction) -> float:
        return self.q_table[(self._state_key(state), action)]

    # ── Agent interface ──────────────────────────────────

    def get_action(self, state: GameState) -> Direction:
        legal = state.get_legal_actions(0)
        moves = [a for a in legal if a != Direction.STOP]
        if not moves:
            moves = [Direction.STOP]

        if self._training and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)

        q_vals = [(self._q(state, a), a) for a in moves]
        max_q = max(q for q, _ in q_vals)
        best_moves = [a for q, a in q_vals if abs(q - max_q) < 1e-6]

        curr_dir = getattr(state, "pacman_direction", Direction.STOP)
        if curr_dir in best_moves:
            return curr_dir

        return self.rng.choice(best_moves)

    def observe_transition(
        self,
        state: GameState,
        action: Direction,
        next_state: GameState,
        reward: float,
    ) -> None:
        if not self._training:
            return
        legal = next_state.get_legal_actions(0)
        moves = [a for a in legal if a != Direction.STOP]
        if not moves:
            moves = [Direction.STOP]
        max_q_next = max(self._q(next_state, a) for a in moves)
        key = (self._state_key(state), action)
        old = self.q_table[key]
        self.q_table[key] = old + self.alpha * (
            reward + self.gamma * max_q_next - old
        )

    def set_training(self, training: bool) -> None:
        self._training = training

    # ── Save / Load (FR-30) ──────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle Q-table to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path: str | Path) -> None:
        """Load Q-table from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data)


# ── Approximate Q-learning (FR-28) ──────────────────────

class ApproxQLearningAgent(Agent):
    """Approximate Q-learning with a linear feature-based function."""

    name = "Approx-Q"

    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.weights: Dict[str, float] = defaultdict(float)
        self._training = True

    def _q(self, state: GameState, action: Direction) -> float:
        features = extract_features(state, action)
        return sum(self.weights[k] * v for k, v in features.items())

    def get_action(self, state: GameState) -> Direction:
        legal = state.get_legal_actions(0)
        moves = [a for a in legal if a != Direction.STOP]
        if not moves:
            moves = [Direction.STOP]

        if self._training and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)

        curr_dir = getattr(state, "pacman_direction", Direction.STOP)
        best_a = moves[0]
        best_q = -1e9
        for a in moves:
            q = self._q(state, a)
            # Add forward momentum and penalize 180° immediate reversal
            if a == curr_dir:
                q += 0.08
            elif a == curr_dir.opposite():
                q -= 0.08
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def observe_transition(
        self,
        state: GameState,
        action: Direction,
        next_state: GameState,
        reward: float,
    ) -> None:
        if not self._training:
            return
        legal = next_state.get_legal_actions(0)
        moves = [a for a in legal if a != Direction.STOP]
        if not moves:
            moves = [Direction.STOP]
        max_q_next = max(self._q(next_state, a) for a in moves)
        diff = reward + self.gamma * max_q_next - self._q(state, action)
        features = extract_features(state, action)
        for k, v in features.items():
            self.weights[k] += self.alpha * diff * v

    def set_training(self, training: bool) -> None:
        self._training = training

    # ── Save / Load (FR-30) ──────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save weights as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(self.weights), f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load weights from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.weights = defaultdict(float, data)


# ── Deep Q-Learning (FR-29) ──────────────────────────────

class DeepQLearningAgent(Agent):
    """Deep Q-Learning Agent using a Multi-Layer Neural Network with Experience Replay.

    Architecture:
      Input (features) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Output(1, linear)
    """

    name = "Deep Q-Learning"

    FEATURE_KEYS = ["bias", "nearest_food", "nearest_ghost", "ghosts_nearby", "eats_food"]

    def __init__(
        self,
        alpha: float = 0.005,
        gamma: float = 0.9,
        epsilon: float = 0.05,
        batch_size: int = 16,
        memory_size: int = 1000,
        seed: int | None = None,
        **kwargs,
    ):
        import numpy as np
        from collections import deque

        self.np = np
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory: deque = deque(maxlen=memory_size)
        self._training = False  # Evaluation by default for interactive play
        self.rng = random.Random(seed)

        # Seed NumPy generator
        np_rng = np.random.default_rng(seed)

        # Neural Network Weights
        d_in = len(self.FEATURE_KEYS)
        h1 = 32
        h2 = 16

        # He initialization
        self.W1 = np_rng.normal(0, np.sqrt(2.0 / d_in), (d_in, h1)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = np_rng.normal(0, np.sqrt(2.0 / h1), (h1, h2)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = np_rng.normal(0, np.sqrt(2.0 / h2), (h2, 1)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

        # Heuristic prior weights so agent plays smartly out-of-the-box
        # bias: 0, nearest_food: +10, nearest_ghost: -15, ghosts_nearby: -20, eats_food: +25
        self.W1[1, :8] = 2.0   # nearest_food
        self.W1[2, 8:16] = -2.5 # nearest_ghost
        self.W1[3, 16:24] = -3.0 # ghosts_nearby
        self.W1[4, 24:] = 3.5  # eats_food
        self.W2[:] = 0.2
        self.W3[:] = 0.5

    def _features_to_vec(self, state: GameState, action: Direction):
        feat = extract_features(state, action)
        return self.np.array([feat[k] for k in self.FEATURE_KEYS], dtype=self.np.float32)

    def _forward(self, x):
        """Forward pass through 2-layer MLP."""
        np = self.np
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        q = a2 @ self.W3 + self.b3
        return q[0], (x, z1, a1, z2, a2)

    def _q(self, state: GameState, action: Direction) -> float:
        x = self._features_to_vec(state, action)
        q, _ = self._forward(x)
        return float(q)

    def get_action(self, state: GameState) -> Direction:
        legal = state.get_legal_actions(0)
        moves = [a for a in legal if a != Direction.STOP]
        if not moves:
            moves = [Direction.STOP]

        if self._training and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)

        curr_dir = getattr(state, "pacman_direction", Direction.STOP)
        best_a = moves[0]
        best_q = -1e9
        for a in moves:
            q = self._q(state, a)
            # Add forward momentum and penalize 180° reversal to prevent 2-tile ping-ponging
            if a == curr_dir:
                q += 0.12
            elif a == curr_dir.opposite():
                q -= 0.12
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def observe_transition(
        self,
        state: GameState,
        action: Direction,
        next_state: GameState,
        reward: float,
    ) -> None:
        self.memory.append((state, action, next_state, reward))
        if not self._training or len(self.memory) < self.batch_size:
            return

        # Mini-batch gradient descent
        batch = self.rng.sample(list(self.memory), self.batch_size)
        np = self.np

        for s, a, ns, r in batch:
            legal_next = ns.get_legal_actions(0)
            moves_next = [na for na in legal_next if na != Direction.STOP]
            if not moves_next:
                moves_next = [Direction.STOP]
            max_q_next = max(self._q(ns, na) for na in moves_next)
            target = r + self.gamma * max_q_next

            x = self._features_to_vec(s, a)
            q, (x, z1, a1, z2, a2) = self._forward(x)

            # Gradient of MSE: dL/dq = 2 * (q - target)
            dq = np.clip(q - target, -5.0, 5.0)

            # Backprop: layer 3
            dW3 = a2.reshape(-1, 1) * dq
            db3 = np.array([dq])

            # Backprop: layer 2
            da2 = (self.W3.flatten() * dq)
            dz2 = da2 * (z2 > 0)
            dW2 = np.outer(a1, dz2)
            db2 = dz2

            # Backprop: layer 1
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = np.outer(x, dz1)
            db1 = dz1

            # Update weights
            lr = self.alpha
            self.W3 -= lr * dW3
            self.b3 -= lr * db3
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

    def set_training(self, training: bool) -> None:
        self._training = training

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        self.W1 = self.np.array(data["W1"], dtype=self.np.float32)
        self.b1 = self.np.array(data["b1"], dtype=self.np.float32)
        self.W2 = self.np.array(data["W2"], dtype=self.np.float32)
        self.b2 = self.np.array(data["b2"], dtype=self.np.float32)
        self.W3 = self.np.array(data["W3"], dtype=self.np.float32)
        self.b3 = self.np.array(data["b3"], dtype=self.np.float32)
