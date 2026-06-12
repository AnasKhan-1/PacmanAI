"""
Unit tests for the Pacman game engine.

NFR-10: Cover movement, collisions, scoring, win/loss — ≥80% engine coverage.
NFR-5:  Never allow movement into walls.
NFR-6:  Invalid actions handled gracefully.
NFR-4:  Deterministic with same seed.
"""

import unittest

from pacman_ai.engine.maze import Maze
from pacman_ai.engine.state import Direction, GameState, GhostMode, GhostState
from pacman_ai.engine.rules import GameRules


# ── Maze fixtures ────────────────────────────────────────

SIMPLE_MAZE_TEXT = """\
%%%%%
%P .%
% %%%
% . %
%%%%%"""

GHOST_MAZE_TEXT = """\
%%%%%%%
%P . G%
%     %
%  .  %
%%%%%%%"""

POWER_MAZE_TEXT = """\
%%%%%%%
%P o G%
%     %
%  .  %
%%%%%%%"""


class TestMazeLoading(unittest.TestCase):
    """FR-1: Loading a maze from text."""

    def test_basic_load(self):
        maze = Maze.from_string(SIMPLE_MAZE_TEXT)
        self.assertEqual(maze.width, 5)
        self.assertEqual(maze.height, 5)
        self.assertEqual(maze.pacman_start, (1, 1))
        self.assertIn((0, 0), maze.walls)
        self.assertIn((3, 1), maze.pellets)

    def test_no_pacman_raises(self):
        with self.assertRaises(ValueError):
            Maze.from_string("%%%%\n%..%\n%%%%")

    def test_wall_detection(self):
        maze = Maze.from_string(SIMPLE_MAZE_TEXT)
        self.assertTrue(maze.is_wall(0, 0))
        self.assertFalse(maze.is_wall(1, 1))
        # Outside grid
        self.assertTrue(maze.is_wall(-1, 0))
        self.assertTrue(maze.is_wall(100, 100))


class TestGameState(unittest.TestCase):
    """FR-2, FR-3: State representation and legal actions."""

    def setUp(self):
        maze = Maze.from_string(SIMPLE_MAZE_TEXT)
        self.state = GameState.from_maze(maze)

    def test_initial_position(self):
        self.assertEqual(self.state.pacman_position, (1, 1))

    def test_legal_actions_no_wall(self):
        actions = self.state.get_legal_actions(0)
        self.assertIn(Direction.STOP, actions)
        # Pacman at (1,1) — wall at (0,1) and (1,0)
        self.assertNotIn(Direction.WEST, actions)
        self.assertNotIn(Direction.NORTH, actions)
        self.assertIn(Direction.EAST, actions)
        self.assertIn(Direction.SOUTH, actions)


class TestMovement(unittest.TestCase):
    """FR-4, NFR-5: Successor function and wall blocking."""

    def setUp(self):
        maze = Maze.from_string(SIMPLE_MAZE_TEXT)
        self.state = GameState.from_maze(maze)

    def test_move_east(self):
        next_state = GameRules.get_successor(self.state, Direction.EAST)
        self.assertEqual(next_state.pacman_position, (2, 1))

    def test_move_into_wall_blocked(self):
        # NORTH from (1,1) → wall at (1,0) — action is invalid, defaults to STOP
        next_state = GameRules.get_successor(self.state, Direction.NORTH)
        self.assertEqual(next_state.pacman_position, (1, 1))

    def test_stop_action(self):
        next_state = GameRules.get_successor(self.state, Direction.STOP)
        self.assertEqual(next_state.pacman_position, (1, 1))

    def test_state_immutability(self):
        """FR-4: Original state not modified."""
        old_pos = self.state.pacman_position
        _ = GameRules.get_successor(self.state, Direction.EAST)
        self.assertEqual(self.state.pacman_position, old_pos)


class TestScoring(unittest.TestCase):
    """FR-5: Pellet eating and scoring."""

    def test_eat_pellet(self):
        maze = Maze.from_string(SIMPLE_MAZE_TEXT)
        state = GameState.from_maze(maze)
        # Move east twice to reach pellet at (3,1)
        s1 = GameRules.get_successor(state, Direction.EAST)
        s2 = GameRules.get_successor(s1, Direction.EAST)
        # Score should have pellet bonus minus step penalties
        self.assertGreater(s2.score, state.score)
        self.assertEqual(s2.pellets_eaten, 1)
        # Pellet removed
        self.assertNotIn((3, 1), s2.pellets)


class TestGhostCollisions(unittest.TestCase):
    """FR-7: Ghost collisions."""

    def test_collision_with_normal_ghost(self):
        maze = Maze.from_string(GHOST_MAZE_TEXT)
        state = GameState.from_maze(maze, lives=1)
        # Move toward ghost repeatedly
        s = state
        for _ in range(10):
            if s.is_terminal():
                break
            s = GameRules.get_successor(
                s, Direction.EAST, [Direction.WEST] * s.num_ghosts
            )
        # Should lose a life or end
        self.assertTrue(s.lives < state.lives or s.is_lose)

    def test_eat_frightened_ghost(self):
        # Layout: P can move right to eat power pellet (o) then ghost (G)
        text = "%%%%%%%%%\n%P.o.G  %\n%%%%%%%%%"
        maze = Maze.from_string(text)
        state = GameState.from_maze(maze, lives=3)
        # Move east step by step: (1,1)->(2,1)->(3,1: power)->(4,1)->(5,1: ghost)
        s = state
        for _ in range(6):
            if s.is_terminal():
                break
            s = GameRules.get_successor(
                s, Direction.EAST, [Direction.STOP] * s.num_ghosts
            )
        # Score should be positive from pellet + power pellet + ghost eat bonuses
        self.assertGreater(s.score, state.score)


class TestWinLoss(unittest.TestCase):
    """FR-8: Win and loss conditions."""

    def test_win_when_all_pellets_eaten(self):
        # Tiny maze with just one pellet
        tiny = Maze.from_string("%%%\n%P.%\n%%%%")
        # Fix: this maze has width issues. Use a proper one.
        tiny = Maze.from_string("%%%%\n%P.%\n%%%%")
        state = GameState.from_maze(tiny, num_ghosts=0)
        s = GameRules.get_successor(state, Direction.EAST)
        self.assertTrue(s.is_win)

    def test_loss_on_step_limit(self):
        maze = Maze.from_string(SIMPLE_MAZE_TEXT)
        state = GameState.from_maze(maze, num_ghosts=0, step_limit=3)
        s = state
        for _ in range(5):
            if s.is_terminal():
                break
            s = GameRules.get_successor(s, Direction.STOP)
        self.assertTrue(s.is_lose)


class TestDeterminism(unittest.TestCase):
    """NFR-4: Same seed → same results."""

    def test_reproducible(self):
        import random
        from pacman_ai.agents.random_agent import RandomAgent

        maze = Maze.from_string(SIMPLE_MAZE_TEXT)
        results = []
        for _ in range(2):
            agent = RandomAgent(seed=42)
            state = GameState.from_maze(maze, num_ghosts=0)
            for _ in range(50):
                if state.is_terminal():
                    break
                action = agent.get_action(state)
                state = GameRules.get_successor(state, action)
            results.append((state.score, state.step_count, state.is_win))

        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
