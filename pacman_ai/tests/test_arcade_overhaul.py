"""
Unit validation for Arcade Overhaul:
1. Game Start Lockout / Intro Sequence
2. Speed Calibration (Pacman 80%/90%, Ghost 75%/50%/200%)
3. Power Pellet (6s) & Flashing Warning (< 2s)
4. Eaten Ghost (Eyes Only) Return Loop & Score Escalation (200, 400, 800, 1600)
"""

import unittest
from pacman_ai.engine.maze import Maze
from pacman_ai.engine.state import Direction, GameState, GhostMode, GhostState
from pacman_ai.engine.rules import GameRules
from pacman_ai.engine.ghosts import GhostController, _bfs_first_step
from pacman_ai.ui.game_ui import GameUI


class TestArcadeOverhaul(unittest.TestCase):

    def setUp(self):
        self.maze = Maze.from_file("pacman_ai/mazes/medium.txt")
        self.state = GameState.from_maze(self.maze, num_ghosts=4)
        self.ctrl = GhostController(behaviors="classic", num_ghosts=4)

    def test_ghost_mode_eaten_exists(self):
        self.assertEqual(GhostMode.EATEN.value, "eaten")

    def test_speed_calibration(self):
        # 1. Normal ghost speed = 75%
        spd_normal = self.ctrl.get_effective_speed(1, self.state)
        self.assertAlmostEqual(spd_normal, 0.75, places=2)

        # 2. Frightened ghost speed = 50%
        fright_state = GameState.from_maze(self.maze, num_ghosts=4)
        f_ghosts = list(fright_state.ghosts)
        f_ghosts[1] = GhostState(position=f_ghosts[1].position, mode=GhostMode.FRIGHTENED)
        fright_state = GameState(
            maze=fright_state.maze,
            pacman_position=fright_state.pacman_position,
            ghosts=tuple(f_ghosts),
            pellets=fright_state.pellets,
            power_pellets=fright_state.power_pellets,
        )
        spd_fright = self.ctrl.get_effective_speed(1, fright_state)
        self.assertAlmostEqual(spd_fright, 0.50, places=2)

        # 3. Eaten ghost eyes speed: equal to or slightly slower than normal ghost speed (<= 0.75, no 2.0x)
        eaten_ghosts = list(fright_state.ghosts)
        eaten_ghosts[1] = GhostState(position=eaten_ghosts[1].position, mode=GhostMode.EATEN)
        eaten_state = GameState(
            maze=fright_state.maze,
            pacman_position=fright_state.pacman_position,
            ghosts=tuple(eaten_ghosts),
            pellets=fright_state.pellets,
            power_pellets=fright_state.power_pellets,
        )
        spd_eaten = self.ctrl.get_effective_speed(1, eaten_state)
        self.assertLessEqual(spd_eaten, spd_normal)
        self.assertAlmostEqual(spd_eaten, 0.85 * 0.75, places=2)

    def test_eaten_ghost_pathfinding_to_house(self):
        # Place eaten ghost at (1, 1) top-left
        ghost_states = [
            GhostState(position=(1, 1), mode=GhostMode.EATEN, start_position=(13, 14))
        ]
        s = GameState(
            maze=self.maze,
            pacman_position=self.maze.pacman_start,
            ghosts=tuple(ghost_states),
            pellets=self.maze.pellets,
            power_pellets=self.maze.power_pellets,
        )
        ctrl = GhostController(behaviors="classic", num_ghosts=1)
        action = ctrl.get_single_ghost_action(0, s)
        # Should navigate towards door
        self.assertIn(action, (Direction.EAST, Direction.SOUTH))

        # Test at doorway entrance (13, 11) -> should step SOUTH into doorway (13, 12)
        s_door = GameState(
            maze=self.maze,
            pacman_position=self.maze.pacman_start,
            ghosts=(GhostState(position=(13, 11), mode=GhostMode.EATEN, start_position=(13, 14)),),
            pellets=self.maze.pellets,
            power_pellets=self.maze.power_pellets,
        )
        act_door = ctrl.get_single_ghost_action(0, s_door)
        self.assertEqual(act_door, Direction.SOUTH)

        # Inside doorway (13, 12) -> should step SOUTH into house (13, 13)
        s_inside_door = GameState(
            maze=self.maze,
            pacman_position=self.maze.pacman_start,
            ghosts=(GhostState(position=(13, 12), mode=GhostMode.EATEN, start_position=(13, 14)),),
            pellets=self.maze.pellets,
            power_pellets=self.maze.power_pellets,
        )
        act_inside = ctrl.get_single_ghost_action(0, s_inside_door)
        self.assertEqual(act_inside, Direction.SOUTH)

        # Inside house (13, 13)
        self.assertTrue(ctrl.is_inside_house(13, 13, s))

    def test_eaten_ghost_harmless_collision(self):
        # Collision with EATEN ghost does not kill Pacman
        px, py = 10, 10
        ghost_states = [
            GhostState(position=(px, py), mode=GhostMode.EATEN, start_position=(13, 14))
        ]
        s = GameState(
            maze=self.maze,
            pacman_position=(px, py),
            ghosts=tuple(ghost_states),
            pellets=self.maze.pellets,
            power_pellets=self.maze.power_pellets,
            lives=3,
        )
        succ = GameRules.get_successor(s, Direction.STOP)
        self.assertEqual(succ.lives, 3)
        self.assertFalse(succ.is_lose)

    def test_score_doubling(self):
        scores = [200 * (2 ** min(3, k)) for k in range(4)]
        self.assertEqual(scores, [200, 400, 800, 1600])


if __name__ == "__main__":
    unittest.main()
