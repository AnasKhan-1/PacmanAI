"""
Game rules — successor function, collision handling, scoring, and win/loss logic.

FR-4: Successor function.
FR-5: Pellet eating / scoring.
FR-6: Power pellet → frightened ghosts.
FR-7: Ghost collisions.
FR-8: Win / loss conditions.
FR-9: Configurable number of ghosts.
FR-10: No graphics dependency.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Tuple

from .maze import Maze
from .state import Direction, GameState, GhostMode, GhostState

logger = logging.getLogger(__name__)


class GameRules:
    """Pure-function rule engine — all methods are stateless class methods."""

    # ── Successor ────────────────────────────────────────

    @staticmethod
    def get_successor(
        state: GameState,
        pacman_action: Direction,
        ghost_actions: List[Direction] | None = None,
    ) -> GameState:
        """Return the next GameState after Pacman and all ghosts act.

        If *ghost_actions* is None, ghosts do not move this step (useful
        for unit-testing Pacman in isolation).
        """
        if state.is_terminal():
            return state

        # Validate Pacman action
        legal = state.get_legal_actions(0)
        if pacman_action not in legal:
            logger.warning(
                "Invalid Pacman action %s; defaulting to STOP", pacman_action
            )
            pacman_action = Direction.STOP

        # --- Move Pacman ---
        raw_px = state.pacman_position[0] + pacman_action.dx
        raw_py = state.pacman_position[1] + pacman_action.dy
        new_pacman = state.maze.wrap_position(raw_px, raw_py)

        # --- Eat pellets ---
        new_pellets = state.pellets
        new_power = state.power_pellets
        score_delta = state.step_penalty
        pellets_eaten = state.pellets_eaten
        new_frightened_duration = state.frightened_duration

        if new_pacman in new_pellets:
            new_pellets = new_pellets - {new_pacman}
            score_delta += state.pellet_score
            pellets_eaten += 1

        ate_power = False
        if new_pacman in new_power:
            new_power = new_power - {new_pacman}
            score_delta += state.power_pellet_score
            ate_power = True

        # --- Update ghosts ---
        new_ghosts_list: list[GhostState] = []
        if ghost_actions is None:
            ghost_actions = [Direction.STOP] * state.num_ghosts

        for i, ghost in enumerate(state.ghosts):
            g_action = ghost_actions[i] if i < len(ghost_actions) else Direction.STOP
            # Validate ghost action
            g_legal = state.get_legal_actions(i + 1)
            if g_action not in g_legal:
                g_action = Direction.STOP

            raw_gx = ghost.position[0] + g_action.dx
            raw_gy = ghost.position[1] + g_action.dy
            gx, gy = state.maze.wrap_position(raw_gx, raw_gy)

            new_mode = ghost.mode
            timer = ghost.frightened_timer

            if ate_power and ghost.mode != GhostMode.EATEN:
                new_mode = GhostMode.FRIGHTENED
                timer = new_frightened_duration
            elif timer > 0:
                timer -= 1
                if timer == 0 and new_mode == GhostMode.FRIGHTENED:
                    new_mode = GhostMode.CHASE

            new_ghosts_list.append(
                GhostState(
                    position=(gx, gy),
                    mode=new_mode,
                    start_position=ghost.start_position,
                    frightened_timer=timer,
                )
            )

        # --- Check Pacman-ghost collisions ---
        new_lives = state.lives
        is_lose = False

        ghosts_after_collision: list[GhostState] = []
        for ghost in new_ghosts_list:
            if ghost.position == new_pacman:
                if ghost.mode == GhostMode.FRIGHTENED:
                    # Eat the ghost -> switch to EATEN (eyes only)
                    score_delta += state.ghost_eat_score
                    ghosts_after_collision.append(
                        GhostState(
                            position=ghost.position,
                            mode=GhostMode.EATEN,
                            start_position=ghost.start_position,
                            frightened_timer=0,
                        )
                    )
                elif ghost.mode == GhostMode.EATEN:
                    # Harmless eyes returning to ghost house
                    ghosts_after_collision.append(ghost)
                else:
                    # Pacman dies
                    new_lives -= 1
                    score_delta += state.death_penalty
                    if new_lives <= 0:
                        is_lose = True
                    ghosts_after_collision.append(ghost)
            else:
                ghosts_after_collision.append(ghost)

        # Also check the *pre-move* crossing (ghosts and Pacman swap cells)
        for idx, ghost in enumerate(state.ghosts):
            new_g = new_ghosts_list[idx]
            if (
                ghost.position == new_pacman
                and new_g.position == state.pacman_position
            ):
                if new_g.mode == GhostMode.FRIGHTENED:
                    score_delta += state.ghost_eat_score
                    ghosts_after_collision[idx] = GhostState(
                        position=new_g.position,
                        mode=GhostMode.EATEN,
                        start_position=new_g.start_position,
                        frightened_timer=0,
                    )
                elif new_g.mode == GhostMode.EATEN:
                    pass
                else:
                    new_lives -= 1
                    score_delta += state.death_penalty
                    if new_lives <= 0:
                        is_lose = True

        # --- Win condition ---
        is_win = len(new_pellets) == 0 and len(new_power) == 0
        if is_win:
            score_delta += state.win_bonus

        # --- Step limit ---
        new_step = state.step_count + 1
        if new_step >= state.step_limit and not is_win:
            is_lose = True

        # If Pacman died but still has lives, reset position
        if new_lives < state.lives and not is_lose:
            new_pacman = state.maze.pacman_start
            # Reset ghosts too
            ghosts_after_collision = [
                GhostState(
                    position=g.start_position,
                    mode=GhostMode.CHASE,
                    start_position=g.start_position,
                    frightened_timer=0,
                )
                for g in ghosts_after_collision
            ]

        new_facing = pacman_action if pacman_action != Direction.STOP else state.pacman_direction

        return replace(
            state,
            pacman_position=new_pacman,
            pacman_direction=new_facing,
            ghosts=tuple(ghosts_after_collision),
            pellets=new_pellets,
            power_pellets=new_power,
            score=state.score + score_delta,
            lives=new_lives,
            step_count=new_step,
            is_win=is_win,
            is_lose=is_lose,
            pellets_eaten=pellets_eaten,
        )
