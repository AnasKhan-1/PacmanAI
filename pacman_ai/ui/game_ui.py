"""
Main game UI loop — ties the renderer, input handler, and game engine together.

Features:
  - 4 Authentic Ghosts: 🔴 Blinky, 💗 Pinky, 🔷 Inky, 🍊 Clyde with exact targeting & Cruise Elroy.
  - Interactive Agent Selector UI: Human, Deep Q, A*, BFS, Minimax, Expectimax, Q-Learn, Random.
  - In-UI Unlimited Live Training Mode: train Q-Learning or Deep Q models continuously right from the UI.
  - 60 FPS rendering with sub-pixel interpolation between grid steps (no tile snapping).
  - Authentic animated Pacman chomp (mouth opens & closes in direction of movement).
  - Authentic animated ghosts (wavy skirt tentacles & directional pupils).
  - Smooth arcade momentum & pre-buffered steering for human control.
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import replace
from typing import List, Optional, Tuple

import pygame

from pacman_ai.agents.base import Agent
from pacman_ai.agents.registry import get_agent
from pacman_ai.engine.ghosts import GhostController
from pacman_ai.engine.maze import Maze
from pacman_ai.engine.rules import GameRules
from pacman_ai.engine.state import Direction, GameState, GhostMode
from .input_handler import get_human_action
from .renderer import AVAILABLE_AGENTS, PacmanRenderer
from .sound import SoundManager


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class GameUI:
    """Pygame game loop with 4 classic ghosts, 60 FPS sub-pixel interpolation, and In-UI Training."""

    def __init__(
        self,
        maze: Maze,
        agent: Agent | None = None,
        ghost_behavior: str = "classic",
        num_ghosts: int | None = 4,
        seed: int = 42,
        target_fps: int = 10,
        cell_size: int = 22,
        ghost_speed: float = 0.75,
    ):
        self.maze = maze
        self.ghost_behavior = ghost_behavior
        self.num_ghosts = num_ghosts if num_ghosts is not None else 4
        self.seed = seed
        self.ghost_speed = ghost_speed
        self.cell_size = cell_size

        # Arcade baseline speeds (~11 tiles/sec max baseline)
        self.render_fps = 60
        self.base_tiles_per_sec = 11.0
        self.base_step_interval = 1.0 / self.base_tiles_per_sec
        self.step_interval = self.base_step_interval / 0.80  # 80% baseline Pac-Man speed

        self.renderer = PacmanRenderer(cell_size=cell_size)
        self.sound = SoundManager()
        self._rng = random.Random(seed)

        # Game phase control
        self._game_started: bool = False
        self._death_paused: bool = False
        self._is_intro: bool = False
        self._intro_timer: float = 0.0

        # Energizer & Frightened state
        self._frightened_timer: float = 0.0
        self._ghosts_eaten_streak: int = 0

        # Freeze frame & Score popups (200, 400, 800, 1600)
        self._freeze_timer: float = 0.0
        self._score_popups: list[dict] = []

        # Active Agent state
        self.agent = agent
        self.active_agent_key = "human" if agent is None else getattr(agent, "registry_key", "custom")
        if agent is not None:
            name_lower = agent.name.lower()
            for k, _ in AVAILABLE_AGENTS:
                if k in name_lower or name_lower in k:
                    self.active_agent_key = k
                    break

        # Flow control
        self._paused = False
        self._running = True
        self._clock: Optional[pygame.time.Clock] = None
        self._surface: Optional[pygame.Surface] = None
        self._ghost_ctrl: Optional[GhostController] = None

        # In-UI Live Training State
        self._is_training: bool = False
        self._train_stats = {
            "episodes": 0,
            "wins": 0,
            "scores": [],
            "win_rate": 0.0,
            "avg_score": 0.0,
            "epsilon": 0.1,
        }

        # Human steering
        self._current_pacman_dir: Direction = Direction.STOP
        self._buffered_action: Direction | None = None

        # Sub-pixel interpolation variables
        self._pacman_prev: Tuple[float, float] = (0.0, 0.0)
        self._pacman_target: Tuple[float, float] = (0.0, 0.0)
        self._pacman_facing_dir: Direction = Direction.EAST
        self._ghost_prev: List[Tuple[float, float]] = []
        self._ghost_target: List[Tuple[float, float]] = []
        self._ghost_dirs: List[Direction] = []
        self._ghost_timers: List[float] = []
        self._ghost_durations: List[float] = []

        # Animation parameters
        self._step_timer: float = 0.0
        self._mouth_phase: float = 0.0
        self._animation_frame: int = 0
        self._toast_message: Optional[str] = None
        self._toast_timer: float = 0.0

    def toggle_training(self) -> None:
        """Toggle unlimited In-UI training mode for Q-Learning or Deep Q."""
        self._is_training = not self._is_training

        if self._is_training:
            self._game_started = True
            self._death_paused = False
            # If not already an RL agent, auto-switch to Deep Q
            if self.active_agent_key not in ("deep_q", "qlearning", "approx_q"):
                self.set_agent("deep_q")

            if hasattr(self.agent, "set_training"):
                self.agent.set_training(True)
                if hasattr(self.agent, "epsilon"):
                    self.agent.epsilon = max(0.15, self.agent.epsilon)
            self._show_toast("⚡ Unlimited Training Active! Press T to stop", duration=2.2)
        else:
            if hasattr(self.agent, "set_training"):
                self.agent.set_training(False)
            self._show_toast("⚡ Training Paused — Model Playing Live!", duration=2.0)

    def set_agent(self, agent_key: str) -> None:
        """Switch the playing agent live."""
        agent_key = agent_key.lower()
        self.active_agent_key = agent_key

        if agent_key == "human":
            self.agent = None
            if self._is_training:
                self._is_training = False
            self._show_toast("Player: Human (WASD / Arrows)")
            return

        try:
            kwargs = {"seed": self.seed}
            if agent_key in ("minimax", "expectimax"):
                kwargs["depth"] = 2
            self.agent = get_agent(agent_key, **kwargs)
            if self._is_training and hasattr(self.agent, "set_training"):
                self.agent.set_training(True)
            label = self.agent.name
            self._show_toast(f"Playing: {label}")
        except Exception as e:
            self._show_toast(f"Error loading {agent_key}: {e}")

    def _show_toast(self, message: str, duration: float = 1.6) -> None:
        self._toast_message = message
        self._toast_timer = duration

    def _trigger_intro_sequence(self) -> None:
        """Trigger dedicated READY / INTRO sequence state with opening intro music (4.2-4.5s)."""
        self._is_intro = True
        self._intro_timer = 4.4
        self.sound.play_start()
        self._current_pacman_dir = Direction.STOP
        self._buffered_action = None
        self._freeze_timer = 0.0
        self._step_timer = 0.0

    def _start_game(self) -> None:
        """Start active gameplay upon player pressing Start Game or movement key."""
        if not self._game_started:
            self._game_started = True
            self._trigger_intro_sequence()

    def _resume_after_death(self) -> None:
        """Resume game following death pause once player presses any key."""
        if self._death_paused:
            self._death_paused = False
            self._current_pacman_dir = Direction.STOP
            self._buffered_action = None
            self._trigger_intro_sequence()

    def _on_pacman_died(self, state: GameState) -> GameState:
        """Handle death: play death sound, send ghosts back inside box, pause until keypress."""
        self.sound.play_death()
        self._frightened_timer = 0.0
        self._ghosts_eaten_streak = 0
        self._freeze_timer = 0.0
        self._score_popups.clear()
        new_lives = max(0, state.lives - 1)
        new_score = state.score + state.death_penalty

        if new_lives <= 0:
            return replace(state, lives=0, score=new_score, is_lose=True)

        # Reset Pac-Man back to start position
        # Reset all ghosts back inside their box at start_position with CHASE mode
        reset_ghosts = [
            replace(g, position=g.start_position, mode=GhostMode.CHASE, frightened_timer=0)
            for g in state.ghosts
        ]
        new_state = replace(
            state,
            lives=new_lives,
            score=new_score,
            pacman_position=state.maze.pacman_start,
            ghosts=reset_ghosts,
        )

        # Reset GhostController house release timers and flags so ghosts exit from house in order
        if self._ghost_ctrl is not None:
            self._ghost_ctrl._has_exited = [False] * new_state.num_ghosts
            self._ghost_ctrl._spawn_timers = [0, 15, 30, 48]
            while len(self._ghost_ctrl._spawn_timers) < new_state.num_ghosts:
                self._ghost_ctrl._spawn_timers.append(len(self._ghost_ctrl._spawn_timers) * 16)

        # Reset steering & interpolation
        self._current_pacman_dir = Direction.STOP
        self._buffered_action = None
        self._pacman_facing_dir = Direction.EAST
        self._init_positions(new_state)
        self._death_paused = True
        return new_state

    def run(self) -> None:
        """Enter the 60 FPS game loop. Blocks until the user quits."""
        state = self._new_game()
        self._surface = self.renderer.init_display(state)
        self._clock = pygame.time.Clock()
        self._ghost_ctrl = self._make_ghost_controller(state)
        self._init_positions(state)

        while self._running:
            dt_ms = self._clock.tick(self.render_fps)
            dt = min(0.05, dt_ms / 1000.0)
            fps = self._clock.get_fps()
            self._animation_frame += 1

            if self._toast_timer > 0:
                self._toast_timer -= dt
                if self._toast_timer <= 0:
                    self._toast_message = None

            # ── Events ──────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # If paused after death, any click resumes
                    if self._death_paused:
                        self._resume_after_death()
                        continue

                    mx, my = event.pos

                    # If not started, check click on Start Game button
                    if not self._game_started:
                        start_rect = self.renderer.get_start_button_rect(
                            self._surface.get_width(), self._surface.get_height()
                        )
                        if start_rect.collidepoint(mx, my):
                            self._start_game()
                            continue

                    # Check click on Agent Selector & Train buttons
                    buttons = self.renderer.get_button_rects(self._surface.get_width(), self._is_training)
                    for key, label, rect in buttons:
                        if rect.collidepoint(mx, my):
                            if key == "train":
                                self.toggle_training()
                            else:
                                self.set_agent(key)
                            break
                elif event.type == pygame.KEYDOWN:
                    if self._death_paused:
                        self._resume_after_death()
                        continue
                    if not self._game_started:
                        if event.key in (
                            pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER,
                            pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d,
                            pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT
                        ):
                            self._start_game()
                    self._handle_key(event.key, state)

            if not self._running:
                break

            # ── Training Mode (Fast-Forward Learning Loop) ──
            if self._is_training and not self._paused:
                # Execute a burst of training steps each frame for rapid learning
                steps_per_frame = 6
                for _ in range(steps_per_frame):
                    pacman_action = self._get_pacman_action(state)
                    ghost_actions = self._ghost_ctrl.get_actions(state)
                    old_state = state
                    new_state = GameRules.get_successor(state, pacman_action, ghost_actions)

                    # RL observation & learning
                    if self.agent is not None and hasattr(self.agent, "observe_transition"):
                        reward = new_state.score - old_state.score
                        self.agent.observe_transition(
                            old_state, pacman_action, new_state, reward
                        )

                    state = new_state

                    # Episode ended in training mode -> record stats & immediately restart
                    if state.is_terminal():
                        self._train_stats["episodes"] += 1
                        if state.is_win:
                            self._train_stats["wins"] += 1
                        self._train_stats["scores"].append(state.score)
                        if len(self._train_stats["scores"]) > 50:
                            self._train_stats["scores"].pop(0)

                        ep = self._train_stats["episodes"]
                        self._train_stats["win_rate"] = (self._train_stats["wins"] / ep) * 100.0
                        self._train_stats["avg_score"] = sum(self._train_stats["scores"]) / len(self._train_stats["scores"])
                        self._train_stats["epsilon"] = getattr(self.agent, "epsilon", 0.0)

                        # Gentle epsilon decay
                        if hasattr(self.agent, "epsilon"):
                            self.agent.epsilon = max(0.02, self.agent.epsilon * 0.998)

                        # Auto reset for continuous learning
                        state = self._new_game()
                        self._ghost_ctrl = self._make_ghost_controller(state)
                        self._init_positions(state)
                        break

                # Update live positions for rendering
                self._pacman_prev = self._pacman_target
                self._pacman_target = state.pacman_position
                self._ghost_prev = list(self._ghost_target)
                self._ghost_target = [g.position for g in state.ghosts]

            # ── Normal Evaluation / Play (Step-by-step 60 FPS) ──
            elif not self._paused and not state.is_terminal() and self._game_started and not self._death_paused:
                # 0. Intro lockout sequence check
                if self._is_intro:
                    self._intro_timer -= dt
                    if self._intro_timer <= 0.0 and not self.sound.is_start_playing():
                        self._is_intro = False
                        self._step_timer = 0.0

                # 0b. Momentary freeze frame on ghost eat
                if self._freeze_timer > 0.0:
                    self._freeze_timer -= dt

                # Update score popups
                for pop in self._score_popups:
                    pop["timer"] -= dt
                self._score_popups = [p for p in self._score_popups if p["timer"] > 0]

                # Update frightened timer countdown (Level 1: 6.0s duration)
                if self._frightened_timer > 0.0 and not self._is_intro and self._freeze_timer <= 0.0:
                    self._frightened_timer -= dt
                    if self._frightened_timer <= 0.0:
                        self._frightened_timer = 0.0
                        # Revert surviving frightened ghosts back to CHASE mode
                        reverted = [
                            replace(g, mode=GhostMode.CHASE, frightened_timer=0) if g.mode == GhostMode.FRIGHTENED else g
                            for g in state.ghosts
                        ]
                        state = replace(state, ghosts=reverted)

                # Active entity updates only when NOT locked by intro or freeze frame
                if not self._is_intro and self._freeze_timer <= 0.0:
                    # 1. Update Pac-Man step with arcade calibrated speed
                    # Pacman Normal: 80% base speed (~8.8 tiles/s); Frightened: 90% base speed (~9.9 tiles/s)
                    pacman_speed_pct = 0.90 if self._frightened_timer > 0.0 else 0.80
                    pacman_step_interval = self.base_step_interval / pacman_speed_pct

                    self._step_timer += dt
                    if self._step_timer >= pacman_step_interval:
                        self._step_timer -= pacman_step_interval

                        pacman_action = self._get_pacman_action(state)
                        old_state = state
                        new_state = GameRules.get_successor(state, pacman_action, [Direction.STOP] * state.num_ghosts)

                        # Check if Pacman stepped into a ghost
                        if new_state.lives < state.lives:
                            state = self._on_pacman_died(state)
                            render_px, render_py = self._pacman_target
                            render_ghosts = list(self._ghost_target)
                            continue

                        # Sound effects for eating dots & power pellets
                        if len(new_state.pellets) < len(old_state.pellets):
                            self.sound.play_chomp()

                        if len(new_state.power_pellets) < len(old_state.power_pellets):
                            self.sound.play_power()
                            self._frightened_timer = 6.0  # FRIGHTENED_DURATION = 6.0s
                            self._ghosts_eaten_streak = 0
                            # Switch all non-EATEN ghosts to FRIGHTENED
                            fright_ghosts = [
                                replace(g, mode=GhostMode.FRIGHTENED, frightened_timer=60) if g.mode != GhostMode.EATEN else g
                                for g in new_state.ghosts
                            ]
                            new_state = replace(new_state, ghosts=fright_ghosts)

                        if self.agent is not None and hasattr(self.agent, "observe_transition"):
                            reward = new_state.score - old_state.score
                            self.agent.observe_transition(
                                old_state, pacman_action, new_state, reward
                            )

                        self._pacman_prev = self._pacman_target
                        self._pacman_target = new_state.pacman_position
                        if pacman_action != Direction.STOP:
                            self._pacman_facing_dir = pacman_action

                        state = new_state

                    # 2. Update Ghosts continuous movement with arcade speed calibration
                    # Normal: 75% | Frightened: 50% | Eaten Eyes: 200%
                    while len(self._ghost_timers) < state.num_ghosts:
                        self._ghost_timers.append(0.0)
                    while len(self._ghost_durations) < state.num_ghosts:
                        self._ghost_durations.append(self.base_step_interval / 0.75)
                    while len(self._ghost_prev) < state.num_ghosts:
                        idx = len(self._ghost_prev)
                        self._ghost_prev.append((float(state.ghosts[idx].position[0]), float(state.ghosts[idx].position[1])))
                    while len(self._ghost_target) < state.num_ghosts:
                        idx = len(self._ghost_target)
                        self._ghost_target.append((float(state.ghosts[idx].position[0]), float(state.ghosts[idx].position[1])))
                    while len(self._ghost_dirs) < state.num_ghosts:
                        self._ghost_dirs.append(Direction.STOP)

                    for i in range(state.num_ghosts):
                        ghost = state.ghosts[i]

                        # ── EATEN / EYES STATE: Wall-phasing direct vector return ──
                        if ghost.mode == GhostMode.EATEN:
                            if state.maze.ghost_starts:
                                target_pen_x = sum(p[0] for p in state.maze.ghost_starts) / len(state.maze.ghost_starts)
                                target_pen_y = sum(p[1] for p in state.maze.ghost_starts) / len(state.maze.ghost_starts)
                            else:
                                target_pen_x = float(state.maze.width // 2)
                                target_pen_y = float(state.maze.height // 2)

                            gx, gy = self._ghost_target[i]
                            dx = target_pen_x - gx
                            dy = target_pen_y - gy
                            dist = math.hypot(dx, dy)

                            # Respawn trigger: within a few pixels (0.25 tiles ~ 5.5px at cell_size 22)
                            if dist < 0.25:
                                respawn_pos = (int(round(target_pen_x)), int(round(target_pen_y)))
                                ghost_list = list(state.ghosts)
                                ghost_list[i] = replace(
                                    ghost_list[i],
                                    position=respawn_pos,
                                    mode=GhostMode.CHASE,
                                    frightened_timer=0,
                                )
                                state = replace(state, ghosts=ghost_list)
                                self._ghost_prev[i] = (float(respawn_pos[0]), float(respawn_pos[1]))
                                self._ghost_target[i] = self._ghost_prev[i]
                                self._ghost_timers[i] = 0.0
                                self._ghost_durations[i] = self.base_step_interval / 0.75
                                if self._ghost_ctrl is not None:
                                    self._ghost_ctrl._has_exited[i] = False
                                    self._ghost_ctrl._spawn_timers[i] = 10
                            else:
                                # Speed: equal to or slightly slower than default normal ghost speed (0.90 * normal)
                                ghost_normal_speed = 0.75 * self.base_tiles_per_sec
                                eyes_speed = 0.90 * ghost_normal_speed
                                move_step = min(dist, eyes_speed * dt)

                                new_gx = gx + (dx / dist) * move_step
                                new_gy = gy + (dy / dist) * move_step

                                self._ghost_prev[i] = (new_gx, new_gy)
                                self._ghost_target[i] = (new_gx, new_gy)

                                # Gaze pupils in vector travel direction
                                if abs(dx) > abs(dy):
                                    self._ghost_dirs[i] = Direction.EAST if dx > 0 else Direction.WEST
                                else:
                                    self._ghost_dirs[i] = Direction.SOUTH if dy > 0 else Direction.NORTH

                                ghost_list = list(state.ghosts)
                                ghost_list[i] = replace(ghost_list[i], position=(int(round(new_gx)), int(round(new_gy))))
                                state = replace(state, ghosts=ghost_list)
                            continue

                        # ── Normal / Frightened Ghosts Grid Movement ──
                        eff_speed = self._ghost_ctrl.get_effective_speed(i, state)
                        self._ghost_durations[i] = self.base_step_interval / max(0.1, eff_speed)
                        self._ghost_timers[i] += dt

                        if self._ghost_timers[i] >= self._ghost_durations[i]:
                            self._ghost_timers[i] -= self._ghost_durations[i]

                            # Arrived at target tile
                            curr_x = int(round(self._ghost_target[i][0]))
                            curr_y = int(round(self._ghost_target[i][1]))

                            ghost_list = list(state.ghosts)
                            ghost_list[i] = replace(ghost_list[i], position=(curr_x, curr_y))
                            state = replace(state, ghosts=ghost_list)

                            self._ghost_prev[i] = self._ghost_target[i]
                            act = self._ghost_ctrl.get_single_ghost_action(i, state)
                            self._ghost_dirs[i] = act
                            nx, ny = state.maze.wrap_position(curr_x + act.dx, curr_y + act.dy)
                            self._ghost_target[i] = (float(nx), float(ny))

            # ── Sub-pixel Interpolation ──────────────────
            if self._death_paused or not self._game_started or self._is_intro or self._freeze_timer > 0.0:
                render_px, render_py = self._pacman_target
                render_ghosts = list(self._ghost_target)
                mouth_angle = 12.0
            else:
                pacman_speed_pct = 0.90 if self._frightened_timer > 0.0 else 0.80
                pacman_step_interval = self.base_step_interval / pacman_speed_pct
                alpha_pacman = min(1.0, self._step_timer / pacman_step_interval) if pacman_step_interval > 0 else 1.0

                # Wrap-around tunnel check: snap if jumping across the screen
                if abs(self._pacman_prev[0] - self._pacman_target[0]) > 2:
                    render_px, render_py = self._pacman_target
                else:
                    render_px = _lerp(self._pacman_prev[0], self._pacman_target[0], alpha_pacman)
                    render_py = _lerp(self._pacman_prev[1], self._pacman_target[1], alpha_pacman)

                # Pacman chomp animation
                is_moving = (self._pacman_target != self._pacman_prev) and not self._paused and not state.is_terminal()
                if is_moving:
                    self._mouth_phase += dt * 22.0
                    mouth_angle = abs(math.sin(self._mouth_phase)) * 42.0
                else:
                    mouth_angle = 12.0

                # Ghosts smooth continuous positions
                render_ghosts = []
                for i in range(state.num_ghosts):
                    if state.ghosts[i].mode == GhostMode.EATEN:
                        render_ghosts.append(self._ghost_target[i])
                        continue
                    gp = self._ghost_prev[i] if i < len(self._ghost_prev) else (float(state.ghosts[i].position[0]), float(state.ghosts[i].position[1]))
                    gt = self._ghost_target[i] if i < len(self._ghost_target) else (float(state.ghosts[i].position[0]), float(state.ghosts[i].position[1]))
                    dur = self._ghost_durations[i] if i < len(self._ghost_durations) else (self.base_step_interval / 0.75)
                    timer = self._ghost_timers[i] if i < len(self._ghost_timers) else 0.0
                    alpha_ghost = min(1.0, timer / max(1e-4, dur))

                    if abs(gp[0] - gt[0]) > 2:
                        rx, ry = gt
                    else:
                        rx = _lerp(gp[0], gt[0], alpha_ghost)
                        ry = _lerp(gp[1], gt[1], alpha_ghost)
                    render_ghosts.append((rx, ry))

            # ── Sub-pixel Real-time Collision Check ─────
            if not self._is_training and not self._paused and not state.is_terminal() and self._game_started and not self._death_paused and not self._is_intro and self._freeze_timer <= 0.0:
                for i, (rx, ry) in enumerate(render_ghosts):
                    dist = math.hypot(render_px - rx, render_py - ry)
                    if dist < 0.75:
                        ghost = state.ghosts[i]
                        if ghost.mode == GhostMode.FRIGHTENED:
                            # Eat ghost!
                            self.sound.play_eat_ghost()
                            self._ghosts_eaten_streak += 1
                            streak_pow = min(3, self._ghosts_eaten_streak - 1)
                            points = 200 * (2 ** streak_pow)  # 200, 400, 800, 1600

                            # Momentary 0.5s freeze frame & score popup
                            self._freeze_timer = 0.5
                            self._score_popups.append({
                                "score": points,
                                "pos": (rx, ry),
                                "timer": 0.5,
                            })

                            # Switch that ghost's state to EATEN (eyes only)
                            ghost_list = list(state.ghosts)
                            ghost_list[i] = replace(
                                ghost,
                                mode=GhostMode.EATEN,
                                frightened_timer=0,
                            )
                            state = replace(state, score=state.score + points, ghosts=ghost_list)

                            # Ensure eyes start at exact eaten position
                            self._ghost_prev[i] = (rx, ry)
                            self._ghost_target[i] = (rx, ry)
                            break
                        elif ghost.mode == GhostMode.EATEN:
                            # Harmless eyes returning to ghost house
                            continue
                        else:
                            # Pacman dies!
                            state = self._on_pacman_died(state)
                            render_px, render_py = self._pacman_target
                            render_ghosts = list(self._ghost_target)
                            break

            # ── Render Frame at 60 FPS ──────────────────
            agent_display = self.agent.name if self.agent else "Human Player"
            self.renderer.draw(
                surface=self._surface,
                state=state,
                render_pacman=(render_px, render_py),
                pacman_direction=self._pacman_facing_dir,
                mouth_angle=mouth_angle,
                render_ghosts=render_ghosts,
                ghost_directions=self._ghost_dirs,
                agent_name=agent_display,
                active_agent_key=self.active_agent_key,
                fps=fps,
                ghost_speed=self.ghost_speed,
                animation_frame=self._animation_frame,
                toast_message=self._toast_message,
                is_training=self._is_training,
                train_stats=self._train_stats,
                is_waiting_to_start=not self._game_started,
                is_death_paused=self._death_paused,
                is_intro_state=self._is_intro,
                frightened_timer=self._frightened_timer,
                score_popups=self._score_popups,
            )

            # Auto-restart prompt after game over (in normal play mode)
            if state.is_terminal() and not self._is_training:
                self._wait_for_restart_or_quit()
                if self._running:
                    state = self._new_game()
                    self._ghost_ctrl = self._make_ghost_controller(state)
                    self._init_positions(state)
                    self._game_started = True
                    self._death_paused = False
                    self._trigger_intro_sequence()

        pygame.quit()

    # ── Helpers ──────────────────────────────────────────

    def _init_positions(self, state: GameState) -> None:
        px, py = state.pacman_position
        self._pacman_prev = (float(px), float(py))
        self._pacman_target = (float(px), float(py))
        self._ghost_prev = [(float(g.position[0]), float(g.position[1])) for g in state.ghosts]
        self._ghost_target = [(float(g.position[0]), float(g.position[1])) for g in state.ghosts]
        self._ghost_dirs = [Direction.STOP] * state.num_ghosts
        self._ghost_timers = [0.0] * state.num_ghosts
        self._ghost_durations = [self.base_step_interval / 0.75] * state.num_ghosts
        self._step_timer = 0.0

    def _new_game(self) -> GameState:
        self._current_pacman_dir = Direction.STOP
        self._buffered_action = None
        state = GameState.from_maze(self.maze, num_ghosts=self.num_ghosts)
        return state

    def _make_ghost_controller(self, state: GameState) -> GhostController:
        # Default to classic 4 distinct ghosts: Blinky, Pinky, Inky, Clyde
        return GhostController(
            behaviors="classic",
            num_ghosts=state.num_ghosts,
            rng=self._rng,
            speed=self.ghost_speed,
        )

    def _get_pacman_action(self, state: GameState) -> Direction:
        if self.agent is not None:
            return self.agent.get_action(state)

        legal = state.get_legal_actions(0)
        held = get_human_action()
        if held is not None:
            self._buffered_action = held

        if self._buffered_action is not None and self._buffered_action in legal:
            self._current_pacman_dir = self._buffered_action
            self._buffered_action = None
            return self._current_pacman_dir

        if self._current_pacman_dir in legal and self._current_pacman_dir != Direction.STOP:
            return self._current_pacman_dir

        self._current_pacman_dir = Direction.STOP
        return Direction.STOP

    def _handle_key(self, key: int, state: GameState) -> None:
        key_dirs = {
            pygame.K_UP: Direction.NORTH,
            pygame.K_w: Direction.NORTH,
            pygame.K_DOWN: Direction.SOUTH,
            pygame.K_s: Direction.SOUTH,
            pygame.K_LEFT: Direction.WEST,
            pygame.K_a: Direction.WEST,
            pygame.K_RIGHT: Direction.EAST,
            pygame.K_d: Direction.EAST,
        }

        hotkey_agents = {
            pygame.K_1: "human",
            pygame.K_2: "astar",
            pygame.K_3: "bfs",
            pygame.K_4: "minimax",
            pygame.K_5: "expectimax",
            pygame.K_6: "deep_q",
            pygame.K_7: "qlearning",
            pygame.K_8: "random",
        }

        if key in hotkey_agents:
            self.set_agent(hotkey_agents[key])
        elif key == pygame.K_t:
            # Hotkey T: toggle in-UI live training mode
            self.toggle_training()
        elif key == pygame.K_TAB:
            all_keys = [k for k, _ in AVAILABLE_AGENTS]
            curr_idx = all_keys.index(self.active_agent_key) if self.active_agent_key in all_keys else 0
            next_key = all_keys[(curr_idx + 1) % len(all_keys)]
            self.set_agent(next_key)
        elif key in key_dirs:
            self._buffered_action = key_dirs[key]
        elif key in (pygame.K_ESCAPE, pygame.K_q):
            self._running = False
        elif key == pygame.K_p:
            self._paused = not self._paused
            self._show_toast("PAUSED" if self._paused else "RESUMED", duration=1.0)
        elif key == pygame.K_r:
            pass
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.step_interval = max(0.04, round(self.step_interval - 0.02, 3))
            self._show_toast(f"Speed: {1.0/self.step_interval:.1f} steps/s", duration=0.8)
        elif key == pygame.K_MINUS:
            self.step_interval = min(0.35, round(self.step_interval + 0.02, 3))
            self._show_toast(f"Speed: {1.0/self.step_interval:.1f} steps/s", duration=0.8)
        elif key in (pygame.K_LEFTBRACKET, pygame.K_9):
            self.ghost_speed = max(0.2, round(self.ghost_speed - 0.05, 2))
            if self._ghost_ctrl:
                self._ghost_ctrl.speed = self.ghost_speed
            self._show_toast(f"Ghost Speed: {int(self.ghost_speed*100)}%", duration=0.8)
        elif key in (pygame.K_RIGHTBRACKET, pygame.K_0):
            self.ghost_speed = min(1.2, round(self.ghost_speed + 0.05, 2))
            if self._ghost_ctrl:
                self._ghost_ctrl.speed = self.ghost_speed
            self._show_toast(f"Ghost Speed: {int(self.ghost_speed*100)}%", duration=0.8)

    def _wait_for_restart_or_quit(self) -> None:
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self._running = False
                        waiting = False
                    elif event.key == pygame.K_r:
                        waiting = False
            if self._clock:
                self._clock.tick(15)
