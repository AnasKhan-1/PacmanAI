"""
Pygame renderer — draws the 28x30 maze, Pacman with animated chomp, 4 authentic colored ghosts,
interactive Agent Selector UI bar with Train Mode toggle, and HUD.

FR-15: Render maze, Pacman, ghosts, pellets, score, lives.
FR-19: Show agent name and live statistics.
NFR-3: Stable 60 FPS sub-pixel rendering.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pygame

from pacman_ai.engine.state import Direction, GameState, GhostMode

# ── Colour palette ───────────────────────────────────────

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_BLUE = (8, 10, 32)
WALL_BLUE = (25, 45, 180)
WALL_INNER = (55, 85, 235)
PELLET_WHITE = (255, 255, 220)
POWER_PELLET = (255, 184, 82)
PACMAN_YELLOW = (255, 238, 0)
GOLD = (255, 215, 0)
CYAN = (0, 220, 255)
EMERALD = (0, 230, 110)

# Authentic 4 Ghost Colors: 🔴 Red, 💗 Pink, 🔷 Cyan, 🍊 Orange
GHOST_COLORS = [
    (255, 0, 0),       # 🔴 Blinky
    (255, 184, 222),   # 💗 Pinky
    (0, 255, 255),     # 🔷 Inky
    (255, 184, 82),    # 🍊 Clyde
]
FRIGHTENED_BLUE = (40, 60, 220)
HUD_BG = (14, 18, 30)
HUD_BORDER = (35, 45, 75)
HUD_TEXT = (220, 230, 245)
WIN_GREEN = (0, 230, 90)
LOSE_RED = (230, 45, 45)

# Available agent algorithms for UI switcher
AVAILABLE_AGENTS = [
    ("human", "1: Human"),
    ("astar", "2: A*"),
    ("bfs", "3: BFS"),
    ("minimax", "4: Minimax"),
    ("expectimax", "5: Expectimax"),
    ("deep_q", "6: Deep Q"),
    ("qlearning", "7: Q-Learn"),
    ("random", "8: Random"),
]


class PacmanRenderer:
    """Renders the Pacman game state with sub-pixel interpolation, chomp animation, and Agent UI."""

    def __init__(
        self,
        cell_size: int = 22,
        top_bar_height: int = 68,
        hud_height: int = 50,
    ):
        self.cell_size = cell_size
        self.top_bar_height = top_bar_height
        self.hud_height = hud_height
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._btn_font: Optional[pygame.font.Font] = None
        self._title_font: Optional[pygame.font.Font] = None

    def init_display(self, state: GameState) -> pygame.Surface:
        """Create and return the Pygame display surface sized for this maze."""
        pygame.init()
        pygame.display.set_caption("Pacman AI Agents Platform — 4 Ghosts & Training Mode")
        width = state.maze.width * self.cell_size
        height = self.top_bar_height + state.maze.height * self.cell_size + self.hud_height
        surface = pygame.display.set_mode((width, height))
        self._font = pygame.font.SysFont("consolas", 18, bold=True)
        self._small_font = pygame.font.SysFont("consolas", 13)
        self._btn_font = pygame.font.SysFont("consolas", 12, bold=True)
        self._title_font = pygame.font.SysFont("consolas", 22, bold=True)
        return surface

    def get_button_rects(self, width: int, is_training: bool = False) -> list[tuple[str, str, pygame.Rect]]:
        """Return (key, label, rect) for all UI buttons in the top bar.

        Row 1 (5 buttons): Human, A*, BFS, Minimax, Expectimax
        Row 2 (4 buttons): Deep Q, Q-Learn, Random, ⚡ Train Mode
        """
        pad_x = 5
        pad_y = 5
        btn_h = 25

        row1_items = [
            ("human", "1: Human"),
            ("astar", "2: A*"),
            ("bfs", "3: BFS"),
            ("minimax", "4: Minimax"),
            ("expectimax", "5: Expectimax"),
        ]

        train_label = "⚡ Train: ON" if is_training else "⚡ Train: OFF"
        row2_items = [
            ("deep_q", "6: Deep Q"),
            ("qlearning", "7: Q-Learn"),
            ("random", "8: Random"),
            ("train", train_label),
        ]

        buttons = []

        # Row 1
        w1 = (width - (len(row1_items) + 1) * pad_x) // len(row1_items)
        for i, (k, lbl) in enumerate(row1_items):
            rx = pad_x + i * (w1 + pad_x)
            ry = pad_y
            buttons.append((k, lbl, pygame.Rect(rx, ry, w1, btn_h)))

        # Row 2
        w2 = (width - (len(row2_items) + 1) * pad_x) // len(row2_items)
        for i, (k, lbl) in enumerate(row2_items):
            rx = pad_x + i * (w2 + pad_x)
            ry = pad_y + btn_h + 4
            buttons.append((k, lbl, pygame.Rect(rx, ry, w2, btn_h)))

        return buttons

    def draw(
        self,
        surface: pygame.Surface,
        state: GameState,
        render_pacman: Tuple[float, float],
        pacman_direction: Direction = Direction.STOP,
        mouth_angle: float = 0.0,
        render_ghosts: Optional[List[Tuple[float, float]]] = None,
        ghost_directions: Optional[List[Direction]] = None,
        agent_name: str = "Human",
        active_agent_key: str = "human",
        fps: float = 60.0,
        ghost_speed: float = 0.65,
        animation_frame: int = 0,
        toast_message: Optional[str] = None,
        is_training: bool = False,
        train_stats: Optional[dict] = None,
        is_waiting_to_start: bool = False,
        is_death_paused: bool = False,
        is_intro_state: bool = False,
        frightened_timer: float = 0.0,
        score_popups: Optional[List[dict]] = None,
    ) -> None:
        """Draw the complete game frame at 60 FPS with sub-pixel coordinates."""
        surface.fill(DARK_BLUE)

        # 1. Top Agent Selector & Training Bar
        self._draw_agent_selector(surface, active_agent_key, is_training)

        # 2. Maze Walls & Pellets
        self._draw_maze(surface, state)
        self._draw_pellets(surface, state)

        # 3. Pacman with chomp animation & sub-pixel position
        self._draw_pacman(surface, render_pacman, pacman_direction, mouth_angle)

        # 4. 4 Ghosts (Blinky, Pinky, Inky, Clyde) with wavy skirts & directional eyes
        ghost_coords = render_ghosts if render_ghosts is not None else [g.position for g in state.ghosts]
        ghost_dirs = ghost_directions if ghost_directions is not None else [Direction.STOP] * state.num_ghosts
        self._draw_ghosts(surface, state, ghost_coords, ghost_dirs, animation_frame, frightened_timer)

        # 5. Score Popups (200, 400, 800, 1600 on eaten ghost)
        if score_popups:
            self._draw_score_popups(surface, score_popups)

        # 6. Bottom HUD (showing stats or training progress)
        self._draw_hud(surface, state, agent_name, fps, ghost_speed, is_training, train_stats)

        # 7. Toast Notification
        if toast_message:
            self._draw_toast(surface, toast_message)

        # 8. Start Screen / Intro Sequence / Death Pause / Game Over Overlays
        if is_waiting_to_start:
            self._draw_start_screen(surface, agent_name, animation_frame)
        elif is_intro_state:
            self._draw_ready_banner(surface, state)
        elif is_death_paused:
            self._draw_death_screen(surface, state.lives, animation_frame)
        elif state.is_terminal() and not is_training:
            self._draw_game_over(surface, state)

        pygame.display.flip()

    # ── Drawing Helpers ──────────────────────────────────

    def _draw_agent_selector(self, surface: pygame.Surface, active_agent_key: str, is_training: bool) -> None:
        width = surface.get_width()
        bar_rect = pygame.Rect(0, 0, width, self.top_bar_height)
        pygame.draw.rect(surface, HUD_BG, bar_rect)
        pygame.draw.line(surface, HUD_BORDER, (0, self.top_bar_height - 1), (width, self.top_bar_height - 1), 2)

        mouse_pos = pygame.mouse.get_pos()
        buttons = self.get_button_rects(width, is_training)

        for key, label, rect in buttons:
            is_active = (key == active_agent_key)
            is_hover = rect.collidepoint(mouse_pos)

            if key == "train":
                # Special styling for Train button
                if is_training:
                    pygame.draw.rect(surface, EMERALD, rect, border_radius=5)
                    pygame.draw.rect(surface, WHITE, rect, 2, border_radius=5)
                    text_color = BLACK
                elif is_hover:
                    pygame.draw.rect(surface, (60, 40, 20), rect, border_radius=5)
                    pygame.draw.rect(surface, GOLD, rect, 2, border_radius=5)
                    text_color = GOLD
                else:
                    pygame.draw.rect(surface, (40, 25, 15), rect, border_radius=5)
                    pygame.draw.rect(surface, (180, 120, 40), rect, 1, border_radius=5)
                    text_color = (255, 200, 120)
            elif is_active:
                # Active algorithm button: glowing gold badge
                pygame.draw.rect(surface, GOLD, rect, border_radius=5)
                pygame.draw.rect(surface, WHITE, rect, 2, border_radius=5)
                text_color = BLACK
            elif is_hover:
                # Hover state
                pygame.draw.rect(surface, (45, 60, 95), rect, border_radius=5)
                pygame.draw.rect(surface, CYAN, rect, 1, border_radius=5)
                text_color = WHITE
            else:
                # Normal state
                pygame.draw.rect(surface, (25, 33, 52), rect, border_radius=5)
                pygame.draw.rect(surface, (50, 65, 95), rect, 1, border_radius=5)
                text_color = (200, 215, 235)

            if self._btn_font:
                txt = self._btn_font.render(label, True, text_color)
                t_rect = txt.get_rect(center=rect.center)
                surface.blit(txt, t_rect)

    def _draw_maze(self, surface: pygame.Surface, state: GameState) -> None:
        cs = self.cell_size
        y_off = self.top_bar_height
        for wx, wy in state.maze.walls:
            rect = pygame.Rect(wx * cs, y_off + wy * cs, cs, cs)
            pygame.draw.rect(surface, WALL_BLUE, rect)
            inner = rect.inflate(-3, -3)
            pygame.draw.rect(surface, WALL_INNER, inner, 1, border_radius=3)

    def _draw_pellets(self, surface: pygame.Surface, state: GameState) -> None:
        cs = self.cell_size
        y_off = self.top_bar_height

        for px, py in state.pellets:
            cx = px * cs + cs // 2
            cy = y_off + py * cs + cs // 2
            pygame.draw.circle(surface, PELLET_WHITE, (cx, cy), max(2, cs // 8))

        for px, py in state.power_pellets:
            cx = px * cs + cs // 2
            cy = y_off + py * cs + cs // 2
            pygame.draw.circle(surface, POWER_PELLET, (cx, cy), max(4, cs // 3))
            pygame.draw.circle(surface, (255, 240, 160), (cx, cy), max(2, cs // 5))

    def _draw_pacman(
        self,
        surface: pygame.Surface,
        render_pos: Tuple[float, float],
        direction: Direction,
        mouth_angle: float,
    ) -> None:
        cs = self.cell_size
        cx = render_pos[0] * cs + cs / 2.0
        cy = self.top_bar_height + render_pos[1] * cs + cs / 2.0
        radius = cs / 2.0 - 1.5

        dir_angles = {
            Direction.EAST: 0.0,
            Direction.NORTH: 90.0,
            Direction.WEST: 180.0,
            Direction.SOUTH: 270.0,
            Direction.STOP: 0.0,
        }
        base_deg = dir_angles.get(direction, 0.0)

        # Chomp mouth polygon
        if mouth_angle < 3.0:
            pygame.draw.circle(surface, PACMAN_YELLOW, (int(cx), int(cy)), int(radius))
        else:
            start_deg = base_deg + mouth_angle
            end_deg = base_deg + 360.0 - mouth_angle
            num_points = 24
            points = [(cx, cy)]
            for i in range(num_points + 1):
                deg = start_deg + (end_deg - start_deg) * (i / num_points)
                rad = math.radians(deg)
                px = cx + radius * math.cos(rad)
                py = cy - radius * math.sin(rad)
                points.append((px, py))
            pygame.draw.polygon(surface, PACMAN_YELLOW, points)

        # Eye positioned perpendicular to facing direction
        eye_dist = radius * 0.45
        if direction in (Direction.EAST, Direction.STOP):
            ex, ey = cx + 2, cy - eye_dist
        elif direction == Direction.WEST:
            ex, ey = cx - 2, cy - eye_dist
        elif direction == Direction.NORTH:
            ex, ey = cx - eye_dist, cy - 2
        elif direction == Direction.SOUTH:
            ex, ey = cx - eye_dist, cy + 2
        else:
            ex, ey = cx + 2, cy - eye_dist

        pygame.draw.circle(surface, (20, 20, 20), (int(ex), int(ey)), max(2, int(radius * 0.16)))

    def _draw_ghosts(
        self,
        surface: pygame.Surface,
        state: GameState,
        render_ghost_positions: List[Tuple[float, float]],
        ghost_directions: List[Direction],
        animation_frame: int,
        frightened_timer: float = 0.0,
    ) -> None:
        cs = self.cell_size
        radius = cs / 2.0 - 1.5

        for i, ghost in enumerate(state.ghosts):
            gx, gy = render_ghost_positions[i] if i < len(render_ghost_positions) else ghost.position
            cx = gx * cs + cs / 2.0
            cy = self.top_bar_height + gy * cs + cs / 2.0

            g_dir = ghost_directions[i] if i < len(ghost_directions) else Direction.STOP
            dir_eye_offsets = {
                Direction.EAST: (3, 0),
                Direction.WEST: (-3, 0),
                Direction.NORTH: (0, -3),
                Direction.SOUTH: (0, 3),
                Direction.STOP: (0, 0),
            }
            dx_eye, dy_eye = dir_eye_offsets.get(g_dir, (0, 0))

            # ── 1. GhostMode.EATEN (Eyes Only) ──
            if ghost.mode == GhostMode.EATEN:
                eye_spacing = radius * 0.42
                left_eye = (cx - eye_spacing, cy - 1)
                right_eye = (cx + eye_spacing, cy - 1)
                sclera_r = max(3, int(radius * 0.38))
                pygame.draw.circle(surface, WHITE, (int(left_eye[0]), int(left_eye[1])), sclera_r)
                pygame.draw.circle(surface, WHITE, (int(right_eye[0]), int(right_eye[1])), sclera_r)

                pupil_r = max(2, sclera_r // 2)
                pygame.draw.circle(
                    surface, (33, 33, 255),
                    (int(left_eye[0] + dx_eye), int(left_eye[1] + dy_eye)), pupil_r
                )
                pygame.draw.circle(
                    surface, (33, 33, 255),
                    (int(right_eye[0] + dx_eye), int(right_eye[1] + dy_eye)), pupil_r
                )
                continue

            # ── 2. Determine Color & Face for Frightened vs Normal ──
            if ghost.mode == GhostMode.FRIGHTENED:
                # Flashing warning in final 2 seconds at ~3 Hz (3-5 flashes)
                is_flashing = (frightened_timer > 0.0 and frightened_timer <= 2.0)
                is_white = is_flashing and (int(frightened_timer * 4.0) % 2 == 1)

                if is_white:
                    body_color = (240, 240, 255)
                    face_color = (220, 30, 30)  # Red warning face
                else:
                    body_color = (33, 33, 255)  # Solid frightened blue
                    face_color = (255, 180, 180)  # Pink face
            else:
                body_color = GHOST_COLORS[i % len(GHOST_COLORS)]
                face_color = None

            # ── 3. Draw Body (Dome Top + Torso + Tentacles) ──
            # Dome top
            pygame.draw.circle(surface, body_color, (int(cx), int(cy - 2)), int(radius))

            # Torso
            body_rect = pygame.Rect(
                int(cx - radius), int(cy - 2), int(radius * 2), int(radius)
            )
            pygame.draw.rect(surface, body_color, body_rect)

            # Wavy skirt feet
            skirt_y = cy - 2 + radius
            num_tentacles = 3
            step_w = (radius * 2) / num_tentacles
            shift = 2 if (animation_frame // 15) % 2 == 0 else -2
            for t in range(num_tentacles):
                tx = cx - radius + t * step_w
                tri = [
                    (tx, skirt_y),
                    (tx + step_w / 2.0, skirt_y + 3 + (shift if t % 2 == 0 else -shift)),
                    (tx + step_w, skirt_y),
                ]
                pygame.draw.polygon(surface, body_color, tri)

            # ── 4. Draw Face ──
            if ghost.mode == GhostMode.FRIGHTENED:
                # Frightened face: small eyes + squiggly mouth
                pygame.draw.circle(surface, face_color, (int(cx - 3), int(cy - 3)), 2)
                pygame.draw.circle(surface, face_color, (int(cx + 3), int(cy - 3)), 2)
                pygame.draw.line(surface, face_color, (int(cx - 4), int(cy + 3)), (int(cx + 4), int(cy + 3)), 2)
            else:
                # Normal face: white sclera + blue pupils gazing in direction
                eye_spacing = radius * 0.42
                left_eye_center = (cx - eye_spacing, cy - 3)
                right_eye_center = (cx + eye_spacing, cy - 3)
                sclera_r = max(2, int(radius * 0.28))
                pygame.draw.circle(surface, WHITE, (int(left_eye_center[0]), int(left_eye_center[1])), sclera_r)
                pygame.draw.circle(surface, WHITE, (int(right_eye_center[0]), int(right_eye_center[1])), sclera_r)

                pupil_r = max(1, sclera_r // 2)
                pygame.draw.circle(
                    surface, (30, 30, 200),
                    (int(left_eye_center[0] + dx_eye), int(left_eye_center[1] + dy_eye)), pupil_r
                )
                pygame.draw.circle(
                    surface, (30, 30, 200),
                    (int(right_eye_center[0] + dx_eye), int(right_eye_center[1] + dy_eye)), pupil_r
                )

    def _draw_ready_banner(self, surface: pygame.Surface, state: GameState) -> None:
        """Render arcade-authentic yellow READY! banner right under the ghost house."""
        if not self._title_font:
            return
        ready_txt = self._title_font.render("READY!", True, PACMAN_YELLOW)
        y_pos = self.top_bar_height + int(17.5 * self.cell_size)
        r_rect = ready_txt.get_rect(center=(surface.get_width() // 2, y_pos))
        surface.blit(ready_txt, r_rect)

    def _draw_score_popups(self, surface: pygame.Surface, score_popups: List[dict]) -> None:
        """Draw floating arcade score awards (200, 400, 800, 1600) on eaten ghosts."""
        if not self._font or not score_popups:
            return
        cs = self.cell_size
        for pop in score_popups:
            gx, gy = pop["pos"]
            score_val = pop["score"]
            cx = int(gx * cs + cs / 2.0)
            cy = int(self.top_bar_height + gy * cs + cs / 2.0)
            txt = self._font.render(str(score_val), True, CYAN)
            t_rect = txt.get_rect(center=(cx, cy))
            surface.blit(txt, t_rect)

    def _draw_hud(
        self,
        surface: pygame.Surface,
        state: GameState,
        agent_name: str,
        fps: float,
        ghost_speed: float,
        is_training: bool = False,
        train_stats: Optional[dict] = None,
    ) -> None:
        cs = self.cell_size
        hud_y = self.top_bar_height + state.maze.height * cs
        hud_rect = pygame.Rect(0, hud_y, surface.get_width(), self.hud_height)
        pygame.draw.rect(surface, HUD_BG, hud_rect)
        pygame.draw.line(surface, HUD_BORDER, (0, hud_y), (surface.get_width(), hud_y), 2)

        if self._font is None:
            return

        if is_training and train_stats is not None:
            # Training HUD display
            ep = train_stats.get("episodes", 0)
            win_r = train_stats.get("win_rate", 0.0)
            avg_s = train_stats.get("avg_score", 0.0)
            eps = train_stats.get("epsilon", 0.0)
            t_text = self._font.render(
                f"⚡ TRAINING {agent_name} | Ep: {ep}", True, EMERALD
            )
            surface.blit(t_text, (10, hud_y + 6))

            if self._small_font:
                t_stats = self._small_font.render(
                    f"Win Rate: {win_r:.1f}% | Avg Score: {avg_s:.0f} | ε: {eps:.3f} | Live Learning Active",
                    True,
                    HUD_TEXT,
                )
                surface.blit(t_stats, (10, hud_y + 28))
            return

        # Normal Play HUD
        score_text = self._font.render(f"Score: {state.score}", True, GOLD)
        surface.blit(score_text, (10, hud_y + 6))

        lives_text = self._font.render(f"Lives: {state.lives}", True, (255, 80, 80))
        surface.blit(lives_text, (150, hud_y + 6))

        if agent_name:
            tag = f"Playing: {agent_name}"
            agent_text = self._small_font.render(tag, True, CYAN)
            surface.blit(agent_text, (260, hud_y + 8))

        if self._small_font:
            pellets_left = len(state.pellets) + len(state.power_pellets)
            stats = self._small_font.render(
                f"Step: {state.step_count}  Pellets: {pellets_left}  Ghost Spd: {int(ghost_speed*100)}%  FPS: {fps:.0f}",
                True,
                HUD_TEXT,
            )
            surface.blit(stats, (10, hud_y + 28))

    def _draw_toast(self, surface: pygame.Surface, message: str) -> None:
        if not self._font:
            return
        txt = self._font.render(message, True, GOLD)
        padding = 16
        bg_rect = pygame.Rect(
            surface.get_width() // 2 - txt.get_width() // 2 - padding,
            self.top_bar_height + 10,
            txt.get_width() + padding * 2,
            txt.get_height() + 8,
        )
        overlay = pygame.Surface((bg_rect.width, bg_rect.height))
        overlay.set_alpha(225)
        overlay.fill((10, 15, 30))
        surface.blit(overlay, bg_rect)
        pygame.draw.rect(surface, CYAN, bg_rect, 2, border_radius=6)
        surface.blit(txt, (bg_rect.x + padding, bg_rect.y + 4))

    def _draw_game_over(self, surface: pygame.Surface, state: GameState) -> None:
        if self._title_font is None:
            return
        if state.is_win:
            text = self._title_font.render("VICTORY! ALL PELLETS CLEARED", True, WIN_GREEN)
        else:
            text = self._title_font.render("GAME OVER — PRESS R TO RESTART", True, LOSE_RED)

        t_rect = text.get_rect(
            center=(surface.get_width() // 2, surface.get_height() // 2)
        )
        overlay = pygame.Surface((t_rect.width + 40, t_rect.height + 24))
        overlay.set_alpha(230)
        overlay.fill(BLACK)
        surface.blit(overlay, (t_rect.x - 20, t_rect.y - 12))
        pygame.draw.rect(surface, GOLD if state.is_win else LOSE_RED, (t_rect.x - 20, t_rect.y - 12, t_rect.width + 40, t_rect.height + 24), 2, border_radius=8)
        surface.blit(text, t_rect)

    def get_start_button_rect(self, width: int, height: int) -> pygame.Rect:
        """Return the clickable Rect for the Start Game button."""
        bw, bh = 240, 50
        bx = (width - bw) // 2
        by = (height - bh) // 2 + 30
        return pygame.Rect(bx, by, bw, bh)

    def _draw_start_screen(self, surface: pygame.Surface, agent_name: str, animation_frame: int = 0) -> None:
        """Render retro arcade start screen with glowing Start Game button."""
        w, h = surface.get_width(), surface.get_height()

        # Dim backdrop
        dim = pygame.Surface((w, h))
        dim.set_alpha(200)
        dim.fill((5, 10, 22))
        surface.blit(dim, (0, 0))

        # Title
        if self._title_font:
            title = self._title_font.render("PAC-MAN AI PLATFORM", True, GOLD)
            t_rect = title.get_rect(center=(w // 2, h // 2 - 80))
            surface.blit(title, t_rect)

        # Agent selected
        if self._font:
            sub = self._font.render(f"Ready with: {agent_name}", True, CYAN)
            s_rect = sub.get_rect(center=(w // 2, h // 2 - 40))
            surface.blit(sub, s_rect)

        # Start Button
        btn_rect = self.get_start_button_rect(w, h)
        pulse = abs(math.sin(animation_frame * 0.08))
        glow_color = (int(0 + 50 * pulse), int(220 + 35 * pulse), int(100 + 50 * pulse))

        pygame.draw.rect(surface, (15, 32, 28), btn_rect, border_radius=10)
        pygame.draw.rect(surface, glow_color, btn_rect, 3, border_radius=10)

        if self._font:
            btn_txt = self._font.render("▶  START GAME", True, WHITE)
            b_rect = btn_txt.get_rect(center=btn_rect.center)
            surface.blit(btn_txt, b_rect)

        # Key prompt
        if self._small_font:
            prompt = self._small_font.render("Click [START GAME] or Press SPACE / ENTER to Play", True, (210, 230, 255))
            p_rect = prompt.get_rect(center=(w // 2, btn_rect.bottom + 26))
            surface.blit(prompt, p_rect)

            hints = self._small_font.render("Controls: WASD / Arrows | 1-8 Choose Agent | T Live Train", True, HUD_TEXT)
            h_rect = hints.get_rect(center=(w // 2, btn_rect.bottom + 50))
            surface.blit(hints, h_rect)

    def _draw_death_screen(self, surface: pygame.Surface, lives_left: int, animation_frame: int = 0) -> None:
        """Render pause overlay when a life is lost waiting for player keypress to resume."""
        w, h = surface.get_width(), surface.get_height()

        dim = pygame.Surface((w, h))
        dim.set_alpha(185)
        dim.fill((22, 6, 6))
        surface.blit(dim, (0, 0))

        if self._title_font:
            title = self._title_font.render("LIFE LOST!", True, LOSE_RED)
            t_rect = title.get_rect(center=(w // 2, h // 2 - 45))
            surface.blit(title, t_rect)

        if self._font:
            lives_msg = f"Lives Remaining: {lives_left}"
            l_txt = self._font.render(lives_msg, True, GOLD)
            l_rect = l_txt.get_rect(center=(w // 2, h // 2))
            surface.blit(l_txt, l_rect)

        if self._small_font:
            pulse = abs(math.sin(animation_frame * 0.10))
            col = (int(200 + 55 * pulse), int(200 + 55 * pulse), int(200 + 55 * pulse))
            prompt = self._small_font.render("Press ANY KEY or CLICK to Continue", True, col)
            p_rect = prompt.get_rect(center=(w // 2, h // 2 + 40))
            surface.blit(prompt, p_rect)
