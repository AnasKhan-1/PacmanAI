"""
Keyboard input handler.

FR-16: Human plays with arrow keys or WASD.
"""

from __future__ import annotations

import pygame

from pacman_ai.engine.state import Direction

# Key mapping
_KEY_MAP = {
    pygame.K_UP: Direction.NORTH,
    pygame.K_DOWN: Direction.SOUTH,
    pygame.K_LEFT: Direction.WEST,
    pygame.K_RIGHT: Direction.EAST,
    pygame.K_w: Direction.NORTH,
    pygame.K_s: Direction.SOUTH,
    pygame.K_a: Direction.WEST,
    pygame.K_d: Direction.EAST,
}


def get_human_action() -> Direction | None:
    """Check pressed keys and return a Direction, or None if no movement key is held."""
    keys = pygame.key.get_pressed()
    for key, direction in _KEY_MAP.items():
        if keys[key]:
            return direction
    return None
