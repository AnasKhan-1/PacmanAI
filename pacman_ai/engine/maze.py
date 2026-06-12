"""
Maze loader — reads a plain text file and produces a structured maze object.

Text file format:
    % = wall
    . = pellet
    o = power pellet
    P = Pacman start
    G = ghost start
    ' ' (space) = empty corridor

FR-1: Load a maze from a text file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple


@dataclass(frozen=True)
class Maze:
    """Immutable representation of a Pacman maze layout."""

    width: int
    height: int
    walls: frozenset[Tuple[int, int]]
    pellets: frozenset[Tuple[int, int]]
    power_pellets: frozenset[Tuple[int, int]]
    pacman_start: Tuple[int, int]
    ghost_starts: Tuple[Tuple[int, int], ...]

    # ── helpers ──────────────────────────────────────────────

    def wrap_position(self, x: int, y: int) -> Tuple[int, int]:
        """Wrap coordinates horizontally for side tunnels."""
        if 0 <= y < self.height:
            if (0, y) not in self.walls and (self.width - 1, y) not in self.walls:
                return (x % self.width, y)
        return (x, y)

    def is_wall(self, x: int, y: int) -> bool:
        """Return True if (x, y) is a wall or outside the grid."""
        wx, wy = self.wrap_position(x, y)
        if wx < 0 or wy < 0 or wx >= self.width or wy >= self.height:
            return True
        return (wx, wy) in self.walls

    def is_valid_position(self, x: int, y: int) -> bool:
        """Return True if (x, y) is inside the grid and NOT a wall."""
        return not self.is_wall(x, y)

    # ── class factory ────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path) -> "Maze":
        """Load a maze from a text file.

        The file is read top-to-bottom; the first line is row 0 (y=0).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Maze file not found: {path}")
        text = path.read_text()
        return cls.from_string(text)

    @classmethod
    def from_string(cls, text: str) -> "Maze":
        """Parse a maze from a multi-line string."""
        lines = text.strip().splitlines()
        height = len(lines)
        width = max(len(line) for line in lines)

        walls: set[Tuple[int, int]] = set()
        pellets: set[Tuple[int, int]] = set()
        power_pellets: set[Tuple[int, int]] = set()
        pacman_start: Tuple[int, int] | None = None
        ghost_starts: list[Tuple[int, int]] = []

        for y, line in enumerate(lines):
            for x, ch in enumerate(line):
                pos = (x, y)
                if ch == "%":
                    walls.add(pos)
                elif ch == ".":
                    pellets.add(pos)
                elif ch == "o":
                    power_pellets.add(pos)
                elif ch == "P":
                    if pacman_start is not None:
                        raise ValueError("Multiple Pacman starts in maze")
                    pacman_start = pos
                elif ch == "G":
                    ghost_starts.append(pos)
                elif ch == " ":
                    pass  # empty corridor
                else:
                    raise ValueError(
                        f"Unknown maze character '{ch}' at ({x}, {y})"
                    )

        if pacman_start is None:
            raise ValueError("Maze has no Pacman start position ('P')")

        return cls(
            width=width,
            height=height,
            walls=frozenset(walls),
            pellets=frozenset(pellets),
            power_pellets=frozenset(power_pellets),
            pacman_start=pacman_start,
            ghost_starts=tuple(ghost_starts),
        )

    def __str__(self) -> str:
        """Pretty-print the maze (useful for debugging)."""
        grid: list[list[str]] = [
            [" "] * self.width for _ in range(self.height)
        ]
        for x, y in self.walls:
            grid[y][x] = "%"
        for x, y in self.pellets:
            grid[y][x] = "."
        for x, y in self.power_pellets:
            grid[y][x] = "o"
        px, py = self.pacman_start
        grid[py][px] = "P"
        for gx, gy in self.ghost_starts:
            grid[gy][gx] = "G"
        return "\n".join("".join(row) for row in grid)
