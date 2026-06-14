# Pacman AI Agents Platform

A complete Pacman game engine with swappable AI agents (search, adversarial, reinforcement learning) and an evaluation framework.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Play the game (human)

```bash
python -m pacman_ai.main play --agent human --maze pacman_ai/mazes/medium.txt
```

**Controls:**
- Arrow keys or WASD — move Pacman
- P — pause / unpause
- R — restart
- ESC / Q — quit
- +/- — adjust game speed (AI mode)

### 3. Watch an AI agent play

```bash
python -m pacman_ai.main play --agent astar --maze pacman_ai/mazes/medium.txt --speed 8
python -m pacman_ai.main play --agent expectimax --depth 2 --speed 5
python -m pacman_ai.main play --agent minimax --depth 2 --speed 5
```

### 4. Run a headless experiment

```bash
python -m pacman_ai.main run --agent bfs --episodes 100 --maze pacman_ai/mazes/medium.txt
```

### 5. Train an RL agent

```bash
python -m pacman_ai.main run --agent approx_q --episodes 500 --train --maze pacman_ai/mazes/medium.txt
```

### 6. Compare all agents

```bash
python -m pacman_ai.main compare --episodes 20 --maze pacman_ai/mazes/medium.txt
```

This produces:
- `output/comparison.png` — bar charts
- Per-agent CSV files in `output/`
- A comparison table in the terminal

### 7. Run unit tests

```bash
python -m pacman_ai.main test
```

## Available Agents

| Agent | Type | Key Parameters |
|-------|------|----------------|
| `random` | Baseline | `seed` |
| `bfs` | Search | — |
| `astar` | Search | — |
| `minimax` | Adversarial | `depth` |
| `expectimax` | Adversarial | `depth` |
| `qlearning` | RL | `alpha`, `gamma`, `epsilon`, `seed` |
| `approx_q` | RL | `alpha`, `gamma`, `epsilon`, `seed` |

## Project Structure

```
pacman_ai/
  engine/        # Game state, rules, maze loading, ghost behavior
  agents/        # Agent interface + all AI implementations
  ui/            # Pygame renderer and game loop
  experiments/   # Headless runner, metrics, plotting
  mazes/         # Text-format maze layouts
  tests/         # Unit tests
  main.py        # CLI entry point
```

## Maze Format

Plain text files where:
- `%` = wall
- `.` = pellet
- `o` = power pellet
- `P` = Pacman start
- `G` = ghost start
- ` ` = empty corridor

## Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────────┐
│   Engine    │────▶│   Agents   │────▶│  Experiments   │
│ state/rules │     │ BFS/A*/RL  │     │ runner/metrics │
└────────────┘     └────────────┘     └────────────────┘
      │                                       │
      ▼                                       ▼
┌────────────┐                        ┌──────────────┐
│  Pygame UI │                        │ CSV + Charts │
└────────────┘                        └──────────────┘
```

The engine has **zero dependency** on the UI — all agents and experiments can run headlessly.
