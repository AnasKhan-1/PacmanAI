"""
Pacman AI Agents Platform — main entry point.

Usage:
  python main.py play [--agent AGENT] [--maze MAZE] [--ghosts N] [--speed FPS]
  python main.py run  [--agent AGENT] [--maze MAZE] [--episodes N] [--seed S] [--train]
  python main.py compare [--maze MAZE] [--episodes N] [--seed S]
  python main.py test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pacman_ai")


def _resolve_maze(maze_path: str) -> str:
    """Resolve maze file whether relative to cwd, pkg_root, or mazes directory."""
    p = Path(maze_path)
    if p.exists():
        return str(p)
    candidates = [
        _pkg_root / maze_path,
        _pkg_root / "pacman_ai" / maze_path,
        _pkg_root / "pacman_ai" / "mazes" / maze_path,
        _pkg_root / "pacman_ai" / "mazes" / f"{maze_path}.txt",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return maze_path


def cmd_play(args: argparse.Namespace) -> None:
    """Launch the Pygame UI for human or AI play."""
    import pygame
    from pacman_ai.engine.maze import Maze
    from pacman_ai.agents.registry import get_agent
    from pacman_ai.ui.game_ui import GameUI

    maze = Maze.from_file(_resolve_maze(args.maze))

    agent = None
    if args.agent and args.agent.lower() != "human":
        agent_kwargs = {}
        if args.depth:
            agent_kwargs["depth"] = args.depth
        agent = get_agent(args.agent, **agent_kwargs)

    ui = GameUI(
        maze=maze,
        agent=agent,
        ghost_behavior=args.ghost_behavior,
        num_ghosts=args.ghosts,
        seed=args.seed,
        target_fps=args.speed,
        ghost_speed=args.ghost_speed,
    )
    ui.run()


def cmd_run(args: argparse.Namespace) -> None:
    """Run a headless experiment."""
    from pacman_ai.experiments.config import ExperimentConfig
    from pacman_ai.experiments.runner import ExperimentRunner
    from pacman_ai.experiments.plotting import plot_learning_curve

    agent_kwargs = {}
    if args.depth:
        agent_kwargs["depth"] = args.depth
    if args.alpha:
        agent_kwargs["alpha"] = args.alpha
    if args.gamma:
        agent_kwargs["gamma"] = args.gamma
    if args.epsilon:
        agent_kwargs["epsilon"] = args.epsilon
    agent_kwargs["seed"] = args.seed

    config = ExperimentConfig(
        maze_file=_resolve_maze(args.maze),
        agent=args.agent,
        agent_kwargs=agent_kwargs,
        num_episodes=args.episodes,
        seed=args.seed,
        num_ghosts=args.ghosts,
        ghost_behavior=args.ghost_behavior,
        training=args.train,
        output_csv=f"{args.agent}_results.csv",
        output_dir=args.output,
    )

    runner = ExperimentRunner(config)
    metrics = runner.run()

    print("\n" + "=" * 50)
    print(f"Results for {args.agent}:")
    for k, v in metrics.summary_dict().items():
        print(f"  {k}: {v}")
    print("=" * 50)

    # Plot learning curve for RL agents
    if args.agent in ("qlearning", "approx_q"):
        csv_path = Path(args.output) / f"{args.agent}_results.csv"
        plot_learning_curve(csv_path, args.agent, args.output)
        print(f"Learning curve saved to {args.output}/")


def cmd_compare(args: argparse.Namespace) -> None:
    """Run all agents and produce comparison charts + table."""
    from pacman_ai.experiments.config import ExperimentConfig
    from pacman_ai.experiments.runner import ExperimentRunner
    from pacman_ai.experiments.plotting import plot_comparison, print_comparison_table

    agents_to_compare = ["random", "bfs", "astar", "minimax", "expectimax"]
    csv_paths = {}

    for agent_name in agents_to_compare:
        agent_kwargs = {"seed": args.seed}
        if agent_name in ("minimax", "expectimax"):
            agent_kwargs["depth"] = args.depth or 2

        config = ExperimentConfig(
            maze_file=_resolve_maze(args.maze),
            agent=agent_name,
            agent_kwargs=agent_kwargs,
            num_episodes=args.episodes,
            seed=args.seed,
            num_ghosts=args.ghosts,
            ghost_behavior=args.ghost_behavior,
            training=False,
            output_csv=f"{agent_name}_results.csv",
            output_dir=args.output,
        )

        print(f"\nRunning {agent_name}...")
        runner = ExperimentRunner(config)
        runner.run()
        csv_paths[agent_name] = Path(args.output) / f"{agent_name}_results.csv"

    # Also run approx Q with training
    print("\nTraining Approx-Q agent...")
    rl_config = ExperimentConfig(
        maze_file=_resolve_maze(args.maze),
        agent="approx_q",
        agent_kwargs={"seed": args.seed, "alpha": 0.01, "gamma": 0.9, "epsilon": 0.1},
        num_episodes=args.episodes * 5,  # More training episodes
        seed=args.seed,
        num_ghosts=args.ghosts,
        ghost_behavior=args.ghost_behavior,
        training=True,
        output_csv="approx_q_results.csv",
        output_dir=args.output,
    )
    runner = ExperimentRunner(rl_config)
    runner.run()
    csv_paths["approx_q"] = Path(args.output) / "approx_q_results.csv"

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print_comparison_table(csv_paths)

    plot_comparison(csv_paths, args.output)
    print(f"\nCharts saved to {args.output}/comparison.png")


def cmd_test(args: argparse.Namespace) -> None:
    """Run unit tests."""
    import unittest

    tests_dir = _pkg_root / "pacman_ai" / "tests"
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern="test_*.py", top_level_dir=str(_pkg_root))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pacman AI Agents Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── play ─────────────────────────────────────────────
    p_play = sub.add_parser("play", help="Play with Pygame UI")
    p_play.add_argument(
        "--agent", default="human", help="Agent name (or 'human')"
    )
    p_play.add_argument(
        "--maze", default="pacman_ai/mazes/medium.txt", help="Maze file"
    )
    p_play.add_argument("--ghosts", type=int, default=4, help="Number of ghosts")
    p_play.add_argument(
        "--ghost-behavior", default="classic", choices=["classic", "chase", "random"]
    )
    p_play.add_argument(
        "--ghost-speed",
        type=float,
        default=0.75,
        help="Ghost speed relative to Pacman (0.1 to 1.0, default: 0.75)",
    )
    p_play.add_argument("--speed", type=int, default=10, help="Game speed (FPS)")
    p_play.add_argument("--seed", type=int, default=42)
    p_play.add_argument("--depth", type=int, default=None, help="Search depth")

    # ── run ──────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Headless experiment")
    p_run.add_argument("--agent", default="random", help="Agent name")
    p_run.add_argument(
        "--maze", default="pacman_ai/mazes/medium.txt", help="Maze file"
    )
    p_run.add_argument("--episodes", type=int, default=100)
    p_run.add_argument("--ghosts", type=int, default=2)
    p_run.add_argument(
        "--ghost-behavior", default="chase", choices=["random", "chase"]
    )
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--train", action="store_true", help="Enable training mode")
    p_run.add_argument("--output", default="output", help="Output directory")
    p_run.add_argument("--depth", type=int, default=None)
    p_run.add_argument("--alpha", type=float, default=None)
    p_run.add_argument("--gamma", type=float, default=None)
    p_run.add_argument("--epsilon", type=float, default=None)

    # ── compare ──────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Compare all agents")
    p_cmp.add_argument(
        "--maze", default="pacman_ai/mazes/medium.txt", help="Maze file"
    )
    p_cmp.add_argument("--episodes", type=int, default=20)
    p_cmp.add_argument("--ghosts", type=int, default=2)
    p_cmp.add_argument(
        "--ghost-behavior", default="chase", choices=["random", "chase"]
    )
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--output", default="output")
    p_cmp.add_argument("--depth", type=int, default=2)

    # ── test ─────────────────────────────────────────────
    sub.add_parser("test", help="Run unit tests")

    args = parser.parse_args()

    if args.command is None:
        print("No command specified. Launching game UI (default: human play on medium maze)...")
        print("Tip: Run with --help or pass subcommands: play, run, compare, test\n")
        args = parser.parse_args(["play"])

    commands = {
        "play": cmd_play,
        "run": cmd_run,
        "compare": cmd_compare,
        "test": cmd_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
