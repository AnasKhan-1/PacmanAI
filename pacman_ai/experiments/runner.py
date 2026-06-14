"""
Headless experiment runner.

FR-31: CLI runner for N episodes.
FR-33: Training mode / evaluation mode.
FR-10: Runs without graphics.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import List

from pacman_ai.engine.ghosts import GhostController, RandomGhost, ChaseGhost
from pacman_ai.engine.maze import Maze
from pacman_ai.engine.rules import GameRules
from pacman_ai.engine.state import Direction, GameState
from pacman_ai.agents.base import Agent
from pacman_ai.agents.registry import get_agent
from .config import ExperimentConfig
from .metrics import EpisodeResult, MetricsCollector

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run many games headlessly and collect statistics."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics = MetricsCollector()

    def run(self) -> MetricsCollector:
        """Execute all episodes and return the collected metrics."""
        cfg = self.config
        maze = Maze.from_file(cfg.maze_file)
        agent = get_agent(cfg.agent, **cfg.agent_kwargs)
        agent.set_training(cfg.training)

        logger.info(
            "Running %d episodes — agent=%s, maze=%s, seed=%d, training=%s",
            cfg.num_episodes,
            agent.name,
            cfg.maze_file,
            cfg.seed,
            cfg.training,
        )

        for ep in range(cfg.num_episodes):
            ep_seed = cfg.seed + ep
            result = self._run_episode(maze, agent, ep, ep_seed, cfg)
            self.metrics.add(result)
            if (ep + 1) % max(1, cfg.num_episodes // 10) == 0:
                logger.info(
                    "  Episode %d/%d — score=%d, won=%s",
                    ep + 1,
                    cfg.num_episodes,
                    result.score,
                    result.won,
                )

        agent.set_training(False)

        # Export CSV
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / cfg.output_csv
        self.metrics.to_csv(csv_path)
        logger.info("Results saved to %s", csv_path)

        # Save RL weights if applicable
        if hasattr(agent, "save"):
            model_path = out_dir / f"{agent.name.lower().replace(' ', '_')}_model"
            agent.save(model_path)
            logger.info("Model saved to %s", model_path)

        return self.metrics

    def _run_episode(
        self,
        maze: Maze,
        agent: Agent,
        episode: int,
        seed: int,
        cfg: ExperimentConfig,
    ) -> EpisodeResult:
        rng = random.Random(seed)
        state = GameState.from_maze(
            maze,
            num_ghosts=cfg.num_ghosts,
            lives=cfg.lives,
            step_limit=cfg.step_limit,
        )

        if cfg.ghost_behavior == "random":
            behavior = RandomGhost()
        else:
            behavior = ChaseGhost()
        ghost_ctrl = GhostController(behavior, state.num_ghosts, rng=rng)

        agent.start_episode()
        t0 = time.perf_counter()

        while not state.is_terminal():
            action = agent.get_action(state)
            ghost_actions = ghost_ctrl.get_actions(state)
            old_state = state
            state = GameRules.get_successor(state, action, ghost_actions)

            reward = state.score - old_state.score
            agent.observe_transition(old_state, action, state, reward)

        t1 = time.perf_counter()
        agent.end_episode()

        return EpisodeResult(
            episode=episode,
            agent=agent.name,
            maze=str(cfg.maze_file),
            seed=seed,
            won=state.is_win,
            score=state.score,
            steps=state.step_count,
            pellets_eaten=state.pellets_eaten,
            runtime_ms=(t1 - t0) * 1000,
        )
