"""Pacman AI agents package."""

from .base import Agent
from .random_agent import RandomAgent
from .search import BFSAgent, AStarAgent
from .adversarial import MinimaxAgent, ExpectimaxAgent
from .rl import QLearningAgent, ApproxQLearningAgent
from .registry import AGENT_REGISTRY, get_agent

__all__ = [
    "Agent",
    "RandomAgent",
    "BFSAgent",
    "AStarAgent",
    "MinimaxAgent",
    "ExpectimaxAgent",
    "QLearningAgent",
    "ApproxQLearningAgent",
    "AGENT_REGISTRY",
    "get_agent",
]
