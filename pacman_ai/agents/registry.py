"""
Agent registry — one-stop lookup so agents can be selected by name / config.

FR-21: Agents interchangeable through configuration.
NFR-9: Adding a new agent requires one class + one registry entry.
"""

from __future__ import annotations

import inspect
from typing import Dict, Type

from .base import Agent
from .random_agent import RandomAgent
from .search import BFSAgent, AStarAgent
from .adversarial import MinimaxAgent, ExpectimaxAgent
from .rl import QLearningAgent, ApproxQLearningAgent, DeepQLearningAgent

# ── Registry ─────────────────────────────────────────────

AGENT_REGISTRY: Dict[str, Type[Agent]] = {
    "random": RandomAgent,
    "bfs": BFSAgent,
    "astar": AStarAgent,
    "minimax": MinimaxAgent,
    "expectimax": ExpectimaxAgent,
    "qlearning": QLearningAgent,
    "approx_q": ApproxQLearningAgent,
    "dqn": DeepQLearningAgent,
    "deep_q": DeepQLearningAgent,
    "deep_q_learning": DeepQLearningAgent,
}


def get_agent(name: str, **kwargs) -> Agent:
    """Instantiate an agent by its registry name.

    Only passes kwargs that the agent's __init__ actually accepts,
    so extra keys (e.g. 'seed' for BFS) are silently dropped.

    Raises ValueError if the name is unknown.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    cls = AGENT_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(AGENT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown agent '{name}'. Available: {available}"
        )

    # If class has no custom __init__, it takes no arguments
    if cls.__init__ is object.__init__:
        return cls()

    # Filter kwargs to only those accepted by the class
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        # Class accepts **kwargs — pass everything
        filtered = kwargs
    else:
        accepted = {
            n for n, p in params.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and n != "self"
        }
        filtered = {k: v for k, v in kwargs.items() if k in accepted}

    return cls(**filtered)
