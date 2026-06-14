"""Experiments package — headless runner, metrics, plotting, config."""

from .runner import ExperimentRunner
from .config import ExperimentConfig
from .plotting import plot_comparison, plot_learning_curve

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "plot_comparison",
    "plot_learning_curve",
]
