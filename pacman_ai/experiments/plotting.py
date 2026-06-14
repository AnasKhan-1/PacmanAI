"""
Plotting — comparison charts and learning curves.

FR-35: Charts (win rate, avg score, survival time, RL learning curves).
FR-36: Side-by-side comparison table.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _load_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def plot_comparison(
    csv_paths: Dict[str, str | Path],
    output_dir: str | Path = "output",
) -> None:
    """Generate comparison bar charts across agents.

    *csv_paths* maps agent-name → CSV file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict[str, float]] = {}
    for agent_name, csv_path in csv_paths.items():
        rows = _load_csv(csv_path)
        if not rows:
            continue
        scores = [int(r["score"]) for r in rows]
        wins = [r["won"].strip().lower() in ("true", "1") for r in rows]
        steps = [int(r["steps"]) for r in rows]
        summaries[agent_name] = {
            "win_rate": sum(wins) / len(wins) * 100,
            "avg_score": sum(scores) / len(scores),
            "avg_steps": sum(steps) / len(steps),
        }

    if not summaries:
        return

    agents = list(summaries.keys())
    metrics = ["win_rate", "avg_score", "avg_steps"]
    titles = ["Win Rate (%)", "Average Score", "Average Survival (steps)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Agent Comparison", fontsize=16, fontweight="bold")

    colors = plt.cm.Set2.colors  # type: ignore

    for ax, metric, title in zip(axes, metrics, titles):
        values = [summaries[a][metric] for a in agents]
        bars = ax.bar(agents, values, color=colors[: len(agents)])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    plt.close()


def plot_learning_curve(
    csv_path: str | Path,
    agent_name: str = "RL Agent",
    output_dir: str | Path = "output",
    window: int = 50,
) -> None:
    """Plot score and win-rate learning curves from a single agent CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_csv(csv_path)
    if not rows:
        return
    scores = [int(r["score"]) for r in rows]
    wins = [1 if r["won"].strip().lower() in ("true", "1") else 0 for r in rows]

    # Rolling average
    def rolling(data: list, w: int) -> list:
        out = []
        for i in range(len(data)):
            start = max(0, i - w + 1)
            out.append(sum(data[start : i + 1]) / (i - start + 1))
        return out

    smooth_scores = rolling(scores, window)
    smooth_wins = rolling(wins, window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Learning Curve — {agent_name}", fontsize=14, fontweight="bold")

    ax1.plot(smooth_scores, color="steelblue", linewidth=1.5)
    ax1.set_ylabel("Score (rolling avg)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(smooth_wins, color="seagreen", linewidth=1.5)
    ax2.set_ylabel("Win Rate (rolling avg)")
    ax2.set_xlabel("Episode")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"learning_curve_{agent_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def print_comparison_table(csv_paths: Dict[str, str | Path]) -> None:
    """Print a formatted comparison table to stdout."""
    rows_out: list[dict] = []
    for agent_name, csv_path in csv_paths.items():
        rows = _load_csv(csv_path)
        if not rows:
            continue
        scores = [int(r["score"]) for r in rows]
        wins = [r["won"].strip().lower() in ("true", "1") for r in rows]
        steps = [int(r["steps"]) for r in rows]
        runtimes = [float(r["runtime_ms"]) for r in rows]
        rows_out.append({
            "Agent": agent_name,
            "Episodes": len(rows),
            "Win Rate": f"{sum(wins)/len(wins)*100:.1f}%",
            "Avg Score": f"{sum(scores)/len(scores):.1f}",
            "Avg Steps": f"{sum(steps)/len(steps):.1f}",
            "Avg Runtime (ms)": f"{sum(runtimes)/len(runtimes):.2f}",
        })

    if not rows_out:
        print("No results to display.")
        return

    # Pretty-print
    headers = list(rows_out[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows_out)) for h in headers}
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)

    print(header_line)
    print(sep_line)
    for r in rows_out:
        print(" | ".join(str(r[h]).ljust(col_widths[h]) for h in headers))
