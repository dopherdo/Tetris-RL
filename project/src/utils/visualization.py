"""
Simple training curve plotting utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_learning_curve(
    rewards: Sequence[float],
    save_path: str | Path,
    window: int = 50,
) -> None:
    if not rewards:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label="Episode reward", alpha=0.5)

    if len(rewards) >= window:
        import numpy as np

        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, window - 1 + len(ma)), ma, label=f"{window}-ep MA")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


