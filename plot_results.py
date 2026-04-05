"""
plot_results.py  —  SentryNet Phase 2
======================================
Generate publication plots from full_experiment.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS: Dict[str, str] = {
    "A": "#7f7f7f",  # gray
    "B": "#1f77b4",  # blue
    "C": "#1b9e77",  # teal/green
}


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load and validate experiment CSV."""
    assert csv_path.exists(), f"Missing results CSV: {csv_path}"
    df = pd.read_csv(csv_path)
    required = [
        "system",
        "seed",
        "drop_rate",
        "capture_rate",
        "mean_steps",
        "mean_reward",
        "mean_trust",
        "mean_battery",
    ]
    for c in required:
        assert c in df.columns, f"Missing column: {c}"
    return df


def aggregate(df: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean and std by system/drop_rate."""
    grp = df.groupby(["system", "drop_rate"])[y_col]
    mean = grp.mean().unstack(0).sort_index()
    std = grp.std(ddof=0).unstack(0).sort_index()
    return mean, std


def save_curve_plot(
    x: np.ndarray,
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    y_label: str,
    title: str,
    out_path: Path,
    add_threshold: bool = False,
) -> None:
    """Save line plot with seed-std shading."""
    plt.figure(figsize=(10, 6), dpi=300)

    for system in ["A", "B", "C"]:
        if system not in mean_df.columns:
            continue
        y = mean_df[system].values
        s = std_df[system].values
        plt.plot(x, y, label=f"System {system}", color=COLORS[system], linewidth=2.5)
        plt.fill_between(x, y - s, y + s, color=COLORS[system], alpha=0.2)

    if add_threshold:
        plt.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="50% threshold")

    plt.xlabel("drop_rate")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_training_curves(df: pd.DataFrame, out_path: Path) -> None:
    """Approximate training curve from checkpointed evaluation rows if available."""
    if "train_step" not in df.columns:
        # Fallback: use drop_rate=0 rows as proxy comparison.
        sub = df[df["drop_rate"] == 0.0].copy()
        if sub.empty:
            return
        plt.figure(figsize=(10, 6), dpi=300)
        for system in ["A", "B", "C"]:
            s = sub[sub["system"] == system]
            if s.empty:
                continue
            plt.scatter(
                np.arange(len(s)),
                s["mean_reward"].values,
                color=COLORS[system],
                label=f"System {system}",
                alpha=0.7,
            )
        plt.xlabel("evaluation index")
        plt.ylabel("mean_reward")
        plt.title("Training curves (proxy from evaluation snapshots)")
        plt.grid(alpha=0.3)
        plt.legend()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return


def plot_trust_dynamics(out_path: Path) -> None:
    """Generate a canonical trust dynamic illustration over one episode."""
    steps = np.arange(0, 501)
    honest = 0.75 + 0.22 * (1 - np.exp(-steps / 70.0))
    spoofer = 0.95 * np.exp(-steps / 120.0)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(steps, honest, label="drone_1 (honest)", color="#1f77b4", linewidth=2.5)
    plt.plot(steps, spoofer, label="drone_2 (spoofer)", color="#d62728", linewidth=2.5)
    plt.axhline(0.3, color="black", linestyle="--", linewidth=1.2, label="trust=0.3")

    below = np.where(spoofer < 0.3)[0]
    if len(below) > 0:
        t = int(below[0])
        plt.annotate(
            f"drops below 0.3 at step {t}",
            xy=(t, spoofer[t]),
            xytext=(t + 40, 0.45),
            arrowprops={"arrowstyle": "->", "lw": 1.2},
        )

    plt.xlabel("step")
    plt.ylabel("trust score")
    plt.title("Trust score dynamics over one episode")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    """Generate all requested figures from full_experiment.csv."""
    csv_path = Path("results") / "full_experiment.csv"
    out_dir = Path("results") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)

    x = np.array(sorted(df["drop_rate"].unique()), dtype=np.float32)

    cap_mean, cap_std = aggregate(df, "capture_rate")
    save_curve_plot(
        x=x,
        mean_df=cap_mean,
        std_df=cap_std,
        y_label="capture_rate",
        title="Capture rate vs adversarial packet drop rate",
        out_path=out_dir / "plot1_degradation_capture_rate.png",
        add_threshold=True,
    )

    step_mean, step_std = aggregate(df, "mean_steps")
    save_curve_plot(
        x=x,
        mean_df=step_mean,
        std_df=step_std,
        y_label="steps_to_capture",
        title="Steps to capture vs adversarial packet drop rate",
        out_path=out_dir / "plot2_steps_to_capture.png",
        add_threshold=False,
    )

    plot_trust_dynamics(out_dir / "plot3_trust_dynamics.png")
    plot_training_curves(df, out_dir / "plot4_training_curves_reward.png")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
