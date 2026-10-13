"""
Attitude-response plotting.

Provides functions to plot desired vs. actual attitude over time
and save the result to a file.
"""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt


def plot_attitude(time_s, desired_roll_deg, desired_pitch_deg, desired_yaw_deg,
                  actual_roll_deg, actual_pitch_deg, actual_yaw_deg,
                  save_path=None, show=False):
    """
    Plot desired vs. actual roll, pitch, and yaw over time.

    Parameters
    ----------
    time_s : array-like
        Time in seconds.
    desired_roll_deg, desired_pitch_deg, desired_yaw_deg : array-like
        Desired attitude in degrees.
    actual_roll_deg, actual_pitch_deg, actual_yaw_deg : array-like
        Actual attitude in degrees.
    save_path : str, optional
        File path to save the plot (e.g. 'results/attitude.png').
    show : bool, optional
        If True, display the plot interactively (requires GUI backend).
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ["Roll", "Pitch", "Yaw"]
    desired = [desired_roll_deg, desired_pitch_deg, desired_yaw_deg]
    actual = [actual_roll_deg, actual_pitch_deg, actual_yaw_deg]
    colors_desired = ["#1f77b4", "#2ca02c", "#d62728"]
    colors_actual = ["#aec7e8", "#98df8a", "#ff9896"]

    for ax, label, des, act, cd, ca in zip(
            axes, labels, desired, actual, colors_desired, colors_actual):
        ax.plot(time_s, des, "--", color=cd, label=f"Desired {label}", linewidth=1.2)
        ax.plot(time_s, act, "-", color=ca, label=f"Actual {label}", linewidth=1.0)
        ax.set_ylabel(f"{label} (deg)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Attitude Response — Desired vs. Actual", fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    plt.close(fig)
