#!/usr/bin/env python3
"""Compare Nicomedia's compatibility API with TDpy's canonical profile."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import nicomedia
import tdpy


def evaluate_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the same double-King profile through both public APIs."""

    angular_deviation = np.linspace(0.0, 5.0, 501)
    core_fraction = 0.35
    core_width = 0.8
    core_index = 2.2
    tail_width = 1.7
    tail_index = 3.4

    nicomedia_profile = nicomedia.retr_doubking(
        angular_deviation,
        core_fraction,
        core_width,
        core_index,
        tail_width,
        tail_index,
    )
    tdpy_profile = tdpy.retr_doubking(
        angular_deviation,
        core_fraction,
        core_width,
        core_index,
        tail_width,
        tail_index,
    )
    return angular_deviation, nicomedia_profile, tdpy_profile


def run_example(output_path: Path) -> float:
    """Plot the shared profile and numerical compatibility residual."""

    angular_deviation, nicomedia_profile, tdpy_profile = evaluate_profiles()
    absolute_difference = np.abs(nicomedia_profile - tdpy_profile)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(
        angular_deviation,
        tdpy_profile,
        color="#16697A",
        linewidth=2.4,
        label="TDpy canonical kernel",
    )
    axes[0].plot(
        angular_deviation,
        nicomedia_profile,
        color="#A51417",
        linewidth=1.5,
        linestyle="--",
        label="Nicomedia compatibility API",
    )
    axes[0].set_ylabel("Double-King profile")
    axes[0].set_title("Nicomedia preserves the canonical TDpy profile")
    axes[0].legend(frameon=True, fancybox=True, framealpha=1.0)

    axes[1].plot(
        angular_deviation,
        absolute_difference,
        color="black",
        linewidth=1.8,
        label="Absolute difference",
    )
    axes[1].set_xlabel("Normalized angular deviation")
    axes[1].set_ylabel("Difference")
    axes[1].legend(frameon=True, fancybox=True, framealpha=1.0)
    for axis in axes:
        axis.grid(False)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return float(absolute_difference.max())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Nicomedia and TDpy double-King profile compatibility."
    )
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(
        f"double_king_compatibility.{arguments.typefileplot}"
    )
    maximum_difference = run_example(output_path)
    print(f"Maximum absolute difference: {maximum_difference:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())