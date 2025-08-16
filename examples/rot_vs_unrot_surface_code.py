# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import scipy.stats
import sinter
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

from graphqec import RotatedSurfaceCode, UnrotatedSurfaceCode, ThresholdLAB


def main(
    configs: dict[str, int | float],
    errors: np.ndarray,
    max_shots: int,
    max_errors: int,
    logic_check: str,
) -> None:
    r"""
    Compare the performance of Rotated and Unrotated Surface Codes.

    Args:
        configs: A dictionary containing the configurations for the surface codes.
        errors: An array of error rates to evaluate.
        max_shots: The maximum number of shots to simulate.
        max_errors: The maximum number of errors to consider.
        logic_check: The logic check to perform.
    """

    # Get stats for the Rotated Surface Code with Z logic
    rot_th_z = ThresholdLAB(
        configurations=configs,
        code=RotatedSurfaceCode,
        error_rates=errors,
        decoder="pymatching",
    )
    rot_th_z.collect_stats(
        max_shots=max_shots, max_errors=max_errors, logic_check=logic_check
    )

    # Get stats for the Unrotated Surface code with Z logic
    unrot_th_z = ThresholdLAB(
        configurations=configs,
        code=UnrotatedSurfaceCode,
        error_rates=errors,
        decoder="pymatching",
    )
    unrot_th_z.collect_stats(
        max_shots=max_shots, max_errors=max_errors, logic_check=logic_check
    )

    # Find the range of values for the fit
    max_n = 0
    min_n = float("inf")
    for sample in unrot_th_z.samples + rot_th_z.samples:
        max_n = max(max_n, np.sqrt(sample.json_metadata["n"]))
        min_n = min(min_n, np.sqrt(sample.json_metadata["n"]))

    # Fit each data and plot them on the same plot alongside the fit
    data = []
    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap(i) for i in range(errors.size)]
    fig, ax = plt.subplots(1, 1)
    for index, error in enumerate(errors):
        if error == 0:
            continue

        fit_unrot_z, x_unrot_z, y_unrot_z = fit(unrot_th_z.samples, error)
        ax.scatter(
            x_unrot_z, y_unrot_z, marker="^", color=colors[index], s=50, zorder=2
        )
        ax.plot(
            [0, max_n],
            [
                np.exp(fit_unrot_z.intercept),
                np.exp(fit_unrot_z.intercept + fit_unrot_z.slope * max_n),
            ],
            linestyle="--",
            label=error,
            color=colors[index],
            dashes=(2, 2),
            alpha=0.8,
            linewidth=1.0,
            zorder=2,
        )
        data.append(
            {
                "n": [int(n**2) for n in x_unrot_z],
                "logical_error": [float(y) for y in y_unrot_z],
                "type": "Unrotated",
                "error_rate": float(error),
                "fit": {
                    "x": [0, max_n],
                    "y": [
                        np.exp(fit_unrot_z.intercept),
                        np.exp(fit_unrot_z.intercept + fit_unrot_z.slope * max_n),
                    ],
                },
            }
        )

        fit_rot_z, x_rot_z, y_rot_z = fit(rot_th_z.samples, error)
        ax.scatter(x_rot_z, y_rot_z, marker="o", color=colors[index], s=50, zorder=2)
        ax.plot(
            [0, max_n],
            [
                np.exp(fit_rot_z.intercept),
                np.exp(fit_rot_z.intercept + fit_rot_z.slope * max_n),
            ],
            linestyle="--",
            label=error,
            color=colors[index],
            dashes=(2, 2),
            alpha=0.8,
            linewidth=1.0,
            zorder=2,
        )
        data.append(
            {
                "n": [int(n**2) for n in x_rot_z],
                "logical_error": [float(y) for y in y_rot_z],
                "type": "Rotated",
                "error_rate": float(error),
                "fit": {
                    "x": [0, max_n],
                    "y": [
                        np.exp(fit_rot_z.intercept),
                        np.exp(fit_rot_z.intercept + fit_rot_z.slope * max_n),
                    ],
                },
            }
        )

    # Build legend
    line_handles = [
        Line2D([0], [0], color=colors[index], lw=2, label=f"{error:.4f}")
        for index, error in enumerate(errors)
    ]

    legend_lines = ax.legend(
        handles=line_handles,
        title="Physical error rate",
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        borderaxespad=0.0,
        handlelength=2,
        frameon=True,
    )
    ax.add_artist(legend_lines)

    marker_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=m,
            linestyle="None",
            markersize=8,
            label=label,
        )
        for m, label in zip(["^", "o"], ["Unrotated", "Rotated"])
    ]

    height_per_entry = 0.055
    extra_for_title = 0.1
    n_lines = len(marker_handles)
    y_marker_legend = 1.0 - (n_lines * height_per_entry + extra_for_title)
    lengend_markers = ax.legend(
        handles=marker_handles,
        title="Code type",
        loc="upper left",
        bbox_to_anchor=(1.05, y_marker_legend),
        borderaxespad=0.0,
        frameon=True,
    )

    ax.semilogy()
    ax.set_xlim(min_n - 2, max_n + 2)
    ax.set_title("")
    ax.set_xlabel(r"$\sqrt{\mathrm{Number\ of\ Physical\ Qubits}}$")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.grid(which="major", zorder=0)
    ax.grid(which="minor", zorder=0)
    plt.tight_layout()
    fig.savefig(
        "rot_vs_unrot_surface_code.png",
        bbox_inches="tight",
        bbox_extra_artists=[legend_lines, lengend_markers],
    )

    with open("rot_vs_unrot_surface_code.json", "w") as f:
        json.dump(data, f, indent=4)


def fit(
    collected_stats: list[sinter.TaskStats], error: float
) -> tuple[scipy.stats.linregress, list[float], list[float]]:
    r"""
    Compute a line fit for the given collected statistics.
    """
    xs = []
    ys = []
    log_ys = []
    for stats in collected_stats:
        if stats.errors == 0 or stats.json_metadata["error"] != error:
            continue
        n = np.sqrt(stats.json_metadata["n"])
        per_shot = stats.errors / stats.shots
        per_round = sinter.shot_error_rate_to_piece_error_rate(
            per_shot, pieces=stats.json_metadata["r"]
        )
        xs.append(n)
        ys.append(per_round)
        log_ys.append(np.log(per_round))
    fit = scipy.stats.linregress(xs, log_ys)
    return fit, xs, ys


if __name__ == "__main__":

    max_shots = 1_000_000
    max_errors = 1000
    logic_check = "Z"
    configs = [{"distance": d} for d in [9, 11, 13, 17, 20, 23]]
    errors = np.linspace(0.001, 0.01, 10)
    main(
        configs=configs,
        errors=errors,
        max_shots=max_shots,
        max_errors=max_errors,
        logic_check=logic_check,
    )
