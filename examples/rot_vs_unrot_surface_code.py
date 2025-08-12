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

import numpy as np
import scipy.stats
import sinter
import matplotlib.pyplot as plt
from graphqec import RotatedSurfaceCode, UnrotatedSurfaceCode, ThresholdLAB


def main(configs: dict[str, int | float], errors: np.ndarray) -> None:
    r"""
    Compare the performance of Rotated and Unrotated Surface Codes.
    """

    # Rotated Surface Code with Z logic
    rot_th_z = ThresholdLAB(
        configurations=configs,
        code=RotatedSurfaceCode,
        error_rates=errors,
        decoder="pymatching",
    )

    rot_th_z.collect_stats(max_shots=10**4, max_errors=1000, logic_check="Z")

    # Unrotated Surface code with Z logic
    unrot_th_z = ThresholdLAB(
        configurations=configs,
        code=UnrotatedSurfaceCode,
        error_rates=errors,
        decoder="pymatching",
    )

    unrot_th_z.collect_stats(max_shots=10**4, max_errors=1000, logic_check="Z")

    max_n = 0
    for sample in unrot_th_z.samples + rot_th_z.samples:
        max_n = max(max_n, sample.json_metadata["n"])

    fig, ax = plt.subplots(1, 1)
    for error in errors:
        if error == 0:
            continue

        fit_unrot_z, x_unrot_z, y_unrot_z = fit(unrot_th_z.samples, error)
        ax.scatter(x_unrot_z, y_unrot_z, marker="^")
        ax.plot(
            [0, max_n],
            [
                np.exp(fit_unrot_z.intercept),
                np.exp(fit_unrot_z.intercept + fit_unrot_z.slope * max_n),
            ],
            linestyle="--",
            label=error,
        )

        fit_rot_z, x_rot_z, y_rot_z = fit(rot_th_z.samples, error)
        ax.scatter(x_rot_z, y_rot_z, marker="o")
        ax.plot(
            [0, max_n],
            [
                np.exp(fit_rot_z.intercept),
                np.exp(fit_rot_z.intercept + fit_rot_z.slope * max_n),
            ],
            linestyle="--",
            label=error,
        )

    ax.semilogy()
    ax.set_title("")
    ax.set_xlabel("Number of Physical Qubits")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.grid(which="major")
    ax.grid(which="minor")
    # ax.legend()
    fig.set_dpi(120)
    fig.savefig("rot_vs_unrot_surface_code.png")


def fit(collected_stats: list[sinter.TaskStats], error):
    r"""
    Compute a line fit for the given collected statistics.
    """
    xs = []
    ys = []
    log_ys = []
    for stats in collected_stats:
        if not stats.errors or stats.json_metadata["error"] != error:
            continue
        n = stats.json_metadata["n"]
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

    configs = [{"distance": d} for d in [9, 11, 13, 15]]
    errors = np.linspace(0.001, 0.005, 3)
    main(configs, errors)
