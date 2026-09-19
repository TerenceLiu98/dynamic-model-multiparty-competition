"""Reproduce the finite-width multiparty Wasserstein phase experiments."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq

from distributional_phase import (
    find_distributional_critical_ratio,
    odd_principal_eigenpair,
    point_limit_components,
    point_pr_eigenvalue,
    run_mode_validation,
    solve_diagonal_state,
)
from plot_style import apply_scienceplots_style

ROOT = Path(__file__).resolve().parents[1]
PHASE_MARKER_SIZE = 3.0
SPECTRAL_MARKER_SIZE = 2.25


def point_critical_ratio(K: int) -> float:
    return float(brentq(lambda r: point_pr_eigenvalue(K, r), 0.25, 1.25, xtol=1e-13))


def brackets() -> dict[int, tuple[float, float]]:
    return {3: (0.46, 0.72), 4: (0.56, 0.84), 5: (0.60, 0.90)}


def point_assumption_check(K: int, derivative_step: float = 1e-5) -> dict[str, float | int]:
    """Evaluate the centered-well and transversality quantities at the point root."""
    root = point_critical_ratio(K)
    components = point_limit_components(K, root)
    derivative = (
        point_pr_eigenvalue(K, root + derivative_step)
        - point_pr_eigenvalue(K, root - derivative_step)
    ) / (2.0 * derivative_step)
    return {
        "K": K,
        "point_critical": root,
        "centering_curvature": components.centering,
        "feedback": components.feedback,
        "lambda_at_root": components.lambda_pr,
        "lambda_derivative": float(derivative),
        "derivative_step": derivative_step,
    }


def run_experiments(output_root: Path, quick: bool = False, *, figures: bool = True) -> None:
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        party_counts = (3,)
        entropies = (0.004,)
        ny, nx, L = 301, 80, 2.0
        scan_count = 7
        root_tol = 2e-4
    else:
        party_counts = (3, 4, 5)
        entropies = (0.0005, 0.001, 0.002, 0.004, 0.008)
        ny, nx, L = 801, 160, 2.2
        scan_count = 15
        root_tol = 2e-5

    assumption_rows: list[dict[str, float | int]] = []
    critical_rows: list[dict[str, float | int]] = []
    scan_rows: list[dict[str, float | int]] = []
    root_lookup: dict[tuple[int, float], float] = {}

    for K in party_counts:
        point_check = point_assumption_check(K)
        assumption_rows.append(point_check)
        point_root = float(point_check["point_critical"])
        for eps in entropies:
            result = find_distributional_critical_ratio(
                K,
                eps,
                brackets()[K],
                L=L,
                ny=ny,
                nx=nx,
                root_tol=root_tol,
            )
            root_lookup[(K, eps)] = result.r_critical
            row = asdict(result)
            row.update(
                point_critical=point_root,
                finite_width_shift=point_root - result.r_critical,
                ny=ny,
                nx=nx,
                L=L,
            )
            critical_rows.append(row)

            scan_half_width = 0.075 if K == 3 else 0.085
            scan_r = np.linspace(
                max(brackets()[K][0], result.r_critical - scan_half_width),
                min(brackets()[K][1], result.r_critical + scan_half_width),
                scan_count,
            )
            previous_density = None
            for r in scan_r:
                state = solve_diagonal_state(
                    K,
                    float(r),
                    eps,
                    L=L,
                    ny=ny,
                    nx=nx,
                    initial_density=previous_density,
                )
                previous_density = state.f
                eig = odd_principal_eigenpair(state)
                scan_rows.append(
                    {
                        "K": K,
                        "eps": eps,
                        "r": float(r),
                        "margin": eig.margin,
                        "nu": eig.nu,
                        "variance": state.variance,
                        "centering_curvature": state.centering_curvature,
                        "translation_overlap": eig.translation_overlap,
                        "stationary_residual_l1": state.residual_l1,
                    }
                )

    pd.DataFrame(assumption_rows).sort_values("K").to_csv(
        results_dir / "distributional_point_assumption_checks.csv", index=False
    )
    critical = pd.DataFrame(critical_rows).sort_values(["K", "eps"])
    scan = pd.DataFrame(scan_rows).sort_values(["K", "eps", "r"])
    critical.to_csv(results_dir / "distributional_critical_curves.csv", index=False)
    scan.to_csv(results_dir / "distributional_spectral_scan.csv", index=False)

    # Dynamic validation and representative mode profiles.
    mode_rows: list[dict[str, float | int | str]] = []
    profile_rows: list[dict[str, float | int | str]] = []
    validation_cases: list[tuple[int, float]]
    if quick:
        validation_cases = [(3, 0.004)]
    else:
        validation_cases = [(3, 0.004), (5, 0.004)]

    for K, eps in validation_cases:
        root = root_lookup[(K, eps)]
        offset = 0.04
        for regime, r in (("unstable", root - offset), ("stable", root + offset)):
            validation = run_mode_validation(K, eps, r, quick=quick)
            for t, amplitude in zip(validation.times, validation.amplitudes):
                mode_rows.append(
                    {
                        "K": K,
                        "eps": eps,
                        "r": r,
                        "regime": regime,
                        "time": float(t),
                        "amplitude": float(amplitude),
                        "leading_growth_rate": validation.leading_growth_rate,
                        "fitted_log_slope": validation.log_amplitude_slope,
                        "mass_error": validation.mass_error,
                    }
                )

        profile_state = solve_diagonal_state(
            K,
            root,
            eps,
            L=1.5,
            ny=301 if not quick else 151,
            nx=140 if not quick else 80,
        )
        profile_eig = odd_principal_eigenpair(profile_state)
        mode = profile_eig.density_vector
        mode_scale = float(np.max(np.abs(mode)))
        normalized_mode = mode / mode_scale if mode_scale > 0.0 else mode
        for y, density, g in zip(profile_state.y, profile_state.f, normalized_mode):
            profile_rows.append(
                {
                    "K": K,
                    "eps": eps,
                    "r": root,
                    "y": float(y),
                    "stationary_density": float(density),
                    "normalized_odd_mode": float(g),
                    "translation_overlap": profile_eig.translation_overlap,
                }
            )

    pd.DataFrame(mode_rows).to_csv(
        results_dir / "distributional_mode_validation.csv", index=False
    )
    pd.DataFrame(profile_rows).to_csv(
        results_dir / "distributional_mode_profiles.csv", index=False
    )

    resolution_rows: list[dict[str, float | int]] = []
    if quick:
        resolution_cases = [(3, 0.004)]
        resolution_grids = [(301, 80, 2.0)]
    else:
        resolution_cases = [(3, 0.001), (3, 0.004), (5, 0.001), (5, 0.004)]
        resolution_grids = [
            (401, 100, 1.8),
            (601, 140, 2.2),
            (801, 160, 2.2),
            (1001, 200, 2.4),
        ]
    for K, eps in resolution_cases:
        for check_ny, check_nx, check_L in resolution_grids:
            check = find_distributional_critical_ratio(
                K,
                eps,
                brackets()[K],
                L=check_L,
                ny=check_ny,
                nx=check_nx,
                root_tol=root_tol,
            )
            resolution_rows.append(
                {
                    "K": K,
                    "eps": eps,
                    "ny": check_ny,
                    "nx": check_nx,
                    "L": check_L,
                    "r_critical": check.r_critical,
                    "margin_root": check.margin_root,
                    "variance_root": check.variance_root,
                    "overlap_root": check.overlap_root,
                }
            )
    pd.DataFrame(resolution_rows).to_csv(
        results_dir / "distributional_resolution_check.csv", index=False
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quick": quick,
        "party_counts": list(party_counts),
        "entropies": list(entropies),
        "grid": {"ny": ny, "nx": nx, "L": L},
        "root_tolerance": root_tol,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    (results_dir / "distributional_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if figures:
        make_figures(output_root)


def make_figures(output_root: Path) -> None:
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    critical = pd.read_csv(results_dir / "distributional_critical_curves.csv")
    scan = pd.read_csv(results_dir / "distributional_spectral_scan.csv")
    modes = pd.read_csv(results_dir / "distributional_mode_validation.csv")
    profiles = pd.read_csv(results_dir / "distributional_mode_profiles.csv")

    apply_scienceplots_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    for K, group in critical.groupby("K"):
        group = group.sort_values("eps")
        axes[0].plot(
            group["eps"],
            group["r_critical"],
            marker="o",
            markersize=PHASE_MARKER_SIZE,
            label=fr"$K={K}$",
        )
        axes[0].axhline(group["point_critical"].iloc[0], linestyle="--", linewidth=0.8)
        axes[1].plot(
            group["eps"],
            group["finite_width_shift"],
            marker="o",
            markersize=PHASE_MARKER_SIZE,
            label=fr"$K={K}$",
        )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_xticks([5e-4, 2e-3, 8e-3])
        ax.set_xticklabels(
            [r"$5\!\times\!10^{-4}$", r"$2\!\times\!10^{-3}$", r"$8\!\times\!10^{-3}$"]
        )
        ax.tick_params(axis="x", which="minor", labelbottom=False)
    axes[0].set_xlabel(r"ideological diffusion $\varepsilon$")
    axes[0].set_ylabel(r"critical tolerance $r_c^{W}$")
    axes[0].set_title("Finite-width phase boundary")
    axes[0].legend(frameon=False)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"ideological diffusion $\varepsilon$")
    axes[1].set_ylabel(r"$r_c(K)-r_c^{W}(K,\varepsilon)$")
    axes[1].set_title("Convergence to the point limit")
    fig.tight_layout()
    fig.savefig(figures_dir / "distributional_phase_diagram.pdf", bbox_inches="tight")
    plt.close(fig)

    selected_K = int(scan["K"].min())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    for eps, group in scan[scan["K"] == selected_K].groupby("eps"):
        axes[0].plot(
            group["r"],
            group["margin"],
            marker="o",
            markersize=SPECTRAL_MARKER_SIZE,
            label=fr"$\varepsilon={eps:g}$",
        )
    axes[0].axhline(0.0, linewidth=0.8, linestyle="--")
    axes[0].set_xlabel(r"tolerance ratio $r$")
    axes[0].set_ylabel(r"spectral margin $\nu/\varepsilon-1$")
    axes[0].set_title(fr"Eigenvalue crossing ($K={selected_K}$)")
    axes[0].legend(frameon=False, fontsize=6)

    profile_case = profiles[(profiles["K"] == selected_K)]
    profile_eps = float(profile_case["eps"].max())
    profile_case = profile_case[np.isclose(profile_case["eps"], profile_eps)]
    axes[1].plot(profile_case["y"], profile_case["stationary_density"], label=r"$f^*$")
    axes[1].plot(profile_case["y"], profile_case["normalized_odd_mode"], linestyle="--", label="odd mode")
    axes[1].set_xlabel("ideology")
    axes[1].set_ylabel("density / normalised mode")
    axes[1].set_title("Distributional separation mode")
    axes[1].legend(frameon=False)

    mode_case = modes[(modes["K"] == selected_K)]
    for regime, group in mode_case.groupby("regime"):
        label = f"{regime}: slope={group['fitted_log_slope'].iloc[0]:.3f}"
        axes[2].semilogy(group["time"], group["amplitude"], label=label)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("odd-mode amplitude")
    axes[2].set_title("Mean-field mode validation")
    axes[2].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(figures_dir / "distributional_spectral_modes.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true", help="compute data for the new unified figure driver")
    parser.add_argument("--quick", action="store_true", help="run a small smoke experiment")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: project root; quick: runs/quick",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="regenerate figures from existing CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or (ROOT / "runs" / "quick" if args.quick else ROOT)
    if args.figures_only:
        make_figures(args.output_dir)
    else:
        run_experiments(args.output_dir, quick=args.quick, figures=not args.no_figures)


if __name__ == "__main__":
    main()
