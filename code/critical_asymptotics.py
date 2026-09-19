#!/usr/bin/env python3
"""Stable evaluation of the PR critical curve and its large-K asymptotics.

Uses the positive integral/moment representation rather than alternating finite
sums.  It also checks integer-K monotonicity over a user-configurable range.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import roots_hermitenorm

from plot_style import apply_scienceplots_style

apply_scienceplots_style()

SQRT_PI = math.sqrt(math.pi)
EULER_GAMMA = 0.5772156649015328606


def hermite_nodes(nq: int = 320):
    return roots_hermitenorm(nq)


def soft_cover(z: np.ndarray, n: float) -> np.ndarray:
    """c_n(z)=1-(1-exp(-z^2/2))^n, evaluated stably for real n>0."""
    s = np.exp(-0.5 * z * z)
    return -np.expm1(n * np.log1p(-s))


def moment_M(K: float, t: float, xh: np.ndarray, wh: np.ndarray) -> float:
    """M_{K-1}(t) under density proportional exp(-t z^2/2)c_{K-1}(z)."""
    n = K - 1.0
    z = xh / math.sqrt(t)
    c = soft_cover(z, n)
    den = float(np.dot(wh, c))
    num = float(np.dot(wh, z * z * c))
    return num / den


def G(K: float, t: float, xh: np.ndarray, wh: np.ndarray) -> float:
    return (K + t) * moment_M(K, t, xh, wh) - K


def critical_t(K: float, xh: np.ndarray, wh: np.ndarray) -> float:
    if K <= 2:
        raise ValueError("The PR crossing exists only for K>2.")
    return brentq(lambda t: G(K, t, xh, wh), 1e-4, 1.05, xtol=2e-13, rtol=2e-13, maxiter=100)


def t_asymptotic(K: float, second_order: bool = True) -> float:
    L = math.log(K)
    gap = (2.0 / SQRT_PI) * math.sqrt(L) / K - 1.0 / K
    if second_order:
        gap += (EULER_GAMMA - 1.0) / (SQRT_PI * K * math.sqrt(L))
    return 1.0 - gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-monotone", type=int, default=5000)
    ap.add_argument("--nq", type=int, default=320)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--figdir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    xh, wh = hermite_nodes(args.nq)

    # Exact roots for every integer K in the monotonicity check range.
    rows = []
    prev = None
    violations = []
    for K in range(3, args.max_monotone + 1):
        t = critical_t(float(K), xh, wh)
        r = math.sqrt(t)
        if prev is not None and not (r > prev):
            violations.append((K, prev, r))
        prev = r
        L = math.log(K)
        rows.append({
            "K": K,
            "r_c": r,
            "t_c": t,
            "gap_1_minus_t": 1.0 - t,
            "scaled_leading": K * (1.0 - t) / math.sqrt(L),
            "scaled_corrected": (K * (1.0 - t) + 1.0) / math.sqrt(L),
            "t_asym_2term": t_asymptotic(K, second_order=False),
            "t_asym_3term": t_asymptotic(K, second_order=True),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "critical_thresholds_extended.csv", index=False)

    # Sparse very-large-K points.
    bigKs = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
    big = []
    for K in bigKs:
        t = critical_t(float(K), xh, wh)
        L = math.log(K)
        big.append({
            "K": K, "r_c": math.sqrt(t), "t_c": t,
            "scaled_leading": K * (1.0 - t) / math.sqrt(L),
            "scaled_corrected": (K * (1.0 - t) + 1.0) / math.sqrt(L),
            "t_asym_3term": t_asymptotic(K, True),
        })
    pd.DataFrame(big).to_csv(outdir / "critical_thresholds_largeK.csv", index=False)

    with open(outdir / "critical_monotonicity_check.txt", "w") as f:
        f.write(f"Gauss-Hermite nodes: {args.nq}\n")
        f.write(f"Checked strict r_c(K+1)>r_c(K) for every integer K=3,...,{args.max_monotone}.\n")
        f.write(f"Violations: {len(violations)}\n")
        for item in violations[:20]: f.write(repr(item)+"\n")

    # Figure: critical curve and asymptotic scaling.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sub = df[df["K"] <= min(args.max_monotone, 400)]
    ax.plot(sub["K"], sub["r_c"], lw=1.8, label=r"exact $r_c(K)$")
    ax.plot(sub["K"], np.sqrt(sub["t_asym_3term"]), ls="--", lw=1.4,
            label="large-$K$ expansion")
    ax.set_xlabel("number of parties $K$")
    ax.set_ylabel(r"critical tolerance ratio $r_c$")
    ax.set_ylim(0.62, 1.005)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "critical_largeK.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # use log-spaced subset to keep plot legible
    pick = np.unique(np.clip(np.round(np.geomspace(3, args.max_monotone, 180)).astype(int), 3, args.max_monotone))
    s = df.set_index("K").loc[pick]
    ax.semilogx(s.index, s["scaled_corrected"], lw=1.8,
                label=r"$[K(1-r_c^2)+1]/\sqrt{\log K}$")
    ax.axhline(2.0/SQRT_PI, ls="--", lw=1.3,
               label=r"$2/\sqrt{\pi}$")
    ax.set_xlabel("$K$")
    ax.set_ylabel("rescaled critical gap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "critical_scaling_collapse.pdf")
    plt.close(fig)

    print(f"wrote {len(df)} exact roots; monotonicity violations={len(violations)}")

if __name__ == "__main__":
    main()
