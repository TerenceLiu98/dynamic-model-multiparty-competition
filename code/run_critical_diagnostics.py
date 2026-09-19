#!/usr/bin/env python3
"""Diagnose finite-time relaxation at the binary critical point.

Reads the paper-profile trajectory, then repeats seven local experiments with
varying perturbation amplitude, time step and spatial resolution. The full
density solvers are used without symmetry projection. These diagnostics do not
replace the paper trajectory or certify its asymptotic convergence.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from distributional_phase import (
    DistributionalState, odd_principal_eigenpair, solve_diagonal_state,
)
from repro_utils import ROOT, environment, sha256, write_csv, write_json
from theta_dynamics import (
    binary_initial_density, density_diagnostics, mixed_potentials,
    odd_growth_eigenpair, sg_rhs, simulate_densities,
)


SETTINGS = (
    (401, 400, 2.2, .1, .01), (401, 400, 2.2, .1, .02),
    (401, 400, 2.2, .1, .04), (401, 400, 2.2, .1, .08),
    (401, 400, 2.2, .05, .02), (401, 400, 2.2, .025, .02),
    (601, 600, 2.6, .05, .02),
)


def fit_cubic_decay(records: pd.DataFrame, start_time: float = 50.) -> dict[str, float]:
    """Fit P(t)^(-2) = intercept + 2 beta t after the initial transient."""
    sample = records.loc[records['time'] >= start_time, ['time', 'P']]
    values = sample.to_numpy(dtype=float)
    if (len(sample) < 2 or not np.isfinite(values).all()
            or (values[:, 1] <= 0).any() or np.ptp(values[:, 0]) <= 0):
        raise ValueError('Cubic fitting requires distinct finite times and positive finite P values')
    slope, intercept = np.polyfit(values[:, 0], values[:, 1] ** -2, 1)
    return {'beta_timeseries': float(slope / 2),
            'inverse_P2_intercept': float(intercept)}


def diagnose_density(state: DistributionalState, densities: np.ndarray) -> dict:
    """Measure instantaneous drift and distance to the common stationary state."""
    feedback = mixed_potentials(state, densities)
    rhs = np.array([
        sg_rhs(fi, potential, state.eps, state.y, state.wy)
        for fi, potential in zip(densities, feedback.potential, strict=True)
    ])
    a = float((densities[1] - densities[0]) @ (state.wy * state.y) / 2)
    adot = float((rhs[1] - rhs[0]) @ (state.wy * state.y) / 2)
    metrics = density_diagnostics(state, densities, feedback)
    return dict(
        P=metrics['P'], signed_a=a, adot=adot,
        beta_rhs=-adot / a ** 3 if a ** 3 != 0. else None,
        rhs_l1=metrics['rhs_l1'], gibbs_l1=metrics['gibbs_l1'],
        l1_distance_common=float(np.max(np.abs(densities - state.f) @ state.wy)),
        mass_error=metrics['mass_error'],
    )


def run(results_dir: Path, output_dir: Path) -> None:
    """Run the seven local diagnostics using the archived paper-profile inputs."""
    results_dir, output_dir = results_dir.resolve(), output_dir.resolve()
    timeseries_path = results_dir / 'theta_nonlinear_timeseries.csv'
    state_path = results_dir / 'nonlinear' / 'critical.npz'
    for path in (timeseries_path, state_path):
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing {path.name}; first run python code/run_all.py --profile paper'
            )
    all_records = pd.read_csv(timeseries_path)
    archive = all_records.loc[all_records.case == 'critical'].copy()
    if archive.empty:
        raise ValueError('The source results contain no critical trajectory')
    fit = fit_cubic_decay(archive)
    with np.load(state_path, allow_pickle=False) as data:
        theta = float(data['theta'])
        r, eps = float(data['r']), float(data['eps'])
        ny, nx = data['y'].size, data['x'].size
        domain = float(np.max(np.abs(data['y'])))
        densities = data['final_densities'].copy()
    if (ny, nx) != (401, 400) or not np.isclose(domain, 2.2):
        raise ValueError('Critical diagnostics require paper-profile inputs, not quick results')
    if not np.isclose(r, .6) or not np.isclose(eps, .002):
        raise ValueError('The diagnostic experiment uses r=0.6 and epsilon=0.002')

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = dict(
        **environment(), status='running', experiment='binary critical relaxation',
        horizon=100., fit_start_time=50., settings=SETTINGS,
        source_sha256={
            'theta_nonlinear_timeseries.csv': sha256(timeseries_path),
            'nonlinear/critical.npz': sha256(state_path),
        },
        convergence_claim='finite-time diagnostics only',
    )
    write_json(output_dir / 'metadata.json', metadata)
    write_csv(output_dir / 'archived_critical_timeseries.csv', archive)
    common = solve_diagonal_state(2, r, eps, theta=theta, ny=ny, nx=nx, L=domain, tol=1e-13)
    baseline = diagnose_density(common, densities)
    baseline.update(
        theta=theta, time=float(archive.time.max()),
        lambda_odd=odd_growth_eigenpair(common).rate, **fit,
        P_50=float(archive.loc[np.isclose(archive.time, 50.), 'P'].iloc[0]),
    )
    write_json(output_dir / 'archive_diagnostic.json', baseline)
    print('Archived trajectory:', baseline, flush=True)

    state_cache = {}

    def state_for(ny, nx, domain):
        key = (ny, nx, domain)
        if key not in state_cache:
            def margin(weight):
                state = solve_diagonal_state(
                    2, r, eps, theta=weight, ny=ny, nx=nx, L=domain, tol=1e-13,
                )
                return odd_principal_eigenpair(state, tol=1e-12).margin
            root = brentq(margin, theta - 1e-5, theta + 1e-5, xtol=5e-14, rtol=1e-14)
            state_cache[key] = solve_diagonal_state(
                2, r, eps, theta=root, ny=ny, nx=nx, L=domain, tol=1e-13,
            )
        return state_cache[key]

    rows = []
    for ny, nx, domain, dt, amplitude in SETTINGS:
        start = time.perf_counter()
        state = state_for(ny, nx, domain)
        trajectory = simulate_densities(
            state, dt=dt, horizon=100., critical=True,
            initial=binary_initial_density(state, amplitude),
            record_every=max(1, int(round(1 / dt))),
        )
        end = diagnose_density(state, trajectory.final_densities)
        records = pd.DataFrame([
            {key: row[key] for key in ('time', 'P', 'rhs_l1', 'gibbs_l1')}
            for row in trajectory.records
        ])
        row = dict(
            ny=ny, nx=nx, L=domain, dt=dt, initial_amplitude=amplitude, theta=state.theta,
            lambda_odd=odd_growth_eigenpair(state).rate,
            beta_timeseries=fit_cubic_decay(records)['beta_timeseries'],
            common_state_residual=state.residual_l1,
            elapsed_seconds=time.perf_counter() - start, **end,
        )
        rows.append(row)
        write_csv(output_dir / f'probe_n{ny}_dt{dt:g}_amp{amplitude:g}.csv', records)
        write_csv(output_dir / 'critical_probe_results.csv', rows)
        print('Probe:', row, flush=True)
    metadata.update(status='success', elapsed_seconds=time.perf_counter() - started)
    metadata['output_sha256'] = {
        path.name: sha256(path) for path in sorted(output_dir.iterdir())
        if path.suffix in ('.csv', '.json') and path.name != 'metadata.json'
    }
    write_json(output_dir / 'metadata.json', metadata)
    print(f'Critical diagnostics written to {output_dir}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results',
                        help='directory containing paper-profile CSV and NPZ results')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'critical',
                        help='directory for additional critical-relaxation diagnostics')
    args = parser.parse_args()
    run(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
