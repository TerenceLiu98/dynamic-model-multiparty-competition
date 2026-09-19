from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from distributional_phase import chi, eta, omega  # noqa: E402


def test_binary_feedback_degenerates() -> None:
    a = np.linspace(0.0, 1.0, 11)
    assert np.allclose(omega(2, a), 0.5)
    assert np.allclose(eta(2, a), 0.0)


def test_three_party_feedback_is_one_sixth() -> None:
    a = np.linspace(0.0, 1.0, 11)
    assert np.allclose(eta(3, a), 1.0 / 6.0)


def test_eta_positive_for_multiparty_systems() -> None:
    a = np.linspace(0.0, 1.0, 101)
    for K in (3, 4, 5, 8):
        assert np.all(eta(K, a) > 0.0)


def test_continuous_limits_at_zero() -> None:
    for K in (2, 3, 5, 9):
        assert np.isclose(omega(K, 0.0), (K - 1) / K)
        assert np.isclose(chi(K, 0.0), 1.0)
        assert np.isclose(eta(K, 0.0), (K - 2) / (2 * K))

from distributional_phase import solve_diagonal_state  # noqa: E402


def test_diagonal_state_is_even_normalized_and_positive() -> None:
    state = solve_diagonal_state(K=3, r=0.68, eps=0.004, ny=401, nx=100)
    assert abs(np.sum(state.wy * state.f) - 1.0) < 1e-9
    assert np.max(np.abs(state.f - state.f[::-1])) < 1e-9
    assert np.min(state.f) > 0.0
    assert state.residual_l1 < 1e-7
    assert state.turnout > 0.0
    assert state.variance > 0.0

from distributional_phase import (  # noqa: E402
    apply_odd_electoral_operator,
    odd_principal_eigenpair,
)


def test_binary_odd_eigenvalue_is_zero() -> None:
    state = solve_diagonal_state(K=2, r=0.7, eps=0.004, ny=301, nx=80)
    eig = odd_principal_eigenpair(state)
    assert abs(eig.nu) < 1e-12
    assert eig.margin == -1.0


def test_odd_operator_preserves_odd_parity() -> None:
    state = solve_diagonal_state(K=3, r=0.68, eps=0.004, ny=301, nx=80)
    z = state.y * np.sqrt(state.wy * state.f)
    out = apply_odd_electoral_operator(state, z)
    assert np.max(np.abs(out + out[::-1])) < 1e-10

from distributional_phase import (  # noqa: E402
    dense_odd_principal_eigenvalue,
    point_limit_components,
)


def test_matrix_free_eigenvalue_matches_dense_operator() -> None:
    state = solve_diagonal_state(K=3, r=0.68, eps=0.006, ny=101, nx=50)
    sparse = odd_principal_eigenpair(state).nu
    dense = dense_odd_principal_eigenvalue(state)
    assert abs(sparse - dense) < 1e-9


def test_point_eigenvalue_splits_into_feedback_minus_centering() -> None:
    for K, r in ((3, 0.66), (4, 0.79), (5, 0.85)):
        comp = point_limit_components(K, r)
        assert abs(comp.lambda_pr - (comp.feedback - comp.centering)) < 1e-9

from distributional_phase import find_distributional_critical_ratio  # noqa: E402


def test_distributional_root_has_opposite_margin_signs() -> None:
    result = find_distributional_critical_ratio(
        K=3,
        eps=0.004,
        bracket=(0.55, 0.78),
        ny=301,
        nx=80,
        root_tol=2e-4,
    )
    assert result.margin_low > 0.0
    assert result.margin_high < 0.0
    assert abs(result.margin_root) < 2e-3
    assert result.r_low < result.r_critical < result.r_high


def test_full_standard_sector_operator_matches_exact_autodiff() -> None:
    import torch

    from distributional_phase import apply_standard_sector_potential

    torch.set_default_dtype(torch.float64)
    K = 4
    state = solve_diagonal_state(K=K, r=0.78, eps=0.004, ny=81, nx=40)
    g = state.f * (state.y**2 - state.variance)
    g /= np.max(np.abs(g))
    labels = np.array([1.0, -1.0, 0.0, 0.0])

    nodes, weights = np.polynomial.legendre.leggauss(K)
    nodes = torch.tensor((nodes + 1.0) / 2.0)
    weights = torch.tensor(weights / 2.0)
    kernel = torch.tensor(state.kernel)
    wx = torch.tensor(state.wx)
    wy = torch.tensor(state.wy)

    def exact_own_potential(delta: float) -> np.ndarray:
        densities = np.stack([state.f + delta * z * g for z in labels])
        f_tensor = torch.tensor(densities, requires_grad=True)
        attraction = kernel @ (f_tensor * wy).T
        factors = 1.0 - (1.0 - nodes[None, :, None]) * attraction[:, None, :]
        prefix = torch.cumprod(factors, dim=2)
        suffix = torch.flip(
            torch.cumprod(torch.flip(factors, dims=[2]), dim=2), dims=[2]
        )
        one = torch.ones((state.x.size, nodes.numel()))
        probabilities = []
        for party in range(K):
            left = prefix[:, :, party - 1] if party > 0 else one
            right = suffix[:, :, party + 1] if party < K - 1 else one
            integral = torch.sum(left * right * weights[None, :], dim=1)
            probabilities.append(attraction[:, party] * integral)
        support = wx @ torch.stack(probabilities, dim=1)
        seats = support / support.sum()
        gradient = torch.autograd.grad(seats[0], f_tensor)[0][0]
        return gradient.detach().numpy() / state.wy

    step = 1e-5
    numerical = (exact_own_potential(step) - exact_own_potential(-step)) / (2 * step)
    analytic = apply_standard_sector_potential(state, g)
    assert np.max(np.abs(numerical - analytic)) < 1e-9

from distributional_phase import run_mode_validation  # noqa: E402


def test_linearized_mode_grows_below_and_decays_above_threshold() -> None:
    below = run_mode_validation(K=3, eps=0.004, r=0.56, quick=True)
    above = run_mode_validation(K=3, eps=0.004, r=0.66, quick=True)
    assert below.log_amplitude_slope > 0.0
    assert above.log_amplitude_slope < 0.0
    assert below.mass_error < 1e-10
    assert above.mass_error < 1e-10


def test_point_assumption_check_is_centered_and_transverse() -> None:
    from run_distributional_phase import point_assumption_check

    for K in (3, 4, 5):
        check = point_assumption_check(K)
        assert check["centering_curvature"] > 0.0
        assert abs(check["lambda_at_root"]) < 1e-9
        assert check["lambda_derivative"] < -0.1
