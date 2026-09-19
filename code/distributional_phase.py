"""Deterministic finite-width phase analysis for the multiparty PR model.

This module implements the coefficient identities that appear when the exact
consideration-set rule is linearized around a diagonal distributional state.
The stationary and spectral solvers are added incrementally and are kept
independent of the stochastic particle approximation in :mod:`model`.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Numeric = Union[float, FloatArray]


def _as_float_array(a: ArrayLike) -> tuple[FloatArray, bool]:
    arr = np.asarray(a, dtype=float)
    return arr, arr.ndim == 0


def _restore_scalar(value: FloatArray, scalar: bool) -> Numeric:
    if scalar:
        return float(np.asarray(value))
    return value


from functools import lru_cache
from scipy.special import roots_hermitenorm


@lru_cache(maxsize=64)
def _choice_quadrature(K: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = np.polynomial.legendre.leggauss((K + 1) // 2)
    return (nodes + 1.0) / 2.0, weights / 2.0


def _coefficient_input(K: int, a: ArrayLike) -> tuple[FloatArray, bool]:
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K must be an integer at least 2")
    arr, scalar = _as_float_array(a)
    if np.any(~np.isfinite(arr)) or np.any((arr < 0) | (arr > 1)):
        raise ValueError("acceptance must lie in [0, 1]")
    return arr, scalar


def omega(K: int, a: ArrayLike) -> Numeric:
    """Eq. (40), evaluated without cancellation at small acceptance."""
    arr, scalar = _coefficient_input(K, a)
    out = np.full_like(arr, (K - 1.0) / K)
    mask = arr > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[mask] = -np.expm1((K - 1) * np.log1p(-arr[mask])) / (K * arr[mask])
    return _restore_scalar(out, scalar)


def chi(K: int, a: ArrayLike) -> Numeric:
    """Eq. (47), own-minus-cross vote derivative."""
    value = np.asarray(omega(K, a)) * K / (K - 1)
    return _restore_scalar(value, np.asarray(a).ndim == 0)


def u_coefficient(K: int, a: ArrayLike) -> Numeric:
    """Eq. (41): diagonal raw-support derivative, exact polynomial quadrature."""
    arr, scalar = _coefficient_input(K, a)
    nodes, weights = _choice_quadrature(K)
    out = np.zeros_like(arr)
    for z, w in zip(nodes, weights):
        out += w * (1 - z * arr) ** (K - 1)
    return _restore_scalar(out, scalar)


def h_coefficient(K: int, a: ArrayLike) -> Numeric:
    """Eq. (41): raw-support divergence feedback; h_2 is exactly 1/2."""
    arr, scalar = _coefficient_input(K, a)
    nodes, weights = _choice_quadrature(K)
    out = np.zeros_like(arr)
    for z, w in zip(nodes, weights):
        out += w * z * (1 - z * arr) ** (K - 2)
    return _restore_scalar(out, scalar)


def eta(K: int, a: ArrayLike) -> Numeric:
    """Eq. (42), using its positive integral rather than subtracting near-equals.

    eta_K(A) = (K-2)/K int_0^1 z(1-zA)^(K-3) dz for K>=3.
    This is algebraically identical to Eq. (42), including A=0 and A=1.
    """
    arr, scalar = _coefficient_input(K, a)
    out = np.zeros_like(arr)
    if K > 2:
        nodes, weights = _choice_quadrature(K)
        for z, w in zip(nodes, weights):
            out += ((K - 2) / K) * w * z * (1 - z * arr) ** (K - 3)
    return _restore_scalar(out, scalar)


def feedback_coefficient(K: int, a: ArrayLike, turnout: float, theta: float = 1.) -> Numeric:
    """Eq. (18); only the binary seat-share channel vanishes."""
    if not np.isfinite(theta) or not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0, 1]")
    if not np.isfinite(turnout) or turnout <= 0:
        raise ValueError("turnout must be positive")
    return theta * eta(K, a) / turnout + (1-theta) * h_coefficient(K, a)


@lru_cache(maxsize=32)
def _gaussian_quadrature(nx: int) -> tuple[FloatArray, FloatArray]:
    # numpy.hermgauss overflows at the Appendix-G orders (400 and 600).
    x, w = roots_hermitenorm(nx)
    return x, w / np.sqrt(2 * np.pi)

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DistributionalState:
    """Numerical diagonal stationary state for a homogeneous Gaussian electorate."""

    K: int
    r: float
    eps: float
    x: FloatArray
    wx: FloatArray
    y: FloatArray
    wy: FloatArray
    kernel: FloatArray
    f: FloatArray
    attraction: FloatArray
    turnout: float
    potential: FloatArray
    residual_l1: float
    iterations: int
    variance: float
    centering_curvature: float
    theta: float = 1.0


def _trapezoid_weights(grid: FloatArray) -> FloatArray:
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("grid must be one-dimensional with at least three points")
    spacing = np.diff(grid)
    if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-14):
        raise ValueError("only uniform grids are supported")
    weights = np.full(grid.size, spacing[0], dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def _normalize_density(f: FloatArray, weights: FloatArray) -> FloatArray:
    f = np.maximum(np.asarray(f, dtype=float), np.finfo(float).tiny)
    mass = float(np.sum(weights * f))
    if not np.isfinite(mass) or mass <= 0.0:
        raise FloatingPointError("density normalization failed")
    return f / mass


def _gibbs_target(
    K: int,
    eps: float,
    kernel: FloatArray,
    wx: FloatArray,
    wy: FloatArray,
    f: FloatArray,
    theta: float = 1.0,
) -> tuple[FloatArray, FloatArray, float, FloatArray]:
    attraction = kernel @ (wy * f)
    attraction = np.clip(attraction, 0.0, 1.0)
    turnout = float(np.sum(wx * (1.0 - np.power(1.0 - attraction, K))))
    if not np.isfinite(turnout) or turnout <= 0.0:
        raise FloatingPointError("turnout must be strictly positive")
    diagonal = theta * np.asarray(omega(K, attraction)) / turnout + (1-theta) * np.asarray(u_coefficient(K, attraction))
    potential = kernel.T @ (wx * diagonal)
    shifted = (potential - float(np.max(potential))) / eps
    target = np.exp(np.clip(shifted, -745.0, 0.0))
    target = 0.5 * (target + target[::-1])
    target = _normalize_density(target, wy)
    return target, attraction, turnout, potential


def solve_diagonal_state(
    K: int,
    r: float,
    eps: float,
    *,
    theta: float = 1.0,
    L: float = 2.5,
    ny: int = 801,
    nx: int = 140,
    damping: float = 0.12,
    tol: float = 1e-10,
    max_iter: int = 20_000,
    initial_density: Optional[ArrayLike] = None,
) -> DistributionalState:
    """Solve the self-consistent diagonal Gibbs equation.

    The electorate is standard Gaussian and is integrated with Gauss--Hermite
    quadrature.  Party ideology is discretized on a symmetric uniform grid.
    Reflection symmetry is imposed at every fixed-point iteration so the solver
    follows the centered diagonal branch.
    """
    if K < 2:
        raise ValueError("K must be at least 2")
    if not np.isfinite(theta) or not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0, 1]")
    if not all(np.isfinite(v) for v in (r, eps, L)) or r <= 0.0 or eps <= 0.0 or L <= 0.0:
        raise ValueError("r, eps, and L must be positive")
    if ny < 51 or ny % 2 == 0:
        raise ValueError("ny must be an odd integer at least 51")
    if nx < 20:
        raise ValueError("nx must be at least 20")
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0, 1]")

    x, wx = _gaussian_quadrature(nx)
    y = np.linspace(-L, L, ny, dtype=float)
    wy = _trapezoid_weights(y)
    kernel = np.exp(-0.5 * np.square((x[:, None] - y[None, :]) / r))

    if initial_density is None:
        sd0 = min(0.45, max(0.06, np.sqrt(eps / 0.08)))
        f = np.exp(-0.5 * np.square(y / sd0))
    else:
        f = np.asarray(initial_density, dtype=float)
        if f.shape != y.shape:
            raise ValueError("initial_density must have shape (ny,)")
    f = 0.5 * (f + f[::-1])
    f = _normalize_density(f, wy)

    residual = np.inf
    iterations = 0
    for iterations in range(1, max_iter + 1):
        target, _, _, _ = _gibbs_target(K, eps, kernel, wx, wy, f, theta)
        residual = float(np.sum(wy * np.abs(target - f)))
        if residual < tol:
            f = target
            break
        f = (1.0 - damping) * f + damping * target
        f = 0.5 * (f + f[::-1])
        f = _normalize_density(f, wy)
    else:
        raise RuntimeError(
            f"diagonal Gibbs iteration did not converge: residual={residual:.3e}"
        )

    target, attraction, turnout, potential = _gibbs_target(
        K, eps, kernel, wx, wy, f, theta
    )
    residual = float(np.sum(wy * np.abs(target - f)))
    mean = float(np.sum(wy * y * f))
    variance = float(np.sum(wy * np.square(y - mean) * f))

    kernel_yy = (
        np.square(x[:, None] - y[None, :]) / r**4 - 1.0 / r**2
    ) * kernel
    diagonal = theta * np.asarray(omega(K, attraction)) / turnout + (1-theta) * np.asarray(u_coefficient(K, attraction))
    potential_yy = kernel_yy.T @ (wx * diagonal)
    center = ny // 2
    centering_curvature = float(-potential_yy[center])

    return DistributionalState(
        K=K,
        r=float(r),
        eps=float(eps),
        x=x,
        wx=wx,
        y=y,
        wy=wy,
        kernel=kernel,
        f=f,
        attraction=attraction,
        turnout=turnout,
        potential=potential,
        residual_l1=residual,
        iterations=iterations,
        variance=variance,
        centering_curvature=centering_curvature,
        theta=float(theta),
    )

from scipy.sparse.linalg import LinearOperator, eigsh


@dataclass(frozen=True)
class OddEigenpair:
    """Principal reflection-odd eigenpair of the electoral Hessian operator."""

    nu: float
    margin: float
    transformed_vector: FloatArray
    density_vector: FloatArray
    translation_overlap: float


def _odd_projection(vector: ArrayLike) -> FloatArray:
    vec = np.asarray(vector, dtype=float)
    if vec.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    out = 0.5 * (vec - vec[::-1])
    if out.size % 2 == 1:
        out[out.size // 2] = 0.0
    return out


def apply_standard_sector_potential(
    state: DistributionalState, density_perturbation: ArrayLike
) -> FloatArray:
    """Apply the exact standard-label linearized strategic potential.

    The input ``g`` is the common density shape in a perturbation
    ``f_i = f* + delta z_i g`` with ``sum_i z_i = 0``.  The returned array is
    the coefficient ``E[g]`` in ``delta psi_i = z_i E[g]`` and includes the
    rank-one valid-turnout normalization term.
    """
    g = np.asarray(density_perturbation, dtype=float)
    if g.shape != state.y.shape:
        raise ValueError("density_perturbation must match the ideology grid")
    b = state.kernel @ (state.wy * g)
    ell = float(
        np.sum(state.wx * np.asarray(chi(state.K, state.attraction)) * b)
    )
    electoral = state.kernel.T @ (state.wx * np.asarray(
        feedback_coefficient(state.K, state.attraction, state.turnout, state.theta)
    ) * b)
    turnout_profile = state.kernel.T @ (
        state.wx * np.power(1.0 - state.attraction, state.K - 1)
    )
    return electoral - state.theta * ell * turnout_profile / state.turnout**2


def apply_odd_electoral_operator(
    state: DistributionalState, vector: ArrayLike
) -> FloatArray:
    """Apply ``sqrt(f) B sqrt(f)`` after projection to odd parity.

    ``vector`` uses the Euclidean coordinates associated with the denominator
    ``integral g^2/f``: ``g = sqrt(f / w_y) * vector``.
    """
    z = np.asarray(vector, dtype=float)
    if z.shape != state.y.shape:
        raise ValueError("vector must have the same shape as the ideology grid")
    z = _odd_projection(z)
    if state.K == 2 and state.theta == 1.0:
        return np.zeros_like(z)
    d_y = np.sqrt(state.wy * state.f)
    b = state.kernel @ (d_y * z)
    weight_x = state.wx * np.asarray(feedback_coefficient(state.K, state.attraction, state.turnout, state.theta))
    out = d_y * (state.kernel.T @ (weight_x * b))
    return _odd_projection(out)


def _translation_vector(state: DistributionalState) -> FloatArray:
    """Stable transformed tangent for a coherent infinitesimal translation."""
    # At a zero-current Gibbs state, f'/f = psi'/eps.  This avoids dividing a
    # numerically tiny boundary density after differentiating f directly.
    potential_y = np.gradient(state.potential, state.y, edge_order=2)
    z = -np.sqrt(state.wy * state.f) * potential_y / state.eps
    z = _odd_projection(z)
    norm = float(np.linalg.norm(z))
    if norm == 0.0:
        return z
    return z / norm


def odd_principal_eigenpair(
    state: DistributionalState,
    *,
    tol: float = 1e-11,
    maxiter: int = 10_000,
) -> OddEigenpair:
    """Compute the principal odd electoral eigenvalue and stability margin."""
    n = state.y.size
    if state.K == 2 and state.theta == 1.0:
        zero = np.zeros(n, dtype=float)
        return OddEigenpair(
            nu=0.0,
            margin=-1.0,
            transformed_vector=zero,
            density_vector=zero,
            translation_overlap=0.0,
        )

    operator = LinearOperator(
        shape=(n, n),
        matvec=lambda v: apply_odd_electoral_operator(state, v),
        dtype=np.float64,
    )
    v0 = _odd_projection(state.y * np.sqrt(state.wy * state.f))
    v0_norm = float(np.linalg.norm(v0))
    if v0_norm == 0.0:
        raise RuntimeError("failed to construct a nonzero odd initial vector")
    v0 /= v0_norm
    values, vectors = eigsh(
        operator,
        k=1,
        which="LA",
        v0=v0,
        tol=tol,
        maxiter=maxiter,
    )
    nu = max(0.0, float(values[0]))
    z = _odd_projection(vectors[:, 0])
    z_norm = float(np.linalg.norm(z))
    if z_norm == 0.0:
        raise RuntimeError("eigensolver returned a zero odd vector")
    z /= z_norm
    g = np.sqrt(state.f / state.wy) * z
    tangent = _translation_vector(state)
    overlap = float(abs(np.dot(z, tangent))) if np.any(tangent) else 0.0
    return OddEigenpair(
        nu=nu,
        margin=nu / state.eps - 1.0,
        transformed_vector=z,
        density_vector=g,
        translation_overlap=overlap,
    )

import math
from scipy.integrate import quad


def dense_odd_principal_eigenvalue(state: DistributionalState) -> float:
    """Dense reference value for testing the matrix-free odd eigensolver."""
    if state.K == 2 and state.theta == 1.0:
        return 0.0
    d_y = np.sqrt(state.wy * state.f)
    weight_x = state.wx * np.asarray(feedback_coefficient(state.K, state.attraction, state.turnout, state.theta))
    weighted_kernel = weight_x[:, None] * state.kernel
    matrix = (d_y[:, None] * (state.kernel.T @ weighted_kernel)) * d_y[None, :]
    n = state.y.size
    reversal = np.eye(n)[::-1]
    projection = 0.5 * (np.eye(n) - reversal)
    odd_matrix = projection @ matrix @ projection
    value = float(np.linalg.eigvalsh(odd_matrix)[-1])
    return max(0.0, value)


@dataclass(frozen=True)
class PointLimitComponents:
    K: int
    r: float
    turnout: float
    feedback: float
    centering: float
    lambda_pr: float


def point_pr_eigenvalue(K: int, r: float) -> float:
    """Closed Gaussian point-party PR differentiation eigenvalue.

    This is Eq. (lambdaPR) of the baseline manuscript with adaptation speed and
    voter standard deviation both set to one.
    """
    if K < 2 or r <= 0.0:
        raise ValueError("K must be at least 2 and r must be positive")
    r2 = r * r
    turnout = r * sum(
        (-1) ** (m + 1) * math.comb(K, m) / math.sqrt(r2 + m)
        for m in range(1, K + 1)
    )
    vote_curvature = (
        sum(
            (-1) ** (m + 1)
            * math.comb(K, m)
            * (1.0 - r2 - m)
            / (r2 + m) ** 1.5
            for m in range(1, K + 1)
        )
        / (K * r)
        + sum(
            (-1) ** n
            * math.comb(K - 2, n)
            / ((n + 2) * (r2 + n + 2) ** 1.5)
            for n in range(0, K - 1)
        )
        / r
    )
    turnout_curvature = (
        sum(
            (-1) ** m
            * math.comb(K - 1, m)
            * (1.0 - r2 - (m + 1))
            / (r2 + m + 1) ** 1.5
            for m in range(0, K)
        )
        / r
        + sum(
            (-1) ** m
            * math.comb(K - 2, m)
            / (r2 + m + 2) ** 1.5
            for m in range(0, K - 1)
        )
        / r
    )
    return (vote_curvature - turnout_curvature / K) / turnout


def point_limit_components(K: int, r: float) -> PointLimitComponents:
    """Decompose the point eigenvalue into feedback minus centering curvature."""
    if K < 2 or r <= 0.0:
        raise ValueError("K must be at least 2 and r must be positive")

    normalizer = math.sqrt(2.0 * math.pi)

    def rho(x: float) -> float:
        return math.exp(-0.5 * x * x) / normalizer

    def satisfaction(x: float) -> float:
        return math.exp(-0.5 * x * x / (r * r))

    turnout = quad(
        lambda x: rho(x) * (1.0 - (1.0 - satisfaction(x)) ** K),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=300,
    )[0]

    feedback = quad(
        lambda x: rho(x)
        * float(eta(K, satisfaction(x)))
        * (x * satisfaction(x) / (r * r)) ** 2,
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=300,
    )[0] / turnout

    potential_second = quad(
        lambda x: rho(x)
        * float(omega(K, satisfaction(x)))
        * ((x * x / r**4 - 1.0 / r**2) * satisfaction(x)),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=300,
    )[0] / turnout
    centering = -potential_second
    return PointLimitComponents(
        K=K,
        r=float(r),
        turnout=float(turnout),
        feedback=float(feedback),
        centering=float(centering),
        lambda_pr=float(point_pr_eigenvalue(K, r)),
    )

from scipy.optimize import brentq


@dataclass(frozen=True)
class CriticalRatioResult:
    K: int
    eps: float
    r_low: float
    r_high: float
    margin_low: float
    margin_high: float
    r_critical: float
    margin_root: float
    variance_root: float
    overlap_root: float
    iterations_root: int
    theta: float = 1.0


def find_distributional_critical_ratio(
    K: int,
    eps: float,
    bracket: tuple[float, float],
    *,
    theta: float = 1.0,
    L: float = 2.5,
    ny: int = 801,
    nx: int = 140,
    damping: float = 0.12,
    state_tol: float = 1e-10,
    root_tol: float = 5e-5,
) -> CriticalRatioResult:
    """Locate the finite-width odd-sector stability boundary.

    The root solves ``nu_K(f*) / eps - 1 = 0``.  Stationary densities are
    cached and the closest solved density seeds each subsequent fixed-point
    solve, which provides continuation without making the result path
    dependent.
    """
    if K < 2 or (K == 2 and theta == 1):
        raise ValueError("binary seat-share maximisation has no odd crossing")
    r_low, r_high = map(float, bracket)
    if not (0.0 < r_low < r_high):
        raise ValueError("bracket must satisfy 0 < low < high")

    cache: dict[float, tuple[DistributionalState, OddEigenpair]] = {}

    def evaluate(r_value: float) -> float:
        key = float(r_value)
        if key in cache:
            return cache[key][1].margin
        initial = None
        if cache:
            nearest = min(cache, key=lambda existing: abs(existing - key))
            initial = cache[nearest][0].f
        state = solve_diagonal_state(
            K=K,
            theta=theta,
            r=key,
            eps=eps,
            L=L,
            ny=ny,
            nx=nx,
            damping=damping,
            tol=state_tol,
            initial_density=initial,
        )
        eigenpair = odd_principal_eigenpair(state)
        cache[key] = (state, eigenpair)
        return eigenpair.margin

    margin_low = float(evaluate(r_low))
    margin_high = float(evaluate(r_high))
    if margin_low <= 0.0 or margin_high >= 0.0:
        raise ValueError(
            "critical bracket must have positive margin at the lower endpoint "
            "and negative margin at the upper endpoint; "
            f"received ({margin_low:.6g}, {margin_high:.6g})"
        )

    r_critical = float(
        brentq(
            evaluate,
            r_low,
            r_high,
            xtol=root_tol,
            rtol=max(4 * np.finfo(float).eps, root_tol * 0.1),
            maxiter=80,
        )
    )
    margin_root = float(evaluate(r_critical))
    root_state, root_eigenpair = cache[r_critical]
    return CriticalRatioResult(
        K=K,
        eps=float(eps),
        r_low=r_low,
        r_high=r_high,
        margin_low=margin_low,
        margin_high=margin_high,
        r_critical=r_critical,
        margin_root=margin_root,
        variance_root=root_state.variance,
        overlap_root=root_eigenpair.translation_overlap,
        iterations_root=root_state.iterations,
        theta=float(theta),
    )

from scipy.linalg import eig
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class ModeValidationResult:
    K: int
    r: float
    eps: float
    leading_growth_rate: float
    log_amplitude_slope: float
    mass_error: float
    times: FloatArray
    amplitudes: FloatArray
    initial_density_mode: FloatArray
    stationary_density: FloatArray
    y: FloatArray


def _odd_basis(n: int) -> FloatArray:
    if n % 2 == 0:
        raise ValueError("odd-parity basis requires an odd grid size")
    half = n // 2
    basis = np.zeros((n, half), dtype=float)
    scale = 1.0 / math.sqrt(2.0)
    for j in range(half):
        basis[j, j] = scale
        basis[n - 1 - j, j] = -scale
    return basis


def _dense_odd_linearized_generator(state: DistributionalState) -> tuple[FloatArray, FloatArray]:
    """Return the conservative finite-volume generator on the odd subspace."""
    n = state.y.size
    dy = float(state.y[1] - state.y[0])
    if not np.allclose(np.diff(state.y), dy):
        raise ValueError("linearized generator requires a uniform ideology grid")

    weight_x = state.wx * np.asarray(feedback_coefficient(state.K, state.attraction, state.turnout, state.theta))
    # B maps nodal density perturbations to nodal strategic-potential changes.
    b_matrix = (
        state.kernel.T @ (weight_x[:, None] * state.kernel)
    ) * state.wy[None, :]

    gradient = np.zeros((n - 1, n), dtype=float)
    row = np.arange(n - 1)
    gradient[row, row] = -1.0 / dy
    gradient[row, row + 1] = 1.0 / dy
    face_density = 0.5 * (state.f[:-1] + state.f[1:])
    stiffness = gradient.T @ ((face_density * dy)[:, None] * gradient)

    safe_density = np.maximum(state.f, float(np.max(state.f)) * 1e-15)
    hessian = state.eps * np.diag(1.0 / safe_density) - b_matrix
    generator = -np.diag(1.0 / state.wy) @ stiffness @ hessian

    basis = _odd_basis(n)
    odd_generator = basis.T @ generator @ basis
    return odd_generator, basis


def run_mode_validation(
    K: int,
    eps: float,
    r: float,
    *,
    quick: bool = False,
) -> ModeValidationResult:
    """Evolve the discrete exact odd linearization and fit modal growth.

    The conservative finite-volume discretization uses zero boundary flux.  It
    is a diagnostic of the mean-field linearized PDE, not a particle estimate.
    """
    ny = 151 if quick else 301
    nx = 80 if quick else 140
    horizon = 30.0 if quick else 60.0
    samples = 31 if quick else 61
    state = solve_diagonal_state(
        K=K,
        r=r,
        eps=eps,
        L=1.5,
        ny=ny,
        nx=nx,
        damping=0.12,
        tol=1e-10,
    )
    odd_generator, basis = _dense_odd_linearized_generator(state)
    values, vectors = eig(odd_generator)
    index = int(np.argmax(values.real))
    leading = float(values[index].real)
    if abs(float(values[index].imag)) > 1e-8:
        raise RuntimeError("leading odd growth rate is unexpectedly complex")
    initial = np.asarray(vectors[:, index].real, dtype=float)
    initial /= np.linalg.norm(initial)

    times = np.linspace(0.0, horizon, samples)
    trajectory = np.asarray(
        expm_multiply(odd_generator, initial, start=0.0, stop=horizon, num=samples),
        dtype=float,
    )
    amplitudes = np.linalg.norm(trajectory, axis=1)
    fit_start = samples // 3
    slope = float(np.polyfit(times[fit_start:], np.log(amplitudes[fit_start:]), 1)[0])

    density_mode = basis @ initial
    density_mode /= math.sqrt(float(np.sum(state.wy * density_mode * density_mode / state.f)))
    mass_error = float(abs(np.sum(state.wy * density_mode)))
    return ModeValidationResult(
        K=K,
        r=float(r),
        eps=float(eps),
        leading_growth_rate=leading,
        log_amplitude_slope=slope,
        mass_error=mass_error,
        times=times,
        amplitudes=amplitudes,
        initial_density_mode=density_mode,
        stationary_density=state.f.copy(),
        y=state.y.copy(),
    )
