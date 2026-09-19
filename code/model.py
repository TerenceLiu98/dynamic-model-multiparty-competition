"""Core numerical implementation for the multiparty distributional satisficing model.

The code implements the exact integral representation of uniform choice from an
independently generated satisficing/consideration set, three smooth electoral
seat maps (PR, FPTP/majoritarian, and compensatory MMP), point-party dynamics,
and particle approximations to entropy-regularized Wasserstein dynamics.
"""
from __future__ import annotations
import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def legendre01(n: int, device: str = "cpu"):
    z, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor((z + 1.0) / 2.0, device=device), torch.tensor(w / 2.0, device=device)


def normal_pdf(x, mean, sd):
    return torch.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


class ElectoralModel:
    def __init__(
        self,
        K: int,
        D: int = 21,
        L: float = 4.0,
        nx: int = 121,
        geo: float = 0.0,
        total_sd: float = 1.0,
        rule: str = "PR",
        beta: float = 25.0,
        threshold: float = 0.0,
        thresh_kappa: float = 80.0,
        mmp_alpha: float = 0.6,
        mmp_gamma: float = 40.0,
        device: str = "cpu",
        homogeneous_control: bool = False,
    ):
        if K < 2 or D < 1 or nx < 3 or L <= 0:
            raise ValueError("require K>=2, D>=1, nx>=3 and L>0")
        self.K = K
        self.D = D
        self.L = L
        self.rule = rule.upper()
        self.beta = beta
        self.threshold = threshold
        self.thresh_kappa = thresh_kappa
        self.mmp_alpha = mmp_alpha
        self.mmp_gamma = mmp_gamma
        self.device = device

        self.x = torch.linspace(-L, L, nx, device=device)
        if D == 1 or geo == 0:
            means = np.zeros(D)
        else:
            from scipy.stats import norm
            probs = (np.arange(D) + 0.5) / D
            means = geo * norm.ppf(probs)
        within = max(0.15, math.sqrt(max(total_sd**2 - geo**2, 0.0225)))
        rho = []
        for m in means:
            r = normal_pdf(self.x, float(m), within)
            r = r / torch.trapezoid(r, self.x)
            rho.append(r)
        self.rho = torch.stack(rho, 0)  # D x nx
        self.dw = torch.ones(D, device=device) / D
        if homogeneous_control:
            national = (self.dw[:, None] * self.rho).sum(dim=0)
            self.rho = national[None, :].repeat(D, 1)
        # Eq. (4): degree K-1 is exact with ceil(K/2) nodes.
        self.z, self.zw = legendre01((K + 1) // 2, device)

    def choice(self, a):
        """Exact choice probabilities from average satisfaction a (nx x K)."""
        nx, K = a.shape
        z, zw = self.z, self.zw
        factors = 1.0 - (1.0 - z[None, :, None]) * a[:, None, :]  # nx x nz x K
        prefix = torch.cumprod(factors, dim=2)
        suffix = torch.flip(torch.cumprod(torch.flip(factors, dims=[2]), dim=2), dims=[2])
        one = torch.ones((nx, len(z)), device=a.device, dtype=a.dtype)
        ps = []
        for i in range(K):
            left = prefix[:, :, i - 1] if i > 0 else one
            right = suffix[:, :, i + 1] if i < K - 1 else one
            integral = torch.sum(left * right * zw[None, :], dim=1)
            ps.append(a[:, i] * integral)
        p = torch.stack(ps, dim=1)
        p0 = torch.prod(1.0 - a, dim=1)
        return p, p0

    def votes_from_a(self, a):
        p, p0 = self.choice(a)
        dx = self.x[1] - self.x[0]
        trap = torch.ones_like(self.x)
        trap[0] = trap[-1] = 0.5
        W = self.rho * trap[None, :] * dx
        q = W @ p  # D x K, unconditional mass of potential voters
        turnout = q.sum(dim=1)
        v = q / turnout[:, None].clamp_min(1e-12)  # valid-vote shares
        return q, v, p0

    def seat_map(self, v, q=None):
        """Smooth electoral institution maps valid votes to national seat shares."""
        national = (self.dw[:, None] * v).sum(dim=0)
        if self.threshold > 0:
            gate = torch.sigmoid(self.thresh_kappa * (national - self.threshold))
            raw = national * gate
            P = raw / raw.sum()
        else:
            P = national / national.sum()

        C = (self.dw[:, None] * torch.softmax(self.beta * v, dim=1)).sum(dim=0)
        C = C / C.sum()

        if self.rule == "PR":
            return P
        if self.rule in {"FPTP", "MAJ", "MAJORITARIAN"}:
            return C
        if self.rule == "MMP":
            direct_floor = self.mmp_alpha * C
            # Stylized compensatory MMP with a smooth overhang floor:
            # in the hard limit u_i = max(P_i, alpha C_i).
            u = P + torch.nn.functional.softplus(
                self.mmp_gamma * (direct_floor - P)
            ) / self.mmp_gamma
            return u / u.sum()
        raise ValueError(f"Unknown rule: {self.rule}")

    def point_seats(self, y, sigma):
        a = torch.exp(-0.5 * ((self.x[:, None] - y[None, :]) / sigma) ** 2)
        q, v, p0 = self.votes_from_a(a)
        return self.seat_map(v, q), q, v, p0

    def particle_seats(self, Y, sigma):
        # Y: K x N, empirical measure for each party
        a = torch.exp(-0.5 * ((self.x[:, None, None] - Y[None, :, :]) / sigma) ** 2).mean(dim=2)
        q, v, p0 = self.votes_from_a(a)
        return self.seat_map(v, q), q, v, p0


def diagonal_grad(seats, y, create_graph=False):
    """Own-party strategic gradient; ``seats`` may be any vector of objectives."""
    K = len(seats)
    out = []
    for i in range(K):
        g = torch.autograd.grad(seats[i], y, retain_graph=True, create_graph=create_graph)[0]
        out.append(g[i])
    return torch.stack(out)


def simulate_point(
    K=5,
    rule="PR",
    sigma=0.65,
    geo=0.0,
    steps=450,
    dt=0.15,
    seed=0,
    init_scale=0.12,
    sort_initial=True,
    grad_tol=1e-8,
    early_stop_after=120,
    diagnostic_grad_tol=1e-8,
    theta=1.0,
    **kwargs,
):
    if not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0, 1]")
    torch.manual_seed(seed)
    model = ElectoralModel(K, rule=rule, geo=geo, **kwargs)
    y0 = init_scale * torch.randn(K)
    if sort_initial:
        y0 = y0.sort().values
    initial_positions = y0.detach().cpu().numpy().copy()
    y = y0.detach().clone().requires_grad_(True)
    traj = []
    stop_reason = "max_steps"
    convergence_step = None
    for t in range(steps):
        seats, q, v, p0 = model.point_seats(y, sigma)
        objectives = (1-theta) * (model.dw[:, None] * q).sum(dim=0) + theta * seats
        g = diagonal_grad(objectives, y)
        max_gradient = torch.max(torch.abs(g)).item()
        if (
            convergence_step is None
            and max_gradient < diagnostic_grad_tol
            and t > early_stop_after
        ):
            convergence_step = t + 1
        with torch.no_grad():
            y += dt * g
            y.clamp_(-model.L + 0.05, model.L - 0.05)
        y.requires_grad_(True)
        if t % 10 == 0 or t == steps - 1:
            snapshot_seats = model.point_seats(y, sigma)[0]
            traj.append((t + 1, y.detach().cpu().numpy().copy(), snapshot_seats.detach().cpu().numpy().copy()))
        if grad_tol is not None and max_gradient < grad_tol and t > early_stop_after:
            stop_reason = "gradient_tolerance"
            break
    seats, q, v, p0 = model.point_seats(y, sigma)
    objectives = (1-theta) * (model.dw[:, None] * q).sum(dim=0) + theta * seats
    final_grad = diagonal_grad(objectives, y)
    final_max_gradient = float(torch.max(torch.abs(final_grad)).detach())
    # national abstention mass (district weighted)
    dx = model.x[1] - model.x[0]
    trap = torch.ones_like(model.x); trap[0] = trap[-1] = 0.5
    abst_d = (model.rho * trap[None, :] * dx) @ p0
    abst = float((model.dw * abst_d).sum().detach())
    return {
        "theta": float(theta),
        "initial_positions": initial_positions,
        "positions": y.detach().cpu().numpy(),
        "seats": seats.detach().cpu().numpy(),
        "q": q.detach().cpu().numpy(),
        "v": v.detach().cpu().numpy(),
        "abstention": abst,
        "trajectory": traj,
        "steps_run": t + 1,
        "physical_time": (t + 1) * dt,
        "convergence_step": convergence_step,
        "convergence_time": None if convergence_step is None else convergence_step * dt,
        "stop_reason": stop_reason,
        "converged": final_max_gradient < diagnostic_grad_tol,
        "final_max_gradient": final_max_gradient,
    }


def simulate_particles(
    K=5,
    rule="PR",
    sigma=0.65,
    geo=0.55,
    N=40,
    eps=0.004,
    steps=600,
    dt=0.025,
    seed=0,
    theta=1.0,
    **kwargs,
):
    if not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0, 1]")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ElectoralModel(K, rule=rule, geo=geo, **kwargs)
    centers = torch.linspace(-0.25, 0.25, K)[:, None]
    Y = (centers + 0.12 * torch.randn(K, N)).detach().clone().requires_grad_(True)
    traj = []
    for t in range(steps):
        seats, q, v, p0 = model.particle_seats(Y, sigma)
        objectives = (1-theta) * (model.dw[:, None] * q).sum(dim=0) + theta * seats
        grads = []
        for i in range(K):
            gi = torch.autograd.grad(objectives[i], Y, retain_graph=True)[0][i]
            grads.append(gi)
        # dJ/dY_n = (1/N) grad(delta J/dmu), hence multiply by N.
        g = torch.stack(grads) * N
        with torch.no_grad():
            Y += dt * g + math.sqrt(2.0 * eps * dt) * torch.randn_like(Y)
            hi = Y > model.L
            Y[hi] = 2 * model.L - Y[hi]
            lo = Y < -model.L
            Y[lo] = -2 * model.L - Y[lo]
        Y.requires_grad_(True)
        if t % 10 == 0 or t == steps - 1:
            snapshot_seats = model.particle_seats(Y, sigma)[0]
            traj.append((t + 1, Y.detach().cpu().numpy().copy(), snapshot_seats.detach().cpu().numpy().copy()))
    seats, q, v, p0 = model.particle_seats(Y, sigma)
    dx = model.x[1] - model.x[0]
    trap = torch.ones_like(model.x); trap[0] = trap[-1] = 0.5
    abst_d = (model.rho * trap[None, :] * dx) @ p0
    abst = float((model.dw * abst_d).sum().detach())
    return {
        "theta": float(theta),
        "particles": Y.detach().cpu().numpy(),
        "seats": seats.detach().cpu().numpy(),
        "q": q.detach().cpu().numpy(),
        "v": v.detach().cpu().numpy(),
        "abstention": abst,
        "trajectory": traj,
    }
