"""Full-density mixed-objective dynamics and SG-consistent odd growth spectrum.

Implements Eqs. (11), (32)--(33), (52), (116)--(117) of the manuscript.
The full evolution never projects party profiles back to reflection symmetry.
The diagonal solver imposes evenness only while finding the reference state.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal, solve_banded
from distributional_phase import DistributionalState, feedback_coefficient
from electoral_feedback import choice_and_jacobian


@dataclass(frozen=True)
class DensityFeedback:
    potential: np.ndarray
    support: np.ndarray
    seats: np.ndarray
    turnout: float
    acceptance: np.ndarray


def mixed_potentials(state: DistributionalState, densities: np.ndarray) -> DensityFeedback:
    """Exact unilateral potentials for homogeneous, threshold-free PR.

    Densities are K x ny. Seat shares are observables, not the mixed objective.
    The derivative of turnout is retained for each unilateral party variation.
    """
    f=np.asarray(densities,dtype=float)
    if f.shape != (state.K,state.y.size):
        raise ValueError('densities must have shape (K, ny)')
    if not np.all(np.isfinite(f)) or np.min(f)<0:
        raise ValueError('densities must be finite and nonnegative')
    a=state.kernel @ (f*state.wy).T
    p,_,M=choice_and_jacobian(a)
    q=state.wx@p; Q=float(q.sum())
    if not np.isfinite(Q) or Q<=0: raise FloatingPointError('nonpositive turnout')
    seats=q/Q
    own=np.diagonal(M,axis1=1,axis2=2)
    turnout_derivative=M.sum(axis=1)
    coefficient=(1-state.theta)*own + (state.theta/Q)*(own-seats*turnout_derivative)
    potential=(state.kernel.T @ (state.wx[:,None]*coefficient)).T
    return DensityFeedback(potential,q,seats,Q,a)


def bernoulli(s):
    """b(s)=s/expm1(s), stable at zero and at large positive/negative s."""
    x=np.asarray(s,dtype=float); out=np.empty_like(x)
    small=np.abs(x)<1e-5; pos=x>50; neg=x < -50
    t=x[small]
    out[small]=1-t/2+t*t/12-t**4/720+t**6/30240
    out[pos]=x[pos]*np.exp(-x[pos])/(-np.expm1(-x[pos]))
    out[neg]=-x[neg]/(-np.expm1(x[neg]))
    middle=~(small|pos|neg)
    out[middle]=x[middle]/np.expm1(x[middle])
    return float(out) if x.ndim==0 else out


def sg_coefficients(potential,eps,y,weights):
    """Lower/diagonal/upper density-generator entries with zero boundary flux."""
    if eps<=0: raise ValueError('SG diffusion must be positive')
    dy=float(y[1]-y[0]); d=np.diff(potential)/eps
    left=eps/dy*bernoulli(-d); right=eps/dy*bernoulli(d)
    diagonal=np.zeros_like(potential,dtype=float)
    diagonal[:-1]-=left/weights[:-1]
    diagonal[1:]-=right/weights[1:]
    lower=left/weights[1:]; upper=right/weights[:-1]
    return lower,diagonal,upper


def sg_rhs(density,potential,eps,y,weights):
    """Conservative RHS: the outward face flux in Eq. (117)."""
    dy=float(y[1]-y[0]); d=np.diff(potential)/eps
    flux=eps/dy*(bernoulli(-d)*density[:-1]-bernoulli(d)*density[1:])
    rhs=np.zeros_like(density)
    rhs[:-1]-=flux/weights[:-1]; rhs[1:]+=flux/weights[1:]
    return rhs


def implicit_sg_step(density,potential,eps,y,weights,dt):
    """Frozen-current-potential, implicit density update; no clipping/renormalising."""
    if not np.isfinite(dt) or dt<=0: raise ValueError('dt must be positive')
    lower,diagonal,upper=sg_coefficients(potential,eps,y,weights)
    band=np.zeros((3,len(y)))
    band[0,1:]=-dt*upper; band[1]=1-dt*diagonal; band[2,:-1]=-dt*lower
    out=solve_banded((1,1),band,density,overwrite_ab=True,check_finite=False)
    if not np.all(np.isfinite(out)) or np.min(out)<0:
        raise FloatingPointError('SG implicit solve lost positivity or finiteness')
    return out


def _odd_matrices(state):
    """Symmetric frozen L0 and C in the weighted odd-density coordinates."""
    n=len(state.y); h=n//2
    if n%2!=1: raise ValueError('an odd, reflected ideology grid is required')
    dy=float(state.y[1]-state.y[0])
    _,full_diag,_=sg_coefficients(state.potential,state.eps,state.y,state.wy)
    d=np.diff(state.potential)/state.eps
    # b(d)*exp(d/2) = b(-|d|)*exp(-|d|/2), without overflow.
    off=(state.eps/dy)*bernoulli(-np.abs(d))*np.exp(-np.abs(d)/2)
    off/=np.sqrt(state.wy[:-1]*state.wy[1:])
    diagonal=full_diag[:h].copy(); off=off[:h-1].copy()
    if state.K==2 and state.theta==1:
        C=np.zeros((h,h))
    else:
        root=np.sqrt(state.wy[:h]*state.f[:h])/np.sqrt(2.)
        X=(state.kernel[:,:h]-state.kernel[:,::-1][:,:h])*root
        wx=state.wx*np.asarray(feedback_coefficient(state.K,state.attraction,state.turnout,state.theta))
        C=X.T @ (wx[:,None]*X)
        C=(C+C.T)/2
    return diagonal,off,C


def _tridiagonal_product(diagonal,off,z):
    out=diagonal*z
    out[:-1]+=off*z[1:]; out[1:]+=off*z[:-1]
    return out


def expand_odd(z):
    return np.concatenate((z,[0.],-z[::-1]))/np.sqrt(2.)


def apply_odd_generator(state,density_perturbation):
    """Apply L0(I-C/epsilon) and return the nodal density RHS."""
    z=np.sqrt(state.wy/state.f)*density_perturbation
    z=(z[:len(z)//2]-z[::-1][:len(z)//2])/np.sqrt(2.)
    diagonal,off,C=_odd_matrices(state)
    out=_tridiagonal_product(diagonal,off,z-C@z/state.eps)
    return np.sqrt(state.f/state.wy)*expand_odd(out)


@dataclass(frozen=True)
class OddGrowthEigenpair:
    rate: float
    residual: float
    transformed_vector: np.ndarray
    density_vector: np.ndarray


def odd_growth_eigenpair(state: DistributionalState) -> OddGrowthEigenpair:
    """Actual time-generator eigenvalue, distinct from nu/epsilon-1.

    Set M=-L0=U.T U on odd parity, with upper bidiagonal Cholesky U.
    L0(I-C/epsilon) is similar to U(C/epsilon-I)U.T, a real symmetric
    matrix. This avoids sorting potentially noisy complex eigenvalues and
    never drops the mobility factor.
    """
    diagonal,off,C=_odd_matrices(state); h=len(diagonal)
    if not np.any(C):
        values,vectors=eigh_tridiagonal(diagonal,off,select='i',select_range=(h-1,h-1))
        rate=float(values[0]); z=vectors[:,0]
    else:
        # Cholesky of the positive, odd-restricted mobility matrix -L0.
        d=np.empty(h); u=np.empty(h-1)
        d[0]=np.sqrt(-diagonal[0])
        for j in range(h-1):
            u[j]=-off[j]/d[j]
            pivot=-diagonal[j+1]-u[j]*u[j]
            if pivot<=0: raise FloatingPointError('odd mobility is not positive definite')
            d[j+1]=np.sqrt(pivot)
        UC=d[:,None]*C
        UC[:-1]+=u[:,None]*C[1:]
        S=UC*d[None,:]
        S[:,:-1]+=UC[:,1:]*u[None,:]
        S/=state.eps
        diag_UUt=d*d; diag_UUt[:-1]+=u*u
        S[np.diag_indices(h)]-=diag_UUt
        jj=np.arange(h-1); cross=u*d[1:]
        S[jj,jj+1]-=cross; S[jj+1,jj]-=cross
        S=(S+S.T)/2
        values,vectors=eigh(S,subset_by_index=(h-1,h-1),check_finite=False,driver='evr')
        rate=float(values[0]); v=vectors[:,0]
        z=d*v; z[1:]+=u*v[:-1]   # z=U.T v, the generator's eigenvector
    z/=np.linalg.norm(z)
    residual=float(np.linalg.norm(_tridiagonal_product(diagonal,off,z-C@z/state.eps)-rate*z))
    full=expand_odd(z)
    return OddGrowthEigenpair(rate,residual,full,np.sqrt(state.f/state.wy)*full)


def density_diagnostics(state,densities,feedback=None):
    feedback=mixed_potentials(state,densities) if feedback is None else feedback
    w=state.wy; y=state.y; f=densities
    mass=f@w; means=f@(w*y); variances=(f@(w*y*y))-means**2
    average=float(feedback.seats@means)
    P=float(np.sqrt(max(0.,feedback.seats@(means-average)**2)))
    residual=max(float(w@np.abs(sg_rhs(f[i],feedback.potential[i],state.eps,y,w))) for i in range(state.K))
    target=np.exp((feedback.potential-feedback.potential.max(axis=1)[:,None])/state.eps)
    target/= (target@w)[:,None]
    gibbs=float(np.max(np.abs(target-f)@w))
    return dict(P=P,turnout=feedback.turnout,rhs_l1=residual,gibbs_l1=gibbs,
                mass_error=float(np.max(np.abs(mass-1))),minimum_density=float(f.min()),
                means=means,variances=variances,seats=feedback.seats,support=feedback.support)


@dataclass
class NonlinearRun:
    times: np.ndarray
    records: list[dict]
    initial_densities: np.ndarray
    final_densities: np.ndarray
    stop_reason: str
    steps: int
    dt: float


def binary_initial_density(state,amplitude=.02):
    if state.K!=2: raise ValueError('Eq. (116) is the binary initial condition')
    if not 0<amplitude<1: raise ValueError('initial amplitude must lie in (0,1)')
    t=np.tanh(state.y/np.sqrt(state.variance))
    return state.f[None,:]*(1+amplitude*np.array([-1.,1.])[:,None]*t)


def simulate_densities(state,*,dt=.1,horizon=1600.,critical=False,
                       initial=None,record_every=10,stationarity_tol=1e-9,
                       min_stop_time=50.) -> NonlinearRun:
    """Appendix G nonlinear experiment, all potentials evaluated simultaneously."""
    if dt<=0 or horizon<=0 or record_every<1: raise ValueError('invalid time parameters')
    steps=int(round(horizon/dt))
    if not np.isclose(steps*dt,horizon,rtol=0,atol=1e-10):
        raise ValueError('horizon must be an integer multiple of dt')
    f=binary_initial_density(state) if initial is None else np.array(initial,dtype=float,copy=True)
    if f.shape != (state.K,len(state.y)) or np.min(f)<0: raise ValueError('invalid initial densities')
    if np.max(np.abs(f@state.wy-1))>1e-9: raise ValueError('initial densities must have unit mass')
    f_initial=f.copy(); feedback=mixed_potentials(state,f)
    records=[]; reason='max_horizon'
    for step in range(steps+1):
        time=step*dt
        # Full RHS each step controls stopping, independent of plotted cadence.
        diag=density_diagnostics(state,f,feedback)
        stop=not critical and time>=min_stop_time and diag['rhs_l1']<stationarity_tol
        if step%record_every==0 or step==steps or stop:
            records.append(dict(time=float(time),**diag))
        if stop:
            reason='stationarity_tolerance'; break
        if step==steps: break
        f=np.stack([implicit_sg_step(f[i],feedback.potential[i],state.eps,state.y,state.wy,dt)
                    for i in range(state.K)])
        if np.max(np.abs(f@state.wy-1))>1e-8:
            raise FloatingPointError('mass drift exceeded tolerance; no renormalisation was applied')
        feedback=mixed_potentials(state,f)
    return NonlinearRun(np.array([r['time'] for r in records]),records,f_initial,f,reason,step,dt)
