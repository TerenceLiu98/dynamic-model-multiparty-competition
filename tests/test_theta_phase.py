from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import distributional_phase as dp


def test_mixed_binary_feedback_is_not_silently_cancelled():
    assert hasattr(dp, 'h_coefficient'), 'New manuscript requires the vote-feedback coefficient h_K'
    a=np.linspace(0,1,21)
    assert np.allclose(dp.h_coefficient(2,a), .5, atol=1e-15)
    s=dp.solve_diagonal_state(2,.6,.002,theta=0,ny=101,nx=240,L=2.2)
    assert dp.odd_principal_eigenpair(s).nu > s.eps
    s1=dp.solve_diagonal_state(2,.6,.002,theta=1,ny=101,nx=240,L=2.2)
    assert dp.odd_principal_eigenpair(s1).nu == 0


def test_high_order_gaussian_quadrature_and_mixed_dense_eigenvalue():
    s=dp.solve_diagonal_state(3,.75,.002,theta=.4,ny=101,nx=400,L=2.2)
    assert np.isfinite(s.f).all() and abs(s.wx.sum()-1)<1e-13
    assert abs(dp.odd_principal_eigenpair(s).nu-dp.dense_odd_principal_eigenvalue(s)) < 1e-11
    assert s.residual_l1 < 1e-10


def test_bernoulli_sg_gibbs_balance_and_implicit_mass_positivity():
    from theta_dynamics import bernoulli, sg_rhs, implicit_sg_step
    y=np.linspace(-2,2,101); w=dp._trapezoid_weights(y); eps=.003
    psi=-.5*y*y
    f=np.exp((psi-psi.max())/eps); f/=w@f
    assert bernoulli(0.) == 1.
    assert np.isfinite(bernoulli(np.array([-1000.,-1e-12,0,1e-12,1000.]))).all()
    assert np.sum(w*np.abs(sg_rhs(f,psi,eps,y,w))) < 1e-11
    f0=np.exp(-.5*(y-.35)**2/.08); f0/=w@f0
    f1=implicit_sg_step(f0,psi,eps,y,w,5.)
    assert np.min(f1)>=0 and abs(w@f1-1)<2e-13


@pytest.mark.parametrize('theta', [0.,.35,1.])
def test_sg_generator_matches_finite_difference_of_full_coupled_rhs(theta):
    from theta_dynamics import mixed_potentials, sg_rhs, odd_growth_eigenpair
    s=dp.solve_diagonal_state(2,.6,.002,theta=theta,ny=101,nx=240,L=2.2,tol=1e-13)
    e=odd_growth_eigenpair(s)
    g=e.density_vector
    # Use a bounded, odd test deformation to keep tails in the density domain.
    g=s.f*np.tanh(s.y/np.sqrt(s.variance)); step=1e-5
    def rhs(delta):
        f=np.stack([s.f+delta*g,s.f-delta*g])
        p=mixed_potentials(s,f).potential
        return sg_rhs(f[0],p[0],s.eps,s.y,s.wy)
    numerical=(rhs(step)-rhs(-step))/(2*step)
    from theta_dynamics import apply_odd_generator
    analytic=apply_odd_generator(s,g)
    assert np.sum(s.wy*np.abs(numerical-analytic)) < 2e-7
    h=dp.odd_principal_eigenpair(s)
    assert np.sign(e.rate) == np.sign(h.margin)
    assert e.residual < 1e-9


@pytest.mark.parametrize('K', [2,3,5])
@pytest.mark.parametrize('theta', [0.,.37,1.])
def test_exact_mixed_potential_and_standard_sector_autodiff(K,theta):
    import torch
    from theta_dynamics import mixed_potentials
    torch.set_default_dtype(torch.float64)
    s=dp.solve_diagonal_state(K,.7,.004,theta=theta,ny=81,nx=80,L=2.2)
    kernel=torch.tensor(s.kernel); wy=torch.tensor(s.wy); wx=torch.tensor(s.wx)
    zz,ww=np.polynomial.legendre.leggauss((K+1)//2)
    z=torch.tensor((zz+1)/2); weights=torch.tensor(ww/2)
    def autograd_potential(f):
        f=torch.tensor(f,requires_grad=True)
        a=kernel@(f*wy).T
        p=[]
        for i in range(K):
            others=[j for j in range(K) if j !=i]
            fac=1-z[None,:,None]*a[:,None,others]
            p.append(a[:,i]*(fac.prod(dim=2)*weights).sum(dim=1))
        q=wx@torch.stack(p,dim=1); J=(1-theta)*q+theta*q/q.sum()
        return torch.autograd.grad(J[0],f)[0][0].detach().numpy()/s.wy
    f=np.tile(s.f,(K,1))
    assert np.max(np.abs(autograd_potential(f)-mixed_potentials(s,f).potential[0]))<2e-12
    for g in [s.f*np.tanh(s.y/np.sqrt(s.variance)), s.f*(s.y**2-s.variance)]:
        h=np.zeros_like(f); h[0]=g; h[1]=-g
        delta=2e-5
        derivative=(autograd_potential(f+delta*h)-autograd_potential(f-delta*h))/(2*delta)
        actual=dp.apply_standard_sector_potential(s,g)
        assert np.max(np.abs(derivative-actual))<2e-9


def test_paper_binary_vote_critical_ratio():
    root=dp.find_distributional_critical_ratio(2,.002,(.7,.9),theta=0,ny=201,nx=400,L=2.2,root_tol=1e-7)
    assert abs(root.r_critical-.7903)<5e-5


def test_mixed_objective_validation():
    with pytest.raises(ValueError):
        dp.solve_diagonal_state(2,.6,.002,theta=1.2,ny=81,nx=80)
