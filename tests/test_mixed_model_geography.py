from pathlib import Path
import sys
import itertools
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from model import ElectoralModel, simulate_point, simulate_particles
from electoral_feedback import choice_and_jacobian


def test_exact_choice_matches_subset_enumeration_and_jacobian():
    rng=np.random.default_rng(3409)
    for K in (2,3,5,8):
        a=rng.uniform(.01,.99,size=(3,K)); p,p0,M=choice_and_jacobian(a)
        brute=np.zeros_like(p)
        for bits in itertools.product([0,1],repeat=K):
            mask=np.array(bits,dtype=bool)
            if not mask.any(): continue
            probability=np.prod(np.where(mask,a,1-a),axis=1)
            brute[:,mask]+=probability[:,None]/mask.sum()
        assert np.max(abs(brute-p))<2e-15
        assert np.max(abs(p.sum(axis=1)+p0-1))<2e-15
        for i in range(K):
            ap=a.copy(); am=a.copy(); ap[:,i]+=1e-6; am[:,i]-=1e-6
            numeric=(choice_and_jacobian(ap)[0]-choice_and_jacobian(am)[0])/2e-6
            assert np.max(abs(numeric-M[:,:,i]))<3e-10


@pytest.mark.parametrize('rule',['PR','FPTP','MMP'])
@pytest.mark.parametrize('theta',[0.,.4,1.])
def test_point_analytic_force_equals_exact_torch_own_objective_gradient(rule,theta):
    from geographic_experiments import NumpyPointModel
    m=ElectoralModel(5,D=9,nx=61,geo=.7,rule=rule,beta=24,threshold=.05,mmp_alpha=.6,mmp_gamma=45)
    p=NumpyPointModel.from_torch(m)
    y=torch.tensor([-.27,-.03,.11,.14,.39],requires_grad=True)
    S,q,_,_=m.point_seats(y,.65)
    J=(1-theta)*(m.dw[:,None]*q).sum(dim=0)+theta*S
    expected=np.array([float(torch.autograd.grad(J[i],y,retain_graph=True)[0][i]) for i in range(5)])
    actual=p.evaluate(y.detach().numpy(),.65,theta)
    assert np.max(abs(actual['gradient']-expected))<2e-13
    assert np.max(abs(actual['seats']-S.detach().numpy()))<2e-14


def test_matched_control_has_identical_national_electorate():
    m=ElectoralModel(5,D=9,nx=61,geo=.7)
    c=ElectoralModel(5,D=9,nx=61,geo=.7,homogeneous_control=True)
    a=(m.dw[:,None]*m.rho).sum(dim=0)
    b=(c.dw[:,None]*c.rho).sum(dim=0)
    assert float(torch.max(torch.abs(a-b)))<3e-16
    assert float(torch.max(torch.abs(c.rho-c.rho[0])))==0


def test_point_and_particle_simulators_accept_theta_and_keep_seats_separate():
    raw=simulate_point(K=2,theta=0,steps=2,D=1,nx=41,seed=1)
    seat=simulate_point(K=2,theta=1,steps=2,D=1,nx=41,seed=1)
    assert not np.allclose(raw['positions'],seat['positions'])
    assert abs(raw['seats'].sum()-1)<1e-14
    p=simulate_particles(K=2,theta=.4,N=4,steps=2,D=1,nx=41,seed=1)
    assert np.isfinite(p['particles']).all() and abs(p['seats'].sum()-1)<1e-14


def test_binary_geographic_restoring_rates_match_table_1():
    from geographic_experiments import local_rates
    expected={'PR':-.401571,'FPTP':-4.818856,'MMP':-.401876}
    for rule in expected:
        result=local_rates(.7,rule,nx=61)
        assert abs(result['lambda_div']-expected[rule])<5.1e-7
        assert result['lambda_common']<0
        assert result['central_difference_error']<9e-8

@pytest.mark.parametrize('kind',['point','particles'])
def test_legacy_positional_arguments_keep_their_meaning(kind):
    """The new theta keyword must not reinterpret the old fourth argument geo."""
    from model import simulate_point,simulate_particles
    if kind=='point':
        positional=simulate_point(2,'PR',.65,.55,steps=2,D=9,nx=41,seed=3)
        keyword=simulate_point(K=2,rule='PR',sigma=.65,geo=.55,steps=2,D=9,nx=41,seed=3)
        assert np.allclose(positional['positions'],keyword['positions'],rtol=0,atol=1e-14)
    else:
        positional=simulate_particles(2,'PR',.65,.55,N=4,steps=2,D=9,nx=41,seed=3)
        keyword=simulate_particles(K=2,rule='PR',sigma=.65,geo=.55,N=4,steps=2,D=9,nx=41,seed=3)
        assert np.allclose(positional['particles'],keyword['particles'],rtol=0,atol=1e-14)
