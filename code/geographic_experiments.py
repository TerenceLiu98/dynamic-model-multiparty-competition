"""Matched-national-density point-platform experiments, Appendix F.1--F.2.

The fast NumPy force is an exact analytical chain rule. Tests independently
compare it with the inherited PyTorch automatic-differentiation force. All
reported local restoring eigenvalues are computed directly with PyTorch AD.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.special import expit, softmax
from electoral_feedback import choice_and_jacobian
from model import ElectoralModel, diagonal_grad


@dataclass
class NumpyPointModel:
    x: np.ndarray
    W: np.ndarray
    dw: np.ndarray
    K: int
    L: float
    rule: str
    beta: float
    threshold: float
    thresh_kappa: float
    mmp_alpha: float
    mmp_gamma: float

    @classmethod
    def from_torch(cls,model: ElectoralModel):
        x=model.x.detach().cpu().numpy().copy()
        w=np.ones_like(x)*(x[1]-x[0]); w[[0,-1]]*=.5
        return cls(x,model.rho.detach().cpu().numpy()*w,
                   model.dw.detach().cpu().numpy().copy(),model.K,model.L,
                   model.rule,model.beta,model.threshold,model.thresh_kappa,
                   model.mmp_alpha,model.mmp_gamma)

    def seats_and_jacobian(self,v):
        K=self.K; dw=self.dw; national=dw@v
        if self.threshold>0:
            gate=expit(self.thresh_kappa*(national-self.threshold))
            raw=national*gate
            raw_derivative=gate+national*self.thresh_kappa*gate*(1-gate)
        else:
            raw=national; raw_derivative=np.ones(K)
        P=raw/raw.sum()
        dPdn=(np.eye(K)-P[:,None])*raw_derivative[None,:]/raw.sum()
        dP=dPdn[:,None,:]*dw[None,:,None]
        if self.rule=='PR': return P,dP
        c=softmax(self.beta*v,axis=1); C=dw@c
        dC=self.beta*dw[None,:,None]*c.T[:,:,None]*(np.eye(K)[:,None,:]-c[None,:,:])
        if self.rule in {'FPTP','MAJ','MAJORITARIAN'}: return C,dC
        if self.rule !='MMP': raise ValueError('unknown seat rule')
        delta=self.mmp_alpha*C-P; gate=expit(self.mmp_gamma*delta)
        u=P+np.logaddexp(0,self.mmp_gamma*delta)/self.mmp_gamma
        du=(1-gate)[:,None,None]*dP+self.mmp_alpha*gate[:,None,None]*dC
        S=u/u.sum()
        R=(du-S[:,None,None]*du.sum(axis=0)[None,:,:])/u.sum()
        return S,R

    def evaluate(self,y,sigma=.65,theta=1.):
        if not 0<=theta<=1 or sigma<=0: raise ValueError('invalid theta or tolerance')
        a=np.exp(-.5*((self.x[:,None]-y[None,:])/sigma)**2)
        p,_,M=choice_and_jacobian(a)
        q=self.W@p; T=q.sum(axis=1); v=q/T[:,None]
        S,R=self.seats_and_jacobian(v)
        correction=R-np.einsum('idl,dl->id',R,v)[:,:,None]
        weights=np.einsum('idl,dx->ixl',correction/T[None,:,None],self.W)
        coefficient=theta*np.einsum('ixl,xli->xi',weights,M)
        own=np.diagonal(M,axis1=1,axis2=2)
        coefficient+=(1-theta)*(self.dw@self.W)[:,None]*own
        derivative=(self.x[:,None]-y[None,:])*a/sigma**2
        gradient=(coefficient*derivative).sum(axis=0)
        mean=float(S@y); P=float(np.sqrt(max(0.,S@(y-mean)**2)))
        national=self.dw@v
        return dict(gradient=gradient,seats=S,q=q,v=v,positions=np.array(y,copy=True),
                    P=P,turnout=float(self.dw@T),abstention=1-float(self.dw@T),
                    distance=float(abs(y[1]-y[0])) if self.K==2 else float(np.ptp(y)),
                    final_max_gradient=float(np.max(np.abs(gradient))),
                    effective_electoral_parties=1/float(national@national),
                    effective_parliamentary_parties=1/float(S@S))


def initial_positions(K,seed,scale=.12):
    # Exactly matches the CPU torch.manual_seed + randn convention of the old archive.
    gen=torch.Generator(device='cpu').manual_seed(int(seed))
    return (scale*torch.randn(K,generator=gen,dtype=torch.float64)).sort().values.numpy()


def simulate_numpy_point(model,initial,*,dt=.15,horizon=42.,sigma=.65,theta=1.,record_every=10):
    if dt<=0 or horizon<=0: raise ValueError('positive dt and horizon required')
    steps=int(round(horizon/dt))
    if not np.isclose(steps*dt,horizon,atol=1e-10,rtol=0):
        raise ValueError('horizon must be an integer multiple of dt')
    y=np.array(initial,dtype=float,copy=True); trajectory=[]
    info=model.evaluate(y,sigma,theta)
    for step in range(steps+1):
        if record_every and (step%record_every==0 or step==steps):
            trajectory.append(dict(time=step*dt,**info))
        if step==steps: break
        y=np.clip(y+dt*info['gradient'],-model.L+.05,model.L-.05)
        info=model.evaluate(y,sigma,theta)
    return dict(**info,initial_positions=np.array(initial,copy=True),trajectory=trajectory,
                physical_time=steps*dt,steps_run=steps,theta=theta,stop_reason='fixed_horizon')


def make_model(K,g,rule,*,nx=61,homogeneous=False):
    return ElectoralModel(K,D=9,L=4,nx=nx,geo=g,rule=rule,beta=24,
                          threshold=.05,thresh_kappa=80,mmp_alpha=.6,mmp_gamma=45,
                          homogeneous_control=homogeneous)


def local_rates(g,rule,*,nx=61,homogeneous=False,step=1e-5):
    model=make_model(2,g,rule,nx=nx,homogeneous=homogeneous)
    def force(y,graph=False):
        S,_,_,_=model.point_seats(y,.65)
        return diagonal_grad(S,y,create_graph=graph)
    y=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    G=force(y,True)
    J=torch.stack([torch.autograd.grad(G[i],y,retain_graph=True)[0] for i in range(2)]).detach().numpy()
    e=np.array([1.,-1.])/np.sqrt(2); common=np.ones(2)/np.sqrt(2)
    jp=np.empty((2,2))
    for i in range(2):
        plus=np.zeros(2); plus[i]=step
        gp=force(torch.tensor(plus,requires_grad=True)).detach().numpy()
        gm=force(torch.tensor(-plus,requires_grad=True)).detach().numpy()
        jp[:,i]=(gp-gm)/(2*step)
    return dict(g=g,rule=rule,nx=nx,configuration='homogeneous' if homogeneous else 'geographic',
                lambda_div=float(e@J@e),lambda_common=float(common@J@common),
                center_force=float(torch.max(torch.abs(G)).detach()),
                central_difference_error=float(np.max(np.abs(jp-J))),
                euler_multiplier=float(1+.15*(e@J@e)))
