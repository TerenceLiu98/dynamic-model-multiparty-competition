"""Exact consideration-set probabilities and their acceptance Jacobian (Eqs. 4, 29).

All arrays are float64. M[x, ell, i] = derivative of p_ell with respect to a_i.
The polynomial integral uses ceil(K/2) Gauss--Legendre nodes, with no subset
sampling or division by an acceptance probability.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=64)
def choice_quadrature(K: int):
    z, w = np.polynomial.legendre.leggauss((K + 1)//2)
    return (z+1)/2, w/2


def choice_and_jacobian(a: np.ndarray):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError('a must have shape (voter nodes, K), K>=2')
    if not np.all(np.isfinite(a)) or np.any(a < -1e-12) or np.any(a > 1+1e-12):
        raise ValueError('acceptance probabilities must be finite and lie in [0,1]')
    nx, K = a.shape
    if K == 2:
        own = 1 - .5*a[:, ::-1]
        p = a*own
        M = np.empty((nx,2,2))
        M[:,0,0] = own[:,0]; M[:,1,1] = own[:,1]
        M[:,0,1] = -.5*a[:,0]; M[:,1,0] = -.5*a[:,1]
        return p, np.prod(1-a,axis=1), M
    z, w = choice_quadrature(K)
    factors = 1 - z[:,None,None]*a[None,:,:]
    p=np.empty_like(a); M=np.empty((nx,K,K))
    for ell in range(K):
        others=[j for j in range(K) if j!=ell]
        own = w @ np.prod(factors[:,:,others],axis=2)
        p[:,ell]=a[:,ell]*own; M[:,ell,ell]=own
        for i in range(K):
            if i==ell: continue
            keep=[j for j in others if j!=i]
            partial=(w*z) @ np.prod(factors[:,:,keep],axis=2)
            M[:,ell,i]=-a[:,ell]*partial
    return p,np.prod(1-a,axis=1),M
