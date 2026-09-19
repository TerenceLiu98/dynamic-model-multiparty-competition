#!/usr/bin/env python3
"""Benchmark effective electoral and parliamentary party numbers."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from model import simulate_point
BASE = dict(K=5, D=9, nx=61, L=4.0, threshold=0.05, thresh_kappa=80.0,
            beta=24.0, mmp_alpha=0.6, mmp_gamma=45.0,
            geo=0.55, sigma=0.65, steps=280, dt=0.15, init_scale=0.12)
RULES = ["PR", "FPTP", "MMP"]
def inv_concentration(x):
    x = np.asarray(x, float); x = x/x.sum(); return float(1.0/np.sum(x*x))
rows=[]
for rule in RULES:
    for seed in range(5):
        out=simulate_point(rule=rule, seed=seed, **BASE)
        seats=np.asarray(out['seats'], float)
        v=np.asarray(out['v'], float)
        national_v=v.mean(axis=0)
        positions=np.asarray(out['positions'], float)
        sw=seats/seats.sum(); mean=float(np.sum(sw*positions))
        P=float(np.sqrt(np.sum(sw*(positions-mean)**2)))
        enep=inv_concentration(national_v); enpp=inv_concentration(seats)
        rows.append(dict(rule=rule,seed=seed,P=P,ENEP=enep,ENPP=enpp,
                         gap_ENEP_minus_ENPP=enep-enpp,abstention=float(out['abstention'])))
df=pd.DataFrame(rows); df.to_csv(ROOT/'results'/'effective_parties_benchmark.csv',index=False)
summary=df.groupby('rule').agg(P_mean=('P','mean'),P_sd=('P','std'),ENEP_mean=('ENEP','mean'),ENEP_sd=('ENEP','std'),ENPP_mean=('ENPP','mean'),ENPP_sd=('ENPP','std'),gap_mean=('gap_ENEP_minus_ENPP','mean'),gap_sd=('gap_ENEP_minus_ENPP','std'))
summary.to_csv(ROOT/'results'/'effective_parties_benchmark_summary.csv')
print(summary.to_string())
