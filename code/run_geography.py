#!/usr/bin/env python3
"""Recompute Figure 7, binary geographic convergence, and Table 1."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from geographic_experiments import make_model,NumpyPointModel,initial_positions,simulate_numpy_point,local_rates
from repro_utils import ROOT,write_csv,write_json,environment,flatten_metrics


def run(out,quick=False):
    result=out/'results';result.mkdir(parents=True,exist_ok=True)
    gs=[0.,.55,.7,.85] if not quick else [0.,.7]
    seeds=range(5) if not quick else range(1)
    horizon=42. if not quick else 3.
    baseline=[];trajectories=[];pair_rows=[];models={};runs={};national=[];raw={}
    for K in [5,2]:
        bundle_initial=[];bundle_positions=[];bundle_q=[];bundle_v=[];bundle_S=[];bundle_labels=[]
        for g in gs:
            for rule in ['PR','FPTP','MMP']:
                pair_models={}
                for hom in [False,True]:
                    m=make_model(K,g,rule,homogeneous=hom)
                    p=NumpyPointModel.from_torch(m);models[(K,g,rule,hom)]=p;pair_models[hom]=p
                ng=pair_models[False].dw@pair_models[False].W
                nh=pair_models[True].dw@pair_models[True].W
                national.append(dict(K=K,g=g,rule=rule,national_density_mass_error=float(np.max(np.abs(ng-nh))),
                                     national_sd=float(np.sqrt(ng@(pair_models[False].x**2))),
                                     reflection_error=float(np.max(np.abs(ng-ng[::-1])))))
                for seed in seeds:
                    y0=initial_positions(K,seed);comparison={}
                    for hom in [False,True]:
                        label='homogeneous' if hom else 'geographic'
                        r=simulate_numpy_point(pair_models[hom],y0,horizon=horizon)
                        runs[(K,g,rule,seed,hom)]=r;comparison[hom]=r
                        row=dict(K=K,g=g,rule=rule,seed=seed,configuration=label,dt=.15,**flatten_metrics(r))
                        baseline.append(row)
                        trajectories.extend(dict(K=K,g=g,rule=rule,seed=seed,configuration=label,
                                                 **flatten_metrics(t)) for t in r['trajectory'])
                        bundle_initial.append(y0);bundle_positions.append(r['positions'])
                        bundle_q.append(r['q']);bundle_v.append(r['v']);bundle_S.append(r['seats'])
                        bundle_labels.append([g,['PR','FPTP','MMP'].index(rule),seed,int(hom)])
                    pair_rows.append(dict(K=K,g=g,rule=rule,seed=seed,P_geographic=comparison[False]['P'],
                                          P_homogeneous=comparison[True]['P'],
                                          delta_P=comparison[False]['P']-comparison[True]['P'],
                                          initial_max_difference=float(np.max(np.abs(comparison[False]['initial_positions']-
                                                                                      comparison[True]['initial_positions'])))))
                print(f'geography K={K}, g={g:.2f}, {rule} complete',flush=True)
        np.savez_compressed(result/f'geography_K{K}_raw.npz',labels=np.array(bundle_labels),
                            label_columns=np.array(['g','rule_index_PR_FPTP_MMP','seed','homogeneous']),
                            initial_positions=np.array(bundle_initial),positions=np.array(bundle_positions),
                            q=np.array(bundle_q),v=np.array(bundle_v),seats=np.array(bundle_S))
    write_csv(result/'geography_baseline.csv',baseline)
    write_csv(result/'geography_trajectories.csv',trajectories)
    pairs=pd.DataFrame(pair_rows);write_csv(result/'geography_matched_pairs.csv',pairs)
    write_csv(result/'geography_national_checks.csv',national)
    summary=pairs.groupby(['K','g','rule'],sort=True).agg(
        P_geographic_mean=('P_geographic','mean'),P_homogeneous_mean=('P_homogeneous','mean'),
        delta_mean=('delta_P','mean'),delta_sd=('delta_P','std'),n=('seed','count')).reset_index()
    if quick:summary['delta_sd']=summary['delta_sd'].fillna(0.)
    write_csv(result/'geography_summary.csv',summary)

    # Actual AD of the full strategic force, not a formula assuming binary cancellation.
    rates=[]
    for g in gs:
        for rule in ['PR','FPTP','MMP']:
            for hom in [False,True]:rates.append(local_rates(g,rule,homogeneous=hom))
    rate_frame=pd.DataFrame(rates);write_csv(result/'binary_geography_rates.csv',rate_frame)
    table=rate_frame[rate_frame.configuration=='geographic'].pivot(index='g',columns='rule',values='lambda_div')[['PR','FPTP','MMP']]
    write_csv(result/'table1.csv',table.reset_index())
    table_rows=['\\begin{tabular}{rrrr}','\\hline',' $g$ & PR & Plurality & MMP \\\\', '\\hline']
    table_rows.extend(f'{g:.2f} & {r.PR:.6f} & {r.FPTP:.6f} & {r.MMP:.6f} \\\\' for g,r in table.iterrows())
    table_rows.extend(['\\hline','\\end{tabular}'])
    (result/'table1.tex').write_text('\n'.join(table_rows)+'\n')
    dt_rows=[];extended=[];resolution=[]
    if not quick:
        for g in [.7,.85]:
            for rule in ['PR','FPTP','MMP']:
                for hom in [False,True]:
                    new=local_rates(g,rule,nx=121,homogeneous=hom)
                    old=next(r for r in rates if r['g']==g and r['rule']==rule and r['configuration']==new['configuration'])
                    resolution.append(dict(**new,delta_lambda_div=new['lambda_div']-old['lambda_div'],
                                           delta_lambda_common=new['lambda_common']-old['lambda_common']))
                    for seed in seeds:
                        original=runs[(2,g,rule,seed,hom)]
                        r=simulate_numpy_point(models[(2,g,rule,hom)],initial_positions(2,seed),dt=.075,horizon=42.,record_every=0)
                        dt_rows.append(dict(g=g,rule=rule,seed=seed,configuration='homogeneous' if hom else 'geographic',
                                            dt=.075,delta_P=r['P']-original['P'],delta_distance=r['distance']-original['distance'],
                                            **flatten_metrics(r)))
        for g in gs:
            for rule in ['PR','FPTP','MMP']:
                for seed in seeds:
                    pair=[runs[(2,g,rule,seed,hom)] for hom in [False,True]]
                    if any(r['final_max_gradient']>1e-8 for r in pair):
                        for hom in [False,True]:
                            r=simulate_numpy_point(models[(2,g,rule,hom)],initial_positions(2,seed),horizon=84.,record_every=0)
                            extended.append(dict(g=g,rule=rule,seed=seed,configuration='homogeneous' if hom else 'geographic',
                                                 dt=.15,**flatten_metrics(r)))
        write_csv(result/'binary_geography_dt_checks.csv',dt_rows)
        write_csv(result/'binary_geography_extensions.csv',extended)
        write_csv(result/'binary_geography_grid_checks.csv',resolution)
    binary=[r for r in baseline if r['K']==2]
    write_json(result/'geography_metadata.json',dict(**environment(),profile='quick' if quick else 'paper',
               K=[5,2],D=9,nx=61,L=4.,sigma=.65,geography=gs,seeds=list(seeds),dt=.15,horizon=horizon,
               beta=24,threshold=.05,threshold_kappa=80,mmp_alpha=.6,mmp_gamma=45,
               initialization='sorted .12*torch.randn(K), CPU float64, seed 0..4',
               error_bars='sample standard deviation of the five paired differences, ddof=1',
               early_stopping=False,baseline_runs=len(baseline),binary_baseline_runs=len(binary),
               max_binary_P=max(r['P'] for r in binary),
               binary_final_gradient_above_1e_8=sum(r['final_max_gradient']>1e-8 for r in binary),
               dt_halving_runs=len(dt_rows),extension_runs=len(extended),
               max_extended_P=max([r['P'] for r in extended],default=0.),
               max_extended_gradient=max([r['final_max_gradient'] for r in extended],default=0.),
               max_central_difference_error=max(r['central_difference_error'] for r in rates)))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    out=args.output_dir or (ROOT/'runs/quick' if args.quick else ROOT)
    t=time.perf_counter();run(out,args.quick);print(f'geography completed in {time.perf_counter()-t:.2f}s',flush=True)

if __name__=='__main__':main()
