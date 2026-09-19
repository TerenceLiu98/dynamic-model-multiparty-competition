#!/usr/bin/env python3
"""Appendix G: institutional-payoff phase diagram, SG rates and binary endpoints.

Default settings are the paper profile. --quick is a separate smoke profile,
not evidence of manuscript-precision reproduction. No archived result is used
as solver input and no reference value is substituted for a computation.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from distributional_phase import solve_diagonal_state, odd_principal_eigenpair
from theta_dynamics import odd_growth_eigenpair, simulate_densities,simulate_stochastic_particles
from repro_utils import ROOT,write_json,write_csv,environment,flatten_metrics


@dataclass(frozen=True)
class ThetaSettings:
    ny:int=401
    nx:int=400
    L:float=2.2
    eps:float=.002
    state_tol:float=1e-10
    root_tol:float=2e-6
    dt:float=.1
    horizon:float=1600.
    profile:str='paper'

    def state(self,K,r,theta,initial=None,**overrides):
        opts=dict(ny=self.ny,nx=self.nx,L=self.L,tol=self.state_tol,theta=float(theta))
        opts.update(overrides)
        return solve_diagonal_state(K,float(r),self.eps,initial_density=initial,**opts)


def scan_critical_roots(K,theta,settings,grid=None):
    window=(.2,1.1) if K==2 else (.5,1.1)
    grid=np.linspace(*window,19) if grid is None else np.asarray(grid)
    cache={}
    def evaluate(r):
        key=float(r)
        if key not in cache:
            f=cache[min(cache,key=lambda t:abs(t-key))][0].f if cache else None
            s=settings.state(K,key,theta,f)
            cache[key]=(s,odd_principal_eigenpair(s))
        return cache[key][1].margin
    margins=np.array([evaluate(r) for r in grid]); brackets=[]
    for i in range(len(grid)-1):
        if margins[i]*margins[i+1]<0: brackets.append((float(grid[i]),float(grid[i+1])))
        elif margins[i]==0: brackets.append((float(grid[i]),float(grid[i])))
    if margins[-1]==0: brackets.append((float(grid[-1]),float(grid[-1])))
    rows=[]; states=[]
    for index,(lo,hi) in enumerate(brackets):
        root=lo if lo==hi else brentq(evaluate,lo,hi,xtol=settings.root_tol,rtol=1e-12)
        evaluate(root); s,e=cache[float(root)]
        rows.append(dict(K=K,theta=theta,eps=settings.eps,status='crossing',crossing_index=index,
                         r_critical=root,r_low=lo,r_high=hi,margin_low=evaluate(lo),margin_high=evaluate(hi),
                         margin_root=e.margin,nu=e.nu,variance=s.variance,translation_overlap=e.translation_overlap,
                         stationary_residual_l1=s.residual_l1,ny=settings.ny,nx=settings.nx,L=settings.L))
        states.append(s)
    if not rows:
        rows.append(dict(K=K,theta=theta,eps=settings.eps,status='no_crossing_in_window',crossing_index=-1,
                         r_critical=np.nan,r_low=window[0],r_high=window[1],margin_low=float(margins[0]),
                         margin_high=float(margins[-1]),margin_root=np.nan,ny=settings.ny,nx=settings.nx,L=settings.L))
    return rows,states


def critical_theta(settings):
    cache={}
    def margin(theta):
        key=float(theta)
        if key not in cache:
            s=settings.state(2,.6,key)
            cache[key]=odd_principal_eigenpair(s).margin
        return cache[key]
    # Tighter than tolerance-root plotting precision, to avoid artificial drift
    # during the critical T=1600 trajectory.
    root=brentq(margin,0,1,xtol=1e-11,rtol=1e-12)
    s=settings.state(2,.6,root)
    rate=odd_growth_eigenpair(s)
    return dict(K=2,r=.6,eps=settings.eps,theta_c=float(root),margin=float(margin(root)),
                lambda_odd=rate.rate,theta_solver_xtol=1e-11,settings=asdict(settings))


def run_phase(out,settings):
    result=out/'results'; result.mkdir(parents=True,exist_ok=True)
    theta_grid=np.linspace(0,1,21 if settings.profile=='paper' else 5)
    r_grid=np.linspace(.2,1.1,37 if settings.profile=='paper' else 7)
    rows=[]; density=[]; shapes=[]
    for theta in theta_grid:
        previous=None
        for r in r_grid:
            s=settings.state(2,r,theta,previous); previous=s.f
            e=odd_principal_eigenpair(s); g=odd_growth_eigenpair(s)
            rows.append(dict(K=2,theta=theta,r=r,eps=settings.eps,nu=e.nu,margin=e.margin,
                             lambda_odd=g.rate,eigenpair_residual=g.residual,
                             sign_agreement=bool(np.sign(g.rate)==np.sign(e.margin)),
                             stationary_residual_l1=s.residual_l1,variance=s.variance,turnout=s.turnout))
            density.append(s.f); shapes.append(g.density_vector)
        print(f'theta map: theta={theta:.2f}, {len(rows)} states',flush=True)
    write_csv(result/'theta_binary_map.csv',rows)
    np.savez_compressed(result/'theta_binary_map_states.npz',y=s.y,wy=s.wy,x=s.x,wx=s.wx,
                        parameters=np.array([[r['theta'],r['r']] for r in rows]),
                        f=np.stack(density),odd_growth_mode=np.stack(shapes))
    roots=[]; root_states=[]; root_params=[]
    for K in (2,3,4,5):
        for theta in theta_grid:
            rr,ss=scan_critical_roots(K,float(theta),settings,
                                    r_grid if K==2 else np.linspace(.5,1.1,13))
            roots.extend(rr)
            for root_state in ss:
                root_states.append(root_state.f); root_params.append([K,theta,root_state.r])
        print(f'critical curves: K={K} complete',flush=True)
    write_csv(result/'theta_critical_tolerance.csv',roots)
    np.savez_compressed(result/'theta_critical_profiles.npz',y=s.y,wy=s.wy,
                        parameters=np.array(root_params),f=np.stack(root_states))
    tc=critical_theta(settings);write_json(result/'theta_critical.json',tc)
    write_json(result/'theta_phase_metadata.json',dict(**environment(),settings=asdict(settings),
                grid_shape=[len(r_grid),len(theta_grid)],count=len(rows),
                sign_disagreements=sum(not r['sign_agreement'] for r in rows),
                max_stationary_residual=max(r['stationary_residual_l1'] for r in rows),
                max_eigenpair_residual=max(r['eigenpair_residual'] for r in rows),
                no_crossing_semantics='No root detected in stated scan window; not a claim outside that window.'))
    return tc


def run_nonlinear(out,settings,tc=None):
    result=out/'results';result.mkdir(parents=True,exist_ok=True)
    raw=result/'nonlinear';raw.mkdir(exist_ok=True)
    tc=critical_theta(settings) if tc is None else tc
    write_json(result/'theta_critical.json',tc)
    theta_c=tc['theta_c']; summary=[]; time_rows=[]; check_rows=[]
    labels=[('seat',1.),('stable',theta_c+.04),('critical',theta_c),('unstable',theta_c-.04),('vote',0.)]
    for name,theta in labels:
        s=settings.state(2,.6,theta,tol=1e-12)
        eigen=odd_growth_eigenpair(s)
        trajectory=simulate_densities(s,dt=settings.dt,horizon=settings.horizon,critical=name=='critical')
        metrics=flatten_metrics(trajectory.records[-1]); initial=trajectory.records[0]
        fitted=np.nan; err=np.nan
        # Explicit measurement window; sufficiently late to suppress initial
        # mode mixtures, early enough to remain in the local regime.
        selected=[r for r in trajectory.records if 5<=r['time']<=35 and r['P']>1e-12 and r['P']<.05]
        if len(selected)>=5 and abs(eigen.rate)>1e-7:
            fitted=float(np.polyfit([r['time'] for r in selected],np.log([r['P'] for r in selected]),1)[0])
            err=abs(fitted-eigen.rate)/abs(eigen.rate)
        row=dict(case=name,theta=theta,eps=s.eps,r=s.r,dt=settings.dt,steps=trajectory.steps,
                 stop_reason=trajectory.stop_reason,lambda_odd=eigen.rate,fitted_growth_rate=fitted,
                 relative_rate_error=err,initial_P=initial['P'],initial_turnout=initial['turnout'],**metrics)
        summary.append(row)
        time_rows.extend(dict(case=name,theta=theta,**flatten_metrics(record)) for record in trajectory.records)
        np.savez_compressed(raw/f'{name}.npz',y=s.y,wy=s.wy,x=s.x,wx=s.wx,common_density=s.f,
                            initial_densities=trajectory.initial_densities,final_densities=trajectory.final_densities,
                            theta=theta,eps=s.eps,r=s.r,times=trajectory.times,dt=settings.dt)
        print(f'nonlinear {name}: theta={theta:.9f}, t={metrics["time"]:g}, P={metrics["P"]:.9g}, '
              f'turnout={metrics["turnout"]:.9g}, residual={metrics["rhs_l1"]:.2g}',flush=True)
        if name in {'unstable','vote'} and settings.profile=='paper':
            initial_tilt=trajectory.final_densities*np.exp(np.array([.03,.01])[:,None]*s.y)
            initial_tilt/=(initial_tilt@s.wy)[:,None]
            experiments=[('dt_halved',settings.dt/2,None),('asymmetric_restart',settings.dt,initial_tilt)]
            for check_name,dt,f0 in experiments:
                check=simulate_densities(s,dt=dt,horizon=settings.horizon,initial=f0)
                terminal=flatten_metrics(check.records[-1])
                check_rows.append(dict(case=name,theta=theta,check=check_name,dt=dt,stop_reason=check.stop_reason,
                                       delta_P=terminal['P']-metrics['P'],delta_turnout=terminal['turnout']-metrics['turnout'],
                                       **terminal))
                np.savez_compressed(raw/f'{name}_{check_name}.npz',y=s.y,wy=s.wy,
                                    initial_densities=check.initial_densities,final_densities=check.final_densities,
                                    theta=theta,eps=s.eps,r=s.r,dt=dt)
                print(f'  check {check_name}: delta_P={check_rows[-1]["delta_P"]:.3g}, '
                      f'delta_turnout={check_rows[-1]["delta_turnout"]:.3g}',flush=True)
    write_csv(result/'theta_nonlinear_summary.csv',summary)
    write_csv(result/'theta_nonlinear_timeseries.csv',time_rows)
    if check_rows:write_csv(result/'theta_endpoint_checks.csv',check_rows)
    write_json(result/'theta_nonlinear_metadata.json',dict(**environment(),settings=asdict(settings),
               initial_condition='Eq.116, amplitude .02, z=(-1,1)',
               symmetry_projection_after_initialization=False,renormalisation_during_dynamics=False,
               stationarity='max partywise full SG RHS weighted L1 <1e-9 after t>=50; critical never early-stopped',
               growth_rate_fit='OLS log P vs time, 5<=t<=35, 1e-12<P<.05; only |lambda|>1e-7'))
    particle_count=200 if settings.profile=='paper' else 40
    particle_nodes=160 if settings.profile=='paper' else 60
    particle_horizon=800. if settings.profile=='paper' else settings.horizon
    particle_state=settings.state(2,.6,theta_c-.04,tol=1e-12,nx=particle_nodes)
    checked_seeds=range(12) if settings.profile=='paper' else range(4)
    particle_runs=[];trajectory_rows=[];seed_rows=[]
    for seed in checked_seeds:
        particle=simulate_stochastic_particles(
            particle_state,count=particle_count,dt=settings.dt,horizon=particle_horizon,
            seed=seed,record_every=max(1,int(round(1/settings.dt))))
        particle_runs.append(particle)
        trajectory_rows.extend(
            dict(seed=seed,time=time,mean_0=means[0],mean_1=means[1],common_shift=common,
                 differentiation=difference,turnout=turnout)
            for time,means,common,difference,turnout in zip(
                particle.times,particle.means,particle.common_shift,particle.differentiation,
                particle.turnout,strict=True))
        seed_rows.append(dict(seed=seed,terminal_common_shift=particle.common_shift[-1],
                              terminal_differentiation=particle.differentiation[-1],
                              terminal_turnout=particle.turnout[-1],
                              branch=int(np.sign(particle.differentiation[-1]))))
    write_csv(result/'theta_stochastic_trajectory.csv',trajectory_rows)
    representative=particle_runs[0]
    np.savez_compressed(raw/'stochastic_unstable.npz',
                        initial_particles=representative.initial_particles,
                        final_particles=np.stack([run.final_particles for run in particle_runs]),
                        seeds=np.asarray(list(checked_seeds)))
    write_csv(result/'theta_stochastic_seed_checks.csv',seed_rows)
    write_json(result/'theta_stochastic_metadata.json',dict(**environment(),K=2,r=.6,
               eps=particle_state.eps,theta=particle_state.theta,theta_c=theta_c,
               particle_count=particle_count,voter_nodes=particle_nodes,ideology_nodes=settings.ny,
               ideology_domain=[-settings.L,settings.L],dt=settings.dt,horizon=particle_horizon,
               highlighted_seeds=[0,1],checked_seeds=list(checked_seeds),
               record_every=max(1,int(round(1/settings.dt))),
               initial_condition='both parties share the same symmetric quantile approximation to f_theta^*',
               interpretation='one finite-particle realisation of Eq. (34); illustrative, not an estimator'))
    branches=[row['branch'] for row in seed_rows]
    print(f'stochastic trajectories: N={particle_count}, t={particle_horizon:g}, '
          f'positive={branches.count(1)}, negative={branches.count(-1)}',flush=True)


def run_checks(out,settings):
    result=out/'results';result.mkdir(parents=True,exist_ok=True)
    cases=[(2,0.),(2,.5),(3,1.),(5,1.)]
    refinement=[(301,240,2.2),(401,400,2.2),(601,600,2.6)] if settings.profile=='paper' else [(101,120,2.2)]
    roots=[];branch=[];rates=[];diffusion=[]
    for K,theta in cases:
        for ny,nx,L in refinement:
            cfg=ThetaSettings(ny=ny,nx=nx,L=L,state_tol=1e-12,root_tol=1e-8,profile=settings.profile)
            row,states=scan_critical_roots(K,theta,cfg)
            roots.extend(row)
            if ny==401:
                s=states[0]; differences=[]
                for width in (.06,.5):
                    initial=np.exp(-.5*(s.y/width)**2)
                    check=cfg.state(K,s.r,theta,initial,tol=1e-13)
                    differences.append(check)
                branch.append(dict(K=K,theta=theta,r=s.r,ny=ny,nx=nx,L=L,
                                   initial_width_narrow=.06,initial_width_broad=.5,
                                   branch_l1=float(s.wy@np.abs(differences[0].f-differences[1].f)),
                                   narrow_residual=differences[0].residual_l1,broad_residual=differences[1].residual_l1))
        for eps in (.001,.002,.004):
            cfg=ThetaSettings(ny=settings.ny,nx=settings.nx,L=settings.L,eps=eps,profile=settings.profile)
            row,_=scan_critical_roots(K,theta,cfg);diffusion.extend(row)
    for K,theta in [(2,0.),(2,.28),(2,.36),(2,1.),(5,0.),(5,1.)]:
        for ny in ([201,401,801] if settings.profile=='paper' else [101]):
            s=solve_diagonal_state(K,.6,.002,theta=theta,ny=ny,nx=settings.nx,L=settings.L)
            e=odd_growth_eigenpair(s)
            rates.append(dict(K=K,theta=theta,r=.6,eps=.002,ny=ny,nx=settings.nx,L=settings.L,
                              lambda_odd=e.rate,eigenpair_residual=e.residual))
    write_csv(result/'theta_root_resolution.csv',roots)
    if branch:write_csv(result/'theta_branch_checks.csv',branch)
    write_csv(result/'theta_rate_resolution.csv',rates)
    write_csv(result/'theta_diffusion_checks.csv',diffusion)
    write_json(result/'theta_checks_metadata.json',dict(**environment(),profile=settings.profile,
         case_selection='PDF specifies refinement grids but does not enumerate the four root cases; '
                        'this reproduction explicitly uses (K,theta)=(2,0),(2,.5),(3,1),(5,1).',
         branch_initial_widths=[.06,.5],rate_cases=[(2,0.),(2,.28),(2,.36),(2,1.),(5,0.),(5,1.)]))
    print('theta resolution, branch and diffusion checks complete',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--mode',choices=['all','phase','nonlinear','checks'],default='all')
    args=parser.parse_args()
    settings=ThetaSettings(ny=101,nx=120,horizon=60.,profile='quick') if args.quick else ThetaSettings()
    out=args.output_dir or (ROOT/'runs/quick' if args.quick else ROOT)
    start=time.perf_counter();tc=None
    if args.mode in {'all','phase'}:tc=run_phase(out,settings)
    if args.mode in {'all','nonlinear'}:run_nonlinear(out,settings,tc)
    if args.mode in {'all','checks'}:run_checks(out,settings)
    print(f'theta stage completed in {time.perf_counter()-start:.2f}s',flush=True)

if __name__=='__main__':main()
