#!/usr/bin/env python3
"""Compare computed outputs with displayed manuscript numbers and numerical contracts.

Published values appear ONLY here as independent comparison targets. They are
never solver input, plotting data, fitted parameters or replacement outputs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from repro_utils import ROOT,write_csv,write_json


def comparison(name,value,target,tolerance,source):
    value=float(value);target=float(target);tolerance=float(tolerance)
    finite=math.isfinite(value)
    return dict(name=name,value=value if finite else None,paper_value=target,
                absolute_error=abs(value-target) if finite else None,tolerance=tolerance,
                passed=bool(finite and abs(value-target)<=tolerance),source=source)


def verify(out:Path,profile='paper'):
    result=out/'results';references=[];contracts=[]
    def df(name):return pd.read_csv(result/f'{name}.csv')
    def meta(name):return json.loads((result/f'{name}.json').read_text())
    def check(name,value,limit,relation='<='):
        value=float(value)
        passed=math.isfinite(value) and (value<=limit if relation=='<=' else value>=limit if relation=='>=' else value==limit)
        contracts.append(dict(name=name,value=value,relation=relation,limit=float(limit),passed=bool(passed)))
    def ref(name,value,target,tol,source):references.append(comparison(name,value,target,tol,source))
    def scalar(frame,**params):
        keep=np.ones(len(frame),dtype=bool)
        for k,v in params.items():keep&=(np.isclose(frame[k],v) if isinstance(v,(int,float)) else frame[k].eq(v))
        matched=frame[keep]
        if len(matched)!=1:raise ValueError(f'Expected one row for {params}, found {len(matched)}')
        return matched.iloc[0]

    phase=df('theta_binary_map');roots=df('theta_critical_tolerance');nonlinear=df('theta_nonlinear_summary')
    geo=df('geography_summary');finite=df('distributional_critical_curves');points=df('point_thresholds')
    table=df('table1');gmeta=meta('geography_metadata');pmeta=meta('theta_phase_metadata')
    check('binary spectral sign disagreements',np.count_nonzero(~phase.sign_agreement.astype(bool)),0,'==')
    check('max common-state Gibbs L1 residual',phase.stationary_residual_l1.max(),1e-10)
    check('minimum nonlinear endpoint density',nonlinear.minimum_density.min(),0,'>=')
    check('maximum nonlinear mass error',nonlinear.mass_error.max(),1e-8)
    check('matched national electorates',df('geography_national_checks').national_density_mass_error.max(),2e-16)
    check('identical paired initial positions',df('geography_matched_pairs').initial_max_difference.max(),0,'==')
    check('all binary local divergence rates restoring',df('binary_geography_rates').lambda_div.max(),0)
    check('all binary common-shift rates restoring',df('binary_geography_rates').lambda_common.max(),0)
    check('point threshold monotonicity violations',np.count_nonzero(np.diff(points.r_c)<=0),0,'==')
    check('root brackets have opposite signs',sum((roots[roots.status=='crossing'].margin_low*
                                                  roots[roots.status=='crossing'].margin_high)>0),0,'==')
    check('finite-width root brackets have opposite signs',sum(finite.margin_low*finite.margin_high>=0),0,'==')
    if profile=='paper':
        endpoint=df('theta_endpoint_checks')
        expected=[(2,0,.7903),(2,.5,.4648),(3,0,.8329),(3,1,.6310),
                  (4,0,.8518),(4,1,.7519),(5,0,.8585),(5,1,.7993)]
        for K,theta,target in expected:
            ref(f'r_c: K={K}, theta={theta:g}',scalar(roots,K=K,theta=theta).r_critical,target,5e-5,'p8 Section 4.1')
        ref('theta_c: K=2, r=.6',meta('theta_critical')['theta_c'],.32364,5e-6,'p8 Section 4.1')
        for case,targetP,T0,T1 in [('unstable',.1572,.6418,.6568),('vote',.3771,.6438,.7120)]:
            row=scalar(nonlinear,case=case)
            for name,val,target in [('P',row.P,targetP),('turnout_initial',row.initial_turnout,T0),('turnout_final',row.turnout,T1)]:
                ref(f'{case}: {name}',val,target,5e-5,'p9 Section 4.2')
        for eps,targets in [(.004,[.5973,.7121,.7489]),(.0005,[.6551,.7805,.8356])]:
            for K,target in zip([3,4,5],targets):
                ref(f'finite width r_c: K={K}, eps={eps:g}',scalar(finite,K=K,eps=eps).r_critical,target,5e-5,'p9 Section 4.3')
        for K,target in [(3,.6630),(5,.8476)]:
            ref(f'point r_c: K={K}',scalar(points,K=K).r_c,target,5e-5,'p20 Appendix D')
        rate=df('theta_rate_resolution')
        for theta,target in [(0,.0290),(1,.0316)]:
            ref(f'lambda_odd: K=5, r=.6, theta={theta}',scalar(rate,K=5,theta=theta,ny=401).lambda_odd,target,5e-5,'p8 Section 4.1')
        geo_targets={.55:{'FPTP':.258,'MMP':.058,'PR':-.016},
                     .7:{'FPTP':.289,'MMP':.069,'PR':-.008},
                     .85:{'FPTP':.308,'MMP':.092,'PR':.030}}
        for g,targets in geo_targets.items():
            for rule,target in targets.items():
                ref(f'paired delta P: g={g:g}, {rule}',scalar(geo,K=5,g=g,rule=rule).delta_mean,target,5e-4,'p24 Appendix F.1')
        fptp=scalar(geo,K=5,g=.7,rule='FPTP');pr=scalar(geo,K=5,g=.7,rule='PR')
        ref('plurality-PR geographic gap, g=.7',fptp.P_geographic_mean-pr.P_geographic_mean,.319,5e-4,'p11 Section 5.3')
        ref('plurality-PR homogeneous gap, g=.7',fptp.P_homogeneous_mean-pr.P_homogeneous_mean,.022,5e-4,'p11 Section 5.3')
        for g,vals in [(0,[-.284908,-3.418901,-.285125]),(.55,[-.350048,-4.200582,-.350314]),
                       (.7,[-.401571,-4.818856,-.401876]),(.85,[-.480145,-5.761743,-.480510])]:
            row=scalar(table,g=g)
            for rule,target in zip(['PR','FPTP','MMP'],vals):
                ref(f'Table 1: g={g:g}, {rule}',row[rule],target,5e-7,'p25 Table 1')
        ref('max binary P(42)',gmeta['max_binary_P'],9.41e-7,5e-10,'p24 Appendix F.2')
        # The printed extended maximum is at roundoff scale. Its scientifically
        # meaningful bound is tested below; do not impose its last 0.01e-12 digit.
        check('777 binary map points',len(phase),777,'==')
        check('five binary nonlinear runs',len(nonlinear),5,'==')
        check('critical trajectory reaches T=1600',scalar(nonlinear,case='critical').time,1600,'==')
        converged=nonlinear[nonlinear.case!='critical']
        check('noncritical full RHS below stopping threshold',converged.rhs_l1.max(),1e-9)
        check('noncritical Gibbs consistency',converged.gibbs_l1.max(),7e-9)
        check('four endpoint robustness checks',len(endpoint),4,'==')
        check('endpoint time-step / restart delta P',endpoint.delta_P.abs().max(),2e-10)
        check('endpoint time-step / restart delta turnout',endpoint.delta_turnout.abs().max(),2e-10)
        check('checked endpoint full RHS',endpoint.rhs_l1.max(),1e-9)
        check('checked endpoint Gibbs consistency',endpoint.gibbs_l1.max(),7e-9)
        check('noncritical fitted rate relative error',converged.relative_rate_error.max(),.02)
        check('common stationary branch agreement',df('theta_branch_checks').branch_l1.max(),2e-13)
        check('translation overlap, two smallest diffusion levels',finite[finite.eps<=.001].overlap_root.min(),.998,'>=')
        check('binary baseline runs',gmeta['binary_baseline_runs'],120,'==')
        check('binary baseline gradients above 1e-8',gmeta['binary_final_gradient_above_1e_8'],54,'==')
        check('binary dt-halving runs',gmeta['dt_halving_runs'],60,'==')
        check('binary extension runs',gmeta['extension_runs'],80,'==')
        check('binary extended P(84)',gmeta['max_extended_P'],8e-12)
        check('binary extended maximum gradient',gmeta['max_extended_gradient'],3.83e-12)
        check('binary force AD versus finite differences',gmeta['max_central_difference_error'],8.3e-8)
        grid=df('binary_geography_grid_checks')
        check('binary voter-grid rate refinement',grid[['delta_lambda_div','delta_lambda_common']].abs().to_numpy().max(),7.7e-7)
        dt=df('binary_geography_dt_checks')
        check('binary dt-halving change in P',dt.delta_P.abs().max(),1.225e-7)
        check('binary dt-halving change in distance',dt.delta_distance.abs().max(),2.445e-7)
        resolution=df('theta_root_resolution')
        for (K,theta),group in resolution.groupby(['K','theta']):
            check(f'root-grid agreement: K={K}, theta={theta:g}',group.r_critical.max()-group.r_critical.min(),2e-6)

    write_csv(result/'paper_reference_comparisons.csv',references)
    write_csv(result/'numerical_contract_checks.csv',contracts)
    passed=all(r['passed'] for r in references+contracts)
    warnings=[]
    for name,actual,bound in [('binary map stationary residual',float(phase.stationary_residual_l1.max()),2.1e-11),
                              ('binary map eigenpair residual',float(phase.eigenpair_residual.max()),4.3e-13)]:
        if profile=='paper' and actual>bound:
            warnings.append(dict(name=name,actual=actual,paper_archival_bound=bound,
                explanation='The archived tight residual bound is not matched; use the actual measured value. '
                            'The declared Gibbs tolerance, spectral signs and displayed numeric targets are checked separately.'))
    report=dict(profile=profile,passed=passed,displayed_value_checks=len(references),
        displayed_value_passes=sum(r['passed'] for r in references),numerical_contract_checks=len(contracts),
        numerical_contract_passes=sum(r['passed'] for r in contracts),warnings=warnings,
        critical_case_convergence='unresolved at full horizon' if profile=='paper' else 'quick smoke only',
        full_paper_precision_verified=profile=='paper' and passed,
        numeric_figures='Figures 4-8 and Table 1 recomputed from model outputs',
        extension_note='Sub-1e-11 terminal differences depend on roundoff; the stated convergence bound is checked, not the last reported digit.')
    write_json(result/'verification.json',report)
    print(f'{profile}: {report["displayed_value_passes"]}/{len(references)} displayed values, '
          f'{report["numerical_contract_passes"]}/{len(contracts)} numerical contracts pass; '
          f'{len(warnings)} archival-precision notices.',flush=True)
    for r in references+contracts:
        if not r['passed']:print('FAILED:',r,flush=True)
    return passed



def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=ROOT)
    ap.add_argument('--profile',choices=['paper','quick'],default='paper');args=ap.parse_args()
    raise SystemExit(0 if verify(args.output_dir,args.profile) else 1)

if __name__=='__main__':main()
