#!/usr/bin/env python3
"""Appendix D / Figure 8: positive-moment roots, never alternating sums."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from critical_asymptotics import hermite_nodes,critical_t,t_asymptotic,G
from repro_utils import ROOT,write_csv,write_json,environment


def run(out:Path,quick:bool=False):
    x,w=hermite_nodes(320)
    rows=[]
    for K in range(3,31 if quick else 5001):
        t=critical_t(float(K),x,w)
        rows.append(dict(K=K,r_c=float(np.sqrt(t)),t_c=t,
                         r_asymptotic=float(np.sqrt(t_asymptotic(K))),moment_residual=G(K,t,x,w)))
    write_csv(out/'results/point_thresholds.csv',rows)
    write_json(out/'results/point_thresholds_metadata.json',dict(**environment(),profile='quick' if quick else 'paper',
        quadrature_nodes=320,root_xtol=2e-13,number_of_roots=len(rows),
        monotonicity_violations=int(np.sum(np.diff([r['r_c'] for r in rows])<=0)),
        formula='Appendix D Eqs. 88 and 100; plotted asymptotic curve uses K>=20'))
    print(f'Computed {len(rows)} point thresholds; r_c(3)={rows[0]["r_c"]:.10f}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    run(args.output_dir or (ROOT/'runs/quick' if args.quick else ROOT),args.quick)

if __name__=='__main__':main()
