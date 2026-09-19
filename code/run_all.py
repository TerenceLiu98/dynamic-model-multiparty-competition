#!/usr/bin/env python3
"""One command: tests, computations, figures and independent paper-value checks.

All numerical stages run afresh from the model specification. The default
paper profile reproduces the manuscript grids. Quick outputs are segregated and
explicitly do not establish paper-precision agreement.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from repro_utils import ROOT,environment,write_json,sha256


def default_output(profile:str)->Path:
    if profile not in ['paper','quick']:raise ValueError('Unknown reproduction profile')
    return ROOT if profile=='paper' else ROOT/'runs'/'quick'


def run(profile:str,out:Path,skip_tests=False):
    out=out.resolve();out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    if profile=='quick' and out==ROOT:raise ValueError('Quick runs may not overwrite the default paper output directory')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    py=sys.executable;code=ROOT/'code';quick=['--quick'] if profile=='quick' else []
    stages=[]
    if not skip_tests:stages.append(('tests',[py,'-m','pytest','-q','-ra']))
    for name,extra in [('run_theta_phase',[]),('run_distributional_phase',['--no-figures']),
                       ('run_geography',[]),('run_point_thresholds',[])]:
        stages.append((name,[py,str(code/f'{name}.py'),'--output-dir',str(out),*quick,*extra]))
    for name in ['make_all_figures']:
        stages.append((name,[py,str(code/f'{name}.py'),'--output-dir',str(out)]))
    stages.append(('verify_reproduction',[py,str(code/'verify_reproduction.py'),'--output-dir',str(out),'--profile',profile]))
    manifest=dict(**environment(),profile=profile,status='running',output_directory=Path(os.path.relpath(out, ROOT)).as_posix(),
                  threads=1,test_stage_skipped=bool(skip_tests),stages=[])
    path=out/'results/run_manifest.json';start=time.perf_counter()
    write_json(path,manifest)
    for name,command in stages:
        log=out/'logs'/f'pipeline_{name}.log'
        print(f'\n=== {name} ({profile}) ===',flush=True)
        t=time.perf_counter()
        recorded_command = [
            'python' if token == py else Path(os.path.relpath(token, ROOT)).as_posix()
            if Path(token).is_absolute() else token for token in command
        ]
        with log.open('w',encoding='utf-8') as stream:
            stream.write('Command: '+' '.join(recorded_command)+'\n\n');stream.flush()
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,bufsize=1,encoding='utf-8',errors='replace')
            assert p.stdout is not None
            for line in p.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            rc=p.wait()
        manifest['stages'].append(dict(name=name,command=recorded_command,exit_code=rc,
                                       elapsed_seconds=time.perf_counter()-t,log=str(log.relative_to(out))))
        if rc!=0:
            manifest.update(status='failed',failed_stage=name,elapsed_seconds=time.perf_counter()-start)
            write_json(path,manifest)
            raise RuntimeError(f'{name} failed (exit {rc}); see {log}')
        write_json(path,manifest)
    manifest.update(status='success',elapsed_seconds=time.perf_counter()-start,
        result_sha256={str(p.relative_to(out)):sha256(p) for p in sorted((out/'results').rglob('*'))
                       if p.is_file() and p!=path})
    write_json(path,manifest)
    print(f'\nCompleted {profile} pipeline in {manifest["elapsed_seconds"]:.2f}s. '
          f'Checks: {out/"results/verification.json"}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',choices=['paper','quick'],default='paper')
    ap.add_argument('--output-dir',type=Path,help='default: project root for paper; runs/quick for quick')
    ap.add_argument('--skip-tests',action='store_true',help='explicitly skip the otherwise mandatory test stage')
    args=ap.parse_args();run(args.profile,args.output_dir or default_output(args.profile),args.skip_tests)

if __name__=='__main__':main()
