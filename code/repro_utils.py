"""Small, explicit helpers shared by reproduction entry points."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def json_default(value):
    if is_dataclass(value): return asdict(value)
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=json_default,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path,index=False,float_format='%.16g')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def environment():
    packages={}
    for name in ['numpy','scipy','pandas','matplotlib','SciencePlots','torch','pytest','PyMuPDF','threadpoolctl']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]='not installed'
    return dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                platform=platform.platform(),packages=packages,
                manuscript="Beyond Point Parties: Phase Transitions and Institutional Selection in Multiparty Competition",
                code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'code').glob('*.py'))})


def flatten_metrics(row):
    out={}
    for key,value in row.items():
        if isinstance(value,np.ndarray):
            if value.ndim==1:
                out.update({f'{key}_{i}':float(v) for i,v in enumerate(value)})
        elif isinstance(value,(str,float,int,bool,np.generic)):
            out[key]=value
    return out
