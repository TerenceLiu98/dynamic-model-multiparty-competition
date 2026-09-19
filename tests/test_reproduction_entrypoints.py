"""Entrypoint and provenance contracts, independent of expensive paper runs."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))

@pytest.mark.parametrize('name',['run_all.py','run_point_thresholds.py'])
def test_new_entrypoints_have_help(name):
    p=subprocess.run([sys.executable,str(ROOT/'code'/name),'--help'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert '--output-dir' in p.stdout

def test_quick_profile_cannot_default_to_paper_results():
    import run_all
    assert run_all.default_output('quick')==ROOT/'runs'/'quick'
    assert run_all.default_output('paper')==ROOT

def test_reference_checker_fails_a_wrong_result():
    from verify_reproduction import comparison
    assert comparison('test',.6630355,.6630,5e-5,'p20')['passed']
    assert not comparison('test',.67,.6630,5e-5,'p20')['passed']
    assert not comparison('test',np.nan,.6630,5e-5,'p20')['passed']
