"""Portable entrypoints and critical-relaxation diagnostics."""
from pathlib import Path
import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def critical_module():
    module_path = ROOT / 'code' / 'run_critical_diagnostics.py'
    assert module_path.is_file(), 'Critical diagnostics must be available in code/'
    spec = importlib.util.spec_from_file_location('run_critical_diagnostics', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_does_not_require_the_manuscript_pdf(tmp_path, monkeypatch):
    import repro_utils
    (tmp_path / 'code').mkdir()
    (tmp_path / 'code' / 'example.py').write_text('x = 1\n')
    monkeypatch.setattr(repro_utils, 'ROOT', tmp_path)
    try:
        data = repro_utils.environment()
    except FileNotFoundError:
        pytest.fail('Runtime metadata must not require an external manuscript PDF')
    assert data['python']
    assert 'code/example.py' in data['code_sha256']
    assert not Path(data.get('executable', '')).is_absolute()


@pytest.mark.parametrize('name', [
    'run_all.py', 'run_theta_phase.py', 'run_distributional_phase.py',
    'run_geography.py', 'run_point_thresholds.py',
    'run_critical_diagnostics.py', 'make_all_figures.py',
    'verify_reproduction.py',
])
def test_help_from_an_unrelated_working_directory(name, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'code' / name), '--help'],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '--output-dir' in result.stdout


def test_cubic_fit_recovers_the_analytic_decay_law():
    diagnostics = critical_module()
    time = np.linspace(0., 100., 101)
    beta, amplitude = .62, .05
    spread = 1 / np.sqrt(amplitude ** -2 + 2 * beta * time)
    data = pd.DataFrame({'time': time, 'P': spread})
    fit = diagnostics.fit_cubic_decay(data, start_time=50.)
    assert fit['beta_timeseries'] == pytest.approx(beta, abs=1e-10)
    assert fit['inverse_P2_intercept'] == pytest.approx(amplitude ** -2, abs=1e-9)


@pytest.mark.parametrize('spreads', [[0., .1], [np.nan, .1], [.1, np.inf]])
def test_cubic_fit_rejects_zero_or_nonfinite_amplitudes(spreads):
    diagnostics = critical_module()
    data = pd.DataFrame({'time': [50., 100.], 'P': spreads})
    with pytest.raises(ValueError):
        diagnostics.fit_cubic_decay(data, start_time=50.)


def test_common_state_diagnostics_do_not_divide_by_zero():
    diagnostics = critical_module()
    from distributional_phase import solve_diagonal_state
    state = solve_diagonal_state(2, .6, .002, theta=.4, ny=101, nx=240, tol=1e-13)
    measurements = diagnostics.diagnose_density(state, np.tile(state.f, (2, 1)))
    assert measurements['signed_a'] == 0.
    assert measurements['beta_rhs'] is None
    assert measurements['l1_distance_common'] == 0.
    assert measurements['rhs_l1'] < 1e-10
