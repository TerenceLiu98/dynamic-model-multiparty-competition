from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def test_plot_style_applies_science_and_ieee(monkeypatch) -> None:
    from plot_style import SCIENCEPLOTS_STYLES, apply_scienceplots_style

    applied_styles: list[list[str]] = []
    monkeypatch.setattr(plt.style, "use", lambda styles: applied_styles.append(styles))

    apply_scienceplots_style()

    assert SCIENCEPLOTS_STYLES == ("science", "ieee")
    assert applied_styles == [["science", "ieee"]]


def test_stylized_district_means_are_symmetric() -> None:
    from make_all_figures import (
        binary_district_response,
        geographic_electorate,
    )

    ideology, means, densities, national = geographic_electorate(9, .7)
    response_means = np.linspace(-.1, .1, 5)
    pr, fptp = binary_district_response(response_means)

    assert densities.shape == (9, len(ideology))
    assert np.allclose(np.trapezoid(densities, ideology, axis=1), 1, atol=1e-14)
    assert np.all(np.diff(means) > 0)
    assert np.allclose(means, -means[::-1], atol=1e-15)
    assert abs(means.mean()) < 1e-15
    assert np.allclose(national, national[::-1], atol=1e-15)
    assert pr[2] == pytest.approx(.5, abs=1e-14)
    assert fptp[2] == pytest.approx(.5, abs=1e-14)
    assert (fptp[3]-fptp[1]) > (pr[3]-pr[1])
