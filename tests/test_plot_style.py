from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def test_plot_style_applies_science_and_ieee(monkeypatch) -> None:
    from plot_style import SCIENCEPLOTS_STYLES, apply_scienceplots_style

    applied_styles: list[list[str]] = []
    monkeypatch.setattr(plt.style, "use", lambda styles: applied_styles.append(styles))

    apply_scienceplots_style()

    assert SCIENCEPLOTS_STYLES == ("science", "ieee")
    assert applied_styles == [["science", "ieee"]]
