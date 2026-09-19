"""Shared SciencePlots configuration for publication figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 - importing registers the bundled styles

SCIENCEPLOTS_STYLES = ("science", "ieee")


def apply_scienceplots_style() -> None:
    plt.style.use(list(SCIENCEPLOTS_STYLES))
