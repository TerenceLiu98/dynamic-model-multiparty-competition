# Code for DWCW

Numerical code for the paper *Diversity Within, Competition Without: Phase Transitions and Institutional Selection in Multiparty Dynamics*. The model represents parties as ideological distributions and combines voter acceptance, endogenous turnout, electoral payoffs and drift–diffusion dynamics.

The experiments reproduce the mixed-objective phase boundaries, binary density trajectories, finite-width stability analysis, matched-electorate geographic comparisons and point-platform thresholds in Figures 4–8 and Table 1. All electorates and initial party states are synthetic.

## Structure

```text
beyond-point-parties/
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── pytest.ini
├── .gitignore
├── code/
│   ├── run_all.py                     # Complete reproduction pipeline
│   ├── run_theta_phase.py             # Payoff weights and binary dynamics
│   ├── run_distributional_phase.py    # Finite-width stability and diffusion
│   ├── run_geography.py               # Geographic comparisons and Table 1
│   ├── run_point_thresholds.py        # Point-platform critical thresholds
│   ├── run_critical_diagnostics.py    # Critical-relaxation diagnostics
│   ├── make_all_figures.py            # Figures and PDF overview
│   ├── verify_reproduction.py        # Numerical reference checks
│   ├── model.py                      # Point and particle models
│   ├── distributional_phase.py       # Common states and stability operators
│   ├── theta_dynamics.py              # Full-density evolution
│   ├── electoral_feedback.py         # Exact choice probabilities
│   ├── geographic_experiments.py     # Paired electorates and measurements
│   ├── critical_asymptotics.py        # Critical-threshold asymptotics
│   ├── effective_parties.py          # Effective party-number measures
│   ├── repro_utils.py                # Output and provenance utilities
│   └── plot_style.py                 # Plot settings
├── tests/                            # Model and entrypoint tests
├── results/                          # Generated CSV, NPZ and JSON outputs (ignored)
│   ├── nonlinear/                    # Full density states
│   └── critical/                     # Additional relaxation diagnostics
└── figures/                          # Generated figures and overview PDF (ignored)
```

Only source code, tests, dependency specifications and project configuration are version controlled. Generated numerical results, figures, logs and local experiment runs remain on disk but are excluded from Git.

## Installation

Use Python 3.11 or newer and a virtual environment. The tested versions are recorded in `requirements-lock.txt`; the wider dependency bounds are in `requirements.txt`. A CPU is sufficient, and no manuscript PDF or external data download is required. Figure generation uses SciencePlots with the `science` and `ieee` styles, which require a working LaTeX installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run

From the repository root:

```bash
python code/run_all.py --profile paper
python code/run_critical_diagnostics.py
```

The first command runs the tests, recomputes the numerical experiments, generates the figures and checks the results against the manuscript. The second runs the seven supplementary critical-relaxation experiments. Numerical settings, source hashes and validation results are saved as JSON and CSV, including `results/verification.json` and `results/run_manifest.json`. Runtime logs are written to the ignored `logs/` directory.

For a small smoke run, use `python code/run_all.py --profile quick`. Its outputs are isolated in `runs/quick/` and do not establish paper-precision agreement. Use `--output-dir PATH` to select another output location. For critical diagnostics on that run, pass `--results-dir PATH/results --output-dir PATH/results/critical` after a paper-profile run.

Run tests alone with `python -m pytest -q`. After generating the paper-profile outputs, check them without recomputing by running `python code/verify_reproduction.py --profile paper`.

The critical trajectory remains a finite-time result at `t = 1600`; the additional diagnostics quantify slow relaxation without relabelling it as a converged endpoint.
