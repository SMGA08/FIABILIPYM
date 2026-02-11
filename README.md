
# fiabilipym

Reliability system modeling and analysis with graph visualization.

`fiabilipym` is a modernized, **Python 3 compatible distribution of the original
`fiabilipy` library**, preserving the **same public API, mathematical models,
and numerical behavior**, while updating packaging, dependencies, and system
drawing. It also adds **Monte Carlo aging** with Weibull lifetimes for
simulation-based studies.

The goal of this project is **compatibility first**, modernization second.

---

## Compatibility with `fiabilipy`

`fiabilipym` is designed to be a **drop-in replacement** for `fiabilipy`.

**Guaranteed invariants:**
- Same public API (`Component`, `System`, `Markov`, `Voter`, …)
- Same system construction semantics (`E` → components → `S`)
- Same reliability / availability / MTTF mathematics
- Same symbolic and numeric results

Code written for **fiabilipy 2.x** runs unchanged (except for Python 2 → 3 syntax).

The examples below correspond directly to the *“How to build a system”* section
of the original **fiabilipy 2.4 documentation**.

---

## What it does

- Build reliability block diagrams
- Compute reliability, availability, maintainability, and MTTF
- Model systems using Markov processes
- Draw system graphs with Matplotlib
- Simulate aging with Weibull (Monte Carlo), yielding MTTF and R(t) curves

---

## Canonical examples (identical behavior)

### Building a component

```python
from fiabilipym import Component
from sympy import Symbol

t = Symbol("t", positive=True)
comp = Component("C0", 1e-4)

comp.mttf
# 10000.0

comp.reliability(1000)
# 0.904837418035960

comp.reliability(t)
# exp(-0.0001*t)

comp.reliability(t=100)
# 0.990049833749168
```
---

### Building a system (series)

```python
from fiabilipym import Component, System
from sympy import Symbol

t = Symbol("t", positive=True)

power = Component("P0", 1e-6)
motor = Component("M0", 1e-3)

S = System()
S["E"] = [power]
S[power] = [motor]
S[motor] = "S"

S.mttf
# 1000000/1001

S.reliability(t)
# exp(-1001*t/1000000)
```

---

## Visualization


## Why Graphviz/pygraphviz is no longer required

The project moved away from a hard Graphviz/pygraphviz requirement for day-to-day
usage and testing:

- Graphviz-based installs are fragile across environments (native binaries, headers, and PATH issues).
- Reproducibility was inconsistent across platforms, especially Windows/conda setups.
- For this project’s block-diagram needs, Graphviz introduced unnecessary operational overhead.
- A lightweight Matplotlib block-style approach now covers the primary visualization workflow.

`System.draw()` renders reliability block diagrams with Matplotlib.

```python
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend for CI/headless runs

import matplotlib.pyplot as plt
from fiabilipym import Component, System

motor = Component("M", 1e-4, 3e-2)
power = Component("P", 1e-6, 2e-4)

system = System()
system["E"] = [power]
system[power] = [motor]
system[motor] = "S"

system.draw()
plt.tight_layout()
plt.savefig("block_diagram.png", dpi=150)
```

Minimal Matplotlib-only block diagram example:

```python
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 2))
ax.plot([0.1, 0.9], [0.5, 0.5], color="black", linewidth=1.5)
ax.text(0.08, 0.5, "E", va="center", ha="right")
ax.text(0.92, 0.5, "S", va="center", ha="left")
ax.add_patch(plt.Rectangle((0.4, 0.4), 0.2, 0.2, fill=False, linewidth=1.5))
ax.text(0.5, 0.5, "C1", va="center", ha="center")
ax.set_axis_off()
fig.tight_layout()
fig.savefig("minimal_block_diagram.png", dpi=150)
```

---

## Monte Carlo aging (Weibull)

Use Weibull distributions to sample component lifetimes, then simulate system
MTTF and R(t) curves with Monte Carlo.

```python
import numpy as np
from fiabilipym import Component, System, Weibull, weibull_eta_from_lambda

beta = 2.0
lam0 = 1e-4
eta = weibull_eta_from_lambda(lam0, beta)
dist = Weibull(beta, eta)

C0 = Component("C0", lam0).with_distribution(dist)
C1 = Component("C1", lam0).with_distribution(dist)
C2 = Component("C2", lam0).with_distribution(dist)

S = System()
S["E"] = C0
S[C0] = [C1, C2]
S[C1] = "S"
S[C2] = "S"

T = np.linspace(0, 50000, 50)
mttf_hat, R_hat = S.monte_carlo(n=50000, grid_t=T, seed=42)
```

Voters (k-out-of-n) support Monte Carlo as well:

```python
import numpy as np
from fiabilipym import Component, Voter, Weibull, weibull_eta_from_lambda

beta = 2.0
lam0 = 1e-4
eta = weibull_eta_from_lambda(lam0, beta)
dist = Weibull(beta, eta)

base = Component("C", lam0).with_distribution(dist)
voter = Voter(base, M=2, N=3)

T = np.linspace(0, 50000, 50)
mttf_hat, R_hat = voter.monte_carlo(n=50000, grid_t=T, seed=42)
```

---

## Notebook-style demo (architectures + plots)

The included notebook `fiabilipym_weibull_montecarlo_demo.ipynb` builds the
five reference architectures and generates:

- MTTF bar plot
- Reliability R(t) curves
- Optional MTTF vs baseline lambda sweep (mean-matched)
- System drawings

If you want a code-only version, the snippet below mirrors the notebook's core:

```python
import numpy as np
import matplotlib.pyplot as plt
from fiabilipym import Component, System, Voter, Weibull, weibull_eta_from_lambda

beta = 2.0
lam0 = 1e-4
eta0 = weibull_eta_from_lambda(lam0, beta)

def aging_component(name, lam, beta, eta, age0=0.0):
    c = Component(name, lam).with_distribution(Weibull(beta, eta), age0=age0)
    return c

def build_series3(lam, beta, eta):
    c1, c2, c3 = (aging_component("C1", lam, beta, eta),
                 aging_component("C2", lam, beta, eta),
                 aging_component("C3", lam, beta, eta))
    S = System()
    S["E"] = c1
    S[c1] = c2
    S[c2] = c3
    S[c3] = "S"
    return S

def build_parallel3(lam, beta, eta):
    c1, c2, c3 = (aging_component("C1", lam, beta, eta),
                 aging_component("C2", lam, beta, eta),
                 aging_component("C3", lam, beta, eta))
    S = System()
    S["E"] = [c1, c2, c3]
    S[c1] = S[c2] = S[c3] = "S"
    return S

def build_series_parallel_3stages(lam, beta, eta):
    A1, A2 = aging_component("A1", lam, beta, eta), aging_component("A2", lam, beta, eta)
    B1, B2 = aging_component("B1", lam, beta, eta), aging_component("B2", lam, beta, eta)
    C1, C2 = aging_component("C1", lam, beta, eta), aging_component("C2", lam, beta, eta)
    S = System()
    S["E"] = [A1, A2]
    S[A1] = S[A2] = "N1"
    S["N1"] = [B1, B2]
    S[B1] = S[B2] = "N2"
    S["N2"] = [C1, C2]
    S[C1] = S[C2] = "S"
    return S

def build_parallel_series_3branches(lam, beta, eta):
    A1, A2 = aging_component("A1", lam, beta, eta), aging_component("A2", lam, beta, eta)
    B1, B2 = aging_component("B1", lam, beta, eta), aging_component("B2", lam, beta, eta)
    C1, C2 = aging_component("C1", lam, beta, eta), aging_component("C2", lam, beta, eta)
    S = System()
    S["E"] = [A1, B1, C1]
    S[A1] = A2
    S[A2] = "S"
    S[B1] = B2
    S[B2] = "S"
    S[C1] = C2
    S[C2] = "S"
    return S

def build_voter_2of3(lam, beta, eta):
    base = aging_component("C", lam, beta, eta)
    return Voter(base, M=2, N=3)

ARCH_BUILDERS = {
    "Series (3)": build_series3,
    "Parallel (3)": build_parallel3,
    "Series–Parallel (3 stages)": build_series_parallel_3stages,
    "Parallel–Series (3 branches)": build_parallel_series_3branches,
    "Voter 2-out-of-3": build_voter_2of3,
}

T = np.linspace(0, 3.0 / lam0, 250)
mttf = {}
R_curves = {}
for name, builder in ARCH_BUILDERS.items():
    obj = builder(lam0, beta, eta0)
    m, R = obj.monte_carlo(n=50000, grid_t=T, seed=42)
    mttf[name] = m
    R_curves[name] = R

plt.figure()
plt.bar(list(mttf.keys()), list(mttf.values()))
plt.xticks(rotation=25, ha="right")
plt.ylabel("MTTF (Monte Carlo)")
plt.title("MTTF by architecture")
plt.tight_layout()
plt.show()

plt.figure()
for name, R in R_curves.items():
    plt.plot(T, R, label=name)
plt.xlabel("Time t")
plt.ylabel("Reliability R(t)")
plt.ylim(0, 1.02)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Installation (uv)

```bash
uv venv
uv sync

uv pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple fiabilipym


```


## Tests

Install test dependencies (editable install):

```bash
pip install -e ".[test]"
```

Run with pytest:

```bash
pytest -q
```

Run with unittest discovery:

```bash
python -m unittest discover -s tests
```

Equivalent uv-based commands:

```bash
uv run --extra test pytest -q
```

Alternative (stdlib unittest):

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests
```

---

## Package structure

```
fiabilipym/
├── changes.md
├── pyproject.toml
├── setup.py
├── README.md
├── src/
│   └── fiabilipym/
│       ├── __init__.py
│       ├── component.py
│       ├── distribution.py
│       ├── markov.py
│       ├── system.py
│       └── voter.py
├── fiabilipym_weibull_montecarlo_demo.ipynb
└── tests/
    ├── test_markov.py
    ├── test_system.py
    ├── test_monte_carlo_systems.py
    └── test_voter_monte_carlo.py
```
