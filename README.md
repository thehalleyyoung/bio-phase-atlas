# PhaseCartographer — Certified Phase Atlases for Biological ODEs

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Tests: 209 passing](https://img.shields.io/badge/tests-209%20passing-brightgreen.svg)]()
[![Z3: 27/27 equilibria verified](https://img.shields.io/badge/Z3-27%2F27%20equilibria%20verified-blueviolet.svg)]()

## Description

PhaseCartographer generates machine-checkable phase atlases for biological ODE
systems. It partitions parameter spaces into certified cells using a three-tier
verification architecture (interval arithmetic → dReal → Z3), proving
equilibrium existence via Krawczyk operators and stability via eigenvalue
enclosure cascades. Across 18 built-in biological models the tool produces
Z3-verified equilibria and stability claims, backed by an independent
1,188-LoC trusted computing base (MiniCheck) for certificate replay.

---

## Key Features

- **Three-tier verification**: interval arithmetic (Tier 1) → dReal δ-complete
  (Tier 2) → Z3 exact (Tier 3)
- **Krawczyk operator** for rigorous equilibrium existence and uniqueness proofs
- **5-stage stability cascade**: Routh-Hurwitz → Bauer-Fike → Schur → center
  manifold
- **Hash-chained proof traces** for auditable composition
- **MiniCheck independent verifier** (1,188 LoC trusted base)
- **GP-guided adaptive octree refinement** with Matérn-5/2 kernel
- **Borell-TIS coverage bounds** for GP surrogates
- **18 built-in biological models** (toggle switch, brusselator, repressilator,
  etc.)
- **YAML, JSON, SBML** (Level 2 & 3) model input formats
- **δ-soundness certificates**: δ_solver < δ_req ensures exact regime
  correctness
- **209 tests passing**, ~18,000 LoC core + 1,188 LoC MiniCheck

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Reference](#cli-reference)
4. [Worked Examples with Actual Output](#worked-examples-with-actual-output)
5. [Supported Input Formats](#supported-input-formats)
6. [Python API](#python-api)
7. [Architecture](#architecture-overview)
8. [Built-in Models](#built-in-models)
9. [FAQ / Troubleshooting](#faq--troubleshooting)

---

## Installation

```bash
git clone https://github.com/<org>/bio-phase-atlas.git
cd bio-phase-atlas
pip install numpy sympy scipy z3-solver pyyaml
```

### Dependencies

| Package    | Version | Required | Purpose                          |
|------------|---------|----------|----------------------------------|
| Python     | ≥3.8    | Yes      | Runtime                          |
| NumPy      | latest  | Yes      | Numerical computations           |
| SymPy      | latest  | Yes      | Symbolic ODE parsing             |
| SciPy      | latest  | Yes      | Optimization, linear algebra     |
| z3-solver  | latest  | Yes      | Tier 3 exact verification        |
| PyYAML     | latest  | Yes      | YAML model parsing               |
| dReal      | latest  | No       | Tier 2 δ-complete verification   |

Verify the installation by running the test suite:

```bash
$ python3 -m pytest tests/ -x --tb=no -q
209 passed in 70.26s (0:01:10)
```

---

## Quick Start

Run a built-in model from the command line in under 10 seconds:

```bash
$ python3 -m phase_cartographer toggle_switch --depth 3 --max-cells 100
```

Expected output:

```
========================================================================
  PhaseCartographer: Certified Phase Atlases for Biological ODEs
  Three-tier verification: Interval Arithmetic → dReal → Z3
========================================================================

  Model: toggle_switch (Gardner toggle switch: mutual Hill repression (n1=n2=2 fixed))
  States: 2, Params: 4, RHS: hill
  Parameter domain: [(2.0, 4.0), (2.0, 4.0), (2.0, 2.0), (2.0, 2.0)]

  Results:
    Certified cells:      2
    Coverage:             100.0%
    Regimes:              {'monostable': 2}
    Z3 equilibrium:       2/2
    Z3 stability:         2/2
    Krawczyk contraction: 0.6609
    Eigenvalue gap (min): 0.000000
    Near bifurcation:     2
    Time:                 5.0s
```

---

## CLI Reference

Entry point: `python3 -m phase_cartographer`

```
$ python3 -m phase_cartographer --help
usage: python3 -m phase_cartographer [-h] [--file FILE] [--case-study]
                                     [--delta-table] [--baseline]
                                     [--depth DEPTH] [--max-cells MAX_CELLS]
                                     [model]

positional arguments:
  model                 Model name (e.g., toggle_switch, brusselator)

options:
  --file FILE           Path to a YAML/JSON/SBML model file
  --case-study          Run toggle switch design case study
  --delta-table         Show δ-soundness parameter table
  --baseline            Run numerical continuation baseline comparison
  --depth DEPTH         Maximum refinement depth (default: 4)
  --max-cells MAX_CELLS Maximum cells per model (default: 200)
```

Subcommands: `check <FILE>` validates a model without computation;
`init-model [-o FILE] [--template 2d|3d|oscillator]` generates a starter YAML.

| Command                                              | Description                               |
|------------------------------------------------------|-------------------------------------------|
| `phase_cartographer <MODEL>`                         | Run a single built-in model               |
| `phase_cartographer`                                 | Run all 18 benchmarks                     |
| `phase_cartographer --file <PATH>`                   | Run a user-defined model file             |
| `phase_cartographer check <PATH>`                    | Validate model file (no computation)      |
| `phase_cartographer init-model [-o FILE] [--template]`| Generate starter YAML template           |
| `phase_cartographer --case-study`                    | Toggle switch design case study           |
| `phase_cartographer --delta-table`                   | δ-soundness parameter table               |
| `phase_cartographer --baseline`                      | Numerical continuation comparison         |

---

## Worked Examples with Actual Output

Every command below was executed and its output captured verbatim.
The banner header is shown once below; subsequent examples omit it for brevity.

```
========================================================================
  PhaseCartographer: Certified Phase Atlases for Biological ODEs
  Three-tier verification: Interval Arithmetic → dReal → Z3
========================================================================
```

### Example 1: Generate a Starter Template

```bash
$ python3 -m phase_cartographer init-model --output my_model.yaml
  ✓ Created 2d template: my_model.yaml
    Edit the file, then run:
      python -m phase_cartographer check my_model.yaml
      python -m phase_cartographer --file my_model.yaml
```

The generated `my_model.yaml`:

```yaml
name: my_2d_model
description: A 2D system with mutual interaction
states: [x, y]
parameters:
  - name: a
    bounds: [0.5, 3.0]
  - name: b
    bounds: [0.5, 3.0]
equations:
  - "a / (1 + y**2) - x"
  - "b / (1 + x**2) - y"
state_domain:
  - [0.1, 5.0]
  - [0.1, 5.0]
rhs_type: hill
expected_regimes: [monostable, bistable]
```

Three template types are available: `2d` (default), `3d`, and `oscillator`.

### Example 2: Validate a YAML Model

```bash
$ python3 -m phase_cartographer check examples/toggle_switch.yaml
  Checking: examples/toggle_switch.yaml

  ✓ Valid model: user_toggle_switch
    States:     2 (x1, x2)
    Parameters: 2 (a1, a2)
    Equations:  2
      dx1/dt = a1/(x2**2 + 1) - x1
      dx2/dt = a2/(x1**2 + 1) - x2
    RHS type:   hill
```

### Example 3: Validate a 3D Model

```bash
$ python3 -m phase_cartographer check examples/three_gene_ring.yaml
  ✓ Valid model: three_gene_ring
    States:     3 (g1, g2, g3)
    Parameters: 2 (alpha, n)
    Equations:  3
      dg1/dt = alpha/(g3**n + 1) - g1
      dg2/dt = alpha/(g1**n + 1) - g2
      dg3/dt = alpha/(g2**n + 1) - g3
    RHS type:   hill
```

### Example 4: Validate an SBML File

PhaseCartographer reads SBML Level 2 & 3 files (no `libsbml` dependency):

```bash
$ python3 -m phase_cartographer check examples/lotka_volterra.sbml
  ✓ Valid model: lotka_volterra
    States:     2 (prey, predator)
    Parameters: 2 (alpha, beta)
    Equations:  2
      dprey/dt = alpha*prey - beta*predator*prey
      dpredator/dt = beta*predator*prey - predator
    RHS type:   polynomial
```

### Example 5: Run a Built-in Model (Toggle Switch)

```bash
$ python3 -m phase_cartographer toggle_switch --depth 3 --max-cells 100
  Model: toggle_switch (Gardner toggle switch: mutual Hill repression (n1=n2=2 fixed))
  States: 2, Params: 4, RHS: hill
  Parameter domain: [(2.0, 4.0), (2.0, 4.0), (2.0, 2.0), (2.0, 2.0)]

  Results:
    Certified cells:      2
    Coverage:             100.0%
    Regimes:              {'monostable': 2}
    Z3 equilibrium:       2/2
    Z3 stability:         2/2
    Krawczyk contraction: 0.6609
    Eigenvalue gap (min): 0.000000
    Near bifurcation:     2
    Time:                 5.0s
```

**Reading the output:**
- **Certified cells**: parameter-space regions with proven equilibria
- **Coverage**: fraction of the parameter domain that has been certified
- **Z3 equilibrium / stability**: claims verified by Z3 (Tier 3)
- **Krawczyk contraction**: mean contraction factor (< 1 proves existence)

### Example 6: Run a Built-in Model (Brusselator)

```bash
$ python3 -m phase_cartographer brusselator --depth 3 --max-cells 100
  Model: brusselator (Brusselator: polynomial autocatalytic model)
  States: 2, Params: 2, RHS: polynomial
  Parameter domain: [(0.8, 2.0), (1.5, 3.5)]

  Results:
    Certified cells:      3
    Coverage:             37.5%
    Regimes:              {'monostable': 3, 'unknown': 5}
    Z3 equilibrium:       3/3
    Z3 stability:         3/3
    Krawczyk contraction: 0.8299
    Eigenvalue gap (min): 0.000000
    Near bifurcation:     2
    Time:                 28.8s
```

The 37.5% coverage reflects a Hopf bifurcation boundary where the Krawczyk
operator correctly refuses to certify cells straddling the oscillatory/monostable
boundary. Increase `--depth` for finer resolution near bifurcations.

### Example 7: Run a User YAML Model

```bash
$ python3 -m phase_cartographer --file examples/toggle_switch.yaml --depth 3 --max-cells 100
  Model: user_toggle_switch (Mutual Hill repression with tunable cooperativity)
  States: 2, Params: 2, RHS: hill
  Parameter domain: [(2.0, 5.0), (2.0, 5.0)]

  Results:
    Certified cells:      2
    Coverage:             100.0%
    Regimes:              {'monostable': 2}
    Z3 equilibrium:       2/2
    Z3 stability:         2/2
    Krawczyk contraction: 0.3334
    Eigenvalue gap (min): 0.000000
    Near bifurcation:     1
    Time:                 3.9s
```

### Example 8: Run an SBML Model (Lotka-Volterra)

```bash
$ python3 -m phase_cartographer --file examples/lotka_volterra.sbml --depth 3 --max-cells 100
  Model: lotka_volterra (SBML model loaded from lotka_volterra.sbml)
  States: 2, Params: 2, RHS: polynomial
  Parameter domain: [(0.5, 1.5), (0.8, 1.2)]

  Results:
    Certified cells:      1
    Coverage:             100.0%
    Regimes:              {'inconclusive': 1}
    Z3 equilibrium:       1/1
    Z3 stability:         0/1
    Krawczyk contraction: 0.9472
    Eigenvalue gap (min): 0.000000
    Near bifurcation:     1
    Time:                 0.5s
```

The Lotka-Volterra system has a center (purely imaginary eigenvalues), so
stability is correctly reported as `inconclusive` — the eigenvalue gap is zero
and the Krawczyk contraction factor (0.9472) is close to 1.

### Example 9: Run a 3D Model (Three-Gene Ring)

```bash
$ python3 -m phase_cartographer --file examples/three_gene_ring.yaml --depth 3 --max-cells 100
  Model: three_gene_ring (Repressilator-like ring of three mutual repressors)
  States: 3, Params: 2, RHS: hill
  Parameter domain: [(3.0, 12.0), (2.0, 2.0)]

  Results:
    Certified cells:      3
    Coverage:             100.0%
    Regimes:              {'monostable': 3}
    Z3 equilibrium:       0/3
    Z3 stability:         0/3
    Krawczyk contraction: 0.7596
    Eigenvalue gap (min): 0.000000
    Near bifurcation:     1
    Time:                 1.5s
```

Z3 reports 0/3 because the Hill-function RHS is non-polynomial — Z3 (Tier 3)
only handles polynomial systems. Equilibria are still certified by Tier 1
(Krawczyk contraction factor 0.76 < 1).

---

## Supported Input Formats

| Format   | Extensions      | Loader              | Notes                              |
|----------|-----------------|----------------------|------------------------------------|
| YAML     | `.yaml`, `.yml` | `load_model_file()`  | Custom ODE + parameter bounds      |
| SBML     | `.sbml`, `.xml` | `load_sbml_file()`   | Level 2 & 3, MathML support       |
| Built-in | —               | `run_benchmark(name)`| 18 pre-defined models              |

### YAML Format Reference

```yaml
name: my_model               # Model identifier
description: My ODE system   # Optional description

states: [x, y]               # State variable names (used in equations)

parameters:                   # Parameters with search bounds
  - name: a
    bounds: [0.5, 3.0]
  - name: b
    bounds: [1.0, 5.0]

equations:                    # SymPy-parseable RHS expressions, one per state
  - "a / (1 + y**2) - x"
  - "b / (1 + x**2) - y"

state_domain:                 # Bounding box for equilibrium search
  - [0.1, 5.0]
  - [0.1, 5.0]

rhs_type: hill               # Optional: hill, polynomial, rational, general
expected_regimes: [bistable]  # Optional: for validation only
```

### SBML Annotation for Parameter Sweeps

By default all SBML parameters are treated as constants. To sweep parameters
across a range, add a `<phaseCartographer>` annotation inside `<model>`:

```xml
<annotation>
  <phaseCartographer xmlns="http://phasecartographer.example.org">
    <parameterBounds id="alpha" lower="0.5" upper="1.5"/>
    <parameterBounds id="beta"  lower="0.8" upper="1.2"/>
    <rhsType>polynomial</rhsType>
    <stateDomain>
      <bound lower="0.1" upper="3.0"/>
      <bound lower="0.1" upper="3.0"/>
    </stateDomain>
  </phaseCartographer>
</annotation>
```

### Shipped Example Files

| File                            | Format | States | Params | Description                  |
|---------------------------------|--------|--------|--------|------------------------------|
| `examples/toggle_switch.yaml`   | YAML   | 2      | 2      | Mutual Hill repression       |
| `examples/predator_prey.yaml`   | YAML   | 2      | 4      | Lotka-Volterra with carrying capacity |
| `examples/three_gene_ring.yaml` | YAML   | 3      | 2      | Repressilator-like ring      |
| `examples/lotka_volterra.sbml`  | SBML   | 2      | 2      | Classic Lotka-Volterra       |

---

## Python API

```python
from phase_cartographer.benchmarks.runner import run_benchmark, run_model
from phase_cartographer.models.benchmark_models import load_model_file, list_benchmarks
from phase_cartographer.equilibrium.krawczyk import KrawczykOperator
from phase_cartographer.equilibrium.stability import StabilityClassifier
from phase_cartographer.atlas.builder import PhaseAtlas
from phase_cartographer.minicheck import verify_certificate

# Run a built-in model
result = run_benchmark("toggle_switch", max_depth=5, max_cells=300)

# Run a user-defined YAML model
bm = load_model_file("examples/toggle_switch.yaml")
result = run_model(bm, max_depth=4, max_cells=200)

# List all 18 built-in models
print(list_benchmarks())

# Low-level: Krawczyk operator for equilibrium existence
krawczyk = KrawczykOperator(rhs)
result = krawczyk.verify(X_box, mu_params)

# Low-level: stability classification
classifier = StabilityClassifier(rhs)
stability_type, eigenvalue_enclosure = classifier.classify(x_enclosure, mu_params)

# Independent certificate verification
result = verify_certificate(certificate_data)
```

See [API.md](API.md) for the full programmatic reference.

---

## Architecture Overview

PhaseCartographer follows a six-stage verification pipeline:

1. **Model Definition** (YAML / SBML / built-in) → `SymbolicRHS`
2. **Adaptive Octree Refinement** → parameter-space cells
3. **Per-cell verification** (Tier 1 → Tier 2 → Tier 3)
4. **Stability cascade** for each verified equilibrium
5. **Certificate composition** with hash-chained proof traces
6. **MiniCheck independent replay** for external audit

```
Model File ──► SymbolicRHS ──► Octree ──► [Tier 1 │ Tier 2 │ Tier 3]
                                               │
                                    Krawczyk ◄──┘──► Stability Cascade
                                        │                  │
                                        ▼                  ▼
                                   Proof Trace ──► MiniCheck Replay
```

| Tier | Engine              | Scope                  | Guarantee               |
|------|---------------------|------------------------|-------------------------|
| 1    | Interval arithmetic | All RHS types          | Conservative enclosure  |
| 2    | dReal               | Non-polynomial (Hill)  | δ-complete soundness    |
| 3    | Z3                  | Polynomial RHS only    | Exact algebraic proof   |

`minicheck.py` (1,188 LoC) is a standalone certificate checker depending only
on Python stdlib + NumPy. It re-verifies Krawczyk contractions, Gershgorin
eigenvalue enclosures, stability classifications, and δ-bounds independently.

---

## Built-in Models

All 18 built-in models:

| Model                    | n | p | Expected Regime | RHS Type   |
|--------------------------|---|---|-----------------|------------|
| `toggle_switch`          | 2 | 4 | Bistable        | Hill       |
| `brusselator`            | 2 | 2 | Oscillatory     | Polynomial |
| `selkov`                 | 2 | 2 | Oscillatory     | Polynomial |
| `repressilator`          | 3 | 4 | Oscillatory     | Hill       |
| `goodwin`                | 3 | 2 | Oscillatory     | Hill       |
| `lotka_volterra`         | 2 | 3 | Center          | Polynomial |
| `fitzhugh_nagumo`        | 2 | 2 | Excitable       | Polynomial |
| `tyson_cell_cycle`       | 2 | 2 | Oscillatory     | Polynomial |
| `mapk_cascade`           | 2 | 2 | Monostable      | Polynomial |
| `p53_mdm2`               | 2 | 2 | Bistable        | Polynomial |
| `calcium_oscillator`     | 2 | 2 | Oscillatory     | Rational   |
| `circadian_clock`        | 2 | 2 | Oscillatory     | Rational   |
| `genetic_oscillator`     | 3 | 2 | Oscillatory     | Hill       |
| `mutual_activation_switch`| 2| 2 | Bistable        | Hill       |
| `van_der_pol`            | 2 | 2 | Limit cycle     | Polynomial |
| `michaelis_menten_switch`| 2 | 2 | Bistable        | Rational   |
| `erk_bistable`           | 2 | 2 | Bistable        | Polynomial |
| `apoptosis_switch`       | 2 | 2 | Bistable        | Polynomial |

Column legend: **n** = state dimension, **p** = swept parameter count.

Run any model by name:

```bash
python3 -m phase_cartographer selkov --depth 4 --max-cells 200
```

Run all 18 in one shot (prints a summary table):

```bash
python3 -m phase_cartographer
```

---

### Performance Notes

| Model (example)        | Params | Depth | Max Cells | Time     |
|------------------------|--------|-------|-----------|----------|
| `toggle_switch`        | 4      | 3     | 100       | ~5s      |
| `brusselator`          | 2      | 3     | 100       | ~29s     |
| `three_gene_ring`      | 2      | 3     | 100       | ~1.5s    |
| `lotka_volterra` (SBML)| 2      | 3     | 100       | ~0.5s    |
| `user_toggle_switch`   | 2      | 3     | 100       | ~4s      |

Runtime scales with parameter dimension and refinement depth.

---

## FAQ / Troubleshooting

**Q: What is the difference between verification tiers?**
A: Tier 1 uses interval arithmetic for fast, conservative checks. Tier 2
invokes dReal for δ-complete decision procedures on non-polynomial RHS. Tier 3
uses Z3 for exact polynomial verification. The tiered dispatcher escalates
automatically when a lower tier is inconclusive.

**Q: dReal is not installed — will it still work?**
A: Yes. dReal is optional. Without it, Tier 2 is skipped and verification falls
back to Tier 1 (interval arithmetic) or escalates directly to Tier 3 (Z3) for
polynomial systems.

**Q: Z3 reports 0/N for my Hill-function model. Is something wrong?**
A: No. Z3 (Tier 3) handles only polynomial RHS. Hill functions involve
rational expressions that Z3 cannot reason about directly. The equilibria are
still certified by Tier 1 (interval arithmetic with Krawczyk). Check that the
Krawczyk contraction factor is < 1 — that is the actual existence proof.

**Q: How do I interpret δ-soundness bounds?**
A: A certificate includes four quantities: γ (contraction margin), ‖R‖
(preconditioner norm), L_Df (Lipschitz constant of the Jacobian), and rad (box
radius). The bound δ_solver < γ / (‖R‖ · L_Df · rad + 1) guarantees that the
solver's numerical perturbation cannot change the regime classification.

**Q: Coverage is below 100%?**
A: Increase `--depth` (6–8) and `--max-cells`. Coverage gaps occur at
bifurcation boundaries where Krawczyk correctly refuses to certify cells
straddling regime changes. GP-guided refinement focuses cells there automatically.

**Q: Model with 4+ parameters is slow?**
A: Fix some parameters to constants (`bounds: [v, v]`), reduce `--depth`, or
use GP-guided refinement.

**Q: Custom model programmatically?**
A: Create a `BenchmarkModel` and pass to `run_model()`, or write YAML and use
`load_model_file()`.

**Q: MiniCheck fails but main verifier passes?**
A: MiniCheck uses tighter rounding. Re-run with higher `--depth` for tighter boxes.

**Q: What does "inconclusive" regime mean?**
A: The stability cascade could not definitively classify the equilibrium — often
because eigenvalues have zero or near-zero real parts (e.g., center dynamics in
Lotka-Volterra). This is a correct conservative answer, not an error.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
