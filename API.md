# PhaseCartographer API Reference

## CLI Commands

### Run analysis on a user-defined model
```bash
python3 -m phase_cartographer --file MODEL.yaml [--depth N] [--max-cells N]
python3 -m phase_cartographer --file MODEL.sbml [--depth N] [--max-cells N]
```

### Run analysis on a built-in benchmark
```bash
python3 -m phase_cartographer toggle_switch [--depth N] [--max-cells N]
python3 -m phase_cartographer                 # Run all 18 benchmarks
```

### Validate a model file (parse-only, no computation)
```bash
python3 -m phase_cartographer check MODEL.yaml
python3 -m phase_cartographer check MODEL.sbml
```

### Generate a starter template
```bash
python3 -m phase_cartographer init-model [-o OUTPUT.yaml] [--template 2d|3d|oscillator]
```

### Special reports
```bash
python3 -m phase_cartographer --case-study    # Toggle switch design case study
python3 -m phase_cartographer --delta-table   # δ-soundness parameter table
python3 -m phase_cartographer --baseline      # Numerical continuation comparison
```

## YAML Model File Format

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

### Programmatic model loading
```python
from phase_cartographer.models.benchmark_models import load_model_file, validate_model_file
bm = load_model_file("my_model.yaml")        # Returns BenchmarkModel
bm = load_model_file("my_model.sbml")        # SBML auto-detected by extension
issues = validate_model_file("my_model.yaml") # Returns list of issues (empty = valid)

from phase_cartographer.benchmarks.runner import run_model
result = run_model(bm, max_depth=5, max_cells=500)
```

## SBML Model File Format

[SBML](http://sbml.org) (Systems Biology Markup Language) files are supported via
`load_model_file()` or `load_sbml_file()`.  Both SBML Level 2 and Level 3 are
handled using only `xml.etree.ElementTree` (no `libsbml` dependency).

The importer extracts species names, initial concentrations, compartment volumes,
global/local parameters, and MathML kinetic rate laws.  Supported MathML operators:
`plus`, `minus`, `times`, `divide`, `power`, `exp`, `ln`, `sqrt`, `sin`, `cos`,
`cn`, `ci`, `csymbol`.

By default all parameters are substituted as constants.  To sweep parameters across
a range, add a `<phaseCartographer>` annotation in the `<model>` element:

```xml
<annotation>
  <phaseCartographer xmlns="http://phasecartographer.example.org">
    <parameterBounds id="alpha" lower="0.5" upper="1.5"/>
    <rhsType>polynomial</rhsType>
    <stateDomain>
      <bound lower="0.1" upper="3.0"/>
      <bound lower="0.1" upper="3.0"/>
    </stateDomain>
  </phaseCartographer>
</annotation>
```

### Programmatic SBML loading
```python
from phase_cartographer.models.benchmark_models import load_sbml_file
bm = load_sbml_file("model.sbml")  # Returns BenchmarkModel
```

## Core Pipeline

### `equilibrium.KrawczykOperator`
```python
from phase_cartographer.equilibrium import KrawczykOperator
kop = KrawczykOperator(rhs, max_iter=20)
result = kop.verify(X, mu)          # Single box verification
result = kop.find_equilibria(domain, mu, max_depth=10)  # Recursive search
result = kop.verify_parametric(X, mu_box)               # Parametric verification
```
- `result.verified: bool` — True if K(X) ⊆ X
- `result.enclosure: IntervalVector` — Tightened enclosure
- `result.contraction_factor: float` — Width ratio K(X)/X

### `equilibrium.StabilityClassifier`
```python
from phase_cartographer.equilibrium import StabilityClassifier
sc = StabilityClassifier(rhs)
stab_type, eig_enc = sc.classify(x_enclosure, mu)
```
- Returns `StabilityType` enum and `EigenvalueEnclosure`
- 4-stage cascade: Routh-Hurwitz → Bauer-Fike → midpoint refinement → center manifold
- `eig_enc.classification_method`: which stage resolved the classification
- Methods: `routh_hurwitz_check()`, `compute_eigenvalue_enclosure()`, `_try_midpoint_refinement()`, `_try_center_manifold()`

### `tiered.certificate.CertifiedCell`
```python
from phase_cartographer.tiered.certificate import CertifiedCell, RegimeInferenceRules
cell.to_dict()           # JSON-serializable dict
cell.fingerprint()       # SHA-256 integrity hash
cell.to_minicheck_format()  # Format for independent verification
RegimeInferenceRules.infer(cell.equilibria)   # Derive regime label
RegimeInferenceRules.validate(cell)           # Check label consistency
```

### `tiered.dispatcher`
```python
from phase_cartographer.tiered.dispatcher import verify_cell, select_tier
cell = verify_cell(cell)   # Run tiered verification (Tier 1 always)
tier = select_tier(model_name, rhs_type)  # Auto-select highest tier
```

### `atlas.builder.PhaseAtlas`
```python
from phase_cartographer.atlas.builder import PhaseAtlas
atlas = PhaseAtlas(model_name, parameter_domain)
atlas.add_cell(cell)
atlas.stats()              # AtlasStats with coverage, regime counts
atlas.coverage_fraction()  # Certified volume / total volume
atlas.save("atlas.json")
```

### `atlas.composition`
```python
from phase_cartographer.atlas.composition import verify_atlas_composition, CertificateProofObject
result = verify_atlas_composition(cells, domain)
# result.valid, result.coverage_fraction, result.boundary_cells
proof = CertificateProofObject.from_cell(cell)  # Proof object with derivation
```

### `atlas.composition.ProofTrace`

Machine-readable proof trace with hash-chained integrity verification.

```python
from phase_cartographer.atlas.composition import ProofTrace

# Generate proof trace from an existing atlas
trace = ProofTrace.from_atlas(atlas)

# Verify hash chain integrity (detects tampering)
assert trace.verify_chain()

# Export to JSON for archival or audit
trace.export_json("proof_trace.json")

# Manual construction
trace = ProofTrace()
entry = trace.add_entry(
    operation="krawczyk_apply",
    inputs={"cell_index": 0, "parameter_box": [[1.0, 2.0], [3.0, 4.0]]},
    outputs={"contraction_factor": 0.661, "verified": True},
    method="krawczyk_interval_newton",
)
# entry.hash: SHA-256 chained to previous entry
```

- `ProofTrace.from_atlas(atlas, model=None) -> ProofTrace` — walks all cells, records Krawczyk, eigenvalue, stability, and regime derivation steps
- `ProofTrace.verify_chain() -> bool` — re-computes every hash; returns False if any entry was altered
- `ProofTrace.export_json(path: str) -> None` — writes `{"n_entries", "chain_hash", "entries": [...]}`

### `atlas.confidence`
```python
from phase_cartographer.atlas.confidence import (
    compute_cell_confidence, atlas_confidence_summary,
    eigenvalue_gap_confidence, krawczyk_margin_confidence,
)
# Cell-level confidence score (0 to 1)
conf = compute_cell_confidence(eigenvalue_gap=0.5, contraction_factor=0.7)
# conf.composite: float, conf.eigenvalue_gap_conf, conf.krawczyk_margin_conf

# Atlas-level summary
summary = atlas_confidence_summary(cells)
# summary.mean_confidence, summary.min_confidence, summary.near_bifurcation_count
```

### `refinement.octree`
```python
from phase_cartographer.refinement.octree import (
    adaptive_refine, gp_guided_refine,
    RefinementConfig, GPGuidedRefinementConfig,
    anisotropic_split_box,
)
# Standard refinement
config = RefinementConfig(max_depth=8, target_coverage=0.95, max_cells=10000)
atlas = adaptive_refine(certify_fn, parameter_domain, model_name, config)

# GP-guided refinement (returns atlas + convergence record)
gp_config = GPGuidedRefinementConfig(
    max_depth=8, gp_warmup_cells=10, gp_retrain_interval=10, gp_weight=0.6
)
atlas, convergence = gp_guided_refine(certify_fn, parameter_domain, model_name, gp_config)
```

### `refinement.octree.posterior_informed_refine`

Posterior-informed refinement with covariance-aware splitting.

```python
from phase_cartographer.refinement.octree import (
    posterior_informed_refine, PosteriorRefinementConfig, covariance_aware_split,
)
import numpy as np

# Configure posterior-informed refinement
config = PosteriorRefinementConfig(
    max_depth=8,
    target_coverage=0.95,
    max_cells=10000,
    covariance_matrix=np.array([[1.0, 0.3], [0.3, 2.0]]),  # parameter correlations
    use_pca_split=True,
    posterior_weight=0.5,
)
atlas, convergence = posterior_informed_refine(certify_fn, parameter_domain, model_name, config)

# Standalone covariance-aware box split
box1, box2 = covariance_aware_split(
    box=[(0.0, 1.0), (0.0, 2.0)],
    covariance_matrix=np.array([[1.0, 0.3], [0.3, 2.0]]),
)
```

- `PosteriorRefinementConfig` extends `GPGuidedRefinementConfig` with `covariance_matrix`, `use_pca_split`, `posterior_weight`
- `covariance_aware_split(box, covariance_matrix)` — splits along the axis-aligned dimension capturing the most posterior variance

## Interval Arithmetic

### `interval.Interval`
```python
from phase_cartographer.interval.interval import Interval
a = Interval(1.0, 2.0)
a + b, a * b, a / b, a ** n   # Outward-rounded arithmetic
a.contains(x), a.width, a.mid, a.mag
```

### `interval.IntervalVector`, `interval.IntervalMatrix`
```python
from phase_cartographer.interval.matrix import IntervalVector, IntervalMatrix
v = IntervalVector([Interval(1, 2), Interval(3, 4)])
M = IntervalMatrix.identity(n)
M.gershgorin_disks()   # Eigenvalue enclosure
M.determinant()        # Interval determinant
```

## GP Surrogate (Advisory)

### `gp.GPSurrogate`
```python
from phase_cartographer.gp.surrogate import GPSurrogate
gp = GPSurrogate(use_ard=True)
gp.optimize_ard_length_scales(X, y)  # ARD hyperparameter optimization
gp.fit(X, y)
pred = gp.predict(x_query)   # GPPrediction(mean, variance, std)
ece = gp.calibration_error(X_test, y_test)
loo = gp.loo_cross_validation(X, y)

# Train directly from an atlas
gp = GPSurrogate.train_from_atlas(atlas, use_ard=True)
boundary_prob = gp.predict_regime_boundary(x_query)
```

### `gp.GPSurrogate.coverage_bound`

Borell-TIS–based probabilistic coverage guarantee.

```python
lower, upper, conf = gp.coverage_bound(
    domain_bounds=[(0.0, 1.0), (0.0, 2.0)],
    confidence=0.95,
    n_samples=1000,
)
# lower: coverage lower bound, upper: coverage upper bound, conf: confidence level
```

### `gp.GPSurrogate.regret_bound`

Matérn-5/2 information-gain regret bound.

```python
regret, info_gain = gp.regret_bound(n_observed=50, domain_volume=2.0)
# regret: R_T = O(√(T γ_T β_T)), info_gain: γ_T for Matérn-5/2
```

### `gp.AcquisitionOptimizer`

### `gp.kernel_verification`
```python
from phase_cartographer.gp.kernel_verification import (
    verify_kernel_regularity,
    compute_composed_delta_error,
    KernelVerificationResult,
)
# Verify GP kernel satisfies Borell-TIS conditions
result = verify_kernel_regularity(
    kernel_name="matern52", nu=2.5, dimension=2,
    domain_bounds=[(0.0, 1.0), (0.0, 2.0)],
    length_scales=[0.5, 0.5],
)
# result.borell_tis_applicable: bool
# result.expected_supremum_bound: float
# result.regularity_class: str ("C^2" for Matérn-5/2)

# Compute composed delta-error across atlas cells
error = compute_composed_delta_error(atlas, rhs, state_domain)
# error: dict with max_composed_delta, soundness_rate, per-cell breakdown
```

### `gp.AcquisitionOptimizer`
```python
from phase_cartographer.gp.acquisition import AcquisitionOptimizer, phase_boundary_score
opt = AcquisitionOptimizer(gp, acquisition="boundary_uncertainty")
ranked = opt.rank_boxes(box_midpoints)
next_idx = opt.select_next(box_midpoints)

# Combined score with eigenvalue sensitivity
score = phase_boundary_score(pred, eigenvalue_sensitivity=2.5)
```

## SMT Interface

### `smt.delta_bound`
```python
from phase_cartographer.smt.delta_bound import DeltaBound, compute_required_delta, ComposedErrorBudget
db = DeltaBound(rhs)
result = db.compute(X, mu_box, eigenvalue_real_parts, delta_solver=1e-3)
# result.delta_required, result.eigenvalue_gap, result.is_sound, result.soundness_margin

# Composed error budget (δ-perturbation + interval-enclosure)
budget = ComposedErrorBudget.compose_errors(result, contraction_factor=0.8, enclosure_width=0.01)
# budget.total_composed_error, budget.is_sound
```

### `smt.z3_verifier`
```python
from phase_cartographer.smt.z3_verifier import (
    verify_equilibrium_z3, verify_stability_z3,
    verify_no_other_equilibria_z3, verify_delta_soundness_z3,
)
# Verify equilibrium existence (SAT = exists)
result = verify_equilibrium_z3(model, mu_box, x_box=None, timeout_ms=5000)
# result: Z3VerificationResult with .status ("SAT"/"UNSAT"/"UNKNOWN"), .time_ms, .witness

# Verify stability via Routh-Hurwitz (UNSAT = stable)
result = verify_stability_z3(model, mu_box, x_eq=None, timeout_ms=5000)

# Verify no other equilibria (completeness check)
result = verify_no_other_equilibria_z3(model, mu_box, known_eq, x_domain, timeout_ms=10000)

# Verify δ-soundness bound
result = verify_delta_soundness_z3(model, mu_box, delta_bound, timeout_ms=5000)
```

## MiniCheck (Independent Verifier)

```python
from phase_cartographer.minicheck import verify_certificate, verify_atlas
result = verify_certificate(cert_dict)   # Single cell
result = verify_atlas(cells, domain)     # Atlas-level
# CLI: python3 -m phase_cartographer.minicheck certificate.json
```

## Benchmarks

### Numerical Continuation Baseline
```python
from phase_cartographer.benchmarks.numerical_baseline import (
    NumericalContinuationBaseline,
    run_full_baseline_comparison,
)
# Compare numerical continuation vs certified results
results = run_full_baseline_comparison(output_dir="benchmark_output/")
# Per-model: numerical eq count, certified eq count, agreement rate, errors caught

# Single-model comparison
baseline = NumericalContinuationBaseline(rhs, parameter_domain, state_domain)
num_result = baseline.run_continuation(n_points=80)
# num_result: n_equilibria_found, n_bifurcations, bifurcation_types
```

### Benchmark Runner
```python
from phase_cartographer.benchmarks.runner import (
    run_benchmark, run_all_benchmarks, run_ablation,
    run_comprehensive_ablation, ComprehensiveAblationResult,
    run_depth_scaling, run_delta_validation, run_gp_calibration_study,
)
result = run_benchmark("toggle_switch", max_depth=5, max_cells=300)
results = run_all_benchmarks("benchmark_output/")
ablation = run_ablation("brusselator", max_depth=4, max_cells=200)

# Depth scaling analysis
depth_results = run_depth_scaling("brusselator", depths=[2,3,4,5,6])
# Returns List[DepthScalingPoint] with coverage at each depth

# Empirical δ-soundness validation
delta_result = run_delta_validation("brusselator", max_depth=4)
# delta_result.all_sound, delta_result.n_sound, delta_result.n_equilibria

# GP calibration study
cal_report = run_gp_calibration_study("brusselator", max_depth=5, max_cells=300)
# cal_report["calibration"]["ece"], cal_report["calibration"]["nlpd"]
```

### `benchmarks.runner.run_comprehensive_ablation`

Full ablation study covering all pipeline components.

```python
from phase_cartographer.benchmarks.runner import run_comprehensive_ablation

result = run_comprehensive_ablation("brusselator", max_depth=4, max_cells=200)
# Tier ablation
print(result.tier1_only_coverage, result.tier12_coverage, result.tier123_coverage)
# GP ablation
print(result.uniform_coverage, result.gp_coverage)
# Newton vs bisection (for ≥3-state models)
print(result.bisection_coverage, result.newton_coverage)
# Splitting ablation
print(result.isotropic_coverage, result.anisotropic_coverage)
```

- `ComprehensiveAblationResult` fields: `tier1_only_coverage/time_s`, `tier12_coverage/time_s`, `tier123_coverage/time_s`, `uniform_coverage/time_s`, `gp_coverage/time_s`, `bisection_coverage/time_s`, `newton_coverage/time_s`, `isotropic_coverage/time_s`, `anisotropic_coverage/time_s`

## Models

```python
from phase_cartographer.models.benchmark_models import get_benchmark, list_benchmarks
bm = get_benchmark("toggle_switch")  # BenchmarkModel with RHS, domains, expected regimes
print(list_benchmarks())
# ['toggle_switch', 'brusselator', 'selkov', 'repressilator', 'goodwin',
#  'lotka_volterra', 'fitzhugh_nagumo', 'tyson_cell_cycle', 'mapk_cascade',
#  'p53_mdm2', 'calcium_oscillator', 'circadian_clock', 'genetic_oscillator',
#  'mutual_activation_switch', 'van_der_pol']
```
