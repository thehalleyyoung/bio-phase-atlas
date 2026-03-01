# PhaseCartographer: Certified Phase Atlases for Biological ODEs

Z3-backed formal verification of biological ODE equilibria — **60/60** equilibrium claims verified across 15 models with 209 passing tests.

## 30-Second Quickstart

```bash
pip install numpy sympy scipy z3-solver
python3 -m pytest tests/ -q          # 209 tests, ~20 seconds
python3 -c "
from phase_cartographer.benchmarks.runner import run_benchmark
r = run_benchmark('brusselator', max_depth=6, max_cells=1000)
print(f'Brusselator: {r.certified_cells} cells, {r.coverage_fraction:.0%} coverage')
print(f'Z3 equilibrium: {r.tier3_verified}/{r.tier3_total}')
print(f'Z3 stability: {r.tier2_verified}/{r.tier2_total}')
print(f'Contraction: {r.mean_contraction:.3f}, Time: {r.total_time_s:.1f}s')
"
```

**Expected output:**
```
Brusselator: 17 cells, 86% coverage
Z3 equilibrium: 17/17
Z3 stability: 8/17
Contraction: 0.793, Time: 66.3s
```

Coverage gaps at the Hopf boundary B = 1 + A² are correct conservative behavior — the Krawczyk operator cannot certify boxes straddling a bifurcation.

## Most Impressive: Machine-Checkable Regime Certificates

Every certified cell carries a machine-checkable proof verified through three independent tiers:

```python
from phase_cartographer.benchmarks.runner import run_benchmark
from phase_cartographer.atlas.certificate_algebra import AtlasCertificateBundle

r = run_benchmark('calcium_oscillator', max_depth=5, max_cells=500)
# 3 cells, 100% coverage, Z3 confirms all 3 equilibria AND stability
print(f'Calcium: {r.certified_cells} cells, Z3 eq {r.tier3_verified}/{r.tier3_total}, stab {r.tier2_verified}/{r.tier2_total}')
```

Z3 confirms equilibrium existence in **60/60** claims across 59 certified cells and stability in **34/60** via Routh-Hurwitz encoding.

## What It Does

Given a biological ODE model ẋ = f(x, μ) and a parameter box P ⊂ ℝ^p, PhaseCartographer:

1. **Partitions** P into certified cells via adaptive octree refinement
2. **Certifies** each cell using the Krawczyk operator (existence + uniqueness of equilibria)
3. **Classifies** stability via rigorous eigenvalue enclosure (Gershgorin + Bauer-Fike)
4. **Labels** regimes via formal inference rules (not ad-hoc classification)
5. **Verifies** every certificate independently via MiniCheck (1,000 LoC TCB)
6. **Verifies** via Z3 SMT solver (equilibrium existence, stability, δ-soundness)
7. **Guides** refinement with GP surrogate (advisory, doesn't affect soundness)
8. **Quantifies** uncertainty via probabilistic confidence layer

## Benchmark Results (15 Models)

| Model | n | p | Cells | Coverage | Z3 Eq | Z3 Stab | Time | ρ̄ |
|-------|---|---|-------|----------|-------|---------|------|------|
| Toggle switch | 2 | 4 | 2 | 100% | 2/2 | --- | 3.0s | 0.661 |
| Brusselator | 2 | 2 | 17 | 85.9% | 17/17 | 8/17 | 66.3s | 0.793 |
| Sel'kov | 2 | 2 | 1 | 100% | 1/1 | 1/1 | 0.8s | 0.833 |
| Repressilator | 3 | 4 | 1 | 100% | 1/1 | --- | <0.1s | 0.633 |
| Goodwin | 3 | 2 | 2 | 100% | 2/2 | --- | 0.1s | 0.903 |
| Lotka-Volterra | 2 | 3 | 1 | 100% | 1/1 | --- | 0.2s | 0.612 |
| FitzHugh-Nagumo | 2 | 2 | 15 | 81.2% | 15/15 | 14/15 | 17.1s | 0.857 |
| Tyson cell cycle | 2 | 2 | 1 | 100% | 1/1 | 1/1 | 0.2s | 0.644 |
| MAPK cascade | 2 | 2 | 1 | 100% | 1/1 | --- | <0.1s | 0.910 |
| p53-Mdm2 | 2 | 2 | 2 | 100% | 2/2 | --- | 1.3s | 0.732 |
| Calcium oscillator | 2 | 2 | 3 | 100% | 3/3 | 3/3 | 4.2s | 0.337 |
| Circadian clock | 2 | 2 | 4 | 100% | 4/4 | --- | 3.0s | 0.763 |
| Genetic oscillator | 3 | 2 | 2 | 100% | 2/2 | --- | 0.1s | 0.868 |
| Mutual activation | 2 | 2 | 6 | 90.6% | 7/7 | 6/7 | 45.3s | 0.804 |
| Van der Pol | 2 | 2 | 1 | 100% | 1/1 | 1/1 | <0.1s | 0.000 |
| **Total** | | | **59** | | **60/60** | **34/60** | | |

## Code Structure (~18,400 LoC, 209 tests)

| Module | LoC | Description |
|--------|-----|-------------|
| interval/ | 2,397 | Rigorous interval arithmetic, matrices, Taylor models |
| ode/ | 1,570 | ODE right-hand sides, validated integration |
| equilibrium/ | 1,055 | Krawczyk operator, stability classification |
| smt/ | 1,260 | Z3 verifier, SMT encoding, δ-bound, error composition |
| benchmarks/ | 1,400 | Benchmark runner, ablation, scalability, δ-validation |
| refinement/ | 631 | Adaptive octree, GP-guided, posterior-informed, covariance-aware |
| gp/ | 617 | GP surrogate (ARD Matérn-5/2), calibration, acquisition |
| tiered/ | 462 | Unified certificate format, Z3-integrated dispatcher |
| atlas/ | 900 | Atlas builder, composition, proof trace, certificate algebra |
| models/ | 561 | 15 benchmark biological models |
| minicheck.py | 1,188 | Minimal independent certificate checker (TCB) |
| tests/ | 2,800 | 209 tests (unit, integration, E2E, Z3, math rigor, GP) |

## Key Theorems

- **Theorem 1 (Krawczyk):** K(X) ⊆ X ⟹ unique zero in X
- **Theorem 2 (δ-Soundness):** δ_solver < γ / (‖R‖·L_{Df}·rad + 1) ⟹ exact regime correctness
- **Theorem 3 (Composition):** Disjoint certified cells compose soundly into atlas certificates

## New Features

- **Composed error budget:** Tracks how δ-perturbation and interval-enclosure errors compose across verification tiers (`ComposedErrorBudget`, `error_budget()`)
- **Empirical δ-soundness validation:** Verifies Theorem B1 conditions hold in practice (`run_delta_validation`)
- **Depth scaling analysis:** Measures coverage convergence rate vs refinement depth (`run_depth_scaling`)
- **Posterior-informed refinement:** Covariance-aware splitting aligns octree splits with GP posterior variance (`PosteriorRefinementConfig`, `covariance_aware_split`)
- **GP coverage bounds:** Borell-TIS–based probabilistic coverage guarantees (`GPSurrogate.coverage_bound()`) and Matérn-5/2 regret bounds (`GPSurrogate.regret_bound()`)
- **Proof trace:** Hash-chained machine-readable proof trace with per-step integrity verification (`ProofTrace.from_atlas`, `verify_chain`, `export_json`)
- **Comprehensive ablation:** Tier-by-tier, GP vs uniform, Newton vs bisection, isotropic vs anisotropic (`run_comprehensive_ablation`)
- **GP calibration pipeline:** ECE, NLPD, and model selection integrated into benchmark pipeline (`run_gp_calibration_study`)

## Requirements

- Python 3.8+ with NumPy, SymPy, SciPy
- Z3 (Tier 3 verification): `pip install z3-solver`
- Optional: dReal (Tier 2 δ-complete verification)

## Running Full Benchmarks

```bash
python3 -c "
from phase_cartographer.benchmarks.runner import run_all_benchmarks
run_all_benchmarks('benchmark_output')
"
```

Results saved to `benchmark_output/summary.json`.
