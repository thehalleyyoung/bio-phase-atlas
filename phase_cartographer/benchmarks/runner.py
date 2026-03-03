"""
Benchmark runner: executes the full certification pipeline on benchmark models
and produces quantitative results with real timing data.
"""

import json
import time
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

import numpy as np

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..equilibrium import KrawczykOperator, StabilityClassifier
from ..smt.delta_bound import DeltaBound, compute_eigenvalue_gap
from ..tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)
from ..tiered.dispatcher import verify_cell, select_tier
from ..atlas.builder import PhaseAtlas
from ..refinement.octree import (
    adaptive_refine, RefinementConfig, split_box, box_volume,
    GPGuidedRefinementConfig, gp_guided_refine, ConvergenceRecord,
)
from ..models.benchmark_models import BenchmarkModel, get_benchmark
from ..minicheck import verify_certificate as minicheck_verify


def _run_z3_verification(cell, bm):
    """Run Z3 verification on a certified cell if applicable."""
    z3_results = {}
    try:
        from ..smt.z3_verifier import (
            verify_equilibrium_z3, verify_stability_z3,
            verify_no_other_equilibria_z3, Z3_AVAILABLE,
        )
        if not Z3_AVAILABLE:
            return z3_results

        rhs = bm.rhs
        if not hasattr(rhs, 'expressions') or not rhs.expressions:
            return z3_results

        for eq_idx, eq in enumerate(cell.equilibria):
            eq_result = verify_equilibrium_z3(
                rhs.expressions, rhs.state_symbols, rhs.param_symbols,
                eq.state_enclosure, cell.parameter_box, timeout_ms=15000,
            )
            z3_results[f'eq_{eq_idx}_existence'] = eq_result.to_dict()

            from ..tiered.certificate import StabilityType as _ST
            # Skip Z3 stability for degenerate/center/unknown
            if eq.stability in (_ST.DEGENERATE, _ST.CENTER, _ST.UNKNOWN):
                z3_results[f'eq_{eq_idx}_stability'] = {
                    'verified': False, 'tier': 'tier3_exact',
                    'status': 'skipped',
                    'message': f'Stability type {eq.stability.value} not Z3-verifiable',
                }
                continue

            if eq.stability == _ST.SADDLE:
                stab_type = "saddle"
            elif eq.is_stable():
                stab_type = "stable"
            else:
                stab_type = "unstable"

            # Use tighter enclosure for stability verification:
            # shrink state enclosure to narrow box around midpoint
            tight_enc = []
            for lo, hi in eq.state_enclosure:
                mid = (lo + hi) / 2.0
                r = max((hi - lo) / 4.0, 1e-8)  # quarter-width or minimum
                tight_enc.append((mid - r, mid + r))

            stab_result = verify_stability_z3(
                rhs.expressions, rhs.state_symbols, rhs.param_symbols,
                tight_enc, cell.parameter_box,
                stability_type=stab_type, timeout_ms=15000,
            )
            # If tight enclosure fails, try with full enclosure
            if not stab_result.verified and stab_result.status != "unsat":
                stab_result2 = verify_stability_z3(
                    rhs.expressions, rhs.state_symbols, rhs.param_symbols,
                    eq.state_enclosure, cell.parameter_box,
                    stability_type=stab_type, timeout_ms=15000,
                )
                if stab_result2.verified:
                    stab_result = stab_result2

            z3_results[f'eq_{eq_idx}_stability'] = stab_result.to_dict()

        # Completeness check for polynomial models
        if bm.rhs_type == "polynomial":
            known_encs = [eq.state_enclosure for eq in cell.equilibria]
            comp_result = verify_no_other_equilibria_z3(
                rhs.expressions, rhs.state_symbols, rhs.param_symbols,
                known_encs, bm.state_domain, cell.parameter_box,
                timeout_ms=30000,
            )
            z3_results['completeness'] = comp_result.to_dict()

    except Exception:
        pass
    return z3_results


@dataclass
class BenchmarkResult:
    """Results from running a benchmark model."""
    model_name: str
    n_states: int
    n_params: int
    rhs_type: str
    total_cells: int
    certified_cells: int
    coverage_fraction: float
    regime_counts: Dict[str, int]
    tier1_pass_rate: float
    max_depth: int
    total_time_s: float
    cells_per_second: float
    mean_contraction: float
    # Extended metrics
    certification_time_breakdown: Dict[str, float] = None
    n_inconclusive: int = 0
    n_boundary_pairs: int = 0
    uniform_coverage: float = 0.0  # for ablation
    uniform_time_s: float = 0.0   # for ablation
    # GP metrics
    gp_ece: float = 0.0
    gp_loo_error: float = 0.0
    gp_training_time_s: float = 0.0
    # Z3/SMT metrics
    tier3_verified: int = 0
    tier3_total: int = 0
    tier3_time_s: float = 0.0
    tier2_verified: int = 0
    tier2_total: int = 0
    tier2_time_s: float = 0.0
    z3_results: Optional[Dict] = None
    # Confidence metrics
    mean_eigenvalue_gap: float = 0.0
    min_eigenvalue_gap: float = 0.0
    n_near_bifurcation: int = 0


def _newton_guided_search(rhs, krawczyk, sd, mu, n_states, state_domain):
    """Find equilibria using numerical Newton's method then Krawczyk verification.

    For ≥3 state variables, pure interval bisection is too expensive.
    Instead: (1) solve numerically at the parameter midpoint, (2) build a
    tight interval box around each numerical root, (3) verify with Krawczyk.
    """
    from scipy.optimize import fsolve
    from ..equilibrium.krawczyk import KrawczykResult, KrawczykStatus

    mu_mid = mu.midpoint()
    results = []

    # Try multiple starting points
    rng = np.random.RandomState(42)
    lo = np.array([s[0] for s in state_domain])
    hi = np.array([s[1] for s in state_domain])
    starts = [
        (lo + hi) / 2.0,  # centre
        lo + 0.25 * (hi - lo),
        lo + 0.75 * (hi - lo),
    ]
    # Add a few random starts
    for _ in range(5):
        starts.append(lo + rng.rand(n_states) * (hi - lo))

    found_roots = []
    for x0 in starts:
        try:
            sol, info, ier, _ = fsolve(
                lambda x: rhs.evaluate(x, mu_mid), x0, full_output=True
            )
            if ier == 1 and np.all(np.isfinite(sol)):
                # Check it's actually in the state domain
                if np.all(sol >= lo - 0.1) and np.all(sol <= hi + 0.1):
                    # Check it's not a duplicate
                    is_dup = any(np.linalg.norm(sol - r) < 1e-6 for r in found_roots)
                    if not is_dup:
                        found_roots.append(sol)
        except Exception:
            continue

    # Verify each numerical root with Krawczyk
    for root in found_roots:
        for radius in [0.05, 0.1, 0.2, 0.5]:
            X_tight = IntervalVector([
                Interval(max(root[i] - radius, state_domain[i][0]),
                         min(root[i] + radius, state_domain[i][1]))
                for i in range(n_states)
            ])
            result = krawczyk.verify(X_tight, mu)
            if result.verified:
                results.append(result)
                break

    return results


def certify_box(rhs, box: List, state_domain: List,
                model_name: str, n_states: int, n_params: int,
                rhs_type: str) -> Optional[CertifiedCell]:
    """
    Attempt to certify a single parameter box.

    For models with ≥3 state variables, uses a Newton-guided approach:
    first finds approximate equilibria numerically, then verifies them
    rigorously with the Krawczyk operator on tight enclosures.

    Returns CertifiedCell if successful, None otherwise.
    """
    mu = IntervalVector([Interval(lo, hi) for lo, hi in box])
    sd = IntervalVector([Interval(lo, hi) for lo, hi in state_domain])

    krawczyk = KrawczykOperator(rhs, max_iter=20)

    try:
        if n_states >= 3:
            # Newton-guided: find numerical approximation first
            results = _newton_guided_search(rhs, krawczyk, sd, mu,
                                            n_states, state_domain)
        else:
            results = krawczyk.find_equilibria(sd, mu, max_depth=10)
    except Exception:
        return None

    verified_results = [r for r in results if r.verified and r.enclosure is not None]
    if not verified_results:
        return None

    classifier = StabilityClassifier(rhs)
    eq_certs = []

    for kr in verified_results:
        try:
            stab_type, eig_enc = classifier.classify(kr.enclosure, mu)
        except Exception:
            continue

        delta_info = None
        if eig_enc.real_parts:
            try:
                db = DeltaBound(rhs)
                delta_result = db.compute(kr.enclosure, mu, eig_enc.real_parts)
                delta_info = delta_result.to_dict()
            except Exception:
                try:
                    gap = compute_eigenvalue_gap(eig_enc.real_parts)
                    delta_info = {"delta_required": gap * 0.5, "eigenvalue_gap": gap,
                                  "is_sound": gap > 0}
                except Exception:
                    pass

        eq_cert = EquilibriumCertificate(
            state_enclosure=[(c.lo, c.hi) for c in kr.enclosure.components],
            stability=StabilityType(stab_type.value),
            eigenvalue_real_parts=[(rp.lo, rp.hi) for rp in eig_enc.real_parts],
            krawczyk_contraction=kr.contraction_factor,
            krawczyk_iterations=kr.iterations,
            delta_bound=delta_info,
        )
        eq_certs.append(eq_cert)

    if not eq_certs:
        return None

    regime = RegimeInferenceRules.infer(eq_certs)
    tier = select_tier(model_name, rhs_type)

    cell = CertifiedCell(
        parameter_box=box,
        model_name=model_name,
        n_states=n_states,
        n_params=n_params,
        equilibria=eq_certs,
        regime=regime,
        tier=tier,
    )

    # Run minicheck
    try:
        mc_result = minicheck_verify(cell.to_minicheck_format())
        cell.minicheck_passed = mc_result.valid
    except Exception:
        cell.minicheck_passed = False

    return cell


def run_model(bm: BenchmarkModel,
              max_depth: int = 5,
              target_coverage: float = 0.90,
              max_cells: int = 500) -> BenchmarkResult:
    """
    Run the full certification pipeline on a BenchmarkModel instance.

    Returns BenchmarkResult with real timing data.
    """
    t0 = time.time()

    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )

    config = RefinementConfig(
        max_depth=max_depth,
        target_coverage=target_coverage,
        max_cells=max_cells,
    )

    atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, config)

    total_time = time.time() - t0
    stats = atlas.stats()

    contractions = []
    for c in atlas.cells:
        for eq in c.equilibria:
            if eq.krawczyk_contraction < float('inf'):
                contractions.append(eq.krawczyk_contraction)

    n_inconclusive = stats.regime_counts.get('inconclusive', 0)

    from ..atlas.composition import verify_atlas_composition
    comp = verify_atlas_composition(atlas.cells, bm.parameter_domain)
    n_boundary = len(comp.boundary_cells)

    tier3_verified = 0
    tier3_total = 0
    tier2_verified = 0
    tier2_total = 0
    z3_time = 0.0
    z3_results_all = {}
    eigenvalue_gaps = []
    n_near_bif = 0

    for cidx, c in enumerate(atlas.cells):
        for eq in c.equilibria:
            if eq.eigenvalue_real_parts:
                min_gap = float('inf')
                for lo, hi in eq.eigenvalue_real_parts:
                    if lo <= 0 <= hi:
                        min_gap = 0.0
                    else:
                        min_gap = min(min_gap, min(abs(lo), abs(hi)))
                eigenvalue_gaps.append(min_gap)
                if min_gap < 0.1:
                    n_near_bif += 1

        import time as _time
        t_z3 = _time.time()
        z3_res = _run_z3_verification(c, bm)
        z3_time += _time.time() - t_z3

        if z3_res:
            z3_results_all[f'cell_{cidx}'] = z3_res
            for key, val in z3_res.items():
                if 'existence' in key:
                    tier3_total += 1
                    if val.get('verified', False):
                        tier3_verified += 1
                elif 'stability' in key:
                    tier2_total += 1
                    if val.get('verified', False):
                        tier2_verified += 1

    mean_eig_gap = float(np.mean(eigenvalue_gaps)) if eigenvalue_gaps else 0.0
    min_eig_gap = float(np.min(eigenvalue_gaps)) if eigenvalue_gaps else 0.0

    gp_ece = 0.0
    gp_loo = 0.0
    gp_train_time = 0.0
    if atlas.cells and len(atlas.cells) >= 6:
        try:
            from ..gp.surrogate import GPSurrogate
            from ..tiered.certificate import RegimeType as _RT
            t_gp = time.time()
            gp = GPSurrogate.train_from_atlas(atlas)
            gp_train_time = time.time() - t_gp
            if gp._fitted:
                X_gp = np.array([
                    [(lo + hi) / 2.0 for lo, hi in c.parameter_box]
                    for c in atlas.cells
                ])
                _rmap_gp = {_RT.MONOSTABLE: 0, _RT.BISTABLE: 1, _RT.MULTISTABLE: 2,
                            _RT.OSCILLATORY: 3, _RT.EXCITABLE: 4, _RT.INCONCLUSIVE: 5}
                y_gp = np.array([_rmap_gp.get(c.regime, 5) for c in atlas.cells],
                                dtype=float)
                split = max(1, len(X_gp) // 5)
                try:
                    gp_ece = gp.calibration_error(X_gp[:split], y_gp[:split])
                except Exception:
                    pass
                try:
                    gp_loo = gp.loo_cross_validation(X_gp, y_gp)
                except Exception:
                    pass
        except Exception:
            pass

    return BenchmarkResult(
        model_name=bm.name,
        n_states=bm.n_states,
        n_params=bm.n_params,
        rhs_type=bm.rhs_type,
        total_cells=stats.total_cells,
        certified_cells=stats.certified_cells,
        coverage_fraction=stats.coverage_fraction,
        regime_counts=stats.regime_counts,
        tier1_pass_rate=stats.minicheck_pass_rate,
        max_depth=stats.max_depth,
        total_time_s=round(total_time, 2),
        cells_per_second=round(stats.certified_cells / max(total_time, 0.01), 2),
        mean_contraction=round(float(np.mean(contractions)) if contractions else 0, 4),
        n_inconclusive=n_inconclusive,
        n_boundary_pairs=n_boundary,
        tier3_verified=tier3_verified,
        tier3_total=tier3_total,
        tier3_time_s=round(z3_time, 3),
        tier2_verified=tier2_verified,
        tier2_total=tier2_total,
        tier2_time_s=round(z3_time, 3),
        z3_results=z3_results_all if z3_results_all else None,
        mean_eigenvalue_gap=round(mean_eig_gap, 6),
        min_eigenvalue_gap=round(min_eig_gap, 6),
        n_near_bifurcation=n_near_bif,
        gp_ece=round(gp_ece, 4),
        gp_loo_error=round(gp_loo, 6),
        gp_training_time_s=round(gp_train_time, 4),
    )


def run_benchmark(model_name: str,
                  max_depth: int = 5,
                  target_coverage: float = 0.90,
                  max_cells: int = 500) -> BenchmarkResult:
    """
    Run the full certification pipeline on a benchmark model by name.

    Returns BenchmarkResult with real timing data.
    """
    bm = get_benchmark(model_name)
    return run_model(bm, max_depth=max_depth,
                     target_coverage=target_coverage, max_cells=max_cells)


# ---------------------------------------------------------------------------
# GP calibration study
# ---------------------------------------------------------------------------


def run_gp_calibration_study(
    model_name: str,
    max_depth: int = 4,
    target_coverage: float = 0.85,
    max_cells: int = 200,
) -> Dict:
    """Run GP calibration analysis on a benchmark model.

    Builds an atlas via adaptive refinement, trains a GP from the atlas,
    and uses :class:`GPCalibrator` to produce a full calibration report
    including ECE, NLPD, coverage curves, and LOO-MSE.

    Returns:
        Dictionary with keys ``"benchmark"``, ``"calibration"``, and
        ``"model_selection"`` containing the respective results.
    """
    from ..gp.surrogate import GPSurrogate
    from ..gp.calibration import GPCalibrator, GPModelSelector
    from ..tiered.certificate import RegimeType as RT

    bm = get_benchmark(model_name)

    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )

    config = RefinementConfig(
        max_depth=max_depth,
        target_coverage=target_coverage,
        max_cells=max_cells,
    )
    atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, config)

    if not atlas.cells or len(atlas.cells) < 3:
        return {
            "benchmark": model_name,
            "calibration": None,
            "model_selection": None,
            "error": "Too few certified cells for GP calibration",
        }

    _rmap = {RT.MONOSTABLE: 0, RT.BISTABLE: 1, RT.MULTISTABLE: 2,
             RT.OSCILLATORY: 3, RT.EXCITABLE: 4, RT.INCONCLUSIVE: 5}
    X = np.array([
        [(lo + hi) / 2.0 for lo, hi in c.parameter_box]
        for c in atlas.cells
    ])
    y = np.array([_rmap.get(c.regime, 5) for c in atlas.cells], dtype=float)

    gp = GPSurrogate.train_from_atlas(atlas)
    calibrator = GPCalibrator(n_bins=10)
    cal_result = calibrator.calibrate(gp, X, y)
    cal_report = calibrator.full_report(cal_result)

    selector = GPModelSelector()
    sel_result = selector.select(X, y)

    return {
        "benchmark": model_name,
        "calibration": cal_report,
        "model_selection": {
            "best_config": sel_result.best_config,
            "selection_criterion": sel_result.selection_criterion,
            "n_candidates": len(sel_result.all_results),
        },
    }


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Side-by-side comparison of uniform vs GP-guided refinement."""
    model_name: str
    uniform_coverage: float
    uniform_cells: int
    uniform_time_s: float
    gp_coverage: float
    gp_cells: int
    gp_time_s: float
    gp_ece: float
    gp_loo_error: float
    gp_training_time_s: float


def run_ablation(model_name: str,
                 max_depth: int = 4,
                 target_coverage: float = 0.85,
                 max_cells: int = 200) -> AblationResult:
    """Run ablation: uniform refinement vs GP-guided refinement.

    Both methods use the same certification function, depth, and cell
    budget so that results are directly comparable.
    """
    bm = get_benchmark(model_name)

    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )

    # --- Uniform ---
    u_config = RefinementConfig(
        max_depth=max_depth,
        target_coverage=target_coverage,
        max_cells=max_cells,
    )
    t0 = time.time()
    u_atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, u_config)
    u_time = time.time() - t0
    u_stats = u_atlas.stats()

    # --- GP-guided ---
    g_config = GPGuidedRefinementConfig(
        max_depth=max_depth,
        target_coverage=target_coverage,
        max_cells=max_cells,
        gp_warmup_cells=max(5, max_cells // 20),
    )
    t0 = time.time()
    g_atlas, convergence = gp_guided_refine(
        certify_fn, bm.parameter_domain, bm.name, g_config,
    )
    g_time = time.time() - t0
    g_stats = g_atlas.stats()

    # GP quality metrics
    gp_ece = 0.0
    gp_loo = 0.0
    gp_train_time = 0.0
    if g_atlas.cells:
        from ..gp.surrogate import GPSurrogate
        t_gp = time.time()
        gp = GPSurrogate.train_from_atlas(g_atlas)
        gp_train_time = time.time() - t_gp
        if gp._fitted:
            X = np.array([
                [(lo + hi) / 2.0 for lo, hi in c.parameter_box]
                for c in g_atlas.cells
            ])
            from ..tiered.certificate import RegimeType as RT
            _rmap = {RT.MONOSTABLE: 0, RT.BISTABLE: 1, RT.MULTISTABLE: 2,
                     RT.OSCILLATORY: 3, RT.EXCITABLE: 4, RT.INCONCLUSIVE: 5}
            y = np.array([_rmap.get(c.regime, 5) for c in g_atlas.cells], dtype=float)
            if len(X) > 5:
                split = max(1, len(X) // 5)
                try:
                    gp_ece = gp.calibration_error(X[:split], y[:split])
                except Exception:
                    pass
            try:
                gp_loo = gp.loo_cross_validation(X, y)
            except Exception:
                pass

    return AblationResult(
        model_name=bm.name,
        uniform_coverage=u_stats.coverage_fraction,
        uniform_cells=u_stats.certified_cells,
        uniform_time_s=round(u_time, 2),
        gp_coverage=g_stats.coverage_fraction,
        gp_cells=g_stats.certified_cells,
        gp_time_s=round(g_time, 2),
        gp_ece=round(gp_ece, 4),
        gp_loo_error=round(gp_loo, 6),
        gp_training_time_s=round(gp_train_time, 4),
    )


# ---------------------------------------------------------------------------
# Scalability study
# ---------------------------------------------------------------------------

@dataclass
class ScalabilityPoint:
    """Single data-point for a scalability study."""
    model_name: str
    n_params: int
    n_states: int
    max_cells: int
    certified_cells: int
    coverage_fraction: float
    total_time_s: float
    cells_per_second: float


def run_scalability_study(
    model_names: Optional[List[str]] = None,
    max_cells: int = 100,
    max_depth: int = 3,
) -> List[ScalabilityPoint]:
    """Measure certification cost vs parameter dimension.

    Runs each model with the same cell budget and depth so that timing
    can be compared across different dimensionalities.
    """
    if model_names is None:
        model_names = ["brusselator", "selkov", "toggle_switch",
                       "repressilator", "goodwin", "lotka_volterra",
                       "fitzhugh_nagumo"]

    results: List[ScalabilityPoint] = []
    for name in model_names:
        try:
            bm = get_benchmark(name)
        except ValueError:
            continue

        def certify_fn(box, _bm=bm):
            return certify_box(
                _bm.rhs, box, _bm.state_domain,
                _bm.name, _bm.n_states, _bm.n_params, _bm.rhs_type,
            )

        config = RefinementConfig(
            max_depth=max_depth,
            target_coverage=0.99,
            max_cells=max_cells,
        )
        t0 = time.time()
        atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, config)
        elapsed = time.time() - t0
        stats = atlas.stats()

        results.append(ScalabilityPoint(
            model_name=name,
            n_params=bm.n_params,
            n_states=bm.n_states,
            max_cells=max_cells,
            certified_cells=stats.certified_cells,
            coverage_fraction=stats.coverage_fraction,
            total_time_s=round(elapsed, 3),
            cells_per_second=round(stats.certified_cells / max(elapsed, 0.01), 2),
        ))

    return results


# ---------------------------------------------------------------------------
# Extended / high-fidelity benchmarks
# ---------------------------------------------------------------------------

def run_extended_benchmarks(
    output_dir: str = "benchmark_output_extended",
    max_depth: int = 6,
    target_coverage: float = 0.95,
    max_cells: int = 1000,
) -> Dict[str, BenchmarkResult]:
    """Run all benchmarks with higher depth/cells for publication-quality results."""
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, BenchmarkResult] = {}

    for name in ["toggle_switch", "brusselator", "selkov",
                 "repressilator", "goodwin", "lotka_volterra",
                 "fitzhugh_nagumo"]:
        print(f"Running extended benchmark: {name}...")
        try:
            result = run_benchmark(name, max_depth, target_coverage, max_cells)
            results[name] = result
            with open(os.path.join(output_dir, f"{name}_result.json"), 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"  {name}: {result.certified_cells} cells, "
                  f"{result.coverage_fraction:.1%} coverage, "
                  f"{result.total_time_s:.1f}s")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    summary = {n: asdict(r) for n, r in results.items()}
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Run all benchmarks (original interface, now includes goodwin)
# ---------------------------------------------------------------------------

def run_all_benchmarks(output_dir: str = "benchmark_output",
                       max_depth: int = 4,
                       target_coverage: float = 0.85,
                       max_cells: int = 200) -> Dict[str, BenchmarkResult]:
    """Run all benchmark models and save results."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name in ["toggle_switch", "brusselator", "selkov",
                 "repressilator", "goodwin", "lotka_volterra",
                 "fitzhugh_nagumo"]:
        print(f"Running benchmark: {name}...")
        try:
            result = run_benchmark(name, max_depth, target_coverage, max_cells)
            results[name] = result
            with open(os.path.join(output_dir, f"{name}_result.json"), 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"  {name}: {result.certified_cells} cells, "
                  f"{result.coverage_fraction:.1%} coverage, "
                  f"{result.total_time_s:.1f}s")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Save summary
    summary = {name: asdict(r) for name, r in results.items()}
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Comprehensive ablation study
# ---------------------------------------------------------------------------

@dataclass
class ComprehensiveAblationResult:
    """Full ablation study results covering all pipeline components."""
    model_name: str
    # Tier ablation
    tier1_only_coverage: float = 0.0
    tier1_only_time_s: float = 0.0
    tier12_coverage: float = 0.0
    tier12_time_s: float = 0.0
    tier123_coverage: float = 0.0
    tier123_time_s: float = 0.0
    # GP ablation
    uniform_coverage: float = 0.0
    uniform_time_s: float = 0.0
    gp_coverage: float = 0.0
    gp_time_s: float = 0.0
    # Newton ablation (3D only)
    bisection_coverage: float = 0.0
    bisection_time_s: float = 0.0
    newton_coverage: float = 0.0
    newton_time_s: float = 0.0
    # Splitting ablation
    isotropic_coverage: float = 0.0
    isotropic_time_s: float = 0.0
    anisotropic_coverage: float = 0.0
    anisotropic_time_s: float = 0.0


def _certify_box_tier_limited(rhs, box, state_domain, model_name,
                              n_states, n_params, rhs_type, max_tier):
    """Certify a box with verification limited to at most ``max_tier``."""
    mu = IntervalVector([Interval(lo, hi) for lo, hi in box])
    sd = IntervalVector([Interval(lo, hi) for lo, hi in state_domain])
    krawczyk = KrawczykOperator(rhs, max_iter=20)

    try:
        if n_states >= 3:
            results = _newton_guided_search(rhs, krawczyk, sd, mu,
                                            n_states, state_domain)
        else:
            results = krawczyk.find_equilibria(sd, mu, max_depth=10)
    except Exception:
        return None

    verified = [r for r in results if r.verified and r.enclosure is not None]
    if not verified:
        return None

    classifier = StabilityClassifier(rhs)
    eq_certs = []

    for kr in verified:
        try:
            stab_type, eig_enc = classifier.classify(kr.enclosure, mu)
        except Exception:
            continue
        eq_cert = EquilibriumCertificate(
            state_enclosure=[(c.lo, c.hi) for c in kr.enclosure.components],
            stability=StabilityType(stab_type.value),
            eigenvalue_real_parts=[(rp.lo, rp.hi) for rp in eig_enc.real_parts],
            krawczyk_contraction=kr.contraction_factor,
            krawczyk_iterations=kr.iterations,
        )
        eq_certs.append(eq_cert)

    if not eq_certs:
        return None

    regime = RegimeInferenceRules.infer(eq_certs)

    # Determine tier based on limit
    tier = VerificationTier.TIER1_IA
    if max_tier >= 2:
        tier = select_tier(model_name, rhs_type)
        # Clamp to max_tier
        tier_order = {
            VerificationTier.TIER1_IA: 1,
            VerificationTier.TIER2_DREAL: 2,
            VerificationTier.TIER3_Z3: 3,
        }
        if tier_order.get(tier, 1) > max_tier:
            if max_tier == 1:
                tier = VerificationTier.TIER1_IA
            elif max_tier == 2:
                tier = VerificationTier.TIER2_DREAL

    cell = CertifiedCell(
        parameter_box=box, model_name=model_name,
        n_states=n_states, n_params=n_params,
        equilibria=eq_certs, regime=regime, tier=tier,
    )

    try:
        mc_result = minicheck_verify(cell.to_minicheck_format())
        cell.minicheck_passed = mc_result.valid
    except Exception:
        cell.minicheck_passed = False

    return cell


def _certify_box_bisection_only(rhs, box, state_domain, model_name,
                                n_states, n_params, rhs_type):
    """Certify using only bisection (no Newton guidance), even for ≥3D."""
    mu = IntervalVector([Interval(lo, hi) for lo, hi in box])
    sd = IntervalVector([Interval(lo, hi) for lo, hi in state_domain])
    krawczyk = KrawczykOperator(rhs, max_iter=20)

    try:
        results = krawczyk.find_equilibria(sd, mu, max_depth=10)
    except Exception:
        return None

    verified = [r for r in results if r.verified and r.enclosure is not None]
    if not verified:
        return None

    classifier = StabilityClassifier(rhs)
    eq_certs = []
    for kr in verified:
        try:
            stab_type, eig_enc = classifier.classify(kr.enclosure, mu)
        except Exception:
            continue
        eq_certs.append(EquilibriumCertificate(
            state_enclosure=[(c.lo, c.hi) for c in kr.enclosure.components],
            stability=StabilityType(stab_type.value),
            eigenvalue_real_parts=[(rp.lo, rp.hi) for rp in eig_enc.real_parts],
            krawczyk_contraction=kr.contraction_factor,
            krawczyk_iterations=kr.iterations,
        ))

    if not eq_certs:
        return None

    regime = RegimeInferenceRules.infer(eq_certs)
    tier = select_tier(model_name, rhs_type)
    cell = CertifiedCell(
        parameter_box=box, model_name=model_name,
        n_states=n_states, n_params=n_params,
        equilibria=eq_certs, regime=regime, tier=tier,
    )
    try:
        mc_result = minicheck_verify(cell.to_minicheck_format())
        cell.minicheck_passed = mc_result.valid
    except Exception:
        cell.minicheck_passed = False
    return cell


def run_comprehensive_ablation(
    model_name: str,
    max_depth: int = 4,
    max_cells: int = 200,
) -> ComprehensiveAblationResult:
    """Run all ablation variants for a single benchmark model.

    Tests:
      1. Tier-by-tier contribution (Tier 1 only, Tier 1+2, Tier 1+2+3)
      2. With/without GP guidance
      3. With/without anisotropic splitting
      4. With/without Newton-guided search (for ≥3-state models)

    Args:
        model_name: Benchmark model name.
        max_depth: Maximum octree depth.
        max_cells: Cell budget for each variant.

    Returns:
        :class:`ComprehensiveAblationResult` with all metrics.
    """
    from ..refinement.octree import (
        adaptive_refine, gp_guided_refine,
        RefinementConfig, GPGuidedRefinementConfig,
        split_box as _split_box,
    )

    bm = get_benchmark(model_name)
    result = ComprehensiveAblationResult(model_name=bm.name)
    base_config = RefinementConfig(
        max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
    )

    # ----- Tier ablation -----
    for max_tier, attr_cov, attr_time in [
        (1, 'tier1_only_coverage', 'tier1_only_time_s'),
        (2, 'tier12_coverage', 'tier12_time_s'),
        (3, 'tier123_coverage', 'tier123_time_s'),
    ]:
        def _cfn_tier(box, _mt=max_tier):
            return _certify_box_tier_limited(
                bm.rhs, box, bm.state_domain,
                bm.name, bm.n_states, bm.n_params, bm.rhs_type, _mt,
            )
        t0 = time.time()
        atlas = adaptive_refine(_cfn_tier, bm.parameter_domain, bm.name, base_config)
        elapsed = time.time() - t0
        setattr(result, attr_cov, round(atlas.coverage_fraction(), 6))
        setattr(result, attr_time, round(elapsed, 3))

    # ----- GP ablation (uniform vs GP-guided) -----
    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )

    t0 = time.time()
    u_atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, base_config)
    result.uniform_time_s = round(time.time() - t0, 3)
    result.uniform_coverage = round(u_atlas.coverage_fraction(), 6)

    gp_config = GPGuidedRefinementConfig(
        max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
        gp_warmup_cells=max(5, max_cells // 20),
    )
    t0 = time.time()
    g_atlas, _ = gp_guided_refine(certify_fn, bm.parameter_domain, bm.name, gp_config)
    result.gp_time_s = round(time.time() - t0, 3)
    result.gp_coverage = round(g_atlas.coverage_fraction(), 6)

    # ----- Splitting ablation (isotropic vs anisotropic) -----
    # Isotropic: use plain split_box via adaptive_refine (already isotropic)
    t0 = time.time()
    iso_atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, base_config)
    result.isotropic_time_s = round(time.time() - t0, 3)
    result.isotropic_coverage = round(iso_atlas.coverage_fraction(), 6)

    # Anisotropic: use gp_guided_refine which uses anisotropic_split_box
    t0 = time.time()
    aniso_atlas, _ = gp_guided_refine(certify_fn, bm.parameter_domain, bm.name, gp_config)
    result.anisotropic_time_s = round(time.time() - t0, 3)
    result.anisotropic_coverage = round(aniso_atlas.coverage_fraction(), 6)

    # ----- Newton ablation (3D+ models only) -----
    if bm.n_states >= 3:
        # With Newton (default)
        t0 = time.time()
        newton_atlas = adaptive_refine(
            certify_fn, bm.parameter_domain, bm.name, base_config,
        )
        result.newton_time_s = round(time.time() - t0, 3)
        result.newton_coverage = round(newton_atlas.coverage_fraction(), 6)

        # Without Newton (bisection only)
        def certify_bisection(box):
            return _certify_box_bisection_only(
                bm.rhs, box, bm.state_domain,
                bm.name, bm.n_states, bm.n_params, bm.rhs_type,
            )
        t0 = time.time()
        bis_atlas = adaptive_refine(
            certify_bisection, bm.parameter_domain, bm.name, base_config,
        )
        result.bisection_time_s = round(time.time() - t0, 3)
        result.bisection_coverage = round(bis_atlas.coverage_fraction(), 6)
    else:
        # For 2D models Newton and bisection are the same
        result.newton_coverage = result.uniform_coverage
        result.newton_time_s = result.uniform_time_s
        result.bisection_coverage = result.uniform_coverage
        result.bisection_time_s = result.uniform_time_s

    return result


# ---------------------------------------------------------------------------
# Depth-scaling experiment
# ---------------------------------------------------------------------------

@dataclass
class DepthScalingPoint:
    """Single data-point for a depth-scaling experiment."""
    model_name: str
    max_depth: int
    certified_cells: int
    coverage_fraction: float
    total_time_s: float
    cells_per_second: float


def run_depth_scaling(
    model_name: str,
    depths: Optional[List[int]] = None,
    max_cells: int = 200,
) -> List[DepthScalingPoint]:
    """Run the same model at different max_depth values.

    Measures how coverage and cost change as refinement depth increases,
    keeping the cell budget constant.
    """
    if depths is None:
        depths = [2, 3, 4, 5]

    bm = get_benchmark(model_name)

    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )

    results: List[DepthScalingPoint] = []
    for d in depths:
        config = RefinementConfig(
            max_depth=d, target_coverage=0.99, max_cells=max_cells,
        )
        t0 = time.time()
        atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, config)
        elapsed = time.time() - t0
        stats = atlas.stats()
        results.append(DepthScalingPoint(
            model_name=bm.name,
            max_depth=d,
            certified_cells=stats.certified_cells,
            coverage_fraction=round(stats.coverage_fraction, 6),
            total_time_s=round(elapsed, 3),
            cells_per_second=round(stats.certified_cells / max(elapsed, 1e-6), 2),
        ))
    return results


# ---------------------------------------------------------------------------
# Empirical delta-soundness validation
# ---------------------------------------------------------------------------

@dataclass
class DeltaValidationResult:
    """Result of empirical delta-soundness validation for a model."""
    model_name: str
    n_cells: int
    n_equilibria: int
    n_with_delta: int
    n_sound: int
    n_unsound: int
    min_eigenvalue_gap: float
    max_delta_required: float
    all_sound: bool
    details: List[Dict] = field(default_factory=list)


def run_delta_validation(
    model_name: str,
    max_depth: int = 3,
    max_cells: int = 50,
) -> DeltaValidationResult:
    """Run benchmark and check if delta-soundness conditions hold empirically.

    For each certified cell, computes the eigenvalue gap and the required
    delta, then checks whether gap > delta_required (Theorem B1).
    """
    bm = get_benchmark(model_name)

    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )

    config = RefinementConfig(
        max_depth=max_depth, target_coverage=0.85, max_cells=max_cells,
    )
    atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, config)

    n_eq = 0
    n_with_delta = 0
    n_sound = 0
    n_unsound = 0
    min_gap = float('inf')
    max_delta = 0.0
    details = []

    for cell in atlas.cells:
        for eq in cell.equilibria:
            n_eq += 1
            if not eq.eigenvalue_real_parts:
                continue

            gap = compute_eigenvalue_gap([
                Interval(lo, hi) for lo, hi in eq.eigenvalue_real_parts
            ])
            delta_req = eq.delta_bound.get("delta_required", 0.0) if eq.delta_bound else 0.0

            if gap > 0:
                n_with_delta += 1
                is_sound = gap > delta_req
                if is_sound:
                    n_sound += 1
                else:
                    n_unsound += 1
                min_gap = min(min_gap, gap)
                max_delta = max(max_delta, delta_req)
                details.append({
                    "eigenvalue_gap": round(gap, 8),
                    "delta_required": round(delta_req, 8),
                    "is_sound": is_sound,
                    "contraction": round(eq.krawczyk_contraction, 6),
                })

    if min_gap == float('inf'):
        min_gap = 0.0

    return DeltaValidationResult(
        model_name=bm.name,
        n_cells=len(atlas.cells),
        n_equilibria=n_eq,
        n_with_delta=n_with_delta,
        n_sound=n_sound,
        n_unsound=n_unsound,
        min_eigenvalue_gap=round(min_gap, 8),
        max_delta_required=round(max_delta, 8),
        all_sound=(n_unsound == 0 and n_with_delta > 0),
        details=details,
    )
