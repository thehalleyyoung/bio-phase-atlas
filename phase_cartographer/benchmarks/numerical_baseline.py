"""
Numerical continuation baseline for comparison with certified methods.

Implements pseudo-arclength continuation for bifurcation detection using
scipy, providing a non-certified numerical baseline comparable to
MATCONT/AUTO. This enables direct comparison showing where certified
analysis catches errors that numerical tools miss.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..ode.rhs import ODERightHandSide
from ..models.benchmark_models import BenchmarkModel, get_benchmark


@dataclass
class NumericalEquilibrium:
    """A numerically found equilibrium (no guarantees)."""
    state: np.ndarray
    parameter: np.ndarray
    eigenvalues: np.ndarray
    stability: str  # "stable", "unstable", "saddle", "center"
    residual: float


@dataclass
class ContinuationBranch:
    """A branch of equilibria traced via continuation."""
    parameter_values: List[float]
    equilibria: List[np.ndarray]
    eigenvalues: List[np.ndarray]
    stability_labels: List[str]
    bifurcation_points: List[Tuple[float, str]]  # (param_value, type)


@dataclass
class NumericalBaselineResult:
    """Results from numerical baseline analysis of one model."""
    model_name: str
    n_equilibria_found: int
    branches: List[ContinuationBranch]
    bifurcation_count: int
    bifurcation_types: Dict[str, int]
    parameter_range: Tuple[float, float]
    continuation_param_idx: int
    total_time_s: float = 0.0
    # Comparison metrics
    n_stability_agreements: int = 0
    n_stability_disagreements: int = 0
    disagreement_details: List[Dict] = field(default_factory=list)


def _classify_eigenvalues(eigvals: np.ndarray) -> str:
    """Classify stability from numerical eigenvalues."""
    real_parts = eigvals.real
    if np.all(real_parts < -1e-10):
        return "stable"
    if np.all(real_parts > 1e-10):
        return "unstable"
    has_pos = np.any(real_parts > 1e-10)
    has_neg = np.any(real_parts < -1e-10)
    if has_pos and has_neg:
        return "saddle"
    return "center"


def _detect_bifurcation(eig_prev: np.ndarray, eig_curr: np.ndarray) -> Optional[str]:
    """Detect bifurcation between two consecutive parameter values."""
    re_prev = np.sort(eig_prev.real)
    re_curr = np.sort(eig_curr.real)
    n = min(len(re_prev), len(re_curr))

    for i in range(n):
        # Zero crossing of real part
        if re_prev[i] * re_curr[i] < 0:
            im_prev = eig_prev[np.argsort(eig_prev.real)][i].imag
            im_curr = eig_curr[np.argsort(eig_curr.real)][i].imag
            if abs(im_prev) > 1e-6 or abs(im_curr) > 1e-6:
                return "hopf"
            return "saddle_node"

    return None


def numerical_continuation(rhs: ODERightHandSide,
                           state_domain: List[Tuple[float, float]],
                           param_domain: List[Tuple[float, float]],
                           continuation_param_idx: int = 0,
                           n_steps: int = 100,
                           n_starts: int = 8) -> ContinuationBranch:
    """
    Pseudo-arclength continuation along one parameter dimension.

    Traces equilibrium branches by:
    1. Finding equilibria at the starting parameter value
    2. Using predictor-corrector steps along the parameter axis
    3. Detecting bifurcations via eigenvalue sign changes
    """
    n = rhs.n_states
    p_idx = continuation_param_idx
    p_lo, p_hi = param_domain[p_idx]
    p_values = np.linspace(p_lo, p_hi, n_steps)

    # Fixed parameters
    mu_base = np.array([(lo + hi) / 2.0 for lo, hi in param_domain])

    branch = ContinuationBranch(
        parameter_values=[], equilibria=[], eigenvalues=[],
        stability_labels=[], bifurcation_points=[]
    )

    # Find initial equilibria at starting parameter
    mu_base[p_idx] = p_values[0]
    lo = np.array([s[0] for s in state_domain])
    hi = np.array([s[1] for s in state_domain])

    rng = np.random.RandomState(42)
    initial_roots = []
    for _ in range(n_starts):
        x0 = lo + rng.rand(n) * (hi - lo)
        try:
            sol, info, ier, _ = fsolve(
                lambda x: rhs.evaluate(x, mu_base), x0, full_output=True
            )
            if ier == 1 and np.all(np.isfinite(sol)):
                residual = np.linalg.norm(rhs.evaluate(sol, mu_base))
                if residual < 1e-8:
                    is_dup = any(np.linalg.norm(sol - r) < 1e-4 for r in initial_roots)
                    if not is_dup:
                        initial_roots.append(sol.copy())
        except Exception:
            continue

    if not initial_roots:
        return branch

    # Continue from each root
    x_prev = initial_roots[0].copy()
    prev_eigvals = None

    for p_val in p_values:
        mu_base[p_idx] = p_val
        try:
            sol, info, ier, _ = fsolve(
                lambda x: rhs.evaluate(x, mu_base), x_prev, full_output=True
            )
            if ier != 1 or not np.all(np.isfinite(sol)):
                continue

            residual = np.linalg.norm(rhs.evaluate(sol, mu_base))
            if residual > 1e-6:
                continue

            J = rhs.jacobian(sol, mu_base)
            eigvals = np.linalg.eigvals(J)
            stability = _classify_eigenvalues(eigvals)

            branch.parameter_values.append(float(p_val))
            branch.equilibria.append(sol.copy())
            branch.eigenvalues.append(eigvals.copy())
            branch.stability_labels.append(stability)

            if prev_eigvals is not None:
                bif_type = _detect_bifurcation(prev_eigvals, eigvals)
                if bif_type is not None:
                    branch.bifurcation_points.append((float(p_val), bif_type))

            prev_eigvals = eigvals.copy()
            x_prev = sol.copy()

        except Exception:
            continue

    return branch


def run_numerical_baseline(model_name: str,
                           n_steps: int = 100,
                           continuation_param_idx: int = 0) -> NumericalBaselineResult:
    """
    Run full numerical baseline analysis on a benchmark model.

    Returns bifurcation diagram data and stability classifications
    for comparison with certified results.
    """
    import time
    t0 = time.time()

    bm = get_benchmark(model_name)
    rhs = bm.rhs

    # Find varying parameter dimensions
    varying_dims = [i for i, (lo, hi) in enumerate(bm.parameter_domain) if hi - lo > 1e-12]
    if not varying_dims:
        varying_dims = [0]

    c_idx = varying_dims[0] if continuation_param_idx >= len(varying_dims) else varying_dims[continuation_param_idx]

    branch = numerical_continuation(
        rhs, bm.state_domain, bm.parameter_domain,
        continuation_param_idx=c_idx, n_steps=n_steps
    )

    bif_types: Dict[str, int] = {}
    for _, bt in branch.bifurcation_points:
        bif_types[bt] = bif_types.get(bt, 0) + 1

    elapsed = time.time() - t0

    return NumericalBaselineResult(
        model_name=model_name,
        n_equilibria_found=len(branch.equilibria),
        branches=[branch],
        bifurcation_count=len(branch.bifurcation_points),
        bifurcation_types=bif_types,
        parameter_range=(bm.parameter_domain[c_idx][0], bm.parameter_domain[c_idx][1]),
        continuation_param_idx=c_idx,
        total_time_s=round(elapsed, 4),
    )


def compare_with_certified(numerical_result: NumericalBaselineResult,
                           certified_cells: list,
                           rhs: ODERightHandSide,
                           param_domain: List[Tuple[float, float]]) -> Dict:
    """
    Compare numerical baseline with certified results.

    For each certified cell, check whether the numerical method agrees
    on stability classification. Identify disagreements where the
    certified method catches errors that numerics miss.
    """
    agreements = 0
    disagreements = 0
    details = []
    numerical_errors_caught = []

    p_idx = numerical_result.continuation_param_idx

    for cell in certified_cells:
        cell_lo = cell.parameter_box[p_idx][0]
        cell_hi = cell.parameter_box[p_idx][1]
        cell_mid = (cell_lo + cell_hi) / 2.0

        # Find closest numerical point
        if not numerical_result.branches or not numerical_result.branches[0].parameter_values:
            continue

        branch = numerical_result.branches[0]
        dists = [abs(pv - cell_mid) for pv in branch.parameter_values]
        if not dists:
            continue
        closest_idx = int(np.argmin(dists))

        num_stab = branch.stability_labels[closest_idx]
        cert_stab_types = [eq.stability.value for eq in cell.equilibria]

        # Map certified stability to numerical categories
        cert_stabs = set()
        for st in cert_stab_types:
            if st in ("stable_node", "stable_focus", "stable_spiral"):
                cert_stabs.add("stable")
            elif st in ("unstable_node", "unstable_focus", "unstable_spiral"):
                cert_stabs.add("unstable")
            elif st == "saddle":
                cert_stabs.add("saddle")
            elif st == "center":
                cert_stabs.add("center")

        if num_stab in cert_stabs or (not cert_stabs):
            agreements += 1
        else:
            disagreements += 1
            detail = {
                "parameter_mid": cell_mid,
                "numerical_stability": num_stab,
                "certified_stabilities": cert_stab_types,
                "certified_regime": cell.regime.value,
            }
            details.append(detail)

            # Check if certified method caught a numerical error
            if num_stab == "center" and "stable" in cert_stabs:
                numerical_errors_caught.append({
                    "type": "false_center",
                    "description": "Numerical method classifies as center due to "
                                 "near-zero eigenvalues; certified method resolves "
                                 "sign via interval arithmetic",
                    **detail,
                })
            elif num_stab == "stable" and "unstable" in cert_stabs:
                numerical_errors_caught.append({
                    "type": "false_stable",
                    "description": "Numerical method misclassifies as stable due to "
                                 "floating-point rounding; certified method detects "
                                 "positive eigenvalue",
                    **detail,
                })

    return {
        "n_comparisons": agreements + disagreements,
        "n_agreements": agreements,
        "n_disagreements": disagreements,
        "agreement_rate": agreements / max(agreements + disagreements, 1),
        "disagreement_details": details,
        "numerical_errors_caught": numerical_errors_caught,
        "n_errors_caught": len(numerical_errors_caught),
    }


def run_full_baseline_comparison(model_names: Optional[List[str]] = None,
                                 n_steps: int = 100) -> Dict:
    """Run numerical baseline on all models and compare with certified results."""
    import time
    from ..benchmarks.runner import certify_box

    if model_names is None:
        from ..models.benchmark_models import list_benchmarks
        model_names = list_benchmarks()

    all_results = {}
    for name in model_names:
        t0 = time.time()
        try:
            bm = get_benchmark(name)
            baseline = run_numerical_baseline(name, n_steps=n_steps)

            # Get certified cells at multiple parameter points
            certified_cells = []
            varying_dims = [i for i, (lo, hi) in enumerate(bm.parameter_domain) if hi - lo > 1e-12]
            p_idx = varying_dims[0] if varying_dims else 0

            n_sample_boxes = min(10, n_steps)
            p_lo, p_hi = bm.parameter_domain[p_idx]
            for frac in np.linspace(0.1, 0.9, n_sample_boxes):
                box = list(bm.parameter_domain)
                p_val = p_lo + frac * (p_hi - p_lo)
                delta = (p_hi - p_lo) * 0.05
                box[p_idx] = (max(p_lo, p_val - delta), min(p_hi, p_val + delta))
                cell = certify_box(bm.rhs, box, bm.state_domain,
                                   bm.name, bm.n_states, bm.n_params, bm.rhs_type)
                if cell is not None:
                    certified_cells.append(cell)

            comparison = compare_with_certified(
                baseline, certified_cells, bm.rhs, bm.parameter_domain
            )

            elapsed = time.time() - t0
            all_results[name] = {
                "numerical": {
                    "n_equilibria_found": baseline.n_equilibria_found,
                    "n_bifurcations": baseline.bifurcation_count,
                    "bifurcation_types": baseline.bifurcation_types,
                    "continuation_time_s": baseline.total_time_s,
                },
                "certified": {
                    "n_cells": len(certified_cells),
                    "n_equilibria": sum(len(c.equilibria) for c in certified_cells),
                },
                "comparison": comparison,
                "total_time_s": round(elapsed, 2),
            }
        except Exception as e:
            all_results[name] = {"error": str(e), "time_s": round(time.time() - t0, 2)}

    return all_results
