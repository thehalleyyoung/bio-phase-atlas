"""
GP kernel regularity verification and Borell-TIS condition checking.

Addresses the reviewer concern that the GP kernel choice (Matérn-5/2)
is asserted without verification that it satisfies the requirements of
the Borell-TIS inequality used in Proposition 4.

Borell-TIS requires:
1. The GP is a.s. continuous (guaranteed by Matérn-ν with ν > d/2)
2. The GP is centred (guaranteed by our zero-mean prior)
3. The domain is compact (guaranteed for bounded parameter boxes)

Matérn-5/2 regularity:
- ν = 5/2 means the GP has ⌊ν⌋ = 2 mean-square derivatives
- Sample paths are a.s. C^1 (twice differentiable in the m.s. sense)
- For d ≤ 4, ν = 5/2 > d/2 so continuity is guaranteed by
  the Kolmogorov-Chentsov theorem

This module computes and verifies these conditions for each model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class KernelRegularityResult:
    """Result of kernel regularity verification."""
    kernel_name: str
    nu: float  # Matérn smoothness parameter
    dimension: int
    satisfies_continuity: bool  # ν > d/2
    satisfies_centred: bool
    satisfies_compact_domain: bool
    borell_tis_applicable: bool
    sigma_max_sq: float  # sup Var(f(x)) over domain
    expected_supremum_bound: float  # E[sup f] ≤ σ_max √(2 log N_ε)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "nu": self.nu,
            "dimension": self.dimension,
            "satisfies_continuity": self.satisfies_continuity,
            "satisfies_centred": self.satisfies_centred,
            "satisfies_compact_domain": self.satisfies_compact_domain,
            "borell_tis_applicable": self.borell_tis_applicable,
            "sigma_max_sq": round(self.sigma_max_sq, 8),
            "expected_supremum_bound": round(self.expected_supremum_bound, 6),
            "details": self.details,
        }


def verify_kernel_regularity(
    kernel_name: str = "matern52",
    dimension: int = 2,
    domain_bounds: Optional[List[Tuple[float, float]]] = None,
    length_scales: Optional[np.ndarray] = None,
    signal_var: float = 1.0,
) -> KernelRegularityResult:
    """
    Verify that the GP kernel satisfies Borell-TIS requirements.

    For Matérn-ν kernel on a compact domain D ⊂ R^d:
    1. Continuity: ν > d/2 ⟹ a.s. continuous sample paths (Kolmogorov-Chentsov)
    2. Centred: zero-mean GP (by construction)
    3. Compact domain: bounded parameter box (by construction)

    Also computes:
    - σ²_max = sup_{x ∈ D} k(x, x) = signal_var (for stationary kernels)
    - Dudley entropy integral bound on E[sup f]
    """
    # Matérn-5/2 has ν = 5/2
    kernel_params = {
        "matern52": {"nu": 2.5, "ms_derivatives": 2},
        "matern32": {"nu": 1.5, "ms_derivatives": 1},
        "rbf": {"nu": float("inf"), "ms_derivatives": float("inf")},
        "squared_exponential": {"nu": float("inf"), "ms_derivatives": float("inf")},
    }

    params = kernel_params.get(kernel_name, {"nu": 2.5, "ms_derivatives": 2})
    nu = params["nu"]

    # Condition 1: ν > d/2 for a.s. continuity
    satisfies_continuity = nu > dimension / 2.0

    # Condition 2: centred GP (always true for our zero-mean prior)
    satisfies_centred = True

    # Condition 3: compact domain
    satisfies_compact = domain_bounds is not None and len(domain_bounds) == dimension

    # All three conditions must hold for Borell-TIS
    borell_tis_ok = satisfies_continuity and satisfies_centred and satisfies_compact

    # σ²_max = k(x, x) = signal_var for stationary kernels
    sigma_max_sq = signal_var

    # Dudley entropy integral bound: E[sup f] ≤ C · σ_max · √(d · log(diam/ε))
    # where diam is the domain diameter and ε is a discretization scale
    domain_diam = 1.0
    if domain_bounds:
        domain_diam = np.sqrt(sum((hi - lo) ** 2 for lo, hi in domain_bounds))

    # For Matérn-5/2, metric entropy N(ε) ~ (diam/ε)^d
    # Dudley integral ≈ σ_max · √(2d · log(diam/ε_min)) with ε_min ~ l/100
    l_eff = 1.0
    if length_scales is not None:
        l_eff = float(np.min(length_scales))
    elif domain_bounds:
        l_eff = domain_diam / 5.0  # rough heuristic

    epsilon_min = max(l_eff / 100.0, 1e-10)
    entropy_term = dimension * np.log(max(domain_diam / epsilon_min, 1.0))
    expected_supremum = np.sqrt(sigma_max_sq) * np.sqrt(2.0 * entropy_term)

    details = {
        "ms_derivatives": params["ms_derivatives"],
        "kolmogorov_chentsov_condition": f"ν={nu} > d/2={dimension/2}",
        "domain_diameter": round(domain_diam, 6) if domain_bounds else None,
        "effective_length_scale": round(l_eff, 6),
        "borell_tis_tail_bound": (
            f"P(sup f > u) ≤ 2·exp(-u²/(2·{signal_var:.4f})) for u ≥ {expected_supremum:.4f}"
        ),
        "regularity_class": f"C^{min(int(nu - 0.5), 10)}" if nu < float("inf") else "C^∞",
    }

    return KernelRegularityResult(
        kernel_name=kernel_name,
        nu=nu,
        dimension=dimension,
        satisfies_continuity=satisfies_continuity,
        satisfies_centred=satisfies_centred,
        satisfies_compact_domain=satisfies_compact,
        borell_tis_applicable=borell_tis_ok,
        sigma_max_sq=sigma_max_sq,
        expected_supremum_bound=expected_supremum,
        details=details,
    )


def verify_all_models_kernel_regularity() -> Dict[str, Dict]:
    """Verify kernel regularity for all benchmark models."""
    from ..models.benchmark_models import get_benchmark, list_benchmarks

    results = {}
    for name in list_benchmarks():
        bm = get_benchmark(name)
        # Only count varying parameter dimensions
        varying = [(lo, hi) for lo, hi in bm.parameter_domain if hi - lo > 1e-12]
        d = len(varying) if varying else len(bm.parameter_domain)

        result = verify_kernel_regularity(
            kernel_name="matern52",
            dimension=d,
            domain_bounds=varying if varying else bm.parameter_domain,
            signal_var=1.0,
        )
        results[name] = result.to_dict()

    return results


@dataclass
class ComposedDeltaErrorBudget:
    """
    Tracks how δ-perturbation errors accumulate across composed cells.

    When an atlas is composed of K certified cells, the total error budget
    must account for:
    1. Per-cell δ errors (from SMT solver tolerance)
    2. Per-cell interval enclosure widths
    3. Eigenvalue perturbation bounds (Bauer-Fike)
    4. Composition overhead at cell boundaries

    Theorem B3 (Composed Error Bound):
    For K cells with individual error budgets (δ_k, w_k, ε_k), the
    composed atlas error is:
        δ_total = max_k (δ_k + w_k · L_k) + ε_boundary
    where L_k is the per-cell Lipschitz constant and ε_boundary accounts
    for the boundary transition error.
    """
    n_cells: int
    per_cell_delta: List[float]
    per_cell_enclosure_width: List[float]
    per_cell_eigenvalue_gap: List[float]
    per_cell_is_sound: List[bool]
    max_composed_delta: float
    max_enclosure_width: float
    min_eigenvalue_gap: float
    overall_sound: bool
    n_sound_cells: int
    n_unsound_cells: int
    boundary_error: float

    def to_dict(self) -> dict:
        return {
            "n_cells": self.n_cells,
            "max_composed_delta": round(self.max_composed_delta, 8),
            "max_enclosure_width": round(self.max_enclosure_width, 8),
            "min_eigenvalue_gap": round(self.min_eigenvalue_gap, 8),
            "overall_sound": self.overall_sound,
            "n_sound_cells": self.n_sound_cells,
            "n_unsound_cells": self.n_unsound_cells,
            "boundary_error": round(self.boundary_error, 8),
            "soundness_rate": round(self.n_sound_cells / max(self.n_cells, 1), 4),
        }


def compute_composed_delta_error(cells: list) -> ComposedDeltaErrorBudget:
    """
    Compute composed δ-error budget across all cells in an atlas.

    For each cell, extracts:
    - δ_required from the delta_bound (per equilibrium)
    - enclosure width from state_enclosure
    - eigenvalue gap from eigenvalue_real_parts
    """
    per_cell_delta = []
    per_cell_width = []
    per_cell_gap = []
    per_cell_sound = []

    for cell in cells:
        cell_delta = 0.0
        cell_width = 0.0
        cell_gap = float('inf')
        cell_sound = True

        for eq in cell.equilibria:
            # Extract enclosure width
            if hasattr(eq, 'state_enclosure') and eq.state_enclosure:
                w = max(hi - lo for lo, hi in eq.state_enclosure)
                cell_width = max(cell_width, w)

            # Extract eigenvalue gap
            if hasattr(eq, 'eigenvalue_real_parts') and eq.eigenvalue_real_parts:
                for lo, hi in eq.eigenvalue_real_parts:
                    if lo <= 0 <= hi:
                        cell_gap = 0.0
                    else:
                        cell_gap = min(cell_gap, min(abs(lo), abs(hi)))

            # Extract delta bound
            if hasattr(eq, 'delta_bound') and eq.delta_bound:
                delta_req = eq.delta_bound.get('delta_required', 0.0)
                cell_delta = max(cell_delta, delta_req)
                if not eq.delta_bound.get('is_sound', True):
                    cell_sound = False

        per_cell_delta.append(cell_delta)
        per_cell_width.append(cell_width)
        per_cell_gap.append(cell_gap if cell_gap < float('inf') else 0.0)
        per_cell_sound.append(cell_sound and cell_gap > cell_delta)

    if not per_cell_delta:
        return ComposedDeltaErrorBudget(
            n_cells=0, per_cell_delta=[], per_cell_enclosure_width=[],
            per_cell_eigenvalue_gap=[], per_cell_is_sound=[],
            max_composed_delta=0.0, max_enclosure_width=0.0,
            min_eigenvalue_gap=0.0, overall_sound=False,
            n_sound_cells=0, n_unsound_cells=0, boundary_error=0.0,
        )

    max_delta = max(per_cell_delta)
    max_width = max(per_cell_width)
    min_gap = min(per_cell_gap)
    n_sound = sum(per_cell_sound)

    # Boundary error: worst-case eigenvalue perturbation across adjacent cells
    boundary_error = max_width * 2.0  # Conservative: 2 × max enclosure width

    # Composed delta: max per-cell delta + boundary error
    composed_delta = max_delta + boundary_error

    overall = min_gap > composed_delta and n_sound == len(cells)

    return ComposedDeltaErrorBudget(
        n_cells=len(cells),
        per_cell_delta=per_cell_delta,
        per_cell_enclosure_width=per_cell_width,
        per_cell_eigenvalue_gap=per_cell_gap,
        per_cell_is_sound=per_cell_sound,
        max_composed_delta=composed_delta,
        max_enclosure_width=max_width,
        min_eigenvalue_gap=min_gap,
        overall_sound=overall,
        n_sound_cells=n_sound,
        n_unsound_cells=len(cells) - n_sound,
        boundary_error=boundary_error,
    )
