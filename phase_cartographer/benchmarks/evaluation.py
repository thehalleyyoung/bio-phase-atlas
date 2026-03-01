"""
Comprehensive evaluation infrastructure for certified phase-atlas construction.

Provides quantitative evaluation, ablation studies, baseline comparisons,
and scalability analysis to address reviewer critiques about missing
empirical validation.
"""

import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Any, Tuple

import numpy as np

from ..refinement.octree import (
    adaptive_refine, gp_guided_refine,
    RefinementConfig, GPGuidedRefinementConfig,
    PosteriorRefinementConfig, posterior_informed_refine,
    split_box, anisotropic_split_box, covariance_aware_split,
    box_volume, box_midpoint, ConvergenceRecord,
)
from ..gp.surrogate import GPSurrogate
from ..models.benchmark_models import BenchmarkModel, get_benchmark
from ..tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)
from ..tiered.dispatcher import select_tier
from ..atlas.builder import PhaseAtlas
from .runner import certify_box


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Full quantitative result from a single benchmark evaluation."""
    model_name: str
    n_states: int
    n_params: int
    rhs_type: str

    # Cell / coverage
    n_cells: int = 0
    coverage_fraction: float = 0.0
    n_equilibria: int = 0

    # Z3 / SMT verification
    z3_eq_verified: int = 0
    z3_eq_total: int = 0
    z3_stab_verified: int = 0
    z3_stab_total: int = 0

    # Contraction metrics
    mean_contraction: float = 0.0
    min_contraction: float = 0.0
    max_contraction: float = 0.0

    # Tier-1 pass rate
    tier1_pass_rate: float = 0.0

    # Timing
    total_time_s: float = 0.0

    # GP quality (optional)
    gp_ece: Optional[float] = None
    gp_loo_mse: Optional[float] = None

    # δ-soundness bounds
    delta_bound_results: List[Dict] = field(default_factory=list)

    # Newton speedup (optional, only meaningful for ≥3-state models)
    newton_speedup: Optional[float] = None


@dataclass
class AblationResult:
    """Result from a single ablation configuration."""
    config_name: str
    coverage: float
    time_s: float
    n_cells: int
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_certify_fn(bm: BenchmarkModel) -> Callable:
    """Build the standard certify_fn closure for a benchmark model."""
    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )
    return certify_fn


def _atlas_metrics(atlas: PhaseAtlas) -> Dict[str, Any]:
    """Extract common metrics from a PhaseAtlas."""
    stats = atlas.stats()
    contractions = []
    n_eq = 0
    for c in atlas.cells:
        n_eq += len(c.equilibria)
        for eq in c.equilibria:
            if eq.krawczyk_contraction < float('inf'):
                contractions.append(eq.krawczyk_contraction)
    return {
        "n_cells": stats.certified_cells,
        "coverage_fraction": stats.coverage_fraction,
        "n_equilibria": n_eq,
        "tier1_pass_rate": stats.minicheck_pass_rate,
        "mean_contraction": float(np.mean(contractions)) if contractions else 0.0,
        "min_contraction": float(np.min(contractions)) if contractions else 0.0,
        "max_contraction": float(np.max(contractions)) if contractions else 0.0,
    }


def _run_z3_on_atlas(atlas: PhaseAtlas, bm: BenchmarkModel) -> Dict[str, int]:
    """Run Z3 verification across all cells and return aggregate counts."""
    from .runner import _run_z3_verification

    eq_verified, eq_total = 0, 0
    stab_verified, stab_total = 0, 0
    for cell in atlas.cells:
        z3_res = _run_z3_verification(cell, bm)
        for key, val in z3_res.items():
            if 'existence' in key:
                eq_total += 1
                if val.get('verified', False):
                    eq_verified += 1
            elif 'stability' in key:
                stab_total += 1
                if val.get('verified', False):
                    stab_verified += 1
    return {
        "z3_eq_verified": eq_verified,
        "z3_eq_total": eq_total,
        "z3_stab_verified": stab_verified,
        "z3_stab_total": stab_total,
    }


def _collect_delta_bounds(atlas: PhaseAtlas) -> List[Dict]:
    """Collect δ-bound results across all equilibria in an atlas."""
    results = []
    for cell in atlas.cells:
        for eq in cell.equilibria:
            if eq.delta_bound is not None:
                results.append(eq.delta_bound)
    return results


def _gp_quality_metrics(atlas: PhaseAtlas) -> Dict[str, Optional[float]]:
    """Compute GP ECE and LOO-MSE from an atlas."""
    if not atlas.cells or len(atlas.cells) < 6:
        return {"gp_ece": None, "gp_loo_mse": None}

    try:
        gp = GPSurrogate.train_from_atlas(atlas)
        if not gp._fitted:
            return {"gp_ece": None, "gp_loo_mse": None}

        X = np.array([box_midpoint(c.parameter_box) for c in atlas.cells])
        _rmap = {
            RegimeType.MONOSTABLE: 0, RegimeType.BISTABLE: 1,
            RegimeType.MULTISTABLE: 2, RegimeType.OSCILLATORY: 3,
            RegimeType.EXCITABLE: 4, RegimeType.INCONCLUSIVE: 5,
        }
        y = np.array([_rmap.get(c.regime, 5) for c in atlas.cells], dtype=float)

        split = max(1, len(X) // 5)
        ece = gp.calibration_error(X[:split], y[:split])
        loo = gp.loo_cross_validation(X, y)
        return {"gp_ece": round(ece, 6), "gp_loo_mse": round(loo, 6)}
    except Exception:
        return {"gp_ece": None, "gp_loo_mse": None}


def _make_benchmark_result(
    bm: BenchmarkModel,
    atlas: PhaseAtlas,
    elapsed: float,
    *,
    newton_speedup: Optional[float] = None,
) -> BenchmarkResult:
    """Assemble a full BenchmarkResult from an atlas run."""
    m = _atlas_metrics(atlas)
    z3 = _run_z3_on_atlas(atlas, bm)
    gp_q = _gp_quality_metrics(atlas)
    deltas = _collect_delta_bounds(atlas)

    return BenchmarkResult(
        model_name=bm.name,
        n_states=bm.n_states,
        n_params=bm.n_params,
        rhs_type=bm.rhs_type,
        n_cells=m["n_cells"],
        coverage_fraction=round(m["coverage_fraction"], 6),
        n_equilibria=m["n_equilibria"],
        z3_eq_verified=z3["z3_eq_verified"],
        z3_eq_total=z3["z3_eq_total"],
        z3_stab_verified=z3["z3_stab_verified"],
        z3_stab_total=z3["z3_stab_total"],
        mean_contraction=round(m["mean_contraction"], 6),
        min_contraction=round(m["min_contraction"], 6),
        max_contraction=round(m["max_contraction"], 6),
        tier1_pass_rate=round(m["tier1_pass_rate"], 4),
        total_time_s=round(elapsed, 3),
        gp_ece=gp_q["gp_ece"],
        gp_loo_mse=gp_q["gp_loo_mse"],
        delta_bound_results=deltas,
        newton_speedup=newton_speedup,
    )


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------

class AblationStudy:
    """Ablation experiments isolating the contribution of each pipeline component."""

    @staticmethod
    def tier_ablation(
        model_name: str,
        certify_fn: Callable,
        domain: List[Tuple[float, float]],
        max_depth: int = 4,
        max_cells: int = 200,
    ) -> List[AblationResult]:
        """Compare full tiered verification vs tier-1 (IA) only.

        Returns two AblationResults: ``"tier1_only"`` and ``"full_tiers"``.
        """
        from .runner import _certify_box_tier_limited

        bm = get_benchmark(model_name)
        config = RefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
        )
        results: List[AblationResult] = []

        # Tier-1 only
        def _tier1_fn(box):
            return _certify_box_tier_limited(
                bm.rhs, box, bm.state_domain,
                bm.name, bm.n_states, bm.n_params, bm.rhs_type,
                max_tier=1,
            )

        t0 = time.time()
        atlas_t1 = adaptive_refine(_tier1_fn, domain, model_name, config)
        t1_time = time.time() - t0
        t1_stats = atlas_t1.stats()
        results.append(AblationResult(
            config_name="tier1_only",
            coverage=round(t1_stats.coverage_fraction, 6),
            time_s=round(t1_time, 3),
            n_cells=t1_stats.certified_cells,
            extra_metrics={"pass_rate": t1_stats.minicheck_pass_rate},
        ))

        # Full tiers
        t0 = time.time()
        atlas_full = adaptive_refine(certify_fn, domain, model_name, config)
        full_time = time.time() - t0
        full_stats = atlas_full.stats()
        results.append(AblationResult(
            config_name="full_tiers",
            coverage=round(full_stats.coverage_fraction, 6),
            time_s=round(full_time, 3),
            n_cells=full_stats.certified_cells,
            extra_metrics={"pass_rate": full_stats.minicheck_pass_rate},
        ))

        return results

    @staticmethod
    def gp_vs_uniform(
        model_name: str,
        certify_fn: Callable,
        domain: List[Tuple[float, float]],
        max_cells: int = 200,
        max_depth: int = 4,
    ) -> List[AblationResult]:
        """Compare GP-guided refinement vs uniform (adaptive) refinement."""
        config_u = RefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
        )
        config_gp = GPGuidedRefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
            gp_warmup_cells=max(5, max_cells // 20),
        )

        results: List[AblationResult] = []

        # Uniform
        t0 = time.time()
        u_atlas = adaptive_refine(certify_fn, domain, model_name, config_u)
        u_time = time.time() - t0
        u_stats = u_atlas.stats()
        results.append(AblationResult(
            config_name="uniform",
            coverage=round(u_stats.coverage_fraction, 6),
            time_s=round(u_time, 3),
            n_cells=u_stats.certified_cells,
        ))

        # GP-guided
        t0 = time.time()
        g_atlas, convergence = gp_guided_refine(
            certify_fn, domain, model_name, config_gp,
        )
        g_time = time.time() - t0
        g_stats = g_atlas.stats()

        gp_extra: Dict[str, Any] = {}
        gp_q = _gp_quality_metrics(g_atlas)
        gp_extra["gp_ece"] = gp_q["gp_ece"]
        gp_extra["gp_loo_mse"] = gp_q["gp_loo_mse"]
        gp_extra["convergence_iterations"] = len(convergence.iteration)

        results.append(AblationResult(
            config_name="gp_guided",
            coverage=round(g_stats.coverage_fraction, 6),
            time_s=round(g_time, 3),
            n_cells=g_stats.certified_cells,
            extra_metrics=gp_extra,
        ))

        return results

    @staticmethod
    def newton_vs_bisection(
        model_name: str,
        certify_fn: Callable,
        domain: List[Tuple[float, float]],
        max_depth: int = 4,
        max_cells: int = 200,
    ) -> List[AblationResult]:
        """Compare Newton-guided equilibrium search vs pure bisection.

        Only meaningful for models with ≥3 state variables.
        """
        from .runner import _certify_box_bisection_only

        bm = get_benchmark(model_name)
        config = RefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
        )
        results: List[AblationResult] = []

        # Newton-guided (default certify_fn already uses Newton for ≥3D)
        t0 = time.time()
        n_atlas = adaptive_refine(certify_fn, domain, model_name, config)
        n_time = time.time() - t0
        n_stats = n_atlas.stats()
        results.append(AblationResult(
            config_name="newton_guided",
            coverage=round(n_stats.coverage_fraction, 6),
            time_s=round(n_time, 3),
            n_cells=n_stats.certified_cells,
        ))

        # Pure bisection
        def bisect_fn(box):
            return _certify_box_bisection_only(
                bm.rhs, box, bm.state_domain,
                bm.name, bm.n_states, bm.n_params, bm.rhs_type,
            )

        t0 = time.time()
        b_atlas = adaptive_refine(bisect_fn, domain, model_name, config)
        b_time = time.time() - t0
        b_stats = b_atlas.stats()
        speedup = b_time / max(n_time, 1e-6)
        results.append(AblationResult(
            config_name="bisection_only",
            coverage=round(b_stats.coverage_fraction, 6),
            time_s=round(b_time, 3),
            n_cells=b_stats.certified_cells,
            extra_metrics={"newton_speedup": round(speedup, 2)},
        ))

        return results

    @staticmethod
    def refinement_strategy_comparison(
        model_name: str,
        certify_fn: Callable,
        domain: List[Tuple[float, float]],
        max_depth: int = 4,
        max_cells: int = 200,
    ) -> List[AblationResult]:
        """Compare uniform, anisotropic, and covariance-aware splitting."""
        results: List[AblationResult] = []

        # Uniform (isotropic) splitting via adaptive_refine
        config_u = RefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
        )
        t0 = time.time()
        u_atlas = adaptive_refine(certify_fn, domain, model_name, config_u)
        u_time = time.time() - t0
        u_stats = u_atlas.stats()
        results.append(AblationResult(
            config_name="uniform",
            coverage=round(u_stats.coverage_fraction, 6),
            time_s=round(u_time, 3),
            n_cells=u_stats.certified_cells,
        ))

        # Anisotropic splitting via GP-guided (uses anisotropic_split_box)
        config_gp = GPGuidedRefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
            gp_warmup_cells=max(5, max_cells // 20),
        )
        t0 = time.time()
        a_atlas, _ = gp_guided_refine(certify_fn, domain, model_name, config_gp)
        a_time = time.time() - t0
        a_stats = a_atlas.stats()
        results.append(AblationResult(
            config_name="anisotropic",
            coverage=round(a_stats.coverage_fraction, 6),
            time_s=round(a_time, 3),
            n_cells=a_stats.certified_cells,
        ))

        # Covariance-aware splitting via posterior_informed_refine
        cov = np.eye(len(domain))
        config_post = PosteriorRefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
            gp_warmup_cells=max(5, max_cells // 20),
            covariance_matrix=cov, use_pca_split=True,
        )
        t0 = time.time()
        c_atlas, _ = posterior_informed_refine(
            certify_fn, domain, model_name, config_post,
        )
        c_time = time.time() - t0
        c_stats = c_atlas.stats()
        results.append(AblationResult(
            config_name="covariance_aware",
            coverage=round(c_stats.coverage_fraction, 6),
            time_s=round(c_time, 3),
            n_cells=c_stats.certified_cells,
        ))

        return results


# ---------------------------------------------------------------------------
# ScalabilityAnalysis
# ---------------------------------------------------------------------------

class ScalabilityAnalysis:
    """Measure how certification cost scales with problem dimensions."""

    @staticmethod
    def dimension_scaling(
        model_factory: Callable[[int], BenchmarkModel],
        dims: List[int] = None,
        max_cells: int = 100,
        max_depth: int = 3,
    ) -> List[AblationResult]:
        """Measure certification time vs state dimension.

        ``model_factory(n)`` should return a BenchmarkModel with *n* state
        variables so that timing can be compared across dimensions.
        """
        if dims is None:
            dims = [2, 3, 4, 5]

        results: List[AblationResult] = []
        for d in dims:
            bm = model_factory(d)
            certify_fn = _build_certify_fn(bm)
            config = RefinementConfig(
                max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
            )
            t0 = time.time()
            atlas = adaptive_refine(
                certify_fn, bm.parameter_domain, bm.name, config,
            )
            elapsed = time.time() - t0
            stats = atlas.stats()
            results.append(AblationResult(
                config_name=f"dim_{d}",
                coverage=round(stats.coverage_fraction, 6),
                time_s=round(elapsed, 3),
                n_cells=stats.certified_cells,
                extra_metrics={"n_states": d, "cells_per_second": round(
                    stats.certified_cells / max(elapsed, 1e-6), 2,
                )},
            ))
        return results

    @staticmethod
    def parameter_scaling(
        model: BenchmarkModel,
        param_counts: List[int] = None,
        max_cells: int = 100,
        max_depth: int = 3,
    ) -> List[AblationResult]:
        """Measure coverage vs number of free parameters.

        Uses progressively wider slices of the model's parameter domain.
        """
        if param_counts is None:
            param_counts = [2, 3, 4]

        certify_fn = _build_certify_fn(model)
        results: List[AblationResult] = []

        for np_ in param_counts:
            # Use the first np_ parameter dimensions (pad with unit intervals
            # if model has fewer parameters).
            domain = list(model.parameter_domain[:np_])
            while len(domain) < np_:
                domain.append((0.0, 1.0))

            config = RefinementConfig(
                max_depth=max_depth, target_coverage=0.99, max_cells=max_cells,
            )
            t0 = time.time()
            atlas = adaptive_refine(
                certify_fn, domain, model.name, config,
            )
            elapsed = time.time() - t0
            stats = atlas.stats()
            results.append(AblationResult(
                config_name=f"params_{np_}",
                coverage=round(stats.coverage_fraction, 6),
                time_s=round(elapsed, 3),
                n_cells=stats.certified_cells,
                extra_metrics={"n_params": np_},
            ))
        return results

    @staticmethod
    def depth_scaling(
        model_name: str,
        certify_fn: Callable,
        domain: List[Tuple[float, float]],
        depths: List[int] = None,
        max_cells: int = 500,
    ) -> List[AblationResult]:
        """Measure coverage and cost vs maximum refinement depth."""
        if depths is None:
            depths = [2, 4, 6, 8]

        results: List[AblationResult] = []
        for d in depths:
            config = RefinementConfig(
                max_depth=d, target_coverage=0.99, max_cells=max_cells,
            )
            t0 = time.time()
            atlas = adaptive_refine(certify_fn, domain, model_name, config)
            elapsed = time.time() - t0
            stats = atlas.stats()
            results.append(AblationResult(
                config_name=f"depth_{d}",
                coverage=round(stats.coverage_fraction, 6),
                time_s=round(elapsed, 3),
                n_cells=stats.certified_cells,
                extra_metrics={"max_depth": d},
            ))
        return results


# ---------------------------------------------------------------------------
# BaselineComparison
# ---------------------------------------------------------------------------

class BaselineComparison:
    """Compare GP-guided acquisition against quasi-random baselines."""

    @staticmethod
    def _sample_and_certify(
        certify_fn: Callable,
        domain: List[Tuple[float, float]],
        sample_points: np.ndarray,
        model_name: str,
    ) -> AblationResult:
        """Certify boxes centred on *sample_points* and collect coverage."""
        dim = len(domain)
        lo = np.array([b[0] for b in domain])
        hi = np.array([b[1] for b in domain])
        widths = (hi - lo) / max(1, int(round(len(sample_points) ** (1.0 / dim))))

        cells: List[CertifiedCell] = []
        t0 = time.time()
        for pt in sample_points:
            box = [
                (max(lo[i], pt[i] - widths[i] / 2),
                 min(hi[i], pt[i] + widths[i] / 2))
                for i in range(dim)
            ]
            cell = certify_fn(box)
            if cell is not None:
                cells.append(cell)
        elapsed = time.time() - t0

        total_vol = float(np.prod(hi - lo))
        cert_vol = sum(box_volume(c.parameter_box) for c in cells)
        coverage = cert_vol / total_vol if total_vol > 0 else 0.0

        return AblationResult(
            config_name="",  # filled by caller
            coverage=round(coverage, 6),
            time_s=round(elapsed, 3),
            n_cells=len(cells),
        )

    @staticmethod
    def sobol_vs_gp(
        model_name: str,
        certify_fn: Callable,
        domain: List[Tuple[float, float]],
        n_samples: int = 64,
        max_depth: int = 4,
    ) -> List[AblationResult]:
        """Sobol quasi-random sampling vs GP-guided refinement."""
        from scipy.stats.qmc import Sobol

        dim = len(domain)
        lo = np.array([b[0] for b in domain])
        hi = np.array([b[1] for b in domain])

        # Sobol samples (must be power-of-2 for Sobol)
        m = int(np.ceil(np.log2(max(n_samples, 1))))
        sampler = Sobol(d=dim, scramble=True, seed=42)
        pts_unit = sampler.random_base2(m=m)
        pts = lo + pts_unit * (hi - lo)

        sobol_res = BaselineComparison._sample_and_certify(
            certify_fn, domain, pts, model_name,
        )
        sobol_res.config_name = "sobol"
        sobol_res.extra_metrics["n_samples"] = len(pts)

        # GP-guided with same cell budget
        config_gp = GPGuidedRefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=len(pts),
            gp_warmup_cells=max(5, len(pts) // 20),
        )
        t0 = time.time()
        g_atlas, conv = gp_guided_refine(
            certify_fn, domain, model_name, config_gp,
        )
        g_time = time.time() - t0
        g_stats = g_atlas.stats()
        gp_res = AblationResult(
            config_name="gp_guided",
            coverage=round(g_stats.coverage_fraction, 6),
            time_s=round(g_time, 3),
            n_cells=g_stats.certified_cells,
            extra_metrics={"convergence_iterations": len(conv.iteration)},
        )

        return [sobol_res, gp_res]

    @staticmethod
    def latin_hypercube_vs_gp(
        model_name: str,
        certify_fn: Callable,
        domain: List[Tuple[float, float]],
        n_samples: int = 64,
        max_depth: int = 4,
    ) -> List[AblationResult]:
        """Latin-Hypercube sampling vs GP-guided refinement."""
        from scipy.stats.qmc import LatinHypercube

        dim = len(domain)
        lo = np.array([b[0] for b in domain])
        hi = np.array([b[1] for b in domain])

        sampler = LatinHypercube(d=dim, seed=42)
        pts_unit = sampler.random(n=n_samples)
        pts = lo + pts_unit * (hi - lo)

        lhs_res = BaselineComparison._sample_and_certify(
            certify_fn, domain, pts, model_name,
        )
        lhs_res.config_name = "latin_hypercube"
        lhs_res.extra_metrics["n_samples"] = n_samples

        # GP-guided with same cell budget
        config_gp = GPGuidedRefinementConfig(
            max_depth=max_depth, target_coverage=0.99, max_cells=n_samples,
            gp_warmup_cells=max(5, n_samples // 20),
        )
        t0 = time.time()
        g_atlas, conv = gp_guided_refine(
            certify_fn, domain, model_name, config_gp,
        )
        g_time = time.time() - t0
        g_stats = g_atlas.stats()
        gp_res = AblationResult(
            config_name="gp_guided",
            coverage=round(g_stats.coverage_fraction, 6),
            time_s=round(g_time, 3),
            n_cells=g_stats.certified_cells,
            extra_metrics={"convergence_iterations": len(conv.iteration)},
        )

        return [lhs_res, gp_res]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_benchmark_table(results: List[BenchmarkResult]) -> Dict[str, list]:
    """Format a list of BenchmarkResults as column-oriented dict for LaTeX.

    Returns a dict whose keys are column headers and whose values are lists
    (one entry per model), suitable for conversion to a LaTeX tabular.
    """
    table: Dict[str, list] = {
        "Model": [],
        "$n$": [],
        "$p$": [],
        "RHS": [],
        "Cells": [],
        "Coverage": [],
        "Equil.": [],
        r"Z3 eq.": [],
        r"Z3 stab.": [],
        r"$\bar q$": [],
        "T1%": [],
        "Time (s)": [],
    }
    for r in results:
        table["Model"].append(r.model_name)
        table["$n$"].append(r.n_states)
        table["$p$"].append(r.n_params)
        table["RHS"].append(r.rhs_type)
        table["Cells"].append(r.n_cells)
        table["Coverage"].append(f"{r.coverage_fraction:.1%}")
        table["Equil."].append(r.n_equilibria)
        z3eq = (f"{r.z3_eq_verified}/{r.z3_eq_total}"
                if r.z3_eq_total else "—")
        table[r"Z3 eq."].append(z3eq)
        z3st = (f"{r.z3_stab_verified}/{r.z3_stab_total}"
                if r.z3_stab_total else "—")
        table[r"Z3 stab."].append(z3st)
        table[r"$\bar q$"].append(f"{r.mean_contraction:.4f}")
        table["T1%"].append(f"{r.tier1_pass_rate:.1%}")
        table["Time (s)"].append(f"{r.total_time_s:.1f}")
    return table


def format_ablation_table(
    results: List[AblationResult],
) -> Dict[str, list]:
    """Format a list of AblationResults as column-oriented dict for LaTeX."""
    table: Dict[str, list] = {
        "Config": [],
        "Coverage": [],
        "Cells": [],
        "Time (s)": [],
    }
    extra_keys: set = set()
    for r in results:
        extra_keys.update(r.extra_metrics.keys())

    for k in sorted(extra_keys):
        table[k] = []

    for r in results:
        table["Config"].append(r.config_name)
        table["Coverage"].append(f"{r.coverage:.1%}")
        table["Cells"].append(r.n_cells)
        table["Time (s)"].append(f"{r.time_s:.1f}")
        for k in sorted(extra_keys):
            val = r.extra_metrics.get(k)
            if val is None:
                table[k].append("—")
            elif isinstance(val, float):
                table[k].append(f"{val:.4f}")
            else:
                table[k].append(val)
    return table


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_results(results: Any, path: str) -> None:
    """Save benchmark / ablation results to a JSON file.

    Accepts a single dataclass instance, a list of dataclass instances, or
    a plain dict.
    """
    def _serialise(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        if isinstance(obj, (list, tuple)):
            return [_serialise(o) for o in obj]
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_serialise(results), f, indent=2, default=str)
