#!/usr/bin/env python3
"""
Comprehensive experiment runner for Certified Bio-Phase Cartographer.

Runs all 15 benchmark models through the full certification pipeline,
ablation studies, and GP calibration analysis.
"""

import json
import os
import signal
import sys
import time
from dataclasses import asdict

import numpy as np

from phase_cartographer.models.benchmark_models import (
    get_benchmark, list_benchmarks, ALL_BENCHMARKS,
)
from phase_cartographer.benchmarks.runner import (
    run_benchmark, run_ablation, certify_box,
    _certify_box_bisection_only, BenchmarkResult, AblationResult,
    run_depth_scaling, run_delta_validation,
)
from phase_cartographer.refinement.octree import (
    adaptive_refine, RefinementConfig, GPGuidedRefinementConfig,
    gp_guided_refine,
)
from phase_cartographer.gp.surrogate import GPSurrogate
from phase_cartographer.gp.calibration import GPCalibrator
from phase_cartographer.tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType,
)
from phase_cartographer.atlas.builder import PhaseAtlas
from phase_cartographer.interval.interval import Interval
from phase_cartographer.interval.matrix import IntervalVector
from phase_cartographer.equilibrium import KrawczykOperator, StabilityClassifier
from phase_cartographer.smt.delta_bound import compute_eigenvalue_gap
from phase_cartographer.minicheck import verify_certificate as minicheck_verify

# ---------------------------------------------------------------------------
# Global timeout
# ---------------------------------------------------------------------------
GLOBAL_TIMEOUT_S = 420  # 7 minutes

_start_time = None


def _check_timeout(label=""):
    """Raise if global timeout exceeded."""
    if _start_time and (time.time() - _start_time) > GLOBAL_TIMEOUT_S:
        raise TimeoutError(f"Global timeout ({GLOBAL_TIMEOUT_S}s) exceeded at: {label}")


# ---------------------------------------------------------------------------
# All 15 model names
# ---------------------------------------------------------------------------
ALL_MODEL_NAMES = [
    "toggle_switch", "brusselator", "selkov", "repressilator", "goodwin",
    "lotka_volterra", "fitzhugh_nagumo",
    "tyson_cell_cycle", "mapk_cascade", "p53_mdm2", "calcium_oscillator",
    "circadian_clock", "genetic_oscillator", "mutual_activation_switch",
    "van_der_pol",
]

ABLATION_MODELS = [
    "toggle_switch", "brusselator", "selkov", "fitzhugh_nagumo",
    "tyson_cell_cycle", "mapk_cascade", "calcium_oscillator",
    "mutual_activation_switch", "van_der_pol",
]

THREE_STATE_MODELS = ["repressilator", "goodwin", "genetic_oscillator"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "benchmark_output")


# ---------------------------------------------------------------------------
# certify_cell_for_model
# ---------------------------------------------------------------------------

def certify_cell_for_model(model_name):
    """Return a certify_fn suitable for octree refinement.

    The returned function takes a parameter box and:
      1. Loads the benchmark model
      2. Creates interval boxes from the parameter box
      3. Runs equilibrium finding + Krawczyk verification + stability classification
      4. Runs MiniCheck (tier1) verification
      5. Runs Z3 verification (tier3) where applicable
      6. Computes delta bounds
      7. Returns a CertifiedCell
    """
    bm = get_benchmark(model_name)

    def certify_fn(box):
        # Use the existing certify_box which does steps 2-7
        cell = certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )
        if cell is None:
            return None

        # Z3 verification (tier3) where applicable
        try:
            from phase_cartographer.benchmarks.runner import _run_z3_verification
            z3_res = _run_z3_verification(cell, bm)
            if z3_res:
                cell.z3_results = z3_res
        except Exception:
            pass

        return cell

    return certify_fn


# ---------------------------------------------------------------------------
# run_all_benchmarks
# ---------------------------------------------------------------------------

def run_all_benchmarks():
    """Run certification on all 15 models, save to benchmark_output/benchmark_results.json."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for name in ALL_MODEL_NAMES:
        _check_timeout(f"benchmark:{name}")
        print(f"  [{name}] ", end="", flush=True)
        t0 = time.time()
        try:
            result = run_benchmark(name, max_depth=3, target_coverage=0.85, max_cells=50)
            elapsed = time.time() - t0
            results[name] = asdict(result)
            print(f"{result.certified_cells} cells, "
                  f"{result.coverage_fraction:.1%} cov, "
                  f"tier1={result.tier1_pass_rate:.0%}, "
                  f"{elapsed:.1f}s")
        except TimeoutError:
            raise
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = {"error": str(e), "time_s": round(elapsed, 2)}
            print(f"FAILED ({elapsed:.1f}s): {e}")

    out_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    return results


# ---------------------------------------------------------------------------
# run_ablation_studies
# ---------------------------------------------------------------------------

def run_ablation_studies():
    """Run GP vs uniform ablation and Newton vs bisection ablation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ablation_results = {"gp_vs_uniform": {}, "newton_vs_bisection": {}}

    # GP vs uniform ablation
    print("\n  --- GP vs Uniform ablation ---")
    for name in ABLATION_MODELS:
        _check_timeout(f"ablation_gp:{name}")
        print(f"  [{name}] ", end="", flush=True)
        t0 = time.time()
        try:
            result = run_ablation(name, max_depth=3, target_coverage=0.80, max_cells=30)
            elapsed = time.time() - t0
            ablation_results["gp_vs_uniform"][name] = asdict(result)
            print(f"uniform={result.uniform_coverage:.1%}/{result.uniform_time_s:.1f}s, "
                  f"gp={result.gp_coverage:.1%}/{result.gp_time_s:.1f}s, "
                  f"ECE={result.gp_ece:.4f}")
        except TimeoutError:
            raise
        except Exception as e:
            elapsed = time.time() - t0
            ablation_results["gp_vs_uniform"][name] = {
                "error": str(e), "time_s": round(elapsed, 2)
            }
            print(f"FAILED ({elapsed:.1f}s): {e}")

    # Newton vs bisection on 3-state models
    # Use direct certify_box variants instead of run_comprehensive_ablation
    # to avoid running 8+ full passes per model
    print("\n  --- Newton vs Bisection ablation (3-state models) ---")
    for name in THREE_STATE_MODELS:
        _check_timeout(f"ablation_newton:{name}")
        print(f"  [{name}] ", end="", flush=True)
        t0 = time.time()
        try:
            bm = get_benchmark(name)
            small_config = RefinementConfig(max_depth=2, target_coverage=0.80, max_cells=15)

            # Newton (default)
            def _cfn_newton(box, _bm=bm):
                return certify_box(_bm.rhs, box, _bm.state_domain,
                                   _bm.name, _bm.n_states, _bm.n_params, _bm.rhs_type)
            t_n = time.time()
            n_atlas = adaptive_refine(_cfn_newton, bm.parameter_domain, bm.name, small_config)
            newton_time = time.time() - t_n
            newton_cov = n_atlas.coverage_fraction()

            _check_timeout(f"ablation_bisection:{name}")

            # Bisection only
            def _cfn_bis(box, _bm=bm):
                return _certify_box_bisection_only(_bm.rhs, box, _bm.state_domain,
                                                   _bm.name, _bm.n_states, _bm.n_params, _bm.rhs_type)
            t_b = time.time()
            b_atlas = adaptive_refine(_cfn_bis, bm.parameter_domain, bm.name, small_config)
            bisection_time = time.time() - t_b
            bisection_cov = b_atlas.coverage_fraction()

            elapsed = time.time() - t0
            ablation_results["newton_vs_bisection"][name] = {
                "newton_coverage": round(newton_cov, 6),
                "newton_time_s": round(newton_time, 3),
                "bisection_coverage": round(bisection_cov, 6),
                "bisection_time_s": round(bisection_time, 3),
            }
            print(f"newton={newton_cov:.1%}/{newton_time:.1f}s, "
                  f"bisection={bisection_cov:.1%}/{bisection_time:.1f}s")
        except TimeoutError:
            raise
        except Exception as e:
            elapsed = time.time() - t0
            ablation_results["newton_vs_bisection"][name] = {
                "error": str(e), "time_s": round(elapsed, 2)
            }
            print(f"FAILED ({elapsed:.1f}s): {e}")

    out_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    return ablation_results


# ---------------------------------------------------------------------------
# run_gp_calibration
# ---------------------------------------------------------------------------

def run_gp_calibration():
    """Train GP on each model's atlas, run calibration analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    calibration_results = {}

    for name in ALL_MODEL_NAMES:
        try:
            _check_timeout(f"gp_cal:{name}")
        except TimeoutError:
            print(f"  [{name}] SKIPPED (timeout)")
            break
        print(f"  [{name}] ", end="", flush=True)
        t0 = time.time()
        try:
            bm = get_benchmark(name)

            # Build a small atlas for GP training
            certify_fn = certify_cell_for_model(name)
            config = RefinementConfig(max_depth=2, target_coverage=0.80, max_cells=20)
            atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, config)

            if not atlas.cells:
                calibration_results[name] = {"error": "no cells certified", "time_s": round(time.time() - t0, 2)}
                print("no cells")
                continue

            # Train GP
            t_gp = time.time()
            gp = GPSurrogate.train_from_atlas(atlas)
            gp_train_time = time.time() - t_gp

            if not gp._fitted:
                calibration_results[name] = {"error": "GP not fitted", "time_s": round(time.time() - t0, 2)}
                print("GP not fitted")
                continue

            # Extract data for calibration
            _regime_map = {
                RegimeType.MONOSTABLE: 0, RegimeType.BISTABLE: 1,
                RegimeType.MULTISTABLE: 2, RegimeType.OSCILLATORY: 3,
                RegimeType.EXCITABLE: 4, RegimeType.INCONCLUSIVE: 5,
            }
            X = np.array([
                [(lo + hi) / 2.0 for lo, hi in c.parameter_box]
                for c in atlas.cells
            ])
            y = np.array([_regime_map.get(c.regime, 5) for c in atlas.cells], dtype=float)

            # Run calibration
            calibrator = GPCalibrator(n_bins=5)
            cal_result = calibrator.calibrate(gp, X, y)
            report = calibrator.full_report(cal_result)

            # Coverage bound
            lower, upper, conf = gp.coverage_bound(bm.parameter_domain, confidence=0.95, n_samples=100)

            elapsed = time.time() - t0
            calibration_results[name] = {
                "n_cells": len(atlas.cells),
                "gp_train_time_s": round(gp_train_time, 4),
                "ece": report.get("ece", None),
                "loo_mse": report.get("loo_mse", None),
                "coverage_90": report.get("coverage_90", None),
                "coverage_95": report.get("coverage_95", None),
                "nlpd": report.get("nlpd", None),
                "sharpness": report.get("sharpness", None),
                "coverage_bound_lower": round(lower, 4),
                "coverage_bound_upper": round(upper, 4),
                "total_time_s": round(elapsed, 2),
            }
            print(f"ECE={cal_result.ece:.4f}, LOO={cal_result.loo_mse:.4f}, "
                  f"cov=[{lower:.2f},{upper:.2f}], {elapsed:.1f}s")
        except TimeoutError:
            print("TIMEOUT")
            break
        except Exception as e:
            elapsed = time.time() - t0
            calibration_results[name] = {"error": str(e), "time_s": round(elapsed, 2)}
            print(f"FAILED ({elapsed:.1f}s): {e}")

    _save_gp_calibration(calibration_results)
    return calibration_results


def _save_gp_calibration(calibration_results):
    out_path = os.path.join(OUTPUT_DIR, "gp_calibration.json")
    with open(out_path, "w") as f:
        json.dump(calibration_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# run_depth_scaling_experiment
# ---------------------------------------------------------------------------

def run_depth_scaling_experiment():
    """Run depth-scaling experiment on brusselator: coverage vs max_depth."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    depth_results = {}

    DEPTH_MODELS = ["brusselator", "selkov", "fitzhugh_nagumo"]
    for name in DEPTH_MODELS:
        try:
            _check_timeout(f"depth_scaling:{name}")
        except TimeoutError:
            print(f"  [{name}] SKIPPED (timeout)")
            break
        print(f"  [{name}] ", end="", flush=True)
        t0 = time.time()
        try:
            points = run_depth_scaling(name, depths=[2, 3, 4, 5], max_cells=200)
            elapsed = time.time() - t0
            depth_results[name] = [asdict(p) for p in points]
            coverages = [f"d{p.max_depth}={p.coverage_fraction:.1%}" for p in points]
            print(f"{', '.join(coverages)} ({elapsed:.1f}s)")
        except TimeoutError:
            raise
        except Exception as e:
            elapsed = time.time() - t0
            depth_results[name] = {"error": str(e), "time_s": round(elapsed, 2)}
            print(f"FAILED ({elapsed:.1f}s): {e}")

    out_path = os.path.join(OUTPUT_DIR, "depth_scaling.json")
    with open(out_path, "w") as f:
        json.dump(depth_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    return depth_results


# ---------------------------------------------------------------------------
# run_delta_validation_experiment
# ---------------------------------------------------------------------------

def run_delta_validation_experiment():
    """Run empirical delta-soundness validation on all models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    delta_results = {}

    DELTA_MODELS = ["toggle_switch", "brusselator", "selkov", "fitzhugh_nagumo",
                    "van_der_pol", "mutual_activation_switch"]
    for name in DELTA_MODELS:
        try:
            _check_timeout(f"delta_val:{name}")
        except TimeoutError:
            print(f"  [{name}] SKIPPED (timeout)")
            break
        print(f"  [{name}] ", end="", flush=True)
        t0 = time.time()
        try:
            result = run_delta_validation(name, max_depth=3, max_cells=50)
            elapsed = time.time() - t0
            delta_results[name] = asdict(result)
            status = "ALL SOUND" if result.all_sound else f"{result.n_unsound} UNSOUND"
            print(f"{result.n_with_delta} checked, {status}, "
                  f"gap>={result.min_eigenvalue_gap:.6f}, "
                  f"delta<={result.max_delta_required:.6f} ({elapsed:.1f}s)")
        except TimeoutError:
            raise
        except Exception as e:
            elapsed = time.time() - t0
            delta_results[name] = {"error": str(e), "time_s": round(elapsed, 2)}
            print(f"FAILED ({elapsed:.1f}s): {e}")

    out_path = os.path.join(OUTPUT_DIR, "delta_validation.json")
    with open(out_path, "w") as f:
        json.dump(delta_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    return delta_results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(benchmark_results, ablation_results, gp_results,
                        depth_results=None, delta_results=None):
    """Print a summary table of all results."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Model':<28} {'States':>6} {'Cells':>6} {'Cover':>7} {'T1%':>5} {'Time':>7}")
    print("-" * 90)
    for name in ALL_MODEL_NAMES:
        r = benchmark_results.get(name, {})
        if "error" in r:
            print(f"{name:<28} {'':>6} {'ERR':>6} {'':>7} {'':>5} {r.get('time_s', 0):>6.1f}s")
        else:
            print(f"{name:<28} {r.get('n_states', '?'):>6} "
                  f"{r.get('certified_cells', 0):>6} "
                  f"{r.get('coverage_fraction', 0):>6.1%} "
                  f"{r.get('tier1_pass_rate', 0):>4.0%} "
                  f"{r.get('total_time_s', 0):>6.1f}s")

    print("\n" + "=" * 90)
    print("GP vs UNIFORM ABLATION")
    print("=" * 90)
    gp_abl = ablation_results.get("gp_vs_uniform", {})
    print(f"{'Model':<28} {'Uni Cov':>8} {'Uni T':>7} {'GP Cov':>8} {'GP T':>7} {'ECE':>7}")
    print("-" * 90)
    for name in ABLATION_MODELS:
        r = gp_abl.get(name, {})
        if "error" in r:
            print(f"{name:<28} {'ERR':>8}")
        else:
            print(f"{name:<28} "
                  f"{r.get('uniform_coverage', 0):>7.1%} "
                  f"{r.get('uniform_time_s', 0):>6.1f}s "
                  f"{r.get('gp_coverage', 0):>7.1%} "
                  f"{r.get('gp_time_s', 0):>6.1f}s "
                  f"{r.get('gp_ece', 0):>7.4f}")

    print("\n" + "=" * 90)
    print("NEWTON vs BISECTION (3-state)")
    print("=" * 90)
    nb = ablation_results.get("newton_vs_bisection", {})
    for name in THREE_STATE_MODELS:
        r = nb.get(name, {})
        if "error" in r:
            print(f"  {name}: ERROR - {r.get('error', '?')}")
        else:
            print(f"  {name}: newton={r.get('newton_coverage', 0):.1%}/{r.get('newton_time_s', 0):.1f}s, "
                  f"bisection={r.get('bisection_coverage', 0):.1%}/{r.get('bisection_time_s', 0):.1f}s")

    print("\n" + "=" * 90)
    print("GP CALIBRATION")
    print("=" * 90)
    print(f"{'Model':<28} {'Cells':>6} {'ECE':>8} {'LOO':>8} {'Cov95':>7} {'Time':>7}")
    print("-" * 90)
    for name in ALL_MODEL_NAMES:
        r = gp_results.get(name, {})
        if "error" in r:
            print(f"{name:<28} {'ERR':>6}")
        else:
            print(f"{name:<28} "
                  f"{r.get('n_cells', 0):>6} "
                  f"{r.get('ece', 0):>8.4f} "
                  f"{r.get('loo_mse', 0):>8.4f} "
                  f"{r.get('coverage_95', 0):>6.2f} "
                  f"{r.get('total_time_s', 0):>6.1f}s")

    if depth_results:
        print("\n" + "=" * 90)
        print("DEPTH SCALING")
        print("=" * 90)
        for name, data in depth_results.items():
            if isinstance(data, dict) and "error" in data:
                print(f"  {name}: ERROR - {data.get('error', '?')}")
            elif isinstance(data, list):
                parts = [f"d{p['max_depth']}={p['coverage_fraction']:.1%}/{p['total_time_s']:.1f}s"
                         for p in data]
                print(f"  {name}: {', '.join(parts)}")

    if delta_results:
        print("\n" + "=" * 90)
        print("DELTA-SOUNDNESS VALIDATION")
        print("=" * 90)
        print(f"{'Model':<28} {'Equil':>6} {'Checked':>8} {'Sound':>6} {'Gap':>10} {'Delta':>10}")
        print("-" * 90)
        for name, r in delta_results.items():
            if isinstance(r, dict) and "error" in r:
                print(f"{name:<28} {'ERR':>6}")
            elif isinstance(r, dict):
                print(f"{name:<28} "
                      f"{r.get('n_equilibria', 0):>6} "
                      f"{r.get('n_with_delta', 0):>8} "
                      f"{r.get('n_sound', 0):>6} "
                      f"{r.get('min_eigenvalue_gap', 0):>10.6f} "
                      f"{r.get('max_delta_required', 0):>10.6f}")

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _start_time = time.time()

    print("=" * 70)
    print("PhaseCartographer: Comprehensive Experiment Runner")
    print(f"Timeout: {GLOBAL_TIMEOUT_S}s | Models: {len(ALL_MODEL_NAMES)}")
    print("=" * 70)

    # Phase 1: All benchmarks
    print("\n[Phase 1] Running all 15 benchmarks...")
    try:
        benchmark_results = run_all_benchmarks()
    except TimeoutError as e:
        print(f"\n  TIMEOUT: {e}")
        benchmark_results = {}

    # Phase 2: Ablation studies
    print("\n[Phase 2] Running ablation studies...")
    try:
        ablation_results = run_ablation_studies()
    except TimeoutError as e:
        print(f"\n  TIMEOUT: {e}")
        ablation_results = {"gp_vs_uniform": {}, "newton_vs_bisection": {}}

    # Phase 3: GP calibration
    print("\n[Phase 3] Running GP calibration...")
    gp_results = run_gp_calibration()

    # Phase 4: Depth scaling
    print("\n[Phase 4] Running depth-scaling experiments...")
    try:
        depth_results = run_depth_scaling_experiment()
    except TimeoutError as e:
        print(f"\n  TIMEOUT: {e}")
        depth_results = {}

    # Phase 5: Delta-soundness validation
    print("\n[Phase 5] Running delta-soundness validation...")
    try:
        delta_results = run_delta_validation_experiment()
    except TimeoutError as e:
        print(f"\n  TIMEOUT: {e}")
        delta_results = {}

    total_elapsed = time.time() - _start_time
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Print summary
    print_summary_table(benchmark_results, ablation_results, gp_results,
                        depth_results, delta_results)

    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("  - benchmark_results.json")
    print("  - ablation_results.json")
    print("  - gp_calibration.json")
    print("  - depth_scaling.json")
    print("  - delta_validation.json")
