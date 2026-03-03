#!/usr/bin/env python3
"""
PhaseCartographer CLI: Certified phase atlas generation for biological ODEs.

Usage:
    python -m phase_cartographer                  # Run all 15 benchmarks
    python -m phase_cartographer toggle_switch     # Run a single model
    python -m phase_cartographer --case-study      # Toggle switch design case study
    python -m phase_cartographer --delta-table     # δ-soundness parameter table
    python -m phase_cartographer --baseline        # Numerical vs certified comparison
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def _print_header():
    print("=" * 72)
    print("  PhaseCartographer: Certified Phase Atlases for Biological ODEs")
    print("  Three-tier verification: Interval Arithmetic → dReal → Z3")
    print("=" * 72)
    print()


def _stability_symbol(s):
    MAP = {
        "stable_node": "●", "stable_focus": "◉", "stable_spiral": "◎",
        "unstable_node": "○", "unstable_focus": "◌", "saddle": "⊗",
        "center": "⊙", "degenerate": "?", "unknown": "?",
    }
    return MAP.get(s, "?")


def run_single_model(model_name, max_depth=4, max_cells=200, verbose=True):
    """Run the full certification pipeline on one model and display results."""
    from phase_cartographer.benchmarks.runner import run_benchmark
    from phase_cartographer.models.benchmark_models import get_benchmark

    bm = get_benchmark(model_name)
    if verbose:
        print(f"  Model: {bm.name} ({bm.description})")
        print(f"  States: {bm.n_states}, Params: {bm.n_params}, RHS: {bm.rhs_type}")
        print(f"  Parameter domain: {bm.parameter_domain}")
        print()

    r = run_benchmark(model_name, max_depth=max_depth, max_cells=max_cells)

    if verbose:
        _print_results(r)

    return r


def run_from_file(file_path, max_depth=4, max_cells=200, verbose=True):
    """Run the full certification pipeline on a user-defined model file."""
    from phase_cartographer.benchmarks.runner import run_model
    from phase_cartographer.models.benchmark_models import load_model_file

    bm = load_model_file(file_path)
    if verbose:
        print(f"  Model: {bm.name} ({bm.description})")
        print(f"  States: {bm.n_states}, Params: {bm.n_params}, RHS: {bm.rhs_type}")
        print(f"  Parameter domain: {bm.parameter_domain}")
        print()

    r = run_model(bm, max_depth=max_depth, max_cells=max_cells)

    if verbose:
        _print_results(r)

    return r


def _print_results(r):
    print(f"  Results:")
    print(f"    Certified cells:      {r.certified_cells}")
    print(f"    Coverage:             {r.coverage_fraction:.1%}")
    print(f"    Regimes:              {r.regime_counts}")
    print(f"    Z3 equilibrium:       {r.tier3_verified}/{r.tier3_total}")
    print(f"    Z3 stability:         {r.tier2_verified}/{r.tier2_total}")
    print(f"    Krawczyk contraction: {r.mean_contraction:.4f}")
    print(f"    Eigenvalue gap (min): {r.min_eigenvalue_gap:.6f}")
    print(f"    Near bifurcation:     {r.n_near_bifurcation}")
    print(f"    Time:                 {r.total_time_s:.1f}s")
    print()


def run_all_benchmarks(max_depth=4, max_cells=200):
    """Run all 15 benchmark models."""
    from phase_cartographer.models.benchmark_models import list_benchmarks

    models = list_benchmarks()
    results = {}
    totals = {"cells": 0, "z3_eq": 0, "z3_eq_total": 0,
              "z3_stab": 0, "z3_stab_total": 0, "time": 0.0}

    print(f"Running {len(models)} benchmark models...\n")
    print(f"{'Model':<25} {'n':>2} {'p':>2} {'Cells':>5} {'Cov':>6} "
          f"{'Z3 Eq':>7} {'Z3 Stab':>8} {'Time':>6}")
    print("-" * 72)

    for name in models:
        try:
            r = run_single_model(name, max_depth=max_depth,
                                 max_cells=max_cells, verbose=False)
            results[name] = r
            totals["cells"] += r.certified_cells
            totals["z3_eq"] += r.tier3_verified
            totals["z3_eq_total"] += r.tier3_total
            totals["z3_stab"] += r.tier2_verified
            totals["z3_stab_total"] += r.tier2_total
            totals["time"] += r.total_time_s

            z3_eq = f"{r.tier3_verified}/{r.tier3_total}"
            z3_st = f"{r.tier2_verified}/{r.tier2_total}"
            print(f"{name:<25} {r.n_states:>2} {r.n_params:>2} "
                  f"{r.certified_cells:>5} {r.coverage_fraction:>5.0%} "
                  f"{z3_eq:>7} {z3_st:>8} {r.total_time_s:>5.1f}s")
        except Exception as e:
            print(f"{name:<25}  ERROR: {e}")

    print("-" * 72)
    z3_eq_s = f"{totals['z3_eq']}/{totals['z3_eq_total']}"
    z3_st_s = f"{totals['z3_stab']}/{totals['z3_stab_total']}"
    print(f"{'TOTAL':<25} {'':>2} {'':>2} {totals['cells']:>5} {'':>6} "
          f"{z3_eq_s:>7} {z3_st_s:>8} {totals['time']:>5.1f}s")
    print()

    return results


def run_delta_soundness_table():
    """Compute and display explicit δ-soundness parameters for all models."""
    from phase_cartographer.models.benchmark_models import list_benchmarks, get_benchmark
    from phase_cartographer.benchmarks.runner import certify_box
    from phase_cartographer.smt.delta_bound import DeltaBound, compute_eigenvalue_gap
    from phase_cartographer.interval.interval import Interval
    from phase_cartographer.interval.matrix import IntervalVector

    models = list_benchmarks()
    print("δ-Soundness Parameters (Theorem 2)")
    print("=" * 100)
    print(f"For each model, parameters are computed at a narrow cell near the")
    print(f"parameter-space midpoint. δ_req = γ / (‖R‖·L_Df·rad + 1).")
    print()
    print(f"{'Model':<22} {'γ':>8} {'‖R‖':>8} {'L_Df':>8} {'rad':>8} "
          f"{'δ_req':>8} {'δ_solver':>8} {'Sound':>6} {'Margin':>8}")
    print("-" * 100)

    table = []
    for name in models:
        try:
            bm = get_benchmark(name)
            # Use narrow box near midpoint for meaningful results
            box = []
            for lo, hi in bm.parameter_domain:
                w = hi - lo
                if w > 0:
                    m = (lo + hi) / 2.0
                    r = w * 0.05
                    box.append((m - r, m + r))
                else:
                    box.append((lo, hi))

            cell = certify_box(bm.rhs, box, bm.state_domain,
                               bm.name, bm.n_states, bm.n_params, bm.rhs_type)
            if cell is None or not cell.equilibria:
                # Try full domain
                cell = certify_box(bm.rhs, list(bm.parameter_domain),
                                   bm.state_domain, bm.name, bm.n_states,
                                   bm.n_params, bm.rhs_type)
            if cell is None or not cell.equilibria:
                print(f"{name:<22} {'—':>8} {'—':>8} {'—':>8} {'—':>8} "
                      f"{'—':>8} {'—':>8} {'—':>6} {'—':>8}")
                continue

            eq = cell.equilibria[0]
            if not eq.eigenvalue_real_parts:
                continue

            mu = IntervalVector([Interval(lo, hi) for lo, hi in cell.parameter_box])
            X = IntervalVector([Interval(lo, hi) for lo, hi in eq.state_enclosure])
            eig_rps = [Interval(lo, hi) for lo, hi in eq.eigenvalue_real_parts]

            db = DeltaBound(bm.rhs)
            result = db.compute(X, mu, eig_rps, delta_solver=1e-3)

            # Compute ‖R‖ directly
            x_mid = X.midpoint()
            mu_mid = mu.midpoint()
            try:
                J = bm.rhs.jacobian(x_mid, mu_mid)
                R = np.linalg.inv(J)
                R_norm = float(np.linalg.norm(R, ord=np.inf))
            except Exception:
                R_norm = float('inf')

            rad = max(float(np.max(X.radius())), float(np.max(mu.radius())))

            row = {
                "model": name,
                "gamma": result.eigenvalue_gap,
                "R_norm": R_norm,
                "L_Df": result.lipschitz_Df,
                "rad": rad,
                "delta_required": result.delta_required,
                "delta_solver": 1e-3,
                "is_sound": result.is_sound,
                "margin": result.soundness_margin,
            }
            table.append(row)

            sound_sym = "✓" if result.is_sound else "✗"
            print(f"{name:<22} {result.eigenvalue_gap:>8.4f} {R_norm:>8.2f} "
                  f"{result.lipschitz_Df:>8.2f} {rad:>8.4f} "
                  f"{result.delta_required:>8.4f} {1e-3:>8.4f} "
                  f"{sound_sym:>6} {result.soundness_margin:>8.4f}")
        except Exception as e:
            print(f"{name:<22} ERROR: {e}")

    print("-" * 100)
    n_sound = sum(1 for r in table if r["is_sound"])
    print(f"\nδ-sound: {n_sound}/{len(table)} models")
    print()
    return table


def run_case_study():
    """Run the toggle switch design case study."""
    from phase_cartographer.models.benchmark_models import get_benchmark
    from phase_cartographer.benchmarks.runner import certify_box
    from phase_cartographer.interval.interval import Interval
    from phase_cartographer.interval.matrix import IntervalVector
    from scipy.optimize import fsolve

    print("Toggle Switch Design Case Study")
    print("=" * 72)
    print()
    print("Scenario: A synthetic biologist wants to build a genetic toggle")
    print("switch that exhibits bistability. They need to choose promoter")
    print("strengths (a1, a2) from the range [2.0, 4.0] × [2.0, 4.0].")
    print()
    print("Question: Which (a1, a2) values guarantee a functional bistable")
    print("switch? Numerical simulation suggests ~80% of the space is")
    print("bistable, but how many of those predictions are reliable?")
    print()

    bm = get_benchmark("toggle_switch")
    rhs = bm.rhs

    # Grid of parameter points
    n_grid = 12
    a1_vals = np.linspace(2.1, 3.9, n_grid)
    a2_vals = np.linspace(2.1, 3.9, n_grid)

    numerical_bistable = 0
    certified_bistable = 0
    numerical_not_certified = 0
    total = 0

    print(f"Sampling {n_grid}×{n_grid} = {n_grid**2} parameter points...")
    print()

    for a1 in a1_vals:
        for a2 in a2_vals:
            total += 1
            mu = np.array([a1, a2, 2.0, 2.0])

            # Numerical analysis: find equilibria with fsolve
            roots = []
            for x0 in [np.array([0.5, 0.5]), np.array([3.0, 0.5]),
                        np.array([0.5, 3.0]), np.array([2.0, 2.0])]:
                try:
                    sol = fsolve(lambda x: rhs.evaluate(x, mu), x0, full_output=False)
                    if np.all(np.isfinite(sol)) and np.all(sol > 0):
                        residual = np.linalg.norm(rhs.evaluate(sol, mu))
                        if residual < 1e-8:
                            is_dup = any(np.linalg.norm(sol - r) < 0.1 for r in roots)
                            if not is_dup:
                                roots.append(sol)
                except Exception:
                    pass

            n_stable_num = 0
            for root in roots:
                J = rhs.jacobian(root, mu)
                eigvals = np.linalg.eigvals(J)
                if np.all(eigvals.real < 0):
                    n_stable_num += 1

            is_numerical_bistable = n_stable_num >= 2
            if is_numerical_bistable:
                numerical_bistable += 1

            # Certified analysis: use Krawczyk
            delta = 0.05
            box = [(a1 - delta, a1 + delta), (a2 - delta, a2 + delta),
                   (2.0, 2.0), (2.0, 2.0)]
            cell = certify_box(rhs, box, bm.state_domain,
                               bm.name, bm.n_states, bm.n_params, bm.rhs_type)

            is_certified_bistable = (cell is not None and
                                     cell.n_stable_equilibria() >= 2)
            if is_certified_bistable:
                certified_bistable += 1

            if is_numerical_bistable and not is_certified_bistable:
                numerical_not_certified += 1

    gap_fraction = numerical_not_certified / max(numerical_bistable, 1)

    print("Results:")
    print(f"  Total parameter points:        {total}")
    print(f"  Numerical bistable:            {numerical_bistable} "
          f"({numerical_bistable/total:.1%})")
    print(f"  Certified bistable:            {certified_bistable} "
          f"({certified_bistable/total:.1%})")
    print(f"  Gap (numerical ∧ ¬certified):  {numerical_not_certified} "
          f"({gap_fraction:.1%} of numerical predictions)")
    print()
    print("Interpretation:")
    print(f"  {gap_fraction:.1%} of parameter points where numerical methods predict")
    print(f"  bistability CANNOT be machine-certified as bistable.")
    print(f"  A synthetic biologist relying only on numerics would risk choosing")
    print(f"  parameters from this uncertain region.")
    print()
    print(f"  Safe design region: {certified_bistable} certified bistable points")
    print(f"  out of {total} total ({certified_bistable/total:.1%} of parameter space)")
    print()

    return {
        "total": total,
        "numerical_bistable": numerical_bistable,
        "certified_bistable": certified_bistable,
        "gap": numerical_not_certified,
        "gap_fraction": gap_fraction,
    }


def run_baseline_comparison():
    """Run numerical continuation vs certified comparison."""
    from phase_cartographer.benchmarks.numerical_baseline import (
        run_full_baseline_comparison,
    )

    print("Numerical Continuation Baseline Comparison")
    print("=" * 72)
    print()
    print("Comparing pseudo-arclength continuation (like MATCONT/AUTO)")
    print("with certified interval-arithmetic analysis.\n")

    comparison = run_full_baseline_comparison(n_steps=50)

    total_errors = 0
    total_comparisons = 0
    total_agreements = 0

    print(f"{'Model':<25} {'Num Eq':>7} {'Bifs':>5} "
          f"{'Agree':>6} {'Disagree':>9} {'Errors':>7}")
    print("-" * 72)

    for name, data in comparison.items():
        if "error" in data:
            print(f"{name:<25} ERROR: {data['error']}")
            continue
        num = data.get("numerical", {})
        comp = data.get("comparison", {})
        n_agree = comp.get("n_agreements", 0)
        n_disagree = comp.get("n_disagreements", 0)
        n_errors = comp.get("n_errors_caught", 0)
        total_errors += n_errors
        total_comparisons += n_agree + n_disagree
        total_agreements += n_agree

        print(f"{name:<25} {num.get('n_equilibria_found', 0):>7} "
              f"{num.get('n_bifurcations', 0):>5} "
              f"{n_agree:>6} {n_disagree:>9} {n_errors:>7}")

    print("-" * 72)
    rate = total_agreements / max(total_comparisons, 1)
    print(f"Agreement rate: {rate:.1%} ({total_agreements}/{total_comparisons})")
    print(f"Floating-point errors caught by certification: {total_errors}")
    print()

    return comparison


def main():
    # Handle subcommands (check, init-model) before argparse to avoid
    # conflict between subparsers and the positional 'model' argument.
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        _main_check()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "init-model":
        _main_init_model()
        return

    parser = argparse.ArgumentParser(
        description="PhaseCartographer: Certified Phase Atlases for Biological ODEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", nargs="?", default=None,
                        help="Model name (e.g., toggle_switch, brusselator)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a YAML/JSON/SBML model file (use instead of model name)")
    parser.add_argument("--case-study", action="store_true",
                        help="Run toggle switch design case study")
    parser.add_argument("--delta-table", action="store_true",
                        help="Show δ-soundness parameter table")
    parser.add_argument("--baseline", action="store_true",
                        help="Run numerical continuation baseline comparison")
    parser.add_argument("--depth", type=int, default=4,
                        help="Maximum refinement depth (default: 4)")
    parser.add_argument("--max-cells", type=int, default=200,
                        help="Maximum cells per model (default: 200)")

    args = parser.parse_args()
    _print_header()

    if args.case_study:
        run_case_study()
    elif args.delta_table:
        run_delta_soundness_table()
    elif args.baseline:
        run_baseline_comparison()
    elif args.file:
        run_from_file(args.file, max_depth=args.depth,
                      max_cells=args.max_cells)
    elif args.model:
        run_single_model(args.model, max_depth=args.depth,
                         max_cells=args.max_cells)
    else:
        run_all_benchmarks(max_depth=args.depth, max_cells=args.max_cells)


def _main_check():
    """Entry point for 'check' subcommand."""
    parser = argparse.ArgumentParser(
        prog="phase_cartographer check",
        description="Validate a model YAML/JSON/SBML file without running analysis",
    )
    parser.add_argument("file", help="Path to YAML/JSON/SBML model file")
    args = parser.parse_args(sys.argv[2:])
    _print_header()
    _run_check(args.file)


def _main_init_model():
    """Entry point for 'init-model' subcommand."""
    parser = argparse.ArgumentParser(
        prog="phase_cartographer init-model",
        description="Generate a starter YAML model template",
    )
    parser.add_argument("--output", "-o", default="my_model.yaml",
                        help="Output file path (default: my_model.yaml)")
    parser.add_argument("--template", choices=["2d", "3d", "oscillator"],
                        default="2d",
                        help="Template type (default: 2d)")
    args = parser.parse_args(sys.argv[2:])
    _print_header()
    _run_init_model(args.output, args.template)


def _run_check(file_path):
    """Validate a model file and report issues."""
    from phase_cartographer.models.benchmark_models import validate_model_file

    print(f"  Checking: {file_path}\n")
    issues = validate_model_file(file_path)
    if not issues:
        from phase_cartographer.models.benchmark_models import load_model_file
        bm = load_model_file(file_path)
        print(f"  ✓ Valid model: {bm.name}")
        print(f"    States:     {bm.n_states} ({', '.join(str(s) for s in bm.rhs.state_symbols)})")
        print(f"    Parameters: {bm.n_params} ({', '.join(str(s) for s in bm.rhs.param_symbols)})")
        print(f"    Equations:  {len(bm.rhs.expressions)}")
        for i, expr in enumerate(bm.rhs.expressions):
            print(f"      d{bm.rhs.state_symbols[i]}/dt = {expr}")
        print(f"    RHS type:   {bm.rhs_type}")
        print()
    else:
        print(f"  ✗ {len(issues)} issue(s) found:\n")
        for issue in issues:
            print(f"    • {issue}")
        print()
        sys.exit(1)


_TEMPLATES = {
    "2d": """\
# 2D ODE model template for PhaseCartographer
# Edit the equations and domains to define your system.
name: my_2d_model
description: A 2D system with mutual interaction

states: [x, y]

parameters:
  - name: a
    bounds: [0.5, 3.0]
  - name: b
    bounds: [0.5, 3.0]

# dx/dt and dy/dt as SymPy expressions
equations:
  - "a / (1 + y**2) - x"
  - "b / (1 + x**2) - y"

state_domain:
  - [0.1, 5.0]
  - [0.1, 5.0]

rhs_type: hill
expected_regimes: [monostable, bistable]
""",
    "3d": """\
# 3D ODE model template for PhaseCartographer
name: my_3d_model
description: A 3D system (e.g., repressilator-like)

states: [x, y, z]

parameters:
  - name: alpha
    bounds: [1.0, 10.0]
  - name: n
    bounds: [2.0, 2.0]

equations:
  - "alpha / (1 + z**n) - x"
  - "alpha / (1 + x**n) - y"
  - "alpha / (1 + y**n) - z"

state_domain:
  - [0.01, 12.0]
  - [0.01, 12.0]
  - [0.01, 12.0]

rhs_type: hill
expected_regimes: [oscillatory]
""",
    "oscillator": """\
# Oscillator model template for PhaseCartographer
name: my_oscillator
description: A chemical oscillator

states: [x, y]

parameters:
  - name: A
    bounds: [0.5, 2.0]
  - name: B
    bounds: [1.5, 4.0]

equations:
  - "A - (B + 1)*x + x**2 * y"
  - "B*x - x**2 * y"

state_domain:
  - [0.01, 5.0]
  - [0.01, 10.0]

rhs_type: polynomial
expected_regimes: [oscillatory, monostable]
""",
}


def _run_init_model(output_path, template):
    """Generate a starter YAML model file."""
    content = _TEMPLATES[template]
    if os.path.exists(output_path):
        print(f"  ✗ File already exists: {output_path}")
        print(f"    Use a different --output path or delete the existing file.")
        sys.exit(1)
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"  ✓ Created {template} template: {output_path}")
    print(f"    Edit the file, then run:")
    print(f"      python -m phase_cartographer check {output_path}")
    print(f"      python -m phase_cartographer --file {output_path}")
    print()


if __name__ == "__main__":
    main()
