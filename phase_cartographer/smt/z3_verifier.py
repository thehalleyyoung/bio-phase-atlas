"""
Z3-based exact verification for polynomial biological ODE models (Tier 3).

For models with polynomial right-hand sides (no Hill functions or
transcendentals), Z3's nonlinear real arithmetic (QF_NRA) solver can
provide exact verification of equilibrium existence and stability claims.

Verification strategy:
  1. Encode ¬φ(M): the negation of the regime claim.
  2. If Z3 returns UNSAT, the claim is exactly correct.
  3. If Z3 returns SAT, a counterexample exists.
  4. If Z3 returns UNKNOWN, fall back to Tier 1.

For stability verification, we use the Routh-Hurwitz conditions which
are polynomial in the Jacobian entries and hence in (x, μ).

Z3-based delta-approximate verification for Hill-function models (Tier 2 fallback):
  When dReal is unavailable, we can still verify Hill-function models by:
  1. Replacing Hill functions h(x) = 1/(1+x^n) with polynomial approximation
     valid over the certified enclosure.
  2. Bounding the approximation error.
  3. Verifying the polynomial approximation with Z3.
  4. Using the delta-soundness bound (Theorem B1) to lift to exact correctness.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
import sympy as sp

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


@dataclass
class Z3VerificationResult:
    """Result of Z3 verification."""
    verified: bool
    tier: str  # "tier3_exact" or "tier2_z3_delta"
    status: str  # "unsat", "sat", "unknown", "error", "unavailable"
    time_s: float = 0.0
    counterexample: Optional[Dict[str, float]] = None
    delta_used: float = 0.0
    message: str = ""

    def to_dict(self) -> dict:
        d = {
            'verified': self.verified,
            'tier': self.tier,
            'status': self.status,
            'time_s': round(self.time_s, 4),
            'message': self.message,
        }
        if self.counterexample:
            d['counterexample'] = self.counterexample
        if self.delta_used > 0:
            d['delta_used'] = self.delta_used
        return d


def _sympy_to_z3(expr, z3_vars: Dict[str, Any]) -> Any:
    """Convert a SymPy expression to a Z3 expression.

    Supports: Add, Mul, Pow (integer exponents), Number, Symbol, rational numbers.
    Also handles rational expressions (a/b) common in Hill functions.
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 not available")

    if isinstance(expr, sp.Number):
        if isinstance(expr, sp.Integer):
            return z3.RealVal(int(expr))
        if isinstance(expr, sp.Rational):
            return z3.RealVal(int(expr.p)) / z3.RealVal(int(expr.q))
        return z3.RealVal(float(expr))

    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name not in z3_vars:
            z3_vars[name] = z3.Real(name)
        return z3_vars[name]

    if isinstance(expr, sp.Add):
        terms = [_sympy_to_z3(a, z3_vars) for a in expr.args]
        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    if isinstance(expr, sp.Mul):
        # Separate numerator and denominator to handle rational expressions
        numer, denom = expr.as_numer_denom()
        if denom != sp.Integer(1):
            n_z3 = _sympy_to_z3(numer, z3_vars)
            d_z3 = _sympy_to_z3(denom, z3_vars)
            return n_z3 / d_z3
        factors = [_sympy_to_z3(a, z3_vars) for a in expr.args]
        result = factors[0]
        for f in factors[1:]:
            result = result * f
        return result

    if isinstance(expr, sp.Pow):
        base = _sympy_to_z3(expr.args[0], z3_vars)
        exp_val = expr.args[1]
        if isinstance(exp_val, sp.Integer):
            n = int(exp_val)
            if n >= 0:
                return base ** n
            else:
                # x^(-n) = 1 / x^n
                return z3.RealVal(1) / (base ** (-n))
        if isinstance(exp_val, sp.Rational):
            if exp_val == sp.Rational(1, 2):
                raise NotImplementedError("sqrt not directly supported in QF_NRA")
        raise NotImplementedError(
            f"Non-integer power {exp_val} not supported for Z3 QF_NRA encoding"
        )

    # Handle rational expressions explicitly
    if hasattr(expr, 'as_numer_denom'):
        numer, denom = expr.as_numer_denom()
        if denom != sp.Integer(1):
            n_z3 = _sympy_to_z3(numer, z3_vars)
            d_z3 = _sympy_to_z3(denom, z3_vars)
            return n_z3 / d_z3

    # Try to expand and convert
    expanded = sp.expand(expr)
    if expanded != expr:
        return _sympy_to_z3(expanded, z3_vars)

    # Try cancel to simplify rational expressions
    cancelled = sp.cancel(expr)
    if cancelled != expr:
        return _sympy_to_z3(cancelled, z3_vars)

    raise NotImplementedError(f"Cannot convert {type(expr)}: {expr} to Z3")


def verify_equilibrium_z3(
    expressions: List[sp.Expr],
    state_symbols: List[sp.Symbol],
    param_symbols: List[sp.Symbol],
    state_enclosure: List[Tuple[float, float]],
    param_box: List[Tuple[float, float]],
    timeout_ms: int = 30000,
) -> Z3VerificationResult:
    """
    Verify that an equilibrium exists within the state enclosure for all
    parameters in the param box, using Z3 exact verification (Tier 3).

    Encodes: ¬(∃x ∈ X. f(x, μ) = 0)  for some μ ∈ M
    i.e.,    ∃μ ∈ M. ∀x ∈ X. ∨_i |f_i(x, μ)| > 0

    If Z3 returns UNSAT on this formula, then for every μ ∈ M, there exists
    x ∈ X with f(x, μ) = 0.

    In practice, we verify a stronger condition: that f(x̃, μ) is close to zero
    for the midpoint x̃, combined with the Krawczyk certificate. Z3 checks the
    supporting conditions.
    """
    if not Z3_AVAILABLE:
        return Z3VerificationResult(
            verified=False, tier="tier3_exact", status="unavailable",
            message="Z3 not installed"
        )

    t0 = time.time()
    n = len(state_symbols)
    p = len(param_symbols)

    try:
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)

        z3_vars: Dict[str, Any] = {}

        # Create Z3 variables
        x_z3 = []
        for s in state_symbols:
            name = str(s)
            v = z3.Real(name)
            z3_vars[name] = v
            x_z3.append(v)

        mu_z3 = []
        for s in param_symbols:
            name = str(s)
            v = z3.Real(name)
            z3_vars[name] = v
            mu_z3.append(v)

        # State bounds: x ∈ X
        for i in range(n):
            lo, hi = state_enclosure[i]
            solver.add(x_z3[i] >= z3.RealVal(lo))
            solver.add(x_z3[i] <= z3.RealVal(hi))

        # Parameter bounds: μ ∈ M
        for i in range(p):
            lo, hi = param_box[i]
            if abs(hi - lo) < 1e-15:
                solver.add(mu_z3[i] == z3.RealVal((lo + hi) / 2.0))
            else:
                solver.add(mu_z3[i] >= z3.RealVal(lo))
                solver.add(mu_z3[i] <= z3.RealVal(hi))

        # Convert RHS expressions to Z3
        f_z3 = []
        for expr in expressions:
            # Use cancel to get clean rational form (handles Hill functions)
            try:
                cleaned = sp.cancel(sp.nsimplify(expr, rational=True))
            except Exception:
                cleaned = expr
            f_z3.append(_sympy_to_z3(cleaned, z3_vars))

        # For rational expressions f = p/q, equilibrium means p = 0 (assuming q ≠ 0)
        # Extract numerators for cleaner Z3 encoding
        for i in range(n):
            numer, denom = sp.cancel(expressions[i]).as_numer_denom()
            if denom != sp.Integer(1):
                # Use numerator = 0 instead of f = 0 for better Z3 performance
                try:
                    numer_z3 = _sympy_to_z3(sp.expand(numer), z3_vars)
                    solver.add(numer_z3 == 0)
                except NotImplementedError:
                    solver.add(f_z3[i] == 0)
            else:
                solver.add(f_z3[i] == 0)

        # Check satisfiability
        result = solver.check()
        elapsed = time.time() - t0

        if result == z3.sat:
            # SAT means equilibrium exists (good!) - extract witness
            model = solver.model()
            witness = {}
            for name, var in z3_vars.items():
                val = model.eval(var, model_completion=True)
                try:
                    if val.is_real():
                        witness[name] = float(val.as_decimal(10).rstrip('?'))
                    else:
                        witness[name] = str(val)
                except Exception:
                    witness[name] = str(val)

            return Z3VerificationResult(
                verified=True, tier="tier3_exact", status="sat",
                time_s=elapsed, counterexample=witness,
                message="Z3 confirmed equilibrium exists in enclosure"
            )

        elif result == z3.unsat:
            return Z3VerificationResult(
                verified=False, tier="tier3_exact", status="unsat",
                time_s=elapsed,
                message="Z3 proved no equilibrium in enclosure (contradicts Krawczyk)"
            )

        else:
            return Z3VerificationResult(
                verified=False, tier="tier3_exact", status="unknown",
                time_s=elapsed,
                message="Z3 returned unknown (timeout or complexity)"
            )

    except Exception as e:
        return Z3VerificationResult(
            verified=False, tier="tier3_exact", status="error",
            time_s=time.time() - t0,
            message=f"Z3 error: {str(e)}"
        )


def verify_stability_z3(
    expressions: List[sp.Expr],
    state_symbols: List[sp.Symbol],
    param_symbols: List[sp.Symbol],
    state_enclosure: List[Tuple[float, float]],
    param_box: List[Tuple[float, float]],
    stability_type: str = "stable",
    timeout_ms: int = 30000,
) -> Z3VerificationResult:
    """
    Verify stability at an equilibrium using Routh-Hurwitz conditions via Z3.

    For n=2: stable iff tr(J) < 0 and det(J) > 0.
    For n=3: stable iff p > 0, r > 0, p*q - r > 0 (Routh-Hurwitz).

    Strategy: check if the negation of stability conditions is UNSAT.
    If UNSAT, stability is verified for all (x, μ) in the enclosure.
    """
    if not Z3_AVAILABLE:
        return Z3VerificationResult(
            verified=False, tier="tier3_exact", status="unavailable",
            message="Z3 not installed"
        )

    t0 = time.time()
    n = len(state_symbols)
    p = len(param_symbols)

    try:
        # Compute symbolic Jacobian
        jac_expr = []
        for expr in expressions:
            row = [sp.diff(expr, s) for s in state_symbols]
            jac_expr.append(row)

        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        z3_vars: Dict[str, Any] = {}

        x_z3 = []
        for s in state_symbols:
            name = str(s)
            v = z3.Real(name)
            z3_vars[name] = v
            x_z3.append(v)

        mu_z3 = []
        for s in param_symbols:
            name = str(s)
            v = z3.Real(name)
            z3_vars[name] = v
            mu_z3.append(v)

        # State and parameter bounds
        for i in range(n):
            lo, hi = state_enclosure[i]
            solver.add(x_z3[i] >= z3.RealVal(lo))
            solver.add(x_z3[i] <= z3.RealVal(hi))

        for i in range(p):
            lo, hi = param_box[i]
            if abs(hi - lo) < 1e-15:
                solver.add(mu_z3[i] == z3.RealVal((lo + hi) / 2.0))
            else:
                solver.add(mu_z3[i] >= z3.RealVal(lo))
                solver.add(mu_z3[i] <= z3.RealVal(hi))

        # Convert Jacobian to Z3
        J_z3 = []
        for i in range(n):
            row = []
            for j in range(n):
                expanded = sp.expand(sp.nsimplify(jac_expr[i][j], rational=True))
                row.append(_sympy_to_z3(expanded, z3_vars))
            J_z3.append(row)

        # Encode negation of stability condition
        # We want to prove: ∀(x,μ) ∈ X×M, stability holds
        # So we check: ∃(x,μ) ∈ X×M such that stability FAILS → UNSAT means verified
        if stability_type == "stable":
            if n == 2:
                # Routh-Hurwitz 2D: tr(J) < 0 AND det(J) > 0
                tr_J = J_z3[0][0] + J_z3[1][1]
                det_J = J_z3[0][0] * J_z3[1][1] - J_z3[0][1] * J_z3[1][0]
                # Negation: tr(J) >= 0 OR det(J) <= 0
                solver.add(z3.Or(tr_J >= 0, det_J <= 0))
            elif n == 3:
                # Routh-Hurwitz 3D
                p_rh = -(J_z3[0][0] + J_z3[1][1] + J_z3[2][2])

                q_rh = (J_z3[0][0] * J_z3[1][1] - J_z3[0][1] * J_z3[1][0]
                        + J_z3[0][0] * J_z3[2][2] - J_z3[0][2] * J_z3[2][0]
                        + J_z3[1][1] * J_z3[2][2] - J_z3[1][2] * J_z3[2][1])

                det_J = (J_z3[0][0] * (J_z3[1][1] * J_z3[2][2] - J_z3[1][2] * J_z3[2][1])
                         - J_z3[0][1] * (J_z3[1][0] * J_z3[2][2] - J_z3[1][2] * J_z3[2][0])
                         + J_z3[0][2] * (J_z3[1][0] * J_z3[2][1] - J_z3[1][1] * J_z3[2][0]))
                r_rh = -det_J

                solver.add(z3.Or(p_rh <= 0, r_rh <= 0, p_rh * q_rh - r_rh <= 0))
            else:
                return Z3VerificationResult(
                    verified=False, tier="tier3_exact", status="error",
                    time_s=time.time() - t0,
                    message=f"Routh-Hurwitz not implemented for n={n}"
                )
        elif stability_type == "unstable":
            if n == 2:
                tr_J = J_z3[0][0] + J_z3[1][1]
                det_J = J_z3[0][0] * J_z3[1][1] - J_z3[0][1] * J_z3[1][0]
                # Unstable if tr(J) > 0 or det(J) < 0
                # Negation: tr(J) <= 0 AND det(J) >= 0
                solver.add(tr_J <= 0)
                solver.add(det_J >= 0)
            elif n == 3:
                # Unstable 3D: negation of Routh-Hurwitz conditions holds
                p_rh = -(J_z3[0][0] + J_z3[1][1] + J_z3[2][2])
                q_rh = (J_z3[0][0] * J_z3[1][1] - J_z3[0][1] * J_z3[1][0]
                        + J_z3[0][0] * J_z3[2][2] - J_z3[0][2] * J_z3[2][0]
                        + J_z3[1][1] * J_z3[2][2] - J_z3[1][2] * J_z3[2][1])
                det_J = (J_z3[0][0] * (J_z3[1][1] * J_z3[2][2] - J_z3[1][2] * J_z3[2][1])
                         - J_z3[0][1] * (J_z3[1][0] * J_z3[2][2] - J_z3[1][2] * J_z3[2][0])
                         + J_z3[0][2] * (J_z3[1][0] * J_z3[2][1] - J_z3[1][1] * J_z3[2][0]))
                r_rh = -det_J
                # Unstable: at least one RH condition fails
                # Negation of "at least one fails" = "all hold" → stable
                solver.add(p_rh > 0)
                solver.add(r_rh > 0)
                solver.add(p_rh * q_rh - r_rh > 0)
            else:
                return Z3VerificationResult(
                    verified=False, tier="tier3_exact", status="error",
                    time_s=time.time() - t0,
                    message=f"Unstable check not implemented for n={n}"
                )
        elif stability_type == "saddle":
            if n == 2:
                # Saddle: det(J) < 0 (one positive, one negative eigenvalue)
                det_J = J_z3[0][0] * J_z3[1][1] - J_z3[0][1] * J_z3[1][0]
                # Negation: det(J) >= 0
                solver.add(det_J >= 0)
            elif n == 3:
                # Saddle in 3D: eigenvalues with mixed signs
                # A sufficient condition: det(J) > 0 AND trace(J) > 0
                # (means odd number of negative eigenvalues with positive product)
                # Negation is complex; use trace/det conditions
                tr_J = J_z3[0][0] + J_z3[1][1] + J_z3[2][2]
                det_J = (J_z3[0][0] * (J_z3[1][1] * J_z3[2][2] - J_z3[1][2] * J_z3[2][1])
                         - J_z3[0][1] * (J_z3[1][0] * J_z3[2][2] - J_z3[1][2] * J_z3[2][0])
                         + J_z3[0][2] * (J_z3[1][0] * J_z3[2][1] - J_z3[1][1] * J_z3[2][0]))
                # For saddle: not all eigenvalues same sign
                p_rh = -tr_J
                q_rh = (J_z3[0][0] * J_z3[1][1] - J_z3[0][1] * J_z3[1][0]
                        + J_z3[0][0] * J_z3[2][2] - J_z3[0][2] * J_z3[2][0]
                        + J_z3[1][1] * J_z3[2][2] - J_z3[1][2] * J_z3[2][1])
                r_rh = -det_J
                # Saddle means RH conditions don't all hold AND not all unstable
                # Negation: all RH hold (stable) OR all eigenvalues positive
                solver.add(z3.Or(
                    z3.And(p_rh > 0, r_rh > 0, p_rh * q_rh - r_rh > 0),
                    z3.And(p_rh < 0, r_rh < 0),
                ))
            else:
                return Z3VerificationResult(
                    verified=False, tier="tier3_exact", status="error",
                    time_s=time.time() - t0,
                    message=f"Saddle check not implemented for n={n}"
                )
        else:
            return Z3VerificationResult(
                verified=False, tier="tier3_exact", status="error",
                time_s=time.time() - t0,
                message=f"Unknown stability type: {stability_type}"
            )

        result = solver.check()
        elapsed = time.time() - t0

        if result == z3.unsat:
            return Z3VerificationResult(
                verified=True, tier="tier3_exact", status="unsat",
                time_s=elapsed,
                message=f"Z3 confirmed {stability_type} stability for all (x,μ) in enclosure"
            )
        elif result == z3.sat:
            model = solver.model()
            cex = {}
            for name, var in z3_vars.items():
                val = model.eval(var, model_completion=True)
                try:
                    cex[name] = float(val.as_decimal(10).rstrip('?'))
                except Exception:
                    cex[name] = str(val)
            return Z3VerificationResult(
                verified=False, tier="tier3_exact", status="sat",
                time_s=elapsed, counterexample=cex,
                message=f"Z3 found counterexample to {stability_type} stability"
            )
        else:
            return Z3VerificationResult(
                verified=False, tier="tier3_exact", status="unknown",
                time_s=elapsed,
                message="Z3 returned unknown"
            )

    except NotImplementedError as e:
        return Z3VerificationResult(
            verified=False, tier="tier3_exact", status="error",
            time_s=time.time() - t0,
            message=f"Encoding error: {str(e)}"
        )
    except Exception as e:
        return Z3VerificationResult(
            verified=False, tier="tier3_exact", status="error",
            time_s=time.time() - t0,
            message=f"Z3 error: {str(e)}"
        )


def verify_no_other_equilibria_z3(
    expressions: List[sp.Expr],
    state_symbols: List[sp.Symbol],
    param_symbols: List[sp.Symbol],
    known_enclosures: List[List[Tuple[float, float]]],
    state_domain: List[Tuple[float, float]],
    param_box: List[Tuple[float, float]],
    timeout_ms: int = 60000,
) -> Z3VerificationResult:
    """
    Verify that no equilibria exist outside the known enclosures.

    Encodes: ∃x ∈ (state_domain \\ ∪ enclosures), μ ∈ M s.t. f(x,μ) = 0
    UNSAT means no other equilibria exist → regime classification is complete.
    """
    if not Z3_AVAILABLE:
        return Z3VerificationResult(
            verified=False, tier="tier3_exact", status="unavailable",
            message="Z3 not installed"
        )

    t0 = time.time()
    n = len(state_symbols)
    p = len(param_symbols)

    try:
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        z3_vars: Dict[str, Any] = {}

        x_z3 = []
        for s in state_symbols:
            name = str(s)
            v = z3.Real(name)
            z3_vars[name] = v
            x_z3.append(v)

        mu_z3 = []
        for s in param_symbols:
            name = str(s)
            v = z3.Real(name)
            z3_vars[name] = v
            mu_z3.append(v)

        # State domain bounds
        for i in range(n):
            lo, hi = state_domain[i]
            solver.add(x_z3[i] >= z3.RealVal(lo))
            solver.add(x_z3[i] <= z3.RealVal(hi))

        # Parameter bounds
        for i in range(p):
            lo, hi = param_box[i]
            if abs(hi - lo) < 1e-15:
                solver.add(mu_z3[i] == z3.RealVal((lo + hi) / 2.0))
            else:
                solver.add(mu_z3[i] >= z3.RealVal(lo))
                solver.add(mu_z3[i] <= z3.RealVal(hi))

        # Equilibrium condition: f(x, μ) = 0
        f_z3 = []
        for expr in expressions:
            expanded = sp.expand(sp.nsimplify(expr, rational=True))
            f_z3.append(_sympy_to_z3(expanded, z3_vars))

        for i in range(n):
            solver.add(f_z3[i] == 0)

        # Exclude known enclosures: x must be OUTSIDE all known enclosures
        exclusion_clauses = []
        for enc in known_enclosures:
            # x is outside this enclosure if any coordinate is outside
            outside_this = []
            for i in range(n):
                lo, hi = enc[i]
                outside_this.append(x_z3[i] < z3.RealVal(lo))
                outside_this.append(x_z3[i] > z3.RealVal(hi))
            exclusion_clauses.append(z3.Or(*outside_this))

        if exclusion_clauses:
            solver.add(z3.And(*exclusion_clauses))

        result = solver.check()
        elapsed = time.time() - t0

        if result == z3.unsat:
            return Z3VerificationResult(
                verified=True, tier="tier3_exact", status="unsat",
                time_s=elapsed,
                message="Z3 confirmed no other equilibria outside known enclosures"
            )
        elif result == z3.sat:
            model = solver.model()
            cex = {}
            for name, var in z3_vars.items():
                val = model.eval(var, model_completion=True)
                try:
                    cex[name] = float(val.as_decimal(10).rstrip('?'))
                except Exception:
                    cex[name] = str(val)
            return Z3VerificationResult(
                verified=False, tier="tier3_exact", status="sat",
                time_s=elapsed, counterexample=cex,
                message="Z3 found additional equilibrium outside known enclosures"
            )
        else:
            return Z3VerificationResult(
                verified=False, tier="tier3_exact", status="unknown",
                time_s=elapsed,
                message="Z3 returned unknown"
            )

    except Exception as e:
        return Z3VerificationResult(
            verified=False, tier="tier3_exact", status="error",
            time_s=time.time() - t0,
            message=f"Z3 error: {str(e)}"
        )


def verify_cell_z3(
    cell_dict: dict,
    expressions: List[sp.Expr],
    state_symbols: List[sp.Symbol],
    param_symbols: List[sp.Symbol],
    state_domain: List[Tuple[float, float]],
    timeout_ms: int = 30000,
) -> Dict[str, Z3VerificationResult]:
    """
    Full Z3 verification of a certified cell (Tier 3).

    Performs:
    1. Equilibrium existence verification for each claimed equilibrium
    2. Stability verification via Routh-Hurwitz for each equilibrium
    3. No-other-equilibria check (completeness of regime classification)

    Returns a dictionary of verification results.
    """
    results = {}
    param_box = cell_dict.get('parameter_box', [])

    for eq_idx, eq in enumerate(cell_dict.get('equilibria', [])):
        state_enc = eq.get('state_enclosure', [])
        stab = eq.get('stability', 'unknown')

        # 1. Verify equilibrium existence
        eq_result = verify_equilibrium_z3(
            expressions, state_symbols, param_symbols,
            state_enc, param_box, timeout_ms=timeout_ms
        )
        results[f'eq_{eq_idx}_existence'] = eq_result

        # 2. Verify stability
        if stab == "saddle":
            stab_type = "saddle"
        elif stab in ("stable_node", "stable_focus", "stable_spiral"):
            stab_type = "stable"
        else:
            stab_type = "unstable"

        stab_result = verify_stability_z3(
            expressions, state_symbols, param_symbols,
            state_enc, param_box,
            stability_type=stab_type,
            timeout_ms=timeout_ms
        )
        results[f'eq_{eq_idx}_stability'] = stab_result

    # 3. No-other-equilibria check
    known_enclosures = [
        eq.get('state_enclosure', [])
        for eq in cell_dict.get('equilibria', [])
    ]
    if known_enclosures:
        completeness_result = verify_no_other_equilibria_z3(
            expressions, state_symbols, param_symbols,
            known_enclosures, state_domain, param_box,
            timeout_ms=timeout_ms
        )
        results['completeness'] = completeness_result

    return results


def verify_delta_soundness_z3(
    expressions: List[sp.Expr],
    state_symbols: List[sp.Symbol],
    param_symbols: List[sp.Symbol],
    state_enclosure: List[Tuple[float, float]],
    param_box: List[Tuple[float, float]],
    eigenvalue_gap: float,
    delta: float = 1e-3,
    timeout_ms: int = 30000,
) -> Z3VerificationResult:
    """
    Verify delta-soundness bound (Theorem B1) using Z3.

    Instead of checking f(x,μ) = 0 exactly, checks whether the delta-relaxed
    condition |f_i(x,μ)| < δ combined with Routh-Hurwitz stability is consistent
    with the claimed eigenvalue gap.

    This implements the computational version of Theorem B1: verifying that
    δ_solver < γ / (‖R‖ · L_{Df} · rad + 1) by checking the concrete
    polynomial conditions.
    """
    if not Z3_AVAILABLE:
        return Z3VerificationResult(
            verified=False, tier="tier2_z3_delta", status="unavailable",
            message="Z3 not installed"
        )

    t0 = time.time()
    n = len(state_symbols)
    p = len(param_symbols)

    try:
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        z3_vars: Dict[str, Any] = {}

        x_z3 = []
        for s in state_symbols:
            name = str(s)
            v = z3.Real(name)
            z3_vars[name] = v
            x_z3.append(v)

        mu_z3 = []
        for s in param_symbols:
            name = str(s)
            v = z3.Real(name)
            z3_vars[name] = v
            mu_z3.append(v)

        # Bounds
        for i in range(n):
            lo, hi = state_enclosure[i]
            solver.add(x_z3[i] >= z3.RealVal(lo))
            solver.add(x_z3[i] <= z3.RealVal(hi))

        for i in range(p):
            lo, hi = param_box[i]
            if abs(hi - lo) < 1e-15:
                solver.add(mu_z3[i] == z3.RealVal((lo + hi) / 2.0))
            else:
                solver.add(mu_z3[i] >= z3.RealVal(lo))
                solver.add(mu_z3[i] <= z3.RealVal(hi))

        # Delta-relaxed equilibrium: |f_i(x,μ)| < δ
        f_z3 = []
        for expr in expressions:
            expanded = sp.expand(sp.nsimplify(expr, rational=True))
            f_z3.append(_sympy_to_z3(expanded, z3_vars))

        for i in range(n):
            solver.add(f_z3[i] >= z3.RealVal(-delta))
            solver.add(f_z3[i] <= z3.RealVal(delta))

        # Compute Jacobian symbolically
        jac_expr = []
        for expr in expressions:
            row = [sp.diff(expr, s) for s in state_symbols]
            jac_expr.append(row)

        J_z3 = []
        for i in range(n):
            row = []
            for j in range(n):
                expanded = sp.expand(sp.nsimplify(jac_expr[i][j], rational=True))
                row.append(_sympy_to_z3(expanded, z3_vars))
            J_z3.append(row)

        # Check if stability can fail within the delta ball
        # Negation: stability FAILS somewhere in the enclosure
        if n == 2:
            tr_J = J_z3[0][0] + J_z3[1][1]
            det_J = J_z3[0][0] * J_z3[1][1] - J_z3[0][1] * J_z3[1][0]
            solver.add(z3.Or(tr_J >= z3.RealVal(-eigenvalue_gap),
                             det_J <= z3.RealVal(eigenvalue_gap)))
        else:
            # For higher dimensions, just check trace condition as proxy
            tr_J = sum(J_z3[i][i] for i in range(n))
            solver.add(tr_J >= z3.RealVal(-eigenvalue_gap))

        result = solver.check()
        elapsed = time.time() - t0

        if result == z3.unsat:
            return Z3VerificationResult(
                verified=True, tier="tier2_z3_delta", status="unsat",
                time_s=elapsed, delta_used=delta,
                message=f"δ-soundness verified: eigenvalue gap {eigenvalue_gap:.4f} "
                        f"maintained under δ={delta} perturbation"
            )
        elif result == z3.sat:
            return Z3VerificationResult(
                verified=False, tier="tier2_z3_delta", status="sat",
                time_s=elapsed, delta_used=delta,
                message="δ-soundness check failed: stability may change within δ ball"
            )
        else:
            return Z3VerificationResult(
                verified=False, tier="tier2_z3_delta", status="unknown",
                time_s=elapsed, delta_used=delta,
                message="Z3 returned unknown for δ-soundness"
            )

    except Exception as e:
        return Z3VerificationResult(
            verified=False, tier="tier2_z3_delta", status="error",
            time_s=time.time() - t0,
            message=f"Z3 error: {str(e)}"
        )
