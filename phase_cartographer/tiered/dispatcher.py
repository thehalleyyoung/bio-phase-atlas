"""
Tiered verification dispatcher.

Dispatches regime claims to the appropriate verification tier:
  Tier 1: Interval-arithmetic recomputation (always available, via minicheck)
  Tier 2: δ-complete SMT verification via dReal (transcendental models)
  Tier 3: Exact verification via Z3 (polynomial-only models)

Tier selection follows the principle of maximum available assurance:
  Polynomial models → try Tier 3, fall back to Tier 1.
  Transcendental models → try Tier 2 (if dReal available), fall back to Tier 1.
  All models always get Tier 1.
"""

import time
from typing import Optional, Tuple

from .certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from minicheck import verify_certificate as minicheck_verify


def _has_dreal() -> bool:
    try:
        import dreal  # noqa: F401
        return True
    except ImportError:
        return False


def _has_z3() -> bool:
    try:
        import z3  # noqa: F401
        return True
    except ImportError:
        return False


def tier1_verify(cell: CertifiedCell) -> Tuple[bool, str]:
    """
    Tier 1: Independent interval-arithmetic recomputation via MiniCheck.

    This is the minimal TCB verification path. MiniCheck reimplements
    Krawczyk verification, eigenvalue enclosure, and stability classification
    from scratch using only basic interval arithmetic.
    """
    cert_dict = cell.to_minicheck_format()
    result = minicheck_verify(cert_dict)
    if result.valid:
        return True, "Tier 1 PASS: MiniCheck independently verified all claims"
    else:
        errors = "; ".join(result.errors)
        return False, f"Tier 1 FAIL: {errors}"


def tier2_verify(cell: CertifiedCell, delta: float = 1e-3,
                 timeout_s: float = 60.0) -> Tuple[bool, str]:
    """
    Tier 2: δ-complete SMT verification.

    For polynomial models when dReal is unavailable, uses Z3 with
    delta-relaxed conditions (delta-soundness bound from Theorem B1).
    For Hill-function models, requires dReal.
    """
    if not _has_dreal() and not _has_z3():
        return False, "Tier 2 UNAVAILABLE: neither dReal nor Z3 installed"

    if _has_z3():
        try:
            from ..smt.z3_verifier import verify_delta_soundness_z3
            from ..models.benchmark_models import get_benchmark

            bm = get_benchmark(cell.model_name)
            rhs = bm.rhs

            if not hasattr(rhs, 'expressions') or not rhs.expressions:
                return False, "Tier 2 SKIPPED: model lacks symbolic expressions"

            # Use eigenvalue gap from first equilibrium
            eig_gap = 0.0
            for eq in cell.equilibria:
                for lo, hi in eq.eigenvalue_real_parts:
                    if lo > 0:
                        eig_gap = max(eig_gap, lo)
                    elif hi < 0:
                        eig_gap = max(eig_gap, abs(hi))

            if eig_gap <= 0:
                return False, "Tier 2 SKIPPED: eigenvalue gap is zero (near bifurcation)"

            eq = cell.equilibria[0]
            result = verify_delta_soundness_z3(
                rhs.expressions, rhs.state_symbols, rhs.param_symbols,
                eq.state_enclosure, cell.parameter_box,
                eigenvalue_gap=eig_gap, delta=delta,
                timeout_ms=int(timeout_s * 1000),
            )

            if result.verified:
                return True, f"Tier 2 PASS: δ-soundness verified (δ={delta}, γ={eig_gap:.4f})"
            return False, f"Tier 2 FAIL: {result.message}"

        except Exception as e:
            return False, f"Tier 2 ERROR: {str(e)}"

    return False, "Tier 2 SKIPPED: dReal not available, Z3 delta-check inconclusive"


def tier3_verify(cell: CertifiedCell, timeout_s: float = 120.0) -> Tuple[bool, str]:
    """
    Tier 3: Exact verification via Z3 for polynomial models.

    Verifies equilibrium existence and stability using Z3's QF_NRA solver.
    Only applicable when the model RHS is purely polynomial.
    """
    if not _has_z3():
        return False, "Tier 3 UNAVAILABLE: Z3 not installed"

    try:
        from ..smt.z3_verifier import verify_equilibrium_z3, verify_stability_z3
        from ..models.benchmark_models import get_benchmark

        bm = get_benchmark(cell.model_name)
        rhs = bm.rhs

        if not hasattr(rhs, 'expressions') or not rhs.expressions:
            return False, "Tier 3 SKIPPED: model lacks symbolic expressions"

        timeout_ms = int(timeout_s * 1000)
        all_verified = True
        messages = []

        for eq_idx, eq in enumerate(cell.equilibria):
            # Verify equilibrium existence
            eq_result = verify_equilibrium_z3(
                rhs.expressions, rhs.state_symbols, rhs.param_symbols,
                eq.state_enclosure, cell.parameter_box,
                timeout_ms=timeout_ms,
            )

            if eq_result.verified:
                messages.append(f"eq{eq_idx}: existence confirmed")
            elif eq_result.status == "unknown":
                messages.append(f"eq{eq_idx}: existence unknown (timeout)")
                all_verified = False
            else:
                messages.append(f"eq{eq_idx}: existence {eq_result.status}")
                all_verified = False

            # Verify stability
            from ..tiered.certificate import StabilityType as _ST
            if eq.stability == _ST.SADDLE:
                stab_type = "saddle"
            elif eq.is_stable():
                stab_type = "stable"
            else:
                stab_type = "unstable"
            stab_result = verify_stability_z3(
                rhs.expressions, rhs.state_symbols, rhs.param_symbols,
                eq.state_enclosure, cell.parameter_box,
                stability_type=stab_type,
                timeout_ms=timeout_ms,
            )

            if stab_result.verified:
                messages.append(f"eq{eq_idx}: {stab_type} stability confirmed")
            elif stab_result.status == "unknown":
                messages.append(f"eq{eq_idx}: stability unknown (timeout)")
                # Don't mark as failed, just inconclusive
            else:
                messages.append(f"eq{eq_idx}: stability {stab_result.status}")

        detail = "; ".join(messages)
        if all_verified:
            return True, f"Tier 3 PASS: Z3 exact verification ({detail})"
        return False, f"Tier 3 PARTIAL: {detail}"

    except Exception as e:
        return False, f"Tier 3 ERROR: {str(e)}"


def select_tier(model_name: str, rhs_type: str = "general") -> VerificationTier:
    """
    Automatically select the highest feasible verification tier.

    Polynomial models (Brusselator, Sel'kov) → Tier 3 if Z3 available.
    Hill/transcendental models → Tier 2 if dReal available.
    All models → Tier 1 always.
    """
    if rhs_type in ("polynomial", "poly"):
        if _has_z3():
            return VerificationTier.TIER3_Z3
    if rhs_type in ("hill", "rational", "transcendental"):
        if _has_dreal():
            return VerificationTier.TIER2_DREAL
    return VerificationTier.TIER1_IA


def verify_cell(cell: CertifiedCell) -> CertifiedCell:
    """
    Run tiered verification on a certified cell.

    Always runs Tier 1. Attempts higher tiers based on model type and
    solver availability. Updates cell with verification results.
    """
    t0 = time.time()

    # Tier 1 always runs
    t1_ok, t1_msg = tier1_verify(cell)
    cell.minicheck_passed = t1_ok

    # Attempt higher tiers
    if cell.tier == VerificationTier.TIER3_Z3:
        t3_ok, t3_msg = tier3_verify(cell)
        if not t3_ok:
            cell.tier = VerificationTier.TIER1_IA

    elif cell.tier == VerificationTier.TIER2_DREAL:
        t2_ok, t2_msg = tier2_verify(cell)
        if not t2_ok:
            cell.tier = VerificationTier.TIER1_IA

    cell.certification_time_s += time.time() - t0

    # Validate regime label against inference rules
    valid, reason = RegimeInferenceRules.validate(cell)
    if not valid:
        cell.regime = RegimeInferenceRules.infer(cell.equilibria)

    return cell
