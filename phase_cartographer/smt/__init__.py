"""
SMT certificate generation and δ-bound computation.

Provides:
- δ-bound computation for soundness of δ-complete SMT encoding
- SMT-LIB2 formula generation for regime claims
- Z3-based exact verification for polynomial models (Tier 3)
- Z3-based δ-approximate verification (Tier 2 fallback)
- Interface to dReal and Z3 solvers
"""

from .delta_bound import (
    DeltaBound,
    DeltaBoundResult,
    compute_required_delta,
    compute_eigenvalue_gap,
    soundness_margin,
)

try:
    from .z3_verifier import (
        verify_equilibrium_z3,
        verify_stability_z3,
        verify_no_other_equilibria_z3,
        verify_cell_z3,
        verify_delta_soundness_z3,
        Z3VerificationResult,
        Z3_AVAILABLE,
    )
except ImportError:
    Z3_AVAILABLE = False

__all__ = [
    "DeltaBound",
    "DeltaBoundResult",
    "compute_required_delta",
    "compute_eigenvalue_gap",
    "soundness_margin",
    "Z3_AVAILABLE",
]
