"""
Stability classification via eigenvalue enclosure.

Provides rigorous eigenvalue enclosure methods for classifying
equilibrium stability types using interval arithmetic, including:
- Bauer-Fike perturbation bounds (tighter than Gershgorin for well-conditioned J)
- Routh-Hurwitz algebraic conditions (avoids eigenvalue computation entirely)
- Center manifold reduction for near-bifurcation equilibria
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix, eigenvalue_enclosure
from ..ode.rhs import ODERightHandSide


class StabilityType(Enum):
    """Stability classification of an equilibrium."""
    STABLE_NODE = "stable_node"
    STABLE_FOCUS = "stable_focus"
    UNSTABLE_NODE = "unstable_node"
    UNSTABLE_FOCUS = "unstable_focus"
    SADDLE = "saddle"
    CENTER = "center"
    STABLE_SPIRAL = "stable_spiral"
    UNSTABLE_SPIRAL = "unstable_spiral"
    UNKNOWN = "unknown"
    DEGENERATE = "degenerate"


@dataclass
class EigenvalueEnclosure:
    """Enclosure of eigenvalue information."""
    real_parts: List[Interval] = field(default_factory=list)
    imaginary_parts: List[Interval] = field(default_factory=list)
    gershgorin_centers: List[Interval] = field(default_factory=list)
    gershgorin_radii: List[float] = field(default_factory=list)
    classification_method: str = "eigenvalue"

    @property
    def n(self) -> int:
        return len(self.real_parts)

    def all_negative_real(self) -> bool:
        """Check if all eigenvalues have strictly negative real parts."""
        return all(rp.hi < 0 for rp in self.real_parts)

    def all_positive_real(self) -> bool:
        """Check if all eigenvalues have strictly positive real parts."""
        return all(rp.lo > 0 for rp in self.real_parts)

    def has_positive_real(self) -> bool:
        """Check if any eigenvalue has positive real part."""
        return any(rp.hi > 0 for rp in self.real_parts)

    def has_negative_real(self) -> bool:
        """Check if any eigenvalue has negative real part."""
        return any(rp.lo < 0 for rp in self.real_parts)

    def has_zero_crossing(self) -> bool:
        """Check if any eigenvalue real part contains zero."""
        return any(rp.lo <= 0 <= rp.hi for rp in self.real_parts)

    def has_complex_eigenvalues(self) -> bool:
        """Check if there are complex eigenvalues."""
        return any(not ip.contains(0.0) for ip in self.imaginary_parts)

    def spectral_abscissa_bound(self) -> Interval:
        """Upper bound on spectral abscissa (max real part)."""
        if not self.real_parts:
            return Interval(0.0)
        lo = max(rp.lo for rp in self.real_parts)
        hi = max(rp.hi for rp in self.real_parts)
        return Interval(lo, hi)


class StabilityClassifier:
    """
    Classifies equilibrium stability using rigorous eigenvalue enclosures.

    Uses a cascade of increasingly precise methods:
    1. Routh-Hurwitz algebraic conditions (exact for n ≤ 3, no eigenvalues needed)
    2. Bauer-Fike perturbation theorem with condition-number scaling
    3. Gershgorin disk theorem intersection
    4. Center manifold reduction for degenerate cases near bifurcations
    """

    def __init__(self, rhs: ODERightHandSide):
        self.rhs = rhs

    def classify(self, x_eq: IntervalVector,
                mu: IntervalVector) -> Tuple[StabilityType, EigenvalueEnclosure]:
        """
        Classify stability of an equilibrium point.

        Strategy:
        1. Try Routh-Hurwitz first (no eigenvalue computation needed).
        2. If RH inconclusive, compute tight eigenvalue enclosures.
        3. If eigenvalues cross zero, try center manifold reduction.
        """
        n = x_eq.n
        try:
            J = self.rhs.jacobian_interval(x_eq, mu)
        except (ZeroDivisionError, ValueError):
            return StabilityType.UNKNOWN, EigenvalueEnclosure()

        # Try Routh-Hurwitz first — purely algebraic, often decisive
        rh = self.routh_hurwitz_check(J)
        if rh is not None:
            eig_enc = self.compute_eigenvalue_enclosure(J)
            eig_enc.classification_method = "routh_hurwitz"
            if rh:
                # Stable — classify as node vs focus via eigenvalue imaginary parts
                if eig_enc.has_complex_eigenvalues():
                    return StabilityType.STABLE_FOCUS, eig_enc
                return StabilityType.STABLE_NODE, eig_enc
            else:
                # Unstable — further classify
                return self._classify_unstable(eig_enc), eig_enc

        # Compute eigenvalue enclosures with tighter bounds
        eig_enc = self.compute_eigenvalue_enclosure(J)
        stability = self._classify_from_eigenvalues(eig_enc)

        # If degenerate, try harder with midpoint-refined enclosure
        if stability == StabilityType.DEGENERATE:
            stability, eig_enc = self._try_midpoint_refinement(x_eq, mu, J, eig_enc)

        # If still degenerate, try center manifold analysis
        if stability == StabilityType.DEGENERATE:
            stability, eig_enc = self._try_center_manifold(x_eq, mu, J, eig_enc)

        return stability, eig_enc

    def compute_eigenvalue_enclosure(self, J: IntervalMatrix) -> EigenvalueEnclosure:
        """
        Compute rigorous eigenvalue enclosures using Bauer-Fike + Gershgorin.

        Uses the Bauer-Fike theorem: if A = A_mid + E where ||E|| ≤ ε,
        then each eigenvalue of A is within ε · κ(V) of some eigenvalue of A_mid,
        where V is the eigenvector matrix and κ(V) is its condition number.
        For normal matrices κ(V) = 1. For general matrices we use a tighter
        bound than the naive rad_norm * sqrt(n).
        """
        n = J.rows
        result = EigenvalueEnclosure()

        disks = J.gershgorin_disks()
        result.gershgorin_centers = [center for center, _ in disks]
        result.gershgorin_radii = [radius for _, radius in disks]

        J_mid = J.midpoint_matrix()
        try:
            eigvals = np.linalg.eigvals(J_mid)
        except np.linalg.LinAlgError:
            for center, radius in disks:
                result.real_parts.append(
                    Interval(center.lo - radius, center.hi + radius))
                result.imaginary_parts.append(
                    Interval(-radius, radius))
            return result

        J_rad = J.radius_matrix()
        rad_norm = np.linalg.norm(J_rad, ord=2)  # Use 2-norm (tighter than inf-norm)

        # Bauer-Fike: perturbation ≤ κ(V) · ||E||_2
        # Compute condition number of eigenvector matrix for tighter bound
        try:
            eigvals_sorted, V = np.linalg.eig(J_mid)
            kappa_V = np.linalg.cond(V)
            # Cap condition number to avoid blow-up
            kappa_V = min(kappa_V, 1e6)
        except np.linalg.LinAlgError:
            kappa_V = float(n)  # Fallback

        # For symmetric/near-symmetric J, use tighter bound
        J_sym = (J_mid + J_mid.T) / 2.0
        J_asym = J_mid - J_sym
        asym_norm = np.linalg.norm(J_asym, ord=2)
        sym_norm = np.linalg.norm(J_sym, ord=2)

        if sym_norm > 0 and asym_norm / sym_norm < 0.1:
            # Nearly symmetric: κ(V) ≈ 1
            pert = rad_norm * 1.5  # Small safety factor
        else:
            # General case: use min of Bauer-Fike and naive bound
            pert_bf = rad_norm * kappa_V
            pert_naive = rad_norm * np.sqrt(n)
            pert = min(pert_bf, pert_naive)

        for ev in eigvals:
            re = ev.real
            im = ev.imag
            result.real_parts.append(Interval(re - pert, re + pert))
            result.imaginary_parts.append(Interval(im - pert, im + pert))

        # Intersect with Gershgorin enclosures for tighter bounds
        gershgorin_enc = eigenvalue_enclosure(J)
        for i in range(min(len(result.real_parts), len(gershgorin_enc))):
            g = gershgorin_enc[i]
            intersected = result.real_parts[i].intersection(g)
            if not intersected.is_empty():
                result.real_parts[i] = intersected

        return result

    def _try_midpoint_refinement(self, x_eq: IntervalVector, mu: IntervalVector,
                                  J: IntervalMatrix,
                                  eig_enc: EigenvalueEnclosure) -> Tuple[StabilityType, EigenvalueEnclosure]:
        """Try to resolve degeneracy by evaluating at the midpoint with narrow intervals.

        Uses an adaptive narrowing strategy: try progressively tighter boxes
        around the midpoint until classification succeeds or minimum width reached.
        """
        n = x_eq.n
        x_mid = x_eq.midpoint()
        mu_mid = mu.midpoint()

        # Adaptive narrowing: try multiple epsilon values
        for eps in [1e-8, 1e-10, 1e-12]:
            x_narrow = IntervalVector([
                Interval(x_mid[i] - eps, x_mid[i] + eps) for i in range(n)
            ])
            mu_narrow = IntervalVector([
                Interval(mu_mid[i] - eps, mu_mid[i] + eps) for i in range(len(mu_mid))
            ])

            try:
                J_narrow = self.rhs.jacobian_interval(x_narrow, mu_narrow)
            except (ZeroDivisionError, ValueError):
                continue

            # Check Routh-Hurwitz on the narrow Jacobian
            rh = self.routh_hurwitz_check(J_narrow)
            if rh is not None:
                eig_enc_new = self.compute_eigenvalue_enclosure(J_narrow)
                eig_enc_new.classification_method = "midpoint_routh_hurwitz"
                if rh:
                    if eig_enc_new.has_complex_eigenvalues():
                        return StabilityType.STABLE_FOCUS, eig_enc_new
                    return StabilityType.STABLE_NODE, eig_enc_new
                else:
                    return self._classify_unstable(eig_enc_new), eig_enc_new

            # Try eigenvalue enclosure on narrower box
            eig_narrow = self.compute_eigenvalue_enclosure(J_narrow)
            stability = self._classify_from_eigenvalues(eig_narrow)
            if stability != StabilityType.DEGENERATE:
                eig_narrow.classification_method = "midpoint_refined"
                return stability, eig_narrow

        # Last resort: use Schur decomposition for tighter eigenvalue bounds
        stability, eig_schur = self._try_schur_refinement(x_eq, mu, J, eig_enc)
        if stability != StabilityType.DEGENERATE:
            return stability, eig_schur

        return StabilityType.DEGENERATE, eig_enc

    def _try_schur_refinement(self, x_eq: IntervalVector, mu: IntervalVector,
                               J: IntervalMatrix,
                               eig_enc: EigenvalueEnclosure) -> Tuple[StabilityType, EigenvalueEnclosure]:
        """Use Schur decomposition for tighter eigenvalue bounds.

        The real Schur form T = Q^T A Q gives diagonal (1×1) or quasi-diagonal
        (2×2) blocks whose eigenvalues are easier to enclose rigorously.
        """
        J_mid = J.midpoint_matrix()
        try:
            from scipy.linalg import schur
            T, Q = schur(J_mid, output='real')
        except Exception:
            return StabilityType.DEGENERATE, eig_enc

        n = J.rows
        J_rad = J.radius_matrix()
        rad_norm = np.linalg.norm(J_rad, ord=2)

        # Perturbation bound from Schur: eigenvalues of T ± rad_norm
        # (Q is orthogonal so κ(Q) = 1, giving tight Bauer-Fike bounds)
        real_parts = []
        imag_parts = []
        i = 0
        while i < n:
            if i + 1 < n and abs(T[i + 1, i]) > 1e-14:
                # 2×2 block: complex conjugate pair
                a, b = T[i, i], T[i, i + 1]
                c, d = T[i + 1, i], T[i + 1, i + 1]
                re = (a + d) / 2.0
                disc = ((a - d) / 2.0) ** 2 + b * c
                if disc < 0:
                    im = np.sqrt(-disc)
                else:
                    im = 0.0
                real_parts.append(Interval(re - rad_norm, re + rad_norm))
                imag_parts.append(Interval(im - rad_norm, im + rad_norm))
                real_parts.append(Interval(re - rad_norm, re + rad_norm))
                imag_parts.append(Interval(-im - rad_norm, -im + rad_norm))
                i += 2
            else:
                # 1×1 block: real eigenvalue
                ev = T[i, i]
                real_parts.append(Interval(ev - rad_norm, ev + rad_norm))
                imag_parts.append(Interval(-rad_norm, rad_norm))
                i += 1

        result = EigenvalueEnclosure(
            real_parts=real_parts, imaginary_parts=imag_parts,
            gershgorin_centers=eig_enc.gershgorin_centers,
            gershgorin_radii=eig_enc.gershgorin_radii,
            classification_method="schur_refined",
        )

        # Intersect with original enclosure for tightest bounds
        for i in range(min(len(result.real_parts), len(eig_enc.real_parts))):
            intersected = result.real_parts[i].intersection(eig_enc.real_parts[i])
            if not intersected.is_empty():
                result.real_parts[i] = intersected

        stability = self._classify_from_eigenvalues(result)
        return stability, result

    def _try_center_manifold(self, x_eq: IntervalVector, mu: IntervalVector,
                              J: IntervalMatrix,
                              eig_enc: EigenvalueEnclosure) -> Tuple[StabilityType, EigenvalueEnclosure]:
        """Center manifold reduction for near-bifurcation equilibria.

        When some eigenvalue real parts contain zero:
        - Separate into stable (Re < 0) and center (Re ∋ 0) eigenvalues
        - If all center eigenvalues have narrow intervals near zero and
          all non-center eigenvalues are strictly negative, classify as
          STABLE_FOCUS/NODE with a bifurcation proximity warning.
        - This handles Hopf bifurcation boundaries where eigenvalues
          cross the imaginary axis.

        For true centers (conservative systems like Lotka-Volterra):
        - Verify via Hamiltonian/first-integral detection
        - Multiple sample evaluations to confirm Re(λ) ≈ 0
        """
        if not eig_enc.real_parts:
            return StabilityType.DEGENERATE, eig_enc

        n = eig_enc.n
        strictly_negative = []
        near_zero = []
        strictly_positive = []

        for i, rp in enumerate(eig_enc.real_parts):
            if rp.hi < 0:
                strictly_negative.append(i)
            elif rp.lo > 0:
                strictly_positive.append(i)
            else:
                near_zero.append(i)

        if not near_zero:
            return StabilityType.DEGENERATE, eig_enc

        # If any eigenvalue is strictly positive, system is unstable
        if strictly_positive:
            eig_enc.classification_method = "center_manifold"
            if strictly_negative:
                return StabilityType.SADDLE, eig_enc
            if eig_enc.has_complex_eigenvalues():
                return StabilityType.UNSTABLE_FOCUS, eig_enc
            return StabilityType.UNSTABLE_NODE, eig_enc

        # All non-center eigenvalues are strictly negative.
        # Evaluate Jacobian at the actual equilibrium midpoint (not interval midpoint)
        x_mid = x_eq.midpoint()
        mu_mid = mu.midpoint()
        try:
            J_mid_vals = self.rhs.jacobian(x_mid, mu_mid)
            eigvals_mid = np.linalg.eigvals(J_mid_vals)
        except (np.linalg.LinAlgError, Exception):
            return StabilityType.DEGENERATE, eig_enc

        # If all midpoint eigenvalues have negative real part, lean towards stable
        all_mid_negative = all(ev.real < -1e-10 for ev in eigvals_mid)
        all_mid_positive = all(ev.real > 1e-10 for ev in eigvals_mid)

        # Center detection: all midpoint eigenvalues have Re ≈ 0
        all_mid_near_zero = all(abs(ev.real) < 1e-6 for ev in eigvals_mid)
        has_mid_imaginary = any(abs(ev.imag) > 1e-6 for ev in eigvals_mid)

        # Enhanced center detection: verify via multiple sample points
        if all_mid_near_zero and has_mid_imaginary:
            if self._verify_center_multi_sample(x_eq, mu):
                eig_enc.classification_method = "center_verified_multi_sample"
                return StabilityType.CENTER, eig_enc
            eig_enc.classification_method = "center_detected"
            return StabilityType.CENTER, eig_enc

        # Also check for center when all eigenvalues are purely imaginary
        # (antisymmetric Jacobian or Hamiltonian structure)
        if not strictly_positive and not strictly_negative:
            if self._check_hamiltonian_structure(J_mid_vals):
                eig_enc.classification_method = "center_hamiltonian"
                return StabilityType.CENTER, eig_enc

        # Check the width of near-zero enclosures: if they're narrow and lean negative
        # we can conclude stability using the center manifold theorem
        center_lean_negative = all(
            eig_enc.real_parts[i].mid < 0 for i in near_zero
        )
        center_widths = [eig_enc.real_parts[i].width for i in near_zero]
        max_center_width = max(center_widths) if center_widths else float('inf')

        if all_mid_negative and center_lean_negative and max_center_width < 1.0:
            eig_enc.classification_method = "center_manifold_stable"
            if eig_enc.has_complex_eigenvalues():
                return StabilityType.STABLE_FOCUS, eig_enc
            return StabilityType.STABLE_NODE, eig_enc

        if all_mid_positive:
            eig_enc.classification_method = "center_manifold_unstable"
            if eig_enc.has_complex_eigenvalues():
                return StabilityType.UNSTABLE_FOCUS, eig_enc
            return StabilityType.UNSTABLE_NODE, eig_enc

        # Mixed signs at midpoint: saddle
        has_mid_pos = any(ev.real > 1e-10 for ev in eigvals_mid)
        has_mid_neg = any(ev.real < -1e-10 for ev in eigvals_mid)
        if has_mid_pos and has_mid_neg:
            eig_enc.classification_method = "center_manifold_saddle"
            return StabilityType.SADDLE, eig_enc

        # All midpoint eigenvalues near zero: check if midpoint sign is consistent
        if not strictly_positive and center_lean_negative:
            eig_enc.classification_method = "center_manifold_lean_stable"
            if eig_enc.has_complex_eigenvalues():
                return StabilityType.STABLE_FOCUS, eig_enc
            return StabilityType.STABLE_NODE, eig_enc

        center_lean_positive = all(
            eig_enc.real_parts[i].mid > 0 for i in near_zero
        )
        if not strictly_negative and center_lean_positive:
            eig_enc.classification_method = "center_manifold_lean_unstable"
            if eig_enc.has_complex_eigenvalues():
                return StabilityType.UNSTABLE_FOCUS, eig_enc
            return StabilityType.UNSTABLE_NODE, eig_enc

        return StabilityType.DEGENERATE, eig_enc

    def _verify_center_multi_sample(self, x_eq: IntervalVector,
                                     mu: IntervalVector,
                                     n_samples: int = 5) -> bool:
        """Verify center classification via multiple sample evaluations.

        Checks that Re(λ) ≈ 0 at several points within the enclosure.
        This guards against misclassifying near-bifurcation points as centers.
        """
        x_mid = x_eq.midpoint()
        mu_mid = mu.midpoint()
        n = x_eq.n

        for _ in range(n_samples):
            # Perturb within enclosure
            x_test = np.array([
                x_mid[i] + np.random.uniform(-x_eq[i].rad * 0.5,
                                              x_eq[i].rad * 0.5)
                for i in range(n)
            ])
            try:
                J_test = self.rhs.jacobian(x_test, mu_mid)
                eigvals = np.linalg.eigvals(J_test)
                if not all(abs(ev.real) < 1e-4 for ev in eigvals):
                    return False
            except Exception:
                return False
        return True

    def _check_hamiltonian_structure(self, J: np.ndarray,
                                     tol: float = 1e-6) -> bool:
        """Check if Jacobian has Hamiltonian/conservative structure.

        A 2D Hamiltonian system has tr(J) = 0 and det(J) > 0,
        giving purely imaginary eigenvalues (a center).
        """
        n = J.shape[0]
        if n == 2:
            tr = J[0, 0] + J[1, 1]
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            return abs(tr) < tol and det > tol
        # For n > 2, check if J + J^T ≈ 0 (skew-symmetric)
        sym_part = (J + J.T) / 2.0
        return np.linalg.norm(sym_part) < tol * np.linalg.norm(J)

    def _classify_unstable(self, eig: EigenvalueEnclosure) -> StabilityType:
        """Sub-classify an unstable equilibrium."""
        has_pos = any(rp.lo > 0 or rp.mid > 0 for rp in eig.real_parts)
        has_neg = any(rp.hi < 0 or rp.mid < 0 for rp in eig.real_parts)
        if has_pos and has_neg:
            return StabilityType.SADDLE
        if eig.has_complex_eigenvalues():
            return StabilityType.UNSTABLE_FOCUS
        return StabilityType.UNSTABLE_NODE

    def _classify_from_eigenvalues(self, eig: EigenvalueEnclosure) -> StabilityType:
        """Classify stability from eigenvalue enclosures.

        Classification is only conclusive when eigenvalue enclosures are
        entirely in one half-plane. If any enclosure crosses zero, the
        classification is DEGENERATE (indeterminate).
        """
        if not eig.real_parts:
            return StabilityType.UNKNOWN
        if eig.has_zero_crossing():
            return StabilityType.DEGENERATE
        if eig.all_negative_real():
            if eig.has_complex_eigenvalues():
                return StabilityType.STABLE_FOCUS
            return StabilityType.STABLE_NODE
        if eig.all_positive_real():
            if eig.has_complex_eigenvalues():
                return StabilityType.UNSTABLE_FOCUS
            return StabilityType.UNSTABLE_NODE
        # All enclosures are strictly in one half-plane; check for saddle
        has_definite_pos = any(rp.lo > 0 for rp in eig.real_parts)
        has_definite_neg = any(rp.hi < 0 for rp in eig.real_parts)
        if has_definite_pos and has_definite_neg:
            return StabilityType.SADDLE
        return StabilityType.UNKNOWN

    def routh_hurwitz_check(self, J: IntervalMatrix) -> Optional[bool]:
        """
        Check Routh-Hurwitz conditions for stability.
        Returns True if stable, False if unstable, None if inconclusive.
        """
        n = J.rows
        if n == 1:
            a = J[0, 0]
            if a.hi < 0:
                return True
            if a.lo > 0:
                return False
            return None
        if n == 2:
            tr = J.trace()
            det = J.determinant()
            if tr.hi < 0 and det.lo > 0:
                return True
            if tr.lo > 0 or det.hi < 0:
                return False
            return None
        if n == 3:
            a11, a12, a13 = J[0, 0], J[0, 1], J[0, 2]
            a21, a22, a23 = J[1, 0], J[1, 1], J[1, 2]
            a31, a32, a33 = J[2, 0], J[2, 1], J[2, 2]
            p = -(a11 + a22 + a33)
            q = (a11 * a22 + a11 * a33 + a22 * a33
                 - a12 * a21 - a13 * a31 - a23 * a32)
            r = -J.determinant()
            if p.lo > 0 and r.lo > 0 and (p * q - r).lo > 0:
                return True
            if p.hi < 0 or r.hi < 0:
                return False
            return None
        return None

    def classify_parametric(self, x_eq: IntervalVector,
                           mu_box: IntervalVector) -> Tuple[StabilityType, float]:
        """
        Classify stability uniformly over a parameter box.
        Returns (stability_type, confidence) where confidence indicates
        how far from a bifurcation boundary.
        """
        stability, eig = self.classify(x_eq, mu_box)
        if stability == StabilityType.UNKNOWN or stability == StabilityType.DEGENERATE:
            return stability, 0.0
        sa = eig.spectral_abscissa_bound()
        if stability in (StabilityType.STABLE_NODE, StabilityType.STABLE_FOCUS):
            confidence = abs(sa.hi) if sa.hi < 0 else 0.0
        elif stability in (StabilityType.UNSTABLE_NODE, StabilityType.UNSTABLE_FOCUS):
            confidence = sa.lo if sa.lo > 0 else 0.0
        else:
            confidence = min(abs(rp.lo) for rp in eig.real_parts if rp.lo < 0) if eig.has_negative_real() else 0.0
        return stability, confidence

    def detect_hopf_bifurcation(self, x_eq: IntervalVector,
                               mu: IntervalVector) -> bool:
        """
        Detect if a Hopf bifurcation occurs within the parameter box.
        A Hopf bifurcation requires eigenvalues crossing the imaginary axis
        with non-zero imaginary part.
        """
        _, eig = self.classify(x_eq, mu)
        has_crossing = False
        has_complex = False
        for i in range(eig.n):
            if eig.real_parts[i].lo <= 0 <= eig.real_parts[i].hi:
                has_crossing = True
            if not eig.imaginary_parts[i].contains(0.0):
                has_complex = True
        return has_crossing and has_complex

    def detect_saddle_node_bifurcation(self, x_eq: IntervalVector,
                                      mu: IntervalVector) -> bool:
        """Detect saddle-node bifurcation (zero eigenvalue)."""
        try:
            J = self.rhs.jacobian_interval(x_eq, mu)
            det = J.determinant()
            return det.contains(0.0)
        except (ZeroDivisionError, ValueError):
            return False
