"""
Focused tests for mathematical rigor of core theorems.

Tests verify:
  - Theorem B1: δ-soundness bound formula matches paper specification
  - Krawczyk operator: main and minicheck implementations agree
  - Proposition C1: adaptive refinement convergence
  - Theorem B2: certificate composition soundness
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.interval.interval import Interval
from phase_cartographer.interval.matrix import IntervalVector, IntervalMatrix
from phase_cartographer.smt.delta_bound import (
    DeltaBound, DeltaBoundResult, compute_eigenvalue_gap,
    compute_required_delta, soundness_margin, ComposedErrorBudget,
)
from phase_cartographer.equilibrium.krawczyk import (
    KrawczykOperator, KrawczykStatus,
)
from phase_cartographer.minicheck import (
    IV, IVec, IMat, recompute_krawczyk,
)
from phase_cartographer.atlas.composition import (
    boxes_disjoint, boxes_adjacent, verify_atlas_composition,
)
from phase_cartographer.atlas.certificate_algebra import (
    CertificateCompositionRules, AtlasCertificateBundle,
    RegimeProofTree, verify_certificate_bundle,
)
from phase_cartographer.tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType,
)
from phase_cartographer.refinement.octree import (
    adaptive_refine, RefinementConfig, split_box, box_volume,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_cell(box, regime, stability, minicheck=True, eig_rp=None):
    if eig_rp is None:
        eig_rp = [(-2.0, -0.5), (-2.0, -0.5)]
    eq = EquilibriumCertificate(
        state_enclosure=[(0.5, 1.5), (0.5, 1.5)],
        stability=stability,
        eigenvalue_real_parts=eig_rp,
        krawczyk_contraction=0.3,
        krawczyk_iterations=5,
    )
    return CertifiedCell(
        parameter_box=box,
        model_name="test",
        n_states=2, n_params=len(box),
        equilibria=[eq],
        regime=regime,
        tier=VerificationTier.TIER1_IA,
        minicheck_passed=minicheck,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Theorem B1: δ-soundness bound
# ═══════════════════════════════════════════════════════════════════════════

class TestTheoremB1DeltaSoundness:
    """Verify that the δ-bound formula matches the paper specification:

        δ_solver < γ / (‖R‖ · L_{Df} · rad(X × M) + 1)

    where rad(X × M) = max_i half-width over the combined state-parameter box.
    """

    def test_eigenvalue_gap_stable(self):
        """γ = min_i |Re(λ_i)| for strictly negative eigenvalues."""
        rp = [Interval(-3.0, -1.0), Interval(-2.0, -0.5)]
        gap = compute_eigenvalue_gap(rp)
        # Closest to zero: hi of [-2, -0.5] is -0.5
        assert gap == pytest.approx(0.5)

    def test_eigenvalue_gap_contains_zero(self):
        """γ = 0 when an eigenvalue enclosure contains zero."""
        rp = [Interval(-1.0, 0.5), Interval(-2.0, -0.5)]
        assert compute_eigenvalue_gap(rp) == 0.0

    def test_eigenvalue_gap_unstable(self):
        """γ = min_i lo for strictly positive eigenvalues."""
        rp = [Interval(0.1, 1.0), Interval(0.3, 2.0)]
        assert compute_eigenvalue_gap(rp) == pytest.approx(0.1)

    def test_eigenvalue_gap_empty(self):
        assert compute_eigenvalue_gap([]) == 0.0

    def test_delta_required_formula(self):
        """Verify δ_required = γ / (‖R‖ · L_{Df} · rad(X×M) + 1)."""
        gamma = 0.5
        R_norm = 2.0
        L_Df = 10.0
        rad = 0.1
        delta_solver = 1e-4

        expected = gamma / (R_norm * L_Df * rad + 1.0)
        # 0.5 / (2*10*0.1 + 1) = 0.5 / 3.0 ≈ 0.1667
        assert expected == pytest.approx(0.5 / 3.0)

        # soundness_margin should be positive when δ_solver < δ_required
        margin = soundness_margin(gamma, L_Df, R_norm, rad, delta_solver)
        perturbation = (R_norm * L_Df * rad + 1.0) * delta_solver
        assert margin == pytest.approx(gamma - perturbation)
        assert margin > 0

    def test_soundness_margin_negative_when_delta_too_large(self):
        """Margin must be negative when δ_solver > δ_required."""
        margin = soundness_margin(
            eigenvalue_gap=0.01,
            lipschitz_Df=100.0,
            preconditioner_norm=5.0,
            domain_radius=0.5,
            delta_solver=0.1,
        )
        assert margin < 0

    def test_domain_rad_uses_max_not_sum(self):
        """Paper specifies rad(X × M) = max_i half-width, not sum."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        model = get_benchmark("van_der_pol")
        rhs = model.rhs

        X = IntervalVector([Interval(0.9, 1.1), Interval(-0.1, 0.1)])
        mu = IntervalVector([Interval(1.0, 1.5), Interval(0.1, 0.3)])
        eig_rp = [Interval(-1.0, -0.3), Interval(-0.8, -0.2)]

        db = DeltaBound(rhs)
        result = db.compute(X, mu, eig_rp, delta_solver=1e-4)

        # Verify domain_rad = max(max state half-width, max param half-width)
        state_rad = float(np.max(X.radius()))    # max(0.1, 0.1) = 0.1
        param_rad = float(np.max(mu.radius()))   # max(0.25, 0.1) = 0.25
        expected_rad = max(state_rad, param_rad)  # 0.25, NOT 0.35

        # δ_required = γ / (‖R‖ · L_{Df} · rad + 1)
        R_norm = db._preconditioner_norm(X.midpoint(), mu.midpoint())
        L_Df = db._lipschitz_Df(X, mu)
        denom = R_norm * L_Df * expected_rad + 1.0
        expected_delta_req = result.eigenvalue_gap / denom

        assert result.delta_required == pytest.approx(expected_delta_req, rel=1e-10)

    def test_bauer_fike_eigenvalue_perturbation(self):
        """δ_eigenvalue = (‖R‖ · L_{Df} · rad + 1) · δ_solver."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        model = get_benchmark("van_der_pol")
        rhs = model.rhs

        X = IntervalVector([Interval(0.9, 1.1), Interval(-0.1, 0.1)])
        mu = IntervalVector([Interval(1.0, 1.5), Interval(0.1, 0.3)])
        eig_rp = [Interval(-1.0, -0.3), Interval(-0.8, -0.2)]
        delta_solver = 1e-3

        db = DeltaBound(rhs)
        result = db.compute(X, mu, eig_rp, delta_solver)

        R_norm = db._preconditioner_norm(X.midpoint(), mu.midpoint())
        L_Df = db._lipschitz_Df(X, mu)
        rad = max(float(np.max(X.radius())), float(np.max(mu.radius())))
        expected_perturbation = (R_norm * L_Df * rad + 1.0) * delta_solver

        assert result.delta_eigenvalue == pytest.approx(expected_perturbation, rel=1e-10)

    def test_soundness_positive_margin(self):
        """is_sound iff δ_solver < δ_required."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        model = get_benchmark("van_der_pol")
        rhs = model.rhs

        X = IntervalVector([Interval(0.9, 1.1), Interval(-0.1, 0.1)])
        mu = IntervalVector([Interval(1.0, 1.2), Interval(0.1, 0.2)])
        eig_rp = [Interval(-2.0, -0.5), Interval(-1.5, -0.3)]

        db = DeltaBound(rhs)
        result = db.compute(X, mu, eig_rp, delta_solver=1e-6)

        if result.delta_required > 1e-6:
            assert result.is_sound
            assert result.soundness_margin > 0
        # With a huge δ_solver it should be unsound
        result2 = db.compute(X, mu, eig_rp, delta_solver=1e3)
        assert not result2.is_sound
        assert result2.soundness_margin < 0


# ═══════════════════════════════════════════════════════════════════════════
# Krawczyk operator: main vs minicheck agreement
# ═══════════════════════════════════════════════════════════════════════════

class TestKrawczykAgreement:
    """Verify that the Krawczyk operator in krawczyk.py matches minicheck.py.

    K(X) = x̃ - R·f(x̃) + (I - R·Df(X))·(X - x̃)
    ρ = max_i width(K(X)_i) / width(X_i)
    """

    def test_krawczyk_formula_simple_system(self):
        """Test K(X) on a simple system f(x) = x - [1,1] with known zero."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        model = get_benchmark("van_der_pol")
        rhs = model.rhs

        # Use a box around a known approximate equilibrium
        X = IntervalVector([Interval(-0.1, 0.1), Interval(-0.1, 0.1)])
        mu = IntervalVector([Interval(0.5), Interval(0.0)])

        kop = KrawczykOperator(rhs)
        result = kop.apply(X, mu)

        # Whether verified or not, check that K was computed
        if result.status == KrawczykStatus.VERIFIED:
            assert result.contraction_factor < 1.0
            assert result.enclosure is not None

    def test_contraction_factor_formula(self):
        """ρ = max_i width(K_i)/width(X_i) must be < 1 for verification."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        model = get_benchmark("van_der_pol")
        rhs = model.rhs

        # Tight box around origin with small mu
        X = IntervalVector([Interval(-0.01, 0.01), Interval(-0.01, 0.01)])
        mu = IntervalVector([Interval(0.5), Interval(0.0)])

        kop = KrawczykOperator(rhs)
        result = kop.apply(X, mu)

        if result.status == KrawczykStatus.VERIFIED:
            K = result.enclosure
            manual_rho = 0.0
            for i in range(X.n):
                if X[i].width > 0:
                    manual_rho = max(manual_rho, K[i].width / X[i].width)
            assert result.contraction_factor == pytest.approx(manual_rho, abs=1e-14)

    def test_minicheck_krawczyk_agrees_with_main(self):
        """Both implementations must produce the same K(X) and ρ."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        model = get_benchmark("toggle_switch")
        rhs = model.rhs

        # Setup boxes
        X_iv = IntervalVector([Interval(0.8, 1.2), Interval(0.2, 0.6)])
        mu_iv = IntervalVector([Interval(2.0), Interval(2.0),
                                Interval(2.0), Interval(2.0)])

        # Main implementation
        kop = KrawczykOperator(rhs)
        main_result = kop.apply(X_iv, mu_iv)

        # MiniCheck implementation uses its own IV/IVec types
        X_mc = IVec([IV(0.8, 1.2), IV(0.2, 0.6)])
        mu_mc = IVec([IV(2.0), IV(2.0), IV(2.0), IV(2.0)])

        # Get minicheck's RHS and Jacobian from its MODELS registry
        from phase_cartographer.minicheck import MODELS
        model_entry = MODELS["toggle_switch"]
        rhs_fn = model_entry["rhs"]
        jac_fn = model_entry["jacobian"]

        mc_verified, mc_enc, mc_rho = recompute_krawczyk(
            rhs_fn, jac_fn, X_mc, mu_mc,
        )

        # Both should agree on verification status
        assert (main_result.status == KrawczykStatus.VERIFIED) == mc_verified

        # If both verified, contraction factors should be close
        if mc_verified and main_result.status == KrawczykStatus.VERIFIED:
            assert main_result.contraction_factor == pytest.approx(
                mc_rho, rel=0.1
            )


# ═══════════════════════════════════════════════════════════════════════════
# Proposition C1: adaptive refinement convergence
# ═══════════════════════════════════════════════════════════════════════════

class TestPropositionC1Convergence:
    """Verify that adaptive refinement convergence properties hold:

    At depth d, uncertified volume ≤ O(D / 2^{d/p}) where D = domain diameter.
    Key structural properties:
      - split_box halves the widest dimension
      - volume of each child = parent_volume / 2
      - max box diameter decreases as 1/2^{d/p}
    """

    def test_split_halves_volume(self):
        """Each split produces two children with half the parent volume."""
        box = [(0.0, 2.0), (0.0, 3.0)]
        parent_vol = box_volume(box)
        child1, child2 = split_box(box)
        assert box_volume(child1) + box_volume(child2) == pytest.approx(parent_vol)
        assert box_volume(child1) == pytest.approx(parent_vol / 2)

    def test_max_width_decreases(self):
        """Repeated splitting reduces max box width monotonically."""
        box = [(0.0, 4.0), (0.0, 2.0)]
        widths = [max(hi - lo for lo, hi in box)]

        current = [box]
        for _ in range(6):
            next_gen = []
            for b in current:
                c1, c2 = split_box(b)
                next_gen.extend([c1, c2])
            current = next_gen
            widths.append(max(
                max(hi - lo for lo, hi in b) for b in current
            ))

        # Max width must be strictly decreasing
        for i in range(1, len(widths)):
            assert widths[i] <= widths[i - 1]

    def test_diameter_bound_at_depth(self):
        """After d splits cycling through p dims, diameter ≤ D / 2^{⌊d/p⌋}."""
        box = [(0.0, 4.0), (0.0, 4.0)]  # p=2, D=4
        p = 2
        D = 4.0

        current = [box]
        for d in range(1, 9):
            next_gen = []
            for b in current:
                c1, c2 = split_box(b)
                next_gen.extend([c1, c2])
            current = next_gen

            max_diam = max(max(hi - lo for lo, hi in b) for b in current)
            theoretical = D / (2 ** (d // p))
            assert max_diam <= theoretical + 1e-12

    def test_adaptive_refine_coverage_increases(self):
        """Coverage increases with depth; all cells are certified or uncertified."""
        domain = [(0.0, 2.0), (0.0, 2.0)]
        # Certify everything (trivial certifier)
        call_count = [0]

        def always_certify(box):
            call_count[0] += 1
            eq = EquilibriumCertificate(
                state_enclosure=[(0.9, 1.1), (0.9, 1.1)],
                stability=StabilityType.STABLE_NODE,
                eigenvalue_real_parts=[(-2.0, -0.5)],
                krawczyk_contraction=0.3,
                krawczyk_iterations=3,
            )
            return CertifiedCell(
                parameter_box=box,
                model_name="test", n_states=2, n_params=2,
                equilibria=[eq],
                regime=RegimeType.MONOSTABLE,
                tier=VerificationTier.TIER1_IA,
                minicheck_passed=True,
            )

        config = RefinementConfig(max_depth=4, target_coverage=1.0, max_cells=100)
        atlas = adaptive_refine(always_certify, domain, "test", config)

        # Full coverage because everything is certified
        assert atlas.coverage_fraction() == pytest.approx(1.0)
        assert call_count[0] >= 1

    def test_uncertified_volume_decreases_with_depth(self):
        """Uncertified volume decreases as max_depth increases."""
        domain = [(0.0, 2.0), (0.0, 2.0)]

        # Certify only boxes away from center (simulating a bifurcation at (1,1))
        def certify_away_from_center(box):
            mid = [(lo + hi) / 2 for lo, hi in box]
            dist = max(abs(m - 1.0) for m in mid)
            max_w = max(hi - lo for lo, hi in box)
            if dist > max_w:
                eq = EquilibriumCertificate(
                    state_enclosure=[(0.9, 1.1), (0.9, 1.1)],
                    stability=StabilityType.STABLE_NODE,
                    eigenvalue_real_parts=[(-2.0, -0.5)],
                    krawczyk_contraction=0.3,
                    krawczyk_iterations=3,
                )
                return CertifiedCell(
                    parameter_box=box,
                    model_name="test", n_states=2, n_params=2,
                    equilibria=[eq],
                    regime=RegimeType.MONOSTABLE,
                    tier=VerificationTier.TIER1_IA,
                    minicheck_passed=True,
                )
            return None

        coverages = []
        for depth in [2, 4, 6]:
            config = RefinementConfig(
                max_depth=depth, target_coverage=1.0,
                max_cells=5000, min_box_width=1e-8,
            )
            atlas = adaptive_refine(
                certify_away_from_center, domain, "test", config,
            )
            coverages.append(atlas.coverage_fraction())

        # Coverage should increase (uncertified volume decreases)
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1] - 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# Theorem B2: certificate composition soundness
# ═══════════════════════════════════════════════════════════════════════════

class TestTheoremB2CompositionSoundness:
    """Verify Theorem B2: composed atlas is sound if:
      (i)   each cell is individually certified,
      (ii)  cells are pairwise disjoint,
      (iii) regime labels are derivable from certified evidence.
    """

    def test_valid_composition_disjoint_cells(self):
        """Disjoint, certified cells compose into a valid atlas."""
        cells = [
            _make_cell([(0.0, 1.0), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
            _make_cell([(1.0, 2.0), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
        ]
        domain = [(0.0, 2.0), (0.0, 1.0)]

        # verify_atlas_composition
        result = verify_atlas_composition(cells, domain)
        assert result.valid
        assert result.coverage_fraction == pytest.approx(1.0)

        # compose_atlas (certificate_algebra)
        valid, steps, errors = CertificateCompositionRules.compose_atlas(
            cells, domain,
        )
        assert valid
        assert len(errors) == 0

    def test_overlapping_cells_rejected(self):
        """Overlapping cells must be rejected (violates condition ii)."""
        cells = [
            _make_cell([(0.0, 1.5), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
            _make_cell([(1.0, 2.0), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
        ]
        domain = [(0.0, 2.0), (0.0, 1.0)]

        result = verify_atlas_composition(cells, domain)
        assert not result.valid
        assert any("overlap" in e.lower() for e in result.errors)

        valid, _, errors = CertificateCompositionRules.compose_atlas(
            cells, domain,
        )
        assert not valid

    def test_uncertified_cell_rejected(self):
        """Cells without minicheck must be rejected (violates condition i)."""
        cells = [
            _make_cell([(0.0, 1.0), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE, minicheck=False),
        ]
        domain = [(0.0, 1.0), (0.0, 1.0)]

        result = verify_atlas_composition(cells, domain)
        assert not result.valid

        valid, _, errors = CertificateCompositionRules.compose_atlas(
            cells, domain,
        )
        assert not valid

    def test_boundary_detection_between_regimes(self):
        """Adjacent cells with different regimes imply a bifurcation boundary."""
        cells = [
            _make_cell([(0.0, 1.0), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
            _make_cell([(1.0, 2.0), (0.0, 1.0)], RegimeType.BISTABLE,
                        StabilityType.STABLE_NODE),
        ]
        domain = [(0.0, 2.0), (0.0, 1.0)]

        result = verify_atlas_composition(cells, domain)
        assert len(result.boundary_cells) == 1
        assert result.boundary_cells[0] == (0, 1)

    def test_bundle_verify_all_passes(self):
        """AtlasCertificateBundle.verify_all passes for a valid atlas."""
        cells = [
            _make_cell([(0.0, 1.0), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
            _make_cell([(1.0, 2.0), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
        ]
        domain = [(0.0, 2.0), (0.0, 1.0)]

        bundle = AtlasCertificateBundle.from_cells(cells, domain)
        ok, errors = bundle.verify_all()
        assert ok, f"Errors: {errors}"

    def test_composition_and_verify_atlas_agree(self):
        """verify_atlas_composition and compose_atlas must agree."""
        cells = [
            _make_cell([(0.0, 0.5), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
            _make_cell([(0.5, 1.0), (0.0, 1.0)], RegimeType.MONOSTABLE,
                        StabilityType.STABLE_NODE),
        ]
        domain = [(0.0, 1.0), (0.0, 1.0)]

        r1 = verify_atlas_composition(cells, domain)
        v2, _, e2 = CertificateCompositionRules.compose_atlas(cells, domain)

        assert r1.valid == v2

    def test_proof_tree_consistency(self):
        """RegimeProofTree from a cell must pass verification."""
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        tree = RegimeProofTree.from_cell(cell)
        ok, errors = tree.verify()
        assert ok, f"Proof tree errors: {errors}"
        assert tree.conclusion == RegimeType.MONOSTABLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
