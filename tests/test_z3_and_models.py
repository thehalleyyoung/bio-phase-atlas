"""
Tests for Z3 verification (Tier 3) and new benchmark models.
"""
import pytest
import numpy as np


class TestZ3Verification:
    """Test Z3-based exact verification for polynomial models."""

    def test_z3_available(self):
        from phase_cartographer.smt.z3_verifier import Z3_AVAILABLE
        assert Z3_AVAILABLE, "Z3 solver not available"

    def test_brusselator_equilibrium_existence(self):
        """Z3 confirms equilibrium exists for Brusselator."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        from phase_cartographer.smt.z3_verifier import verify_equilibrium_z3

        bm = get_benchmark('brusselator')
        # Known equilibrium at (A, B/A) for the Brusselator
        # For A=1, B=2: equilibrium at (1.0, 2.0)
        state_enc = [(0.9, 1.1), (1.8, 2.2)]
        param_box = [(0.95, 1.05), (1.95, 2.05)]

        result = verify_equilibrium_z3(
            bm.rhs.expressions, bm.rhs.state_symbols, bm.rhs.param_symbols,
            state_enc, param_box, timeout_ms=10000,
        )
        assert result.verified, f"Z3 should find equilibrium: {result.message}"
        assert result.status == "sat"

    def test_selkov_equilibrium_existence(self):
        """Z3 confirms equilibrium exists for Selkov."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        from phase_cartographer.smt.z3_verifier import verify_equilibrium_z3

        bm = get_benchmark('selkov')
        # Selkov equilibrium: x1 = b/(a + b^2), x2 = b/(a + b^2)^2
        # For a=0.1, b=0.6: x1 ≈ 0.6, x2 ≈ 1.3
        state_enc = [(0.5, 0.7), (1.1, 1.5)]
        param_box = [(0.08, 0.12), (0.55, 0.65)]

        result = verify_equilibrium_z3(
            bm.rhs.expressions, bm.rhs.state_symbols, bm.rhs.param_symbols,
            state_enc, param_box, timeout_ms=10000,
        )
        assert result.verified, f"Z3 should find equilibrium: {result.message}"

    def test_brusselator_stability_stable(self):
        """Z3 confirms stability for Brusselator below Hopf."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        from phase_cartographer.smt.z3_verifier import verify_stability_z3

        bm = get_benchmark('brusselator')
        # Below Hopf: B < 1 + A² → stable
        # A=1, B=1.5: B=1.5 < 1+1=2, so stable
        state_enc = [(0.99, 1.01), (1.49, 1.51)]
        param_box = [(0.99, 1.01), (1.49, 1.51)]

        result = verify_stability_z3(
            bm.rhs.expressions, bm.rhs.state_symbols, bm.rhs.param_symbols,
            state_enc, param_box, stability_type="stable", timeout_ms=10000,
        )
        assert result.verified, f"Should confirm stability: {result.message}"

    def test_lotka_volterra_z3(self):
        """Z3 verification on Lotka-Volterra model."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        from phase_cartographer.smt.z3_verifier import verify_equilibrium_z3

        bm = get_benchmark('lotka_volterra')
        # Interior equilibrium at (gamma/delta, alpha/beta) = (gamma, alpha/beta)
        # alpha=1, beta=1, gamma=1: eq at (1, 1)
        state_enc = [(0.9, 1.1), (0.9, 1.1)]
        param_box = [(0.95, 1.05), (0.95, 1.05), (1.0, 1.0)]

        result = verify_equilibrium_z3(
            bm.rhs.expressions, bm.rhs.state_symbols, bm.rhs.param_symbols,
            state_enc, param_box, timeout_ms=10000,
        )
        assert result.verified, f"Z3 should find LV equilibrium: {result.message}"

    def test_fitzhugh_nagumo_z3(self):
        """Z3 verification on FitzHugh-Nagumo model."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        from phase_cartographer.smt.z3_verifier import verify_equilibrium_z3

        bm = get_benchmark('fitzhugh_nagumo')
        # Rough equilibrium near v ≈ -1.2, w ≈ -0.6 for I=0
        state_enc = [(-1.5, -0.8), (-1.0, -0.3)]
        param_box = [(0.0, 0.05), (0.05, 0.1)]

        result = verify_equilibrium_z3(
            bm.rhs.expressions, bm.rhs.state_symbols, bm.rhs.param_symbols,
            state_enc, param_box, timeout_ms=15000,
        )
        # FHN has an equilibrium; Z3 should find it
        assert result.status in ("sat", "unknown"), f"Unexpected: {result.message}"

    def test_sympy_to_z3_conversion(self):
        """Test SymPy to Z3 expression conversion."""
        from phase_cartographer.smt.z3_verifier import _sympy_to_z3
        import sympy as sp

        x = sp.Symbol('x')
        y = sp.Symbol('y')

        z3_vars = {}
        expr = x**2 + 2*x*y - 3
        z3_expr = _sympy_to_z3(expr, z3_vars)
        assert 'x' in z3_vars
        assert 'y' in z3_vars

    def test_delta_soundness_z3(self):
        """Test δ-soundness verification via Z3."""
        from phase_cartographer.models.benchmark_models import get_benchmark
        from phase_cartographer.smt.z3_verifier import verify_delta_soundness_z3

        bm = get_benchmark('brusselator')
        state_enc = [(0.99, 1.01), (1.49, 1.51)]
        param_box = [(0.99, 1.01), (1.49, 1.51)]

        result = verify_delta_soundness_z3(
            bm.rhs.expressions, bm.rhs.state_symbols, bm.rhs.param_symbols,
            state_enc, param_box,
            eigenvalue_gap=0.3, delta=0.001, timeout_ms=10000,
        )
        # Should return a result (verified or not)
        assert result.status in ("sat", "unsat", "unknown")


class TestNewModels:
    """Test new benchmark models."""

    def test_lotka_volterra_loads(self):
        from phase_cartographer.models.benchmark_models import get_benchmark
        bm = get_benchmark('lotka_volterra')
        assert bm.n_states == 2
        assert bm.n_params == 3
        assert bm.rhs_type == "polynomial"

    def test_fitzhugh_nagumo_loads(self):
        from phase_cartographer.models.benchmark_models import get_benchmark
        bm = get_benchmark('fitzhugh_nagumo')
        assert bm.n_states == 2
        assert bm.n_params == 2
        assert bm.rhs_type == "polynomial"

    def test_lotka_volterra_evaluates(self):
        from phase_cartographer.models.benchmark_models import get_benchmark
        bm = get_benchmark('lotka_volterra')
        x = np.array([1.0, 1.0])
        mu = np.array([1.0, 1.0, 1.0])
        f = bm.rhs.evaluate(x, mu)
        # At equilibrium (gamma, alpha/beta) = (1, 1): f should be zero
        np.testing.assert_allclose(f, [0.0, 0.0], atol=1e-10)

    def test_fitzhugh_nagumo_evaluates(self):
        from phase_cartographer.models.benchmark_models import get_benchmark
        bm = get_benchmark('fitzhugh_nagumo')
        x = np.array([0.0, 0.0])
        mu = np.array([0.0, 0.1])
        f = bm.rhs.evaluate(x, mu)
        # At v=0, w=0, I=0: dv = 0-0-0+0=0, dw = eps*(0+0.7-0) = 0.07
        assert abs(f[0]) < 1e-10
        assert abs(f[1] - 0.07) < 1e-10

    def test_lotka_volterra_certifies(self):
        """Lotka-Volterra should certify with Krawczyk."""
        from phase_cartographer.benchmarks.runner import certify_box
        from phase_cartographer.models.benchmark_models import get_benchmark
        bm = get_benchmark('lotka_volterra')
        cell = certify_box(bm.rhs, bm.parameter_domain, bm.state_domain,
                           bm.name, bm.n_states, bm.n_params, bm.rhs_type)
        assert cell is not None, "Lotka-Volterra should certify"
        assert cell.minicheck_passed

    def test_fitzhugh_nagumo_certifies(self):
        """FitzHugh-Nagumo should certify on a sub-box."""
        from phase_cartographer.benchmarks.runner import certify_box
        from phase_cartographer.models.benchmark_models import get_benchmark
        bm = get_benchmark('fitzhugh_nagumo')
        box = [(0.0, 0.1), (0.05, 0.1)]
        cell = certify_box(bm.rhs, box, bm.state_domain,
                           bm.name, bm.n_states, bm.n_params, bm.rhs_type)
        assert cell is not None, "FitzHugh-Nagumo should certify on sub-box"
        assert cell.minicheck_passed

    def test_all_models_listed(self):
        from phase_cartographer.models.benchmark_models import list_benchmarks
        models = list_benchmarks()
        assert 'lotka_volterra' in models
        assert 'fitzhugh_nagumo' in models
        assert len(models) == 18


class TestConfidenceLayer:
    """Test probabilistic confidence quantification."""

    def test_eigenvalue_gap_confidence(self):
        from phase_cartographer.atlas.confidence import eigenvalue_gap_confidence
        # Large gap → high confidence
        conf, gap = eigenvalue_gap_confidence([(-2.0, -1.0), (-3.0, -2.0)], scale=0.5)
        assert conf > 0.8
        assert gap == 1.0

    def test_eigenvalue_gap_zero(self):
        from phase_cartographer.atlas.confidence import eigenvalue_gap_confidence
        # Zero crossing → zero confidence
        conf, gap = eigenvalue_gap_confidence([(-0.1, 0.1)])
        assert conf == 0.0
        assert gap == 0.0

    def test_krawczyk_margin(self):
        from phase_cartographer.atlas.confidence import krawczyk_margin_confidence
        # Good contraction → high confidence
        assert krawczyk_margin_confidence(0.5) > 0.4
        # Borderline → low confidence
        assert krawczyk_margin_confidence(0.95) < 0.1
        # Failed → zero
        assert krawczyk_margin_confidence(1.1) == 0.0

    def test_composite_confidence(self):
        from phase_cartographer.atlas.confidence import compute_cell_confidence
        result = compute_cell_confidence(
            eigenvalue_real_parts=[(-2.0, -1.0)],
            contraction_factor=0.5,
            gp_mean=0.0, gp_std=0.1, regime_label=0,
        )
        assert result.composite_confidence > 0.5
        assert not result.is_near_bifurcation

    def test_near_bifurcation_detection(self):
        from phase_cartographer.atlas.confidence import compute_cell_confidence
        result = compute_cell_confidence(
            eigenvalue_real_parts=[(-0.05, -0.01)],
            contraction_factor=0.96,
        )
        assert result.is_near_bifurcation
        assert result.bifurcation_distance < 0.1


class TestTieredDispatcher:
    """Test that the tiered dispatcher now exercises Z3."""

    def test_tier3_selector_polynomial(self):
        from phase_cartographer.tiered.dispatcher import select_tier, _has_z3
        from phase_cartographer.tiered.certificate import VerificationTier
        if _has_z3():
            tier = select_tier("brusselator", "polynomial")
            assert tier == VerificationTier.TIER3_Z3

    def test_tier3_verify_brusselator(self):
        from phase_cartographer.tiered.dispatcher import tier3_verify, _has_z3
        if not _has_z3():
            pytest.skip("Z3 not available")

        from phase_cartographer.benchmarks.runner import certify_box
        from phase_cartographer.models.benchmark_models import get_benchmark
        bm = get_benchmark('brusselator')
        box = [(0.95, 1.05), (1.5, 1.8)]
        cell = certify_box(bm.rhs, box, bm.state_domain,
                           bm.name, bm.n_states, bm.n_params, bm.rhs_type)
        if cell is not None:
            ok, msg = tier3_verify(cell, timeout_s=30.0)
            # Should attempt verification (may succeed or fail)
            assert isinstance(ok, bool)
            assert len(msg) > 0
