"""
Comprehensive tests for new features in PhaseCartographer:

1. New benchmark models (8 models)
2. Posterior-informed refinement (covariance-aware splitting)
3. GP coverage/regret bounds
4. Proof trace with hash chain verification
5. Comprehensive ablation
6. MiniCheck with new models
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.models.benchmark_models import (
    get_benchmark, list_benchmarks, ALL_BENCHMARKS, BenchmarkModel,
)
from phase_cartographer.benchmarks.runner import (
    run_benchmark, run_comprehensive_ablation,
)
from phase_cartographer.refinement.octree import (
    covariance_aware_split, posterior_informed_refine,
    split_box, PosteriorRefinementConfig,
)
from phase_cartographer.gp.surrogate import GPSurrogate
from phase_cartographer.atlas.composition import ProofTrace
from phase_cartographer.atlas.builder import PhaseAtlas
from phase_cartographer.minicheck import (
    IV, IVec, IMat, verify_certificate, MODELS,
)
from phase_cartographer.interval.interval import Interval
from phase_cartographer.interval.matrix import IntervalVector


# ============================================================================
# Constants
# ============================================================================

ALL_MODEL_NAMES = [
    "toggle_switch", "brusselator", "selkov", "repressilator", "goodwin",
    "lotka_volterra", "fitzhugh_nagumo",
    "tyson_cell_cycle", "mapk_cascade", "p53_mdm2", "calcium_oscillator",
    "circadian_clock", "genetic_oscillator", "mutual_activation_switch",
    "van_der_pol",
]

NEW_MODEL_NAMES = [
    "tyson_cell_cycle", "mapk_cascade", "p53_mdm2", "calcium_oscillator",
    "circadian_clock", "genetic_oscillator", "mutual_activation_switch",
    "van_der_pol",
]


# ============================================================================
# 1. New benchmark models
# ============================================================================

class TestNewBenchmarkModels:
    """Test all 15 benchmark models, focusing on 8 new models."""

    def test_all_15_models_constructible(self):
        """All 15 models can be constructed via get_benchmark."""
        for name in ALL_MODEL_NAMES:
            bm = get_benchmark(name)
            assert isinstance(bm, BenchmarkModel), f"{name} not a BenchmarkModel"
            assert bm.name == name

    def test_list_benchmarks_complete(self):
        """list_benchmarks returns all 15 names."""
        names = list_benchmarks()
        assert len(names) >= 15
        for name in ALL_MODEL_NAMES:
            assert name in names

    @pytest.mark.parametrize("model_name", NEW_MODEL_NAMES)
    def test_new_model_has_symbolic_expressions(self, model_name):
        """Each new model has properly set symbolic expressions."""
        bm = get_benchmark(model_name)
        rhs = bm.rhs
        assert rhs.expressions is not None, f"{model_name}: expressions not set"
        assert len(rhs.expressions) == bm.n_states, (
            f"{model_name}: expected {bm.n_states} expressions, got {len(rhs.expressions)}"
        )
        assert len(rhs.state_symbols) == bm.n_states
        assert len(rhs.param_symbols) == bm.n_params

    @pytest.mark.parametrize("model_name", NEW_MODEL_NAMES)
    def test_new_model_evaluates_at_midpoint(self, model_name):
        """Evaluate each new model at the midpoint of its domains."""
        bm = get_benchmark(model_name)
        x_mid = np.array([(lo + hi) / 2.0 for lo, hi in bm.state_domain])
        mu_mid = np.array([(lo + hi) / 2.0 for lo, hi in bm.parameter_domain])

        result = bm.rhs.evaluate(x_mid, mu_mid)
        assert result.shape == (bm.n_states,), f"{model_name}: wrong shape"
        assert all(np.isfinite(result)), f"{model_name}: non-finite evaluation"

    @pytest.mark.parametrize("model_name", NEW_MODEL_NAMES)
    def test_new_model_interval_evaluation(self, model_name):
        """Interval evaluation works for each new model."""
        bm = get_benchmark(model_name)
        # Build narrow intervals around the domain midpoint
        x_ivs = []
        for lo, hi in bm.state_domain:
            mid = (lo + hi) / 2.0
            eps = max(abs(mid) * 0.01, 0.01)
            x_ivs.append(Interval(mid - eps, mid + eps))
        x_iv = IntervalVector(x_ivs)

        mu_ivs = []
        for lo, hi in bm.parameter_domain:
            mid = (lo + hi) / 2.0
            eps = max(abs(mid) * 0.01, 0.001)
            mu_ivs.append(Interval(mid - eps, mid + eps))
        mu_iv = IntervalVector(mu_ivs)

        result = bm.rhs.evaluate_interval(x_iv, mu_iv)
        assert result.n == bm.n_states, f"{model_name}: interval result dimension mismatch"

    @pytest.mark.parametrize("model_name", NEW_MODEL_NAMES)
    def test_new_model_jacobian_computation(self, model_name):
        """Jacobian computation works for each new model."""
        bm = get_benchmark(model_name)
        x_mid = np.array([(lo + hi) / 2.0 for lo, hi in bm.state_domain])
        mu_mid = np.array([(lo + hi) / 2.0 for lo, hi in bm.parameter_domain])

        jac = bm.rhs.jacobian(x_mid, mu_mid)
        assert jac.shape == (bm.n_states, bm.n_states), f"{model_name}: wrong Jacobian shape"
        assert np.all(np.isfinite(jac)), f"{model_name}: non-finite Jacobian"

    @pytest.mark.parametrize("model_name", ["tyson_cell_cycle", "mapk_cascade", "van_der_pol"])
    def test_quick_benchmark_run(self, model_name):
        """Quick benchmark run (max_depth=3, max_cells=50) for fast models."""
        result = run_benchmark(model_name, max_depth=3, max_cells=50)
        assert result.model_name == model_name
        assert result.certified_cells >= 1
        assert result.total_time_s > 0
        assert result.tier1_pass_rate >= 0.0

    def test_new_model_attributes(self):
        """Each new model has required attributes (rhs_type, expected_regimes, description)."""
        for name in NEW_MODEL_NAMES:
            bm = get_benchmark(name)
            assert bm.rhs_type in ("hill", "polynomial", "rational"), (
                f"{name}: unknown rhs_type '{bm.rhs_type}'"
            )
            assert len(bm.expected_regimes) >= 1, f"{name}: no expected regimes"
            assert len(bm.description) > 0, f"{name}: empty description"
            assert len(bm.parameter_domain) == bm.n_params
            assert len(bm.state_domain) == bm.n_states


# ============================================================================
# 2. Posterior-informed refinement
# ============================================================================

class TestPosteriorInformedRefinement:
    """Test covariance-aware splitting and posterior-informed refinement."""

    def test_covariance_aware_split_identity(self):
        """With identity covariance, should behave like widest-dim split."""
        box = [(0.0, 2.0), (0.0, 1.0)]  # dim 0 is wider
        cov = np.eye(2)
        left, right = covariance_aware_split(box, cov)

        # Should split along dim 0 (widest * identity = widest)
        assert len(left) == 2
        assert len(right) == 2
        # Verify the split happened along dim 0
        assert left[0][1] < box[0][1]   # left upper bound < original upper
        assert right[0][0] > box[0][0]  # right lower bound > original lower
        # Dim 1 unchanged
        assert left[1] == box[1]
        assert right[1] == box[1]

    def test_covariance_aware_split_nontrivial(self):
        """With non-trivial covariance, should pick different dimension."""
        # Make dim 1 narrower in box but with much higher variance in cov
        box = [(0.0, 2.0), (0.0, 1.0)]  # dim 0 wider
        # Covariance strongly favoring dim 1
        cov = np.array([[0.01, 0.0], [0.0, 100.0]])
        left, right = covariance_aware_split(box, cov)

        # Should split along dim 1 (high variance * extent dominates)
        # dim 1 score = 100.0 * 1.0 = 100, dim 0 score = 0.01 * 2.0 = 0.02
        assert left[0] == box[0], "dim 0 should be unchanged"
        assert right[0] == box[0], "dim 0 should be unchanged"
        assert left[1][1] < box[1][1]  # split happened in dim 1
        assert right[1][0] > box[1][0]

    def test_covariance_aware_split_none_covariance(self):
        """With None covariance, falls back to widest-dim split."""
        box = [(0.0, 1.0), (0.0, 3.0)]  # dim 1 is wider
        left, right = covariance_aware_split(box, None)
        # Should split along dim 1
        assert left[1][1] < box[1][1]
        assert right[1][0] > box[1][0]

    def test_posterior_informed_refine_runs(self):
        """posterior_informed_refine runs without error on a simple model."""
        bm = get_benchmark("selkov")

        def simple_certify(box):
            from phase_cartographer.tiered.certificate import (
                CertifiedCell, EquilibriumCertificate, VerificationTier,
                RegimeType, StabilityType,
            )
            eq = EquilibriumCertificate(
                state_enclosure=[(lo, hi) for lo, hi in zip(
                    [0.5] * bm.n_states, [1.5] * bm.n_states
                )],
                stability=StabilityType.STABLE_NODE,
                eigenvalue_real_parts=[(-2.0, -0.1)] * bm.n_states,
                krawczyk_contraction=0.3,
                krawczyk_iterations=5,
            )
            return CertifiedCell(
                parameter_box=box,
                model_name=bm.name,
                n_states=bm.n_states,
                n_params=bm.n_params,
                equilibria=[eq],
                regime=RegimeType.MONOSTABLE,
                tier=VerificationTier.TIER1_IA,
            )

        config = PosteriorRefinementConfig(
            max_depth=2, target_coverage=0.5, max_cells=10,
        )
        atlas, conv = posterior_informed_refine(
            simple_certify, bm.parameter_domain, bm.name, config,
        )
        assert isinstance(atlas, PhaseAtlas)
        assert atlas.coverage_fraction() > 0

class TestGPCoverageRegretBounds:
    """Test GP-based coverage and regret bounds."""

    def _make_fitted_gp(self):
        """Create a fitted GP surrogate for testing."""
        gp = GPSurrogate(length_scale=1.0, signal_var=1.0, noise_var=1e-4)
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 2, size=(20, 2))
        y = (X[:, 0] > 1.0).astype(float)
        gp.fit(X, y)
        return gp

    def test_coverage_bound_returns_valid_bounds(self):
        """coverage_bound returns bounds in [0, 1]."""
        gp = self._make_fitted_gp()
        domain = [(0.0, 2.0), (0.0, 2.0)]
        lower, upper, conf = gp.coverage_bound(domain, confidence=0.95, n_samples=200)

        assert 0.0 <= lower <= 1.0, f"lower bound {lower} out of [0,1]"
        assert 0.0 <= upper <= 1.0, f"upper bound {upper} out of [0,1]"
        assert lower <= upper, f"lower {lower} > upper {upper}"
        assert conf == 0.95

    def test_coverage_bound_unfitted_gp(self):
        """Unfitted GP returns (0.0, 1.0, confidence)."""
        gp = GPSurrogate()
        lower, upper, conf = gp.coverage_bound([(0, 1), (0, 1)])
        assert lower == 0.0
        assert upper == 1.0

    def test_regret_bound_positive_finite(self):
        """regret_bound returns positive finite bounds."""
        gp = self._make_fitted_gp()
        regret, info_gain = gp.regret_bound(n_observed=20, domain_volume=4.0)

        assert regret > 0, f"regret bound {regret} not positive"
        assert np.isfinite(regret), f"regret bound {regret} not finite"
        assert info_gain > 0, f"info gain {info_gain} not positive"
        assert np.isfinite(info_gain), f"info gain {info_gain} not finite"

    def test_regret_bound_zero_observations(self):
        """regret_bound with 0 observations returns infinity."""
        gp = self._make_fitted_gp()
        regret, info_gain = gp.regret_bound(n_observed=0, domain_volume=1.0)
        assert regret == float('inf')
        assert info_gain == float('inf')

    def test_optimize_ard_length_scales(self):
        """optimize_ard_length_scales with L-BFGS produces reasonable scales."""
        gp = GPSurrogate(length_scale=1.0, signal_var=1.0, noise_var=1e-4,
                         use_ard=True)
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 2, size=(30, 2))
        # Regime depends strongly on dim 0, weakly on dim 1
        y = (X[:, 0] > 1.0).astype(float)

        scales = gp.optimize_ard_length_scales(X, y)
        assert scales.shape == (2,), f"wrong shape {scales.shape}"
        assert np.all(scales > 0), "scales must be positive"
        assert np.all(np.isfinite(scales)), "scales must be finite"

    def test_regret_bound_increases_with_observations(self):
        """Regret bound should increase (sublinearly) with more observations."""
        gp = self._make_fitted_gp()
        r10, _ = gp.regret_bound(n_observed=10, domain_volume=4.0)
        r100, _ = gp.regret_bound(n_observed=100, domain_volume=4.0)
        assert r100 > r10, "regret should increase with more observations"


# ============================================================================
# 4. Proof trace
# ============================================================================

class TestProofTrace:
    """Test proof trace construction, verification, and export."""

    def _make_small_atlas(self):
        """Create a small atlas with a few certified cells."""
        from phase_cartographer.tiered.certificate import (
            CertifiedCell, EquilibriumCertificate, VerificationTier,
            RegimeType, StabilityType,
        )
        bm = get_benchmark("selkov")
        atlas = PhaseAtlas(bm.name, bm.parameter_domain)
        eq = EquilibriumCertificate(
            state_enclosure=[(0.4, 0.6), (0.4, 0.6)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-1.0, -0.5), (-0.8, -0.3)],
            krawczyk_contraction=0.25,
            krawczyk_iterations=5,
        )
        cell = CertifiedCell(
            parameter_box=bm.parameter_domain,
            model_name=bm.name,
            n_states=bm.n_states,
            n_params=bm.n_params,
            equilibria=[eq],
            regime=RegimeType.MONOSTABLE,
            tier=VerificationTier.TIER1_IA,
        )
        atlas.add_cell(cell)
        return atlas

    def test_from_atlas_produces_valid_trace(self):
        """ProofTrace.from_atlas produces a trace with entries."""
        atlas = self._make_small_atlas()
        trace = ProofTrace.from_atlas(atlas)
        assert isinstance(trace, ProofTrace)
        assert len(trace.entries) > 0
        assert trace.chain_hash != ""

    def test_verify_chain_on_fresh_trace(self):
        """verify_chain returns True on a freshly created trace."""
        atlas = self._make_small_atlas()
        trace = ProofTrace.from_atlas(atlas)
        assert trace.verify_chain() is True

    def test_verify_chain_detects_tampering(self):
        """verify_chain returns False when an entry is tampered with."""
        atlas = self._make_small_atlas()
        trace = ProofTrace.from_atlas(atlas)
        # Tamper with the first entry
        if len(trace.entries) > 0:
            trace.entries[0].hash = "tampered_hash_value"
            assert trace.verify_chain() is False

    def test_export_json_creates_valid_file(self):
        """export_json creates a valid JSON file."""
        atlas = self._make_small_atlas()
        trace = ProofTrace.from_atlas(atlas)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            trace.export_json(path)
            assert os.path.exists(path)
            with open(path, 'r') as f:
                data = json.load(f)
            assert "n_entries" in data
            assert "chain_hash" in data
            assert "entries" in data
            assert data["n_entries"] == len(trace.entries)
            assert data["chain_hash"] == trace.chain_hash
        finally:
            os.unlink(path)

    def test_add_entry_manual(self):
        """Manually adding entries builds a valid chain."""
        trace = ProofTrace()
        trace.add_entry(
            operation="test_op",
            inputs={"x": 1.0},
            outputs={"y": 2.0},
            method="test_method",
        )
        trace.add_entry(
            operation="test_op_2",
            inputs={"a": "b"},
            outputs={"c": "d"},
            method="manual",
        )
        assert len(trace.entries) == 2
        assert trace.verify_chain() is True


# ============================================================================
# 5. Comprehensive ablation
# ============================================================================

class TestComprehensiveAblation:
    """Test run_comprehensive_ablation on a fast model."""

    def test_ablation_completes_for_selkov(self):
        """run_comprehensive_ablation completes for selkov with small budget."""
        result = run_comprehensive_ablation(
            "selkov", max_depth=3, max_cells=30,
        )
        assert result.model_name == "selkov"
        assert result.tier1_only_coverage >= 0.0
        assert result.tier1_only_time_s > 0
        assert result.uniform_coverage >= 0.0
        assert result.gp_coverage >= 0.0


# ============================================================================
# 6. MiniCheck with new models
# ============================================================================

class TestMiniCheckNewModels:
    """Test that minicheck can verify certificates from new models."""

    # Model-specific certificate generators
    # Each returns a plausible certificate for the given new model.

    @staticmethod
    def _make_certificate(model_name, n_states, n_params,
                          rhs_type, param_values, state_enclosure,
                          stability="stable_node"):
        """Create a certificate dict for testing minicheck."""
        return {
            "model": {
                "name": model_name,
                "n_states": n_states,
                "n_params": n_params,
                "rhs_type": rhs_type,
            },
            "parameter_box": [[v, v] for v in param_values],
            "equilibria": [{
                "state_enclosure": state_enclosure,
                "stability": stability,
                "eigenvalue_real_parts": [[-2.0, -0.1]] * n_states,
                "krawczyk_contraction": 0.3,
                "krawczyk_iterations": 5,
                "delta_bound": {
                    "delta_required": 0.1,
                    "eigenvalue_gap": 0.5,
                },
            }],
            "regime_label": f"monostable_{stability}",
            "coverage_fraction": 1.0,
        }

    def test_minicheck_tyson_cell_cycle(self):
        """MiniCheck processes tyson_cell_cycle certificate."""
        assert "tyson_cell_cycle" in MODELS
        cert = self._make_certificate(
            "tyson_cell_cycle", 2, 2, "polynomial",
            param_values=[1.0, 0.5],
            state_enclosure=[[0.8, 1.2], [0.8, 1.2]],
        )
        result = verify_certificate(cert)
        assert result.equilibria_total == 1
        assert isinstance(result.valid, bool)

    def test_minicheck_mapk_cascade(self):
        """MiniCheck processes mapk_cascade certificate."""
        assert "mapk_cascade" in MODELS
        cert = self._make_certificate(
            "mapk_cascade", 2, 2, "polynomial",
            param_values=[1.0, 0.5],
            state_enclosure=[[0.4, 0.8], [0.2, 0.6]],
        )
        result = verify_certificate(cert)
        assert result.equilibria_total == 1
        assert isinstance(result.valid, bool)

    def test_minicheck_van_der_pol(self):
        """MiniCheck processes van_der_pol certificate."""
        assert "van_der_pol" in MODELS
        cert = self._make_certificate(
            "van_der_pol", 2, 2, "polynomial",
            param_values=[1.0, 0.5],
            state_enclosure=[[0.3, 0.7], [-0.2, 0.2]],
        )
        result = verify_certificate(cert)
        assert result.equilibria_total == 1
        assert isinstance(result.valid, bool)

    def test_minicheck_p53_mdm2(self):
        """MiniCheck processes p53_mdm2 certificate."""
        assert "p53_mdm2" in MODELS
        cert = self._make_certificate(
            "p53_mdm2", 2, 2, "rational",
            param_values=[1.0, 1.0],
            state_enclosure=[[0.8, 1.2], [0.8, 1.2]],
        )
        result = verify_certificate(cert)
        assert result.equilibria_total == 1
        assert isinstance(result.valid, bool)

    def test_minicheck_unknown_model_rejected(self):
        """MiniCheck rejects certificates for unknown models."""
        cert = self._make_certificate(
            "nonexistent_model_xyz", 2, 2, "polynomial",
            param_values=[1.0, 1.0],
            state_enclosure=[[0.5, 1.5], [0.5, 1.5]],
        )
        result = verify_certificate(cert)
        assert not result.valid
        assert any("Unknown model" in e or "unknown" in e.lower()
                    for e in result.errors)

    def test_all_new_models_registered_in_minicheck(self):
        """All 8 new models are registered in minicheck."""
        for name in NEW_MODEL_NAMES:
            assert name in MODELS, f"{name} not registered in minicheck MODELS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
