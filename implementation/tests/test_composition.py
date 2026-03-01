"""
Tests for certificate composition and regime derivation.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)
from phase_cartographer.atlas.composition import (
    RegimeDerivation, CertificateProofObject,
    CompositionResult, boxes_adjacent, boxes_disjoint,
    verify_atlas_composition,
)


def _make_cell(box, regime, stability, minicheck=True):
    """Helper to create a CertifiedCell."""
    eq = EquilibriumCertificate(
        state_enclosure=[(0.5, 1.5), (0.5, 1.5)],
        stability=stability,
        eigenvalue_real_parts=[(-2.0, -0.5), (-2.0, -0.5)],
        krawczyk_contraction=0.3,
        krawczyk_iterations=5,
    )
    cell = CertifiedCell(
        parameter_box=box,
        model_name="test",
        n_states=2,
        n_params=2,
        equilibria=[eq],
        regime=regime,
        tier=VerificationTier.TIER1_IA,
        minicheck_passed=minicheck,
    )
    return cell


class TestBoxGeometry:
    def test_adjacent_boxes(self):
        box1 = [(0.0, 1.0), (0.0, 1.0)]
        box2 = [(1.0, 2.0), (0.0, 1.0)]
        assert boxes_adjacent(box1, box2)

    def test_non_adjacent_boxes(self):
        box1 = [(0.0, 1.0), (0.0, 1.0)]
        box2 = [(2.0, 3.0), (0.0, 1.0)]
        assert not boxes_adjacent(box1, box2)

    def test_disjoint_boxes(self):
        box1 = [(0.0, 1.0), (0.0, 1.0)]
        box2 = [(2.0, 3.0), (0.0, 1.0)]
        assert boxes_disjoint(box1, box2)

    def test_overlapping_boxes(self):
        box1 = [(0.0, 1.5), (0.0, 1.0)]
        box2 = [(1.0, 2.0), (0.0, 1.0)]
        assert not boxes_disjoint(box1, box2)


class TestRegimeDerivation:
    def test_monostable_derivation(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        deriv = RegimeDerivation.derive(cell)
        assert deriv.rule_name == "MONO"
        assert deriv.conclusion == "monostable"
        assert deriv.n_stable == 1

    def test_bistable_derivation(self):
        eq1 = EquilibriumCertificate(
            state_enclosure=[(0.5, 1.0), (0.5, 1.0)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
        )
        eq2 = EquilibriumCertificate(
            state_enclosure=[(2.0, 3.0), (2.0, 3.0)],
            stability=StabilityType.STABLE_FOCUS,
            eigenvalue_real_parts=[(-1.0, -0.2)],
            krawczyk_contraction=0.4,
            krawczyk_iterations=6,
        )
        cell = CertifiedCell(
            parameter_box=[(0.0, 1.0), (0.0, 1.0)],
            model_name="test",
            n_states=2, n_params=2,
            equilibria=[eq1, eq2],
            regime=RegimeType.BISTABLE,
            tier=VerificationTier.TIER1_IA,
            minicheck_passed=True,
        )
        deriv = RegimeDerivation.derive(cell)
        assert deriv.rule_name == "BI"
        assert deriv.n_stable == 2

    def test_derivation_to_dict(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        deriv = RegimeDerivation.derive(cell)
        d = deriv.to_dict()
        assert "premises" in d
        assert "rule" in d
        assert d["rule"] == "MONO"


class TestCertificateProofObject:
    def test_from_cell(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        proof = CertificateProofObject.from_cell(cell)
        assert proof.derivation.rule_name == "MONO"
        assert proof.tier_evidence["tier1_minicheck"]["passed"]

    def test_to_dict(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        proof = CertificateProofObject.from_cell(cell)
        d = proof.to_dict()
        assert "cell" in d
        assert "derivation" in d


class TestAtlasComposition:
    def test_valid_atlas(self):
        cell1 = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        cell2 = _make_cell(
            [(1.0, 2.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        domain = [(0.0, 2.0), (0.0, 1.0)]
        result = verify_atlas_composition([cell1, cell2], domain)
        assert result.valid
        assert abs(result.coverage_fraction - 1.0) < 1e-10

    def test_overlapping_cells_detected(self):
        cell1 = _make_cell(
            [(0.0, 1.5), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        cell2 = _make_cell(
            [(1.0, 2.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        domain = [(0.0, 2.0), (0.0, 1.0)]
        result = verify_atlas_composition([cell1, cell2], domain)
        assert not result.valid

    def test_boundary_detection(self):
        cell1 = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        cell2 = _make_cell(
            [(1.0, 2.0), (0.0, 1.0)],
            RegimeType.BISTABLE,
            StabilityType.STABLE_NODE,
        )
        domain = [(0.0, 2.0), (0.0, 1.0)]
        result = verify_atlas_composition([cell1, cell2], domain)
        assert len(result.boundary_cells) == 1

    def test_uncertified_cell_flagged(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
            minicheck=False,
        )
        domain = [(0.0, 1.0), (0.0, 1.0)]
        result = verify_atlas_composition([cell], domain)
        assert not result.valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── New tests for I2 improvements ─────────────────────────────────────────

from phase_cartographer.atlas.certificate_algebra import (
    CertificateAtom,
    RegimeProofTree,
    AtlasCertificateBundle,
    CertificateCompositionRules,
    VerificationReport,
    verify_certificate_bundle,
)
from phase_cartographer.atlas.composition import ProofTrace
from phase_cartographer.smt.delta_bound import (
    DeltaBoundResult,
    ComposedErrorBudget,
)


class TestErrorBudget:
    """Tests for EquilibriumCertificate.error_budget()."""

    def test_error_budget_no_delta(self):
        eq = EquilibriumCertificate(
            state_enclosure=[(0.5, 1.5), (0.8, 1.2)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
        )
        budget = eq.error_budget()
        assert budget["delta_solver"] == 0.0
        assert budget["interval_enclosure_width"] == 1.0  # max(1.5-0.5, 1.2-0.8)
        assert budget["is_sound"] is False  # no delta → not sound

    def test_error_budget_with_delta_sound(self):
        eq = EquilibriumCertificate(
            state_enclosure=[(0.9, 1.1), (0.9, 1.1)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
            delta_bound={
                "delta_required": 0.001,
                "eigenvalue_gap": 0.5,
                "delta_eigenvalue": 0.0005,
                "delta_residual": 0.0003,
                "is_sound": True,
            },
        )
        budget = eq.error_budget()
        assert budget["delta_solver"] == 0.001
        assert budget["eigenvalue_gap"] == 0.5
        assert budget["eigenvalue_perturbation"] == 0.0005
        # composed = delta_required + enclosure_width = 0.001 + 0.2
        assert budget["composed_error_bound"] == pytest.approx(0.201)
        assert budget["is_sound"] is True  # 0.201 < 0.5

    def test_error_budget_with_explicit_composed(self):
        eq = EquilibriumCertificate(
            state_enclosure=[(0.9, 1.1), (0.9, 1.1)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
            delta_bound={
                "delta_required": 0.001,
                "eigenvalue_gap": 0.5,
                "is_sound": True,
            },
            composed_error_bound=0.05,
        )
        budget = eq.error_budget()
        assert budget["composed_error_bound"] == 0.05

    def test_interval_enclosure_width_field(self):
        eq = EquilibriumCertificate(
            state_enclosure=[(0.9, 1.1), (0.9, 1.1)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
            interval_enclosure_width=0.15,
        )
        budget = eq.error_budget()
        assert budget["interval_enclosure_width"] == 0.15

    def test_to_dict_includes_new_fields(self):
        eq = EquilibriumCertificate(
            state_enclosure=[(0.9, 1.1), (0.9, 1.1)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
            interval_enclosure_width=0.2,
            composed_error_bound=0.05,
        )
        d = eq.to_dict()
        assert d["interval_enclosure_width"] == 0.2
        assert d["composed_error_bound"] == 0.05

    def test_to_dict_omits_none_fields(self):
        eq = EquilibriumCertificate(
            state_enclosure=[(0.9, 1.1), (0.9, 1.1)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
        )
        d = eq.to_dict()
        assert "interval_enclosure_width" not in d
        assert "composed_error_bound" not in d


class TestComposedErrorBudget:
    """Tests for ComposedErrorBudget.compose_errors()."""

    def test_sound_composition(self):
        dr = DeltaBoundResult(
            delta_required=0.001,
            eigenvalue_gap=0.5,
            delta_residual=0.0003,
            delta_eigenvalue=0.0005,
            is_sound=True,
            soundness_margin=0.499,
        )
        budget = ComposedErrorBudget.compose_errors(dr, 0.3, 0.2)
        assert budget.delta_solver == 0.001
        assert budget.interval_width == 0.2
        assert budget.eigenvalue_perturbation == 0.0005
        assert budget.equilibrium_displacement == 0.0003
        # total = delta_eigenvalue + enclosure_width * contraction
        expected = 0.0005 + 0.2 * 0.3
        assert budget.total_composed_error == pytest.approx(expected)
        assert budget.eigenvalue_gap == 0.5
        assert budget.is_sound is True

    def test_unsound_composition(self):
        dr = DeltaBoundResult(
            delta_required=0.1,
            eigenvalue_gap=0.05,
            delta_residual=0.03,
            delta_eigenvalue=0.04,
            is_sound=False,
            soundness_margin=-0.05,
        )
        budget = ComposedErrorBudget.compose_errors(dr, 0.5, 1.0)
        assert budget.is_sound is False
        assert budget.total_composed_error > budget.eigenvalue_gap

    def test_to_dict(self):
        dr = DeltaBoundResult(
            delta_required=0.001,
            eigenvalue_gap=0.5,
            delta_residual=0.0003,
            delta_eigenvalue=0.0005,
            is_sound=True,
        )
        budget = ComposedErrorBudget.compose_errors(dr, 0.3, 0.2)
        d = budget.to_dict()
        assert "total_composed_error" in d
        assert "is_sound" in d


class TestVerifyCertificateBundle:
    """Tests for verify_certificate_bundle()."""

    def _make_bundle(self, minicheck=True, sound_delta=True):
        delta = None
        if sound_delta:
            delta = {
                "delta_required": 0.001,
                "eigenvalue_gap": 0.5,
                "delta_eigenvalue": 0.0005,
                "delta_residual": 0.0003,
                "is_sound": True,
            }
        eq = EquilibriumCertificate(
            state_enclosure=[(0.9, 1.1), (0.9, 1.1)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
            delta_bound=delta,
        )
        cell = CertifiedCell(
            parameter_box=[(0.0, 1.0), (0.0, 1.0)],
            model_name="test",
            n_states=2,
            n_params=2,
            equilibria=[eq],
            regime=RegimeType.MONOSTABLE,
            tier=VerificationTier.TIER1_IA,
            minicheck_passed=minicheck,
        )
        domain = [(0.0, 1.0), (0.0, 1.0)]
        bundle = AtlasCertificateBundle.from_cells([cell], domain)
        return bundle, domain

    def test_valid_bundle(self):
        bundle, domain = self._make_bundle()
        report = verify_certificate_bundle(bundle, domain)
        assert report.valid
        assert report.atom_errors == []
        assert report.proof_tree_errors == []
        assert report.composition_errors == []

    def test_invalid_minicheck(self):
        bundle, domain = self._make_bundle(minicheck=False)
        report = verify_certificate_bundle(bundle, domain)
        assert not report.valid
        assert len(report.composition_errors) > 0

    def test_hash_chain_valid(self):
        bundle, domain = self._make_bundle()
        # Build a dummy proof trace
        trace = ProofTrace()
        trace.add_entry("test_op", {"a": 1}, {"b": 2}, "test_method")
        report = verify_certificate_bundle(bundle, domain, proof_trace=trace)
        assert report.hash_chain_valid is True

    def test_hash_chain_tampered(self):
        bundle, domain = self._make_bundle()
        trace = ProofTrace()
        trace.add_entry("test_op", {"a": 1}, {"b": 2}, "test_method")
        trace.entries[0].hash = "tampered"
        report = verify_certificate_bundle(bundle, domain, proof_trace=trace)
        assert report.hash_chain_valid is False
        assert not report.valid

    def test_error_composition_unsound(self):
        eq = EquilibriumCertificate(
            state_enclosure=[(0.0, 2.0), (0.0, 2.0)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-0.1, -0.01)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
            delta_bound={
                "delta_required": 0.5,
                "eigenvalue_gap": 0.01,
                "delta_eigenvalue": 0.4,
                "delta_residual": 0.3,
                "is_sound": False,
            },
        )
        cell = CertifiedCell(
            parameter_box=[(0.0, 1.0), (0.0, 1.0)],
            model_name="test",
            n_states=2,
            n_params=2,
            equilibria=[eq],
            regime=RegimeType.MONOSTABLE,
            tier=VerificationTier.TIER1_IA,
            minicheck_passed=True,
        )
        domain = [(0.0, 1.0), (0.0, 1.0)]
        bundle = AtlasCertificateBundle.from_cells([cell], domain)
        report = verify_certificate_bundle(bundle, domain)
        assert len(report.error_composition_errors) > 0
        assert not report.valid

    def test_no_domain_skips_composition(self):
        bundle, _ = self._make_bundle()
        report = verify_certificate_bundle(bundle)
        assert report.composition_errors == []

    def test_report_to_dict(self):
        bundle, domain = self._make_bundle()
        report = verify_certificate_bundle(bundle, domain)
        d = report.to_dict()
        assert "valid" in d
        assert "hash_chain_valid" in d
