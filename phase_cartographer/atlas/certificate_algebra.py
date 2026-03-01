"""
Formal certificate composition algebra for phase atlas construction.

Provides a rigorous algebraic framework for composing, merging, splitting,
and verifying cell-level certificates into atlas-level proofs.  Each
operation is recorded as an auditable CompositionStep so that the full
derivation can be replayed and independently checked.

Addresses the requirement for a unified certificate format and formal
composition rules (Theorem B2).
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from ..tiered.certificate import (
    CertifiedCell,
    EquilibriumCertificate,
    VerificationTier,
    RegimeType,
    StabilityType,
    RegimeInferenceRules,
)
from .composition import boxes_adjacent, boxes_disjoint, ProofTrace


# ── helpers ───────────────────────────────────────────────────────────────

def _sha256(obj: Any) -> str:
    """Deterministic SHA-256 hex digest (first 64 chars) of *obj*."""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _format_interval(iv: Tuple[float, float]) -> str:
    return f"[{iv[0]:.6g}, {iv[1]:.6g}]"


# ── 1. CertificateEvidence ────────────────────────────────────────────────

@dataclass
class CertificateEvidence:
    """All verification evidence collected for a single equilibrium."""

    tier: VerificationTier
    krawczyk_verified: bool
    contraction_factor: float
    eigenvalue_enclosure: List[Tuple[float, float]]
    stability_type: str
    delta_bound: Optional[Dict[str, float]] = None
    z3_equilibrium_verified: bool = False
    z3_stability_verified: bool = False
    minicheck_passed: bool = False
    proof_hash: str = ""

    def __post_init__(self):
        if not self.proof_hash:
            self.proof_hash = self.compute_hash()

    def compute_hash(self) -> str:
        return _sha256({
            "tier": self.tier.value,
            "krawczyk_verified": self.krawczyk_verified,
            "contraction_factor": self.contraction_factor,
            "eigenvalue_enclosure": list(self.eigenvalue_enclosure),
            "stability_type": self.stability_type,
            "delta_bound": self.delta_bound,
            "z3_equilibrium_verified": self.z3_equilibrium_verified,
            "z3_stability_verified": self.z3_stability_verified,
            "minicheck_passed": self.minicheck_passed,
        })

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "krawczyk_verified": self.krawczyk_verified,
            "contraction_factor": self.contraction_factor,
            "eigenvalue_enclosure": list(self.eigenvalue_enclosure),
            "stability_type": self.stability_type,
            "delta_bound": self.delta_bound,
            "z3_equilibrium_verified": self.z3_equilibrium_verified,
            "z3_stability_verified": self.z3_stability_verified,
            "minicheck_passed": self.minicheck_passed,
            "proof_hash": self.proof_hash,
        }

    @staticmethod
    def from_equilibrium(eq: EquilibriumCertificate,
                         tier: VerificationTier,
                         minicheck_passed: bool = False,
                         z3_eq: bool = False,
                         z3_stab: bool = False) -> "CertificateEvidence":
        """Build evidence from an existing EquilibriumCertificate."""
        return CertificateEvidence(
            tier=tier,
            krawczyk_verified=eq.krawczyk_contraction < 1.0,
            contraction_factor=eq.krawczyk_contraction,
            eigenvalue_enclosure=list(eq.eigenvalue_real_parts),
            stability_type=eq.stability.value,
            delta_bound=eq.delta_bound,
            z3_equilibrium_verified=z3_eq,
            z3_stability_verified=z3_stab,
            minicheck_passed=minicheck_passed,
        )


# ── 2. CertificateAtom ───────────────────────────────────────────────────

@dataclass
class CertificateAtom:
    """Atomic certificate for a single equilibrium in a single parameter box."""

    parameter_box: List[Tuple[float, float]]
    state_enclosure: List[Tuple[float, float]]
    evidence: CertificateEvidence

    # ── public API ────────────────────────────────────────────────────

    def verify_self_consistency(self) -> Tuple[bool, List[str]]:
        """Check internal consistency of this atom.

        Returns ``(ok, errors)`` where *errors* is empty when consistent.
        """
        errors: List[str] = []

        if self.evidence.krawczyk_verified and self.evidence.contraction_factor >= 1.0:
            errors.append(
                f"Contraction factor {self.evidence.contraction_factor} >= 1 "
                "but marked Krawczyk-verified"
            )

        if self.evidence.proof_hash != self.evidence.compute_hash():
            errors.append("Stored proof_hash does not match recomputed hash")

        for dim, (lo, hi) in enumerate(self.parameter_box):
            if lo > hi:
                errors.append(f"Parameter box dim {dim}: lo ({lo}) > hi ({hi})")

        for dim, (lo, hi) in enumerate(self.state_enclosure):
            if lo > hi:
                errors.append(f"State enclosure dim {dim}: lo ({lo}) > hi ({hi})")

        return (len(errors) == 0, errors)

    def compute_hash(self) -> str:
        return _sha256({
            "parameter_box": list(self.parameter_box),
            "state_enclosure": list(self.state_enclosure),
            "evidence_hash": self.evidence.proof_hash,
        })

    def to_proof_obligation(self) -> Dict[str, Any]:
        """Return a machine-readable proof obligation for external checkers."""
        return {
            "kind": "equilibrium_existence_and_stability",
            "parameter_box": list(self.parameter_box),
            "state_enclosure": list(self.state_enclosure),
            "required_contraction": "< 1.0",
            "actual_contraction": self.evidence.contraction_factor,
            "stability_claim": self.evidence.stability_type,
            "eigenvalue_enclosure": list(self.evidence.eigenvalue_enclosure),
            "atom_hash": self.compute_hash(),
        }

    def to_dict(self) -> dict:
        return {
            "parameter_box": list(self.parameter_box),
            "state_enclosure": list(self.state_enclosure),
            "evidence": self.evidence.to_dict(),
            "hash": self.compute_hash(),
        }

    # ── convenience constructors ──────────────────────────────────────

    @staticmethod
    def from_cell(cell: CertifiedCell) -> List["CertificateAtom"]:
        """Create one CertificateAtom per equilibrium in *cell*."""
        atoms: List[CertificateAtom] = []
        for eq in cell.equilibria:
            ev = CertificateEvidence.from_equilibrium(
                eq, cell.tier,
                minicheck_passed=bool(cell.minicheck_passed),
            )
            atoms.append(CertificateAtom(
                parameter_box=list(cell.parameter_box),
                state_enclosure=list(eq.state_enclosure),
                evidence=ev,
            ))
        return atoms


# ── 3. RegimeProofTree ────────────────────────────────────────────────────

_VALID_RULES = {"MONO", "BI", "MULTI", "EXC", "OSC", "INC"}

_RULE_TO_REGIME: Dict[str, RegimeType] = {
    "MONO": RegimeType.MONOSTABLE,
    "BI": RegimeType.BISTABLE,
    "MULTI": RegimeType.MULTISTABLE,
    "EXC": RegimeType.EXCITABLE,
    "OSC": RegimeType.OSCILLATORY,
    "INC": RegimeType.INCONCLUSIVE,
}


@dataclass
class RegimeProofTree:
    """Formal derivation tree from leaf atoms to a regime conclusion."""

    leaves: List[CertificateAtom]
    rule_applied: str  # one of MONO, BI, MULTI, EXC, OSC, INC
    conclusion: RegimeType
    premises_satisfied: bool = False

    def verify(self) -> Tuple[bool, List[str]]:
        """Verify that the tree is internally consistent.

        Checks:
        * Every leaf passes self-consistency.
        * The applied rule is valid and maps to the stated conclusion.
        * The leaf evidence actually satisfies the rule's premises.
        """
        errors: List[str] = []

        # Leaf consistency
        for i, leaf in enumerate(self.leaves):
            ok, leaf_errors = leaf.verify_self_consistency()
            if not ok:
                errors.extend(f"Leaf {i}: {e}" for e in leaf_errors)

        # Rule validity
        if self.rule_applied not in _VALID_RULES:
            errors.append(f"Unknown rule: {self.rule_applied}")
        elif _RULE_TO_REGIME[self.rule_applied] != self.conclusion:
            errors.append(
                f"Rule {self.rule_applied} should yield "
                f"{_RULE_TO_REGIME[self.rule_applied].value}, "
                f"not {self.conclusion.value}"
            )

        # Premise check
        n_stable = sum(
            1 for lf in self.leaves
            if lf.evidence.stability_type in (
                StabilityType.STABLE_NODE.value,
                StabilityType.STABLE_FOCUS.value,
                StabilityType.STABLE_SPIRAL.value,
            )
        )
        n_saddle = sum(
            1 for lf in self.leaves
            if lf.evidence.stability_type == StabilityType.SADDLE.value
        )

        rule_ok = False
        if self.rule_applied == "MONO":
            rule_ok = n_stable == 1
        elif self.rule_applied == "BI":
            rule_ok = n_stable >= 2
        elif self.rule_applied == "MULTI":
            rule_ok = n_stable >= 3
        elif self.rule_applied == "EXC":
            rule_ok = n_stable == 1 and n_saddle >= 1
        elif self.rule_applied == "OSC":
            rule_ok = n_stable == 0
        elif self.rule_applied == "INC":
            rule_ok = True  # always admissible

        if not rule_ok:
            errors.append(
                f"Premises not satisfied for rule {self.rule_applied}: "
                f"n_stable={n_stable}, n_saddle={n_saddle}"
            )

        self.premises_satisfied = rule_ok and len(errors) == 0
        return (len(errors) == 0, errors)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_applied,
            "conclusion": self.conclusion.value,
            "premises_satisfied": self.premises_satisfied,
            "n_leaves": len(self.leaves),
            "leaves": [lf.to_dict() for lf in self.leaves],
        }

    def render_ascii(self) -> str:
        """Produce a simple ASCII derivation tree for debugging."""
        lines: List[str] = []
        lines.append(f"┌─ Conclusion: {self.conclusion.value}  "
                      f"(rule {self.rule_applied}, "
                      f"premises_ok={self.premises_satisfied})")
        for i, lf in enumerate(self.leaves):
            prefix = "└" if i == len(self.leaves) - 1 else "├"
            box_str = " × ".join(_format_interval(iv) for iv in lf.parameter_box)
            lines.append(
                f"{prefix}── Leaf {i}: {lf.evidence.stability_type}  "
                f"box={box_str}  κ={lf.evidence.contraction_factor:.4f}"
            )
        return "\n".join(lines)

    @staticmethod
    def from_cell(cell: CertifiedCell) -> "RegimeProofTree":
        """Build a proof tree directly from a CertifiedCell."""
        atoms = CertificateAtom.from_cell(cell)
        inferred = RegimeInferenceRules.infer(cell.equilibria)

        rule_map: Dict[RegimeType, str] = {v: k for k, v in _RULE_TO_REGIME.items()}
        rule = rule_map.get(inferred, "INC")

        tree = RegimeProofTree(
            leaves=atoms,
            rule_applied=rule,
            conclusion=inferred,
        )
        tree.verify()
        return tree


# ── 4. CompositionStep ────────────────────────────────────────────────────

@dataclass
class CompositionStep:
    """An auditable record of one composition operation."""

    operation: str  # merge / split / compose / boundary
    input_hashes: List[str]
    output_hash: str
    valid: bool
    justification: str

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "input_hashes": self.input_hashes,
            "output_hash": self.output_hash,
            "valid": self.valid,
            "justification": self.justification,
        }


# ── 5. CertificateCompositionRules ────────────────────────────────────────

class CertificateCompositionRules:
    """Formal composition rules for cell-level certificates.

    Every public method returns a ``CompositionStep`` so the full
    derivation is replayable.
    """

    @staticmethod
    def merge(cell_a: CertifiedCell,
              cell_b: CertifiedCell) -> Tuple[Optional[CertifiedCell], CompositionStep]:
        """Merge two adjacent cells with the *same* regime.

        Returns the merged cell (or ``None`` if merging is invalid) and the
        recorded composition step.
        """
        in_hashes = [cell_a.fingerprint(), cell_b.fingerprint()]

        # Pre-conditions
        if cell_a.regime != cell_b.regime:
            return None, CompositionStep(
                operation="merge",
                input_hashes=in_hashes,
                output_hash="",
                valid=False,
                justification=(
                    f"Regime mismatch: {cell_a.regime.value} vs "
                    f"{cell_b.regime.value}"
                ),
            )

        if not boxes_adjacent(cell_a.parameter_box, cell_b.parameter_box):
            return None, CompositionStep(
                operation="merge",
                input_hashes=in_hashes,
                output_hash="",
                valid=False,
                justification="Cells are not adjacent",
            )

        # Build merged box (bounding box of the union)
        merged_box: List[Tuple[float, float]] = [
            (min(lo_a, lo_b), max(hi_a, hi_b))
            for (lo_a, hi_a), (lo_b, hi_b) in zip(
                cell_a.parameter_box, cell_b.parameter_box
            )
        ]

        merged_equilibria = list(cell_a.equilibria) + list(cell_b.equilibria)

        merged_cell = CertifiedCell(
            parameter_box=merged_box,
            model_name=cell_a.model_name,
            n_states=cell_a.n_states,
            n_params=cell_a.n_params,
            equilibria=merged_equilibria,
            regime=cell_a.regime,
            tier=min(cell_a.tier, cell_b.tier, key=lambda t: t.value),
            depth=max(cell_a.depth, cell_b.depth),
            certification_time_s=(
                cell_a.certification_time_s + cell_b.certification_time_s
            ),
            minicheck_passed=(
                bool(cell_a.minicheck_passed) and bool(cell_b.minicheck_passed)
            ),
        )

        step = CompositionStep(
            operation="merge",
            input_hashes=in_hashes,
            output_hash=merged_cell.fingerprint(),
            valid=True,
            justification=(
                f"Merged adjacent {cell_a.regime.value} cells; "
                f"{len(merged_equilibria)} equilibria in merged cell"
            ),
        )
        return merged_cell, step

    @staticmethod
    def split(cell: CertifiedCell,
              split_dim: int) -> Tuple[List[CertifiedCell], CompositionStep]:
        """Split *cell* into two sub-cells along dimension *split_dim*.

        Each sub-cell inherits the parent's equilibria whose state
        enclosures lie within the sub-box (a conservative over-
        approximation keeps all equilibria in both halves when
        containment cannot be determined).
        """
        in_hash = cell.fingerprint()
        box = list(cell.parameter_box)

        if split_dim < 0 or split_dim >= len(box):
            return [cell], CompositionStep(
                operation="split",
                input_hashes=[in_hash],
                output_hash=in_hash,
                valid=False,
                justification=f"Invalid split dimension {split_dim}",
            )

        lo, hi = box[split_dim]
        mid = (lo + hi) / 2.0

        box_lo = list(box)
        box_lo[split_dim] = (lo, mid)
        box_hi = list(box)
        box_hi[split_dim] = (mid, hi)

        # Conservative: keep all equilibria in both sub-cells
        child_a = CertifiedCell(
            parameter_box=box_lo,
            model_name=cell.model_name,
            n_states=cell.n_states,
            n_params=cell.n_params,
            equilibria=list(cell.equilibria),
            regime=cell.regime,
            tier=cell.tier,
            depth=cell.depth + 1,
            certification_time_s=cell.certification_time_s / 2.0,
            minicheck_passed=cell.minicheck_passed,
        )
        child_b = CertifiedCell(
            parameter_box=box_hi,
            model_name=cell.model_name,
            n_states=cell.n_states,
            n_params=cell.n_params,
            equilibria=list(cell.equilibria),
            regime=cell.regime,
            tier=cell.tier,
            depth=cell.depth + 1,
            certification_time_s=cell.certification_time_s / 2.0,
            minicheck_passed=cell.minicheck_passed,
        )

        out_hash = _sha256([child_a.fingerprint(), child_b.fingerprint()])
        step = CompositionStep(
            operation="split",
            input_hashes=[in_hash],
            output_hash=out_hash,
            valid=True,
            justification=(
                f"Split along dim {split_dim} at midpoint {mid:.6g}; "
                f"2 children inherit {len(cell.equilibria)} equilibria"
            ),
        )
        return [child_a, child_b], step

    @staticmethod
    def boundary_detect(
        cell_a: CertifiedCell,
        cell_b: CertifiedCell,
    ) -> Tuple[Optional[List[Tuple[float, float]]], CompositionStep]:
        """Detect a bifurcation boundary between adjacent different-regime cells.

        If the two cells are adjacent and carry different regime labels, the
        shared face is returned as the boundary location.  Otherwise returns
        ``None``.
        """
        in_hashes = [cell_a.fingerprint(), cell_b.fingerprint()]

        if not boxes_adjacent(cell_a.parameter_box, cell_b.parameter_box):
            return None, CompositionStep(
                operation="boundary",
                input_hashes=in_hashes,
                output_hash="",
                valid=False,
                justification="Cells are not adjacent",
            )

        if cell_a.regime == cell_b.regime:
            return None, CompositionStep(
                operation="boundary",
                input_hashes=in_hashes,
                output_hash="",
                valid=False,
                justification="Same regime — no bifurcation boundary",
            )

        # Identify the shared face
        tol = 1e-10
        boundary: List[Tuple[float, float]] = []
        for (lo_a, hi_a), (lo_b, hi_b) in zip(
            cell_a.parameter_box, cell_b.parameter_box
        ):
            if abs(hi_a - lo_b) < tol:
                boundary.append((hi_a, hi_a))
            elif abs(hi_b - lo_a) < tol:
                boundary.append((lo_a, lo_a))
            else:
                boundary.append((max(lo_a, lo_b), min(hi_a, hi_b)))

        out_hash = _sha256({
            "boundary": boundary,
            "regimes": [cell_a.regime.value, cell_b.regime.value],
        })
        step = CompositionStep(
            operation="boundary",
            input_hashes=in_hashes,
            output_hash=out_hash,
            valid=True,
            justification=(
                f"Bifurcation boundary between {cell_a.regime.value} and "
                f"{cell_b.regime.value}"
            ),
        )
        return boundary, step

    @staticmethod
    def compose_atlas(
        cells: List[CertifiedCell],
        domain: List[Tuple[float, float]],
    ) -> Tuple[bool, List[CompositionStep], List[str]]:
        """Verify full atlas composition (Theorem B2).

        Checks:
        1. Each cell is individually certified (minicheck passed).
        2. Cells are pairwise disjoint.
        3. Coverage fraction is computed.

        Returns ``(valid, steps, errors)``.
        """
        steps: List[CompositionStep] = []
        errors: List[str] = []
        cell_hashes = [c.fingerprint() for c in cells]

        # (i) Individual certification
        for i, cell in enumerate(cells):
            if not cell.minicheck_passed:
                errors.append(f"Cell {i}: minicheck not passed")
            ok, reason = RegimeInferenceRules.validate(cell)
            if not ok:
                errors.append(f"Cell {i}: {reason}")

        # (ii) Pairwise disjointness
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                if not boxes_disjoint(
                    cells[i].parameter_box, cells[j].parameter_box
                ):
                    errors.append(f"Cells {i} and {j} overlap")

        # (iii) Coverage
        domain_vol = 1.0
        for lo, hi in domain:
            w = hi - lo
            if w > 0:
                domain_vol *= w
        certified_vol = sum(c.volume() for c in cells)
        coverage = min(1.0, certified_vol / domain_vol) if domain_vol > 0 else 0.0

        out_hash = _sha256({
            "cell_hashes": cell_hashes,
            "coverage": coverage,
        })
        steps.append(CompositionStep(
            operation="compose",
            input_hashes=cell_hashes,
            output_hash=out_hash,
            valid=len(errors) == 0,
            justification=(
                f"Atlas composition: {len(cells)} cells, "
                f"coverage={coverage:.4f}, "
                f"{len(errors)} errors"
            ),
        ))
        return (len(errors) == 0, steps, errors)


# ── 6. AtlasCertificateBundle ─────────────────────────────────────────────

@dataclass
class AtlasCertificateBundle:
    """Full atlas-level certificate aggregating all cells and proofs."""

    cells: List[CertifiedCell]
    proof_trees: List[RegimeProofTree] = field(default_factory=list)
    composition_steps: List[CompositionStep] = field(default_factory=list)
    coverage_fraction: float = 0.0
    boundary_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def verify_all(self) -> Tuple[bool, List[str]]:
        """Run every available verification check.

        Returns ``(ok, errors)``.
        """
        errors: List[str] = []

        # Per-cell checks
        for i, cell in enumerate(self.cells):
            ok, reason = RegimeInferenceRules.validate(cell)
            if not ok:
                errors.append(f"Cell {i}: {reason}")

        # Proof-tree checks
        for i, tree in enumerate(self.proof_trees):
            ok, tree_errors = tree.verify()
            if not ok:
                errors.extend(f"ProofTree {i}: {e}" for e in tree_errors)

        # Composition-step validity
        for i, step in enumerate(self.composition_steps):
            if not step.valid:
                errors.append(
                    f"CompositionStep {i} ({step.operation}): {step.justification}"
                )

        return (len(errors) == 0, errors)

    def export_json(self, path: str) -> None:
        """Serialise the entire bundle to *path*."""
        data = {
            "n_cells": len(self.cells),
            "coverage_fraction": self.coverage_fraction,
            "n_proof_trees": len(self.proof_trees),
            "n_composition_steps": len(self.composition_steps),
            "boundary_pairs": list(self.boundary_pairs),
            "cells": [c.to_dict() for c in self.cells],
            "proof_trees": [t.to_dict() for t in self.proof_trees],
            "composition_steps": [s.to_dict() for s in self.composition_steps],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        ok, errs = self.verify_all()
        regime_counts: Dict[str, int] = {}
        for c in self.cells:
            regime_counts[c.regime.value] = regime_counts.get(c.regime.value, 0) + 1
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(regime_counts.items()))
        status = "VALID" if ok else f"INVALID ({len(errs)} errors)"
        return (
            f"AtlasCertificateBundle: {len(self.cells)} cells, "
            f"coverage={self.coverage_fraction:.4f}, "
            f"{len(self.boundary_pairs)} bifurcation boundaries, "
            f"regimes=[{dist}], status={status}"
        )

    @staticmethod
    def from_cells(
        cells: List[CertifiedCell],
        domain: List[Tuple[float, float]],
    ) -> "AtlasCertificateBundle":
        """Build a complete bundle from a list of certified cells."""
        # Proof trees
        proof_trees = [RegimeProofTree.from_cell(c) for c in cells]

        # Composition verification
        valid, comp_steps, _errors = (
            CertificateCompositionRules.compose_atlas(cells, domain)
        )

        # Boundary detection
        boundary_pairs: List[Tuple[int, int]] = []
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                if (cells[i].regime != cells[j].regime
                        and boxes_adjacent(
                            cells[i].parameter_box, cells[j].parameter_box)):
                    boundary_pairs.append((i, j))
                    _bnd, bstep = CertificateCompositionRules.boundary_detect(
                        cells[i], cells[j],
                    )
                    comp_steps.append(bstep)

        # Coverage
        domain_vol = 1.0
        for lo, hi in domain:
            w = hi - lo
            if w > 0:
                domain_vol *= w
        certified_vol = sum(c.volume() for c in cells)
        coverage = min(1.0, certified_vol / domain_vol) if domain_vol > 0 else 0.0

        return AtlasCertificateBundle(
            cells=cells,
            proof_trees=proof_trees,
            composition_steps=comp_steps,
            coverage_fraction=coverage,
            boundary_pairs=boundary_pairs,
        )


# ── 7. VerificationReport & verify_certificate_bundle ─────────────────────

@dataclass
class VerificationReport:
    """Detailed report from verify_certificate_bundle()."""
    valid: bool
    atom_errors: List[str] = field(default_factory=list)
    proof_tree_errors: List[str] = field(default_factory=list)
    composition_errors: List[str] = field(default_factory=list)
    hash_chain_valid: Optional[bool] = None
    error_composition_errors: List[str] = field(default_factory=list)

    @property
    def all_errors(self) -> List[str]:
        errs: List[str] = []
        errs.extend(self.atom_errors)
        errs.extend(self.proof_tree_errors)
        errs.extend(self.composition_errors)
        errs.extend(self.error_composition_errors)
        if self.hash_chain_valid is False:
            errs.append("Hash chain integrity check failed")
        return errs

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "atom_errors": self.atom_errors,
            "proof_tree_errors": self.proof_tree_errors,
            "composition_errors": self.composition_errors,
            "hash_chain_valid": self.hash_chain_valid,
            "error_composition_errors": self.error_composition_errors,
        }


def verify_certificate_bundle(
    bundle: AtlasCertificateBundle,
    domain: Optional[List[Tuple[float, float]]] = None,
    proof_trace: Optional[ProofTrace] = None,
) -> VerificationReport:
    """Run the full verification stack on an AtlasCertificateBundle.

    Checks performed:
    1. Atom self-consistency for every equilibrium in every cell.
    2. Proof tree validity for each tree in the bundle.
    3. Composition soundness (disjointness + coverage) if domain given.
    4. Hash chain integrity if a ProofTrace is provided.
    5. Error composition soundness for each atom with delta_bound data.

    Returns:
        A VerificationReport with detailed per-check results.
    """
    report = VerificationReport(valid=True)

    # 1. Atom self-consistency
    for ci, cell in enumerate(bundle.cells):
        atoms = CertificateAtom.from_cell(cell)
        for ai, atom in enumerate(atoms):
            ok, errs = atom.verify_self_consistency()
            if not ok:
                report.atom_errors.extend(
                    f"Cell {ci} Atom {ai}: {e}" for e in errs
                )

    # 2. Proof tree validity
    for ti, tree in enumerate(bundle.proof_trees):
        ok, errs = tree.verify()
        if not ok:
            report.proof_tree_errors.extend(
                f"ProofTree {ti}: {e}" for e in errs
            )

    # 3. Composition soundness
    if domain is not None:
        _valid, _steps, comp_errs = (
            CertificateCompositionRules.compose_atlas(bundle.cells, domain)
        )
        report.composition_errors.extend(comp_errs)

    # 4. Hash chain integrity
    if proof_trace is not None:
        report.hash_chain_valid = proof_trace.verify_chain()

    # 5. Error composition soundness per atom
    for ci, cell in enumerate(bundle.cells):
        for ei, eq in enumerate(cell.equilibria):
            budget = eq.error_budget()
            if eq.delta_bound is not None and not budget["is_sound"]:
                report.error_composition_errors.append(
                    f"Cell {ci} Eq {ei}: composed error "
                    f"{budget['composed_error_bound']:.6g} >= eigenvalue gap "
                    f"{budget['eigenvalue_gap']:.6g}"
                )

    report.valid = len(report.all_errors) == 0
    return report
