"""
Probabilistic confidence quantification for certified phase atlases.

Addresses the critique that the system issues binary certified/uncertified
verdicts with no graded confidence. This module provides:

1. Eigenvalue-gap confidence: measures distance from bifurcation boundary
   based on the gap between eigenvalue real parts and zero.

2. Krawczyk margin confidence: measures how far the Krawczyk operator
   contraction is from the failure threshold.

3. GP-posterior confidence: uses GP uncertainty to estimate confidence
   in regime classification near uncertified regions.

4. Composite confidence score: weighted combination of all three,
   providing a single [0, 1] confidence for each certified cell.

All confidence scores are INFORMATIONAL — they do not affect the
soundness of certified claims. A cell with low confidence is still
rigorously certified; the score indicates proximity to bifurcation
boundaries where parameter perturbation might change the regime.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class CellConfidence:
    """Confidence metrics for a single certified cell."""
    eigenvalue_gap_confidence: float = 0.0
    krawczyk_margin_confidence: float = 0.0
    gp_posterior_confidence: float = 0.0
    composite_confidence: float = 0.0
    is_near_bifurcation: bool = False
    bifurcation_distance: float = float('inf')
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'eigenvalue_gap_confidence': round(self.eigenvalue_gap_confidence, 4),
            'krawczyk_margin_confidence': round(self.krawczyk_margin_confidence, 4),
            'gp_posterior_confidence': round(self.gp_posterior_confidence, 4),
            'composite_confidence': round(self.composite_confidence, 4),
            'is_near_bifurcation': self.is_near_bifurcation,
            'bifurcation_distance': round(self.bifurcation_distance, 6),
        }


def eigenvalue_gap_confidence(
    eigenvalue_real_parts: List[Tuple[float, float]],
    scale: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute confidence from eigenvalue gap.

    Returns (confidence, gap) where:
    - gap = min_i dist(Re(λ_i), 0)
    - confidence = 1 - exp(-gap / scale)

    confidence → 0 near bifurcation (gap → 0)
    confidence → 1 far from bifurcation (gap → ∞)
    """
    if not eigenvalue_real_parts:
        return 0.0, 0.0

    gap = float('inf')
    for lo, hi in eigenvalue_real_parts:
        if lo <= 0 <= hi:
            return 0.0, 0.0  # eigenvalue crosses zero
        dist = min(abs(lo), abs(hi))
        gap = min(gap, dist)

    confidence = 1.0 - np.exp(-gap / scale)
    return float(confidence), float(gap)


def krawczyk_margin_confidence(
    contraction_factor: float,
    threshold: float = 1.0,
) -> float:
    """
    Compute confidence from Krawczyk contraction margin.

    contraction_factor < 1.0 means verified.
    Margin = 1.0 - contraction_factor.
    Confidence = margin / threshold.
    """
    if contraction_factor >= threshold:
        return 0.0
    margin = threshold - contraction_factor
    return float(min(1.0, margin / threshold))


def gp_posterior_confidence(
    gp_mean: float,
    gp_std: float,
    regime_label: int,
) -> float:
    """
    Compute confidence from GP posterior.

    High confidence when GP posterior mean is close to the regime label
    and uncertainty is low. Low confidence when GP predicts a different
    regime or has high uncertainty.

    confidence = exp(-|mean - label|²) * exp(-std)
    """
    if gp_std <= 0:
        return 1.0
    label_proximity = np.exp(-(gp_mean - regime_label) ** 2)
    uncertainty_penalty = np.exp(-gp_std)
    return float(label_proximity * uncertainty_penalty)


def compute_cell_confidence(
    eigenvalue_real_parts: List[Tuple[float, float]],
    contraction_factor: float,
    gp_mean: float = 0.0,
    gp_std: float = 1.0,
    regime_label: int = 0,
    eigenvalue_scale: float = 0.5,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> CellConfidence:
    """
    Compute composite confidence score for a certified cell.

    Args:
        eigenvalue_real_parts: Interval enclosures of eigenvalue real parts.
        contraction_factor: Krawczyk operator contraction factor.
        gp_mean: GP posterior mean at cell midpoint.
        gp_std: GP posterior std at cell midpoint.
        regime_label: Integer regime label.
        eigenvalue_scale: Scale for eigenvalue gap confidence.
        weights: (w_eig, w_kraw, w_gp) weights for composite score.

    Returns:
        CellConfidence with all metrics.
    """
    w_eig, w_kraw, w_gp = weights

    eig_conf, eig_gap = eigenvalue_gap_confidence(
        eigenvalue_real_parts, scale=eigenvalue_scale
    )
    kraw_conf = krawczyk_margin_confidence(contraction_factor)
    gp_conf = gp_posterior_confidence(gp_mean, gp_std, regime_label)

    composite = w_eig * eig_conf + w_kraw * kraw_conf + w_gp * gp_conf

    # Near bifurcation if eigenvalue gap < 0.1 or contraction > 0.9
    near_bif = eig_gap < 0.1 or contraction_factor > 0.95

    return CellConfidence(
        eigenvalue_gap_confidence=eig_conf,
        krawczyk_margin_confidence=kraw_conf,
        gp_posterior_confidence=gp_conf,
        composite_confidence=composite,
        is_near_bifurcation=near_bif,
        bifurcation_distance=eig_gap,
        details={
            'eigenvalue_gap': eig_gap,
            'contraction_factor': contraction_factor,
            'gp_mean': gp_mean,
            'gp_std': gp_std,
        },
    )


def atlas_confidence_summary(
    cells_confidence: List[CellConfidence],
) -> Dict[str, float]:
    """
    Summarize confidence across all cells in an atlas.

    Returns mean, min, fraction near bifurcation, etc.
    """
    if not cells_confidence:
        return {'mean_confidence': 0.0, 'min_confidence': 0.0,
                'frac_near_bifurcation': 0.0, 'n_cells': 0}

    composites = [c.composite_confidence for c in cells_confidence]
    n_near = sum(1 for c in cells_confidence if c.is_near_bifurcation)

    return {
        'mean_confidence': float(np.mean(composites)),
        'min_confidence': float(np.min(composites)),
        'max_confidence': float(np.max(composites)),
        'std_confidence': float(np.std(composites)),
        'frac_near_bifurcation': n_near / len(cells_confidence),
        'n_cells': len(cells_confidence),
        'n_near_bifurcation': n_near,
        'mean_eigenvalue_gap': float(np.mean([
            c.bifurcation_distance for c in cells_confidence
            if c.bifurcation_distance < float('inf')
        ])) if any(c.bifurcation_distance < float('inf') for c in cells_confidence) else 0.0,
    }
