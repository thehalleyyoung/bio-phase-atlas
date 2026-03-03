"""
Explicit δ-soundness parameter table computation for all benchmark models.

For each model and equilibrium, computes the five parameters required by
Theorem 2 (δ-soundness):
    γ   = eigenvalue gap (min |Re(λ)|)
    ‖R‖ = preconditioner norm (inverse Jacobian norm)
    L_Df = Lipschitz constant of Jacobian over enclosure
    rad  = max radius of state × parameter enclosure
    δ_req = γ / (‖R‖ · L_Df · rad + 1)

The bound guarantees: if δ_solver < δ_req, then the δ-verified regime
claim is exactly correct.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..smt.delta_bound import DeltaBound, DeltaBoundResult, compute_eigenvalue_gap
from ..models.benchmark_models import get_benchmark, list_benchmarks, BenchmarkModel
from ..ode.rhs import ODERightHandSide


@dataclass
class DeltaTableRow:
    """One row of the δ-soundness parameter table."""
    model_name: str
    eq_index: int
    gamma: float  # eigenvalue gap
    R_norm: float  # preconditioner norm
    L_Df: float  # Lipschitz constant of Jacobian
    rad: float  # enclosure radius
    delta_required: float  # maximum δ for soundness
    delta_solver: float  # δ used by solver
    is_sound: bool
    soundness_margin: float
    stability_type: str
    state_enclosure_width: float
    residual_bound: float

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "eq_index": self.eq_index,
            "gamma": round(self.gamma, 8),
            "R_norm": round(self.R_norm, 6),
            "L_Df": round(self.L_Df, 6),
            "rad": round(self.rad, 8),
            "delta_required": round(self.delta_required, 8),
            "delta_solver": self.delta_solver,
            "is_sound": self.is_sound,
            "soundness_margin": round(self.soundness_margin, 8),
            "stability_type": self.stability_type,
            "state_enclosure_width": round(self.state_enclosure_width, 10),
            "residual_bound": round(self.residual_bound, 12),
        }


@dataclass
class DeltaTable:
    """Complete δ-soundness parameter table for all models."""
    rows: List[DeltaTableRow] = field(default_factory=list)
    n_sound: int = 0
    n_total: int = 0
    n_models: int = 0

    def to_dict(self) -> dict:
        return {
            "n_sound": self.n_sound,
            "n_total": self.n_total,
            "n_models": self.n_models,
            "sound_rate": round(self.n_sound / max(self.n_total, 1), 4),
            "rows": [r.to_dict() for r in self.rows],
        }

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_R_norm(rhs: ODERightHandSide, x: np.ndarray,
                   mu: np.ndarray) -> float:
    """Compute ‖R‖ = ‖J(x,μ)⁻¹‖_∞ (preconditioner norm)."""
    try:
        J = rhs.jacobian(x, mu)
        cond = np.linalg.cond(J)
        if cond > 1e14:
            return float('inf')
        R = np.linalg.inv(J)
        return float(np.linalg.norm(R, ord=np.inf))
    except np.linalg.LinAlgError:
        return float('inf')


def compute_delta_table_row(
    rhs: ODERightHandSide,
    model_name: str,
    eq_index: int,
    state_enclosure: List[tuple],
    param_box: List[tuple],
    eigenvalue_real_parts: List[tuple],
    stability_type: str,
    delta_solver: float = 1e-3,
) -> Optional[DeltaTableRow]:
    """Compute one row of the δ-soundness table."""
    if not eigenvalue_real_parts:
        return None

    X = IntervalVector([Interval(lo, hi) for lo, hi in state_enclosure])
    mu = IntervalVector([Interval(lo, hi) for lo, hi in param_box])
    eig_rps = [Interval(lo, hi) for lo, hi in eigenvalue_real_parts]

    # Compute eigenvalue gap
    gamma = compute_eigenvalue_gap(eig_rps)

    # Compute ‖R‖
    x_mid = X.midpoint()
    mu_mid = mu.midpoint()
    R_norm = compute_R_norm(rhs, x_mid, mu_mid)

    # Compute L_Df via DeltaBound
    db = DeltaBound(rhs)
    try:
        result = db.compute(X, mu, eig_rps, delta_solver)
    except Exception:
        return None

    # Compute radius
    state_rad = float(np.max(X.radius()))
    param_rad = float(np.max(mu.radius()))
    rad = max(state_rad, param_rad)

    # Enclosure width
    enc_width = max(hi - lo for lo, hi in state_enclosure)

    return DeltaTableRow(
        model_name=model_name,
        eq_index=eq_index,
        gamma=gamma,
        R_norm=R_norm,
        L_Df=result.lipschitz_Df,
        rad=rad,
        delta_required=result.delta_required,
        delta_solver=delta_solver,
        is_sound=result.is_sound,
        soundness_margin=result.soundness_margin,
        stability_type=stability_type,
        state_enclosure_width=enc_width,
        residual_bound=result.residual_bound,
    )


def compute_full_delta_table(
    delta_solver: float = 1e-3,
    model_names: Optional[List[str]] = None,
) -> DeltaTable:
    """Compute the complete δ-soundness parameter table for all benchmark models.

    Uses adaptive refinement to build an atlas, then computes δ-soundness
    parameters for each certified cell. This provides the explicit computation
    of γ, ‖R‖, L_{Df}, rad required by Theorem 2.
    """
    from ..benchmarks.runner import run_benchmark, certify_box

    if model_names is None:
        model_names = list_benchmarks()

    table = DeltaTable(n_models=len(model_names))

    for name in model_names:
        try:
            bm = get_benchmark(name)

            # First try the full domain box
            box = list(bm.parameter_domain)
            cell = certify_box(bm.rhs, box, bm.state_domain,
                               bm.name, bm.n_states, bm.n_params, bm.rhs_type)

            # If full domain fails, try the midpoint with a narrow box
            if cell is None:
                mid_box = []
                for lo, hi in bm.parameter_domain:
                    w = hi - lo
                    if w > 0:
                        m = (lo + hi) / 2.0
                        r = w * 0.05  # 5% of range
                        mid_box.append((m - r, m + r))
                    else:
                        mid_box.append((lo, hi))
                cell = certify_box(bm.rhs, mid_box, bm.state_domain,
                                   bm.name, bm.n_states, bm.n_params, bm.rhs_type)

            if cell is None:
                continue

            for eq_idx, eq in enumerate(cell.equilibria):
                row = compute_delta_table_row(
                    rhs=bm.rhs,
                    model_name=name,
                    eq_index=eq_idx,
                    state_enclosure=eq.state_enclosure,
                    param_box=cell.parameter_box,
                    eigenvalue_real_parts=eq.eigenvalue_real_parts,
                    stability_type=eq.stability.value,
                    delta_solver=delta_solver,
                )
                if row is not None:
                    table.rows.append(row)
                    table.n_total += 1
                    if row.is_sound:
                        table.n_sound += 1

        except Exception:
            continue

    return table
