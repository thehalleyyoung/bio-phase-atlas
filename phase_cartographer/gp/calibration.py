"""
GP calibration and validation pipeline.

Provides tools for assessing whether the GP surrogate's uncertainty
estimates are well-calibrated, selecting between isotropic and ARD
kernels, and formatting diagnostic tables for publication.

This addresses the concern that the GP surrogate is underspecified by
giving concrete, quantitative calibration metrics (ECE, NLPD, coverage,
sharpness) and a principled model-selection procedure based on marginal
likelihood and leave-one-out error.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from scipy.stats import norm

from .surrogate import GPSurrogate, GPPrediction


# -------------------------------------------------------------------
# Result dataclasses
# -------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Aggregated calibration diagnostics for a fitted GP surrogate.

    Attributes:
        ece: Expected calibration error (lower is better).
        sharpness: Average predictive variance (lower means tighter predictions).
        coverage_90: Fraction of test points within the 90 % credible interval.
        coverage_95: Fraction of test points within the 95 % credible interval.
        nlpd: Negative log predictive density (lower is better).
        loo_mse: Leave-one-out mean squared error.
        ard_length_scales: Per-dimension length scales (None for isotropic).
        signal_variance: Kernel signal variance.
        noise_variance: Observation noise variance.
        n_train: Number of training points used.
        n_test: Number of test points used (may equal n_train for LOO).
        calibration_curve: List of (expected_coverage, observed_coverage) pairs.
    """

    ece: float
    sharpness: float
    coverage_90: float
    coverage_95: float
    nlpd: float
    loo_mse: float
    ard_length_scales: Optional[np.ndarray]
    signal_variance: float
    noise_variance: float
    n_train: int
    n_test: int
    calibration_curve: List[Tuple[float, float]]


@dataclass
class ModelSelectionResult:
    """Result of GP kernel / hyper-parameter model selection.

    Attributes:
        best_config: Dict describing the winning configuration.
        all_results: List of dicts, one per candidate, with metrics.
        selection_criterion: Name of the criterion used for ranking.
    """

    best_config: Dict[str, Any]
    all_results: List[Dict[str, Any]]
    selection_criterion: str


# -------------------------------------------------------------------
# GPCalibrator
# -------------------------------------------------------------------


class GPCalibrator:
    """Calibration assessment for a :class:`GPSurrogate`.

    Computes expected calibration error, negative log predictive density,
    coverage at standard confidence levels, and a full calibration curve.

    If no held-out test set is provided, leave-one-out (LOO) predictions
    are used instead.

    Args:
        n_bins: Number of bins for ECE computation and calibration curve.
        seed: Random seed for any stochastic operations.
    """

    def __init__(self, n_bins: int = 10, seed: int = 42) -> None:
        self.n_bins = n_bins
        self.seed = seed

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        gp: GPSurrogate,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """Run the full calibration pipeline.

        If *X_test* / *y_test* are not supplied the method falls back to
        efficient leave-one-out (LOO) predictions derived from the already-
        inverted kernel matrix.

        Args:
            gp: A :class:`GPSurrogate` instance (will be fitted on *X_train*).
            X_train: Training inputs, shape ``(n, d)``.
            y_train: Training targets, shape ``(n,)``.
            X_test: Optional held-out inputs, shape ``(m, d)``.
            y_test: Optional held-out targets, shape ``(m,)``.

        Returns:
            A :class:`CalibrationResult` with all diagnostics populated.
        """
        # Ensure the GP is fitted on the training data.
        if not gp._fitted:
            gp.fit(X_train, y_train)

        use_loo = X_test is None or y_test is None

        if use_loo:
            predictions, y_true = self._loo_predictions(gp, X_train, y_train)
            n_test = len(y_true)
        else:
            predictions = gp.predict_batch(X_test)
            y_true = y_test
            n_test = len(y_test)

        ece = self._compute_ece(predictions, y_true)
        nlpd = self._compute_nlpd(predictions, y_true)
        coverage_90 = self._compute_coverage(predictions, y_true, 0.90)
        coverage_95 = self._compute_coverage(predictions, y_true, 0.95)
        cal_curve = self._compute_calibration_curve(predictions, y_true)
        sharpness = float(np.mean([p.variance for p in predictions]))
        loo_mse = gp.loo_cross_validation(X_train, y_train)

        ard_ls = gp.length_scales.copy() if gp.length_scales is not None else None

        return CalibrationResult(
            ece=ece,
            sharpness=sharpness,
            coverage_90=coverage_90,
            coverage_95=coverage_95,
            nlpd=nlpd,
            loo_mse=loo_mse,
            ard_length_scales=ard_ls,
            signal_variance=gp.signal_var,
            noise_variance=gp.noise_var,
            n_train=X_train.shape[0],
            n_test=n_test,
            calibration_curve=cal_curve,
        )

    def full_report(self, result: CalibrationResult) -> Dict[str, Any]:
        """Return a formatted dictionary suitable for serialisation or display.

        Args:
            result: A :class:`CalibrationResult` produced by :meth:`calibrate`.

        Returns:
            A plain-dict mirror of *result* with numpy arrays converted to
            lists for JSON compatibility.
        """
        return {
            "ece": round(result.ece, 6),
            "sharpness": round(result.sharpness, 6),
            "coverage_90": round(result.coverage_90, 4),
            "coverage_95": round(result.coverage_95, 4),
            "nlpd": round(result.nlpd, 6),
            "loo_mse": round(result.loo_mse, 6),
            "ard_length_scales": (
                result.ard_length_scales.tolist()
                if result.ard_length_scales is not None
                else None
            ),
            "signal_variance": round(result.signal_variance, 6),
            "noise_variance": result.noise_variance,
            "n_train": result.n_train,
            "n_test": result.n_test,
            "calibration_curve": [
                {"expected": round(e, 4), "observed": round(o, 4)}
                for e, o in result.calibration_curve
            ],
        }

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _loo_predictions(
        self, gp: GPSurrogate, X: np.ndarray, y: np.ndarray
    ) -> Tuple[List[GPPrediction], np.ndarray]:
        """Compute LOO predictive distributions from the inverted kernel matrix.

        Uses the identities:
            μ_{-i} = y_i − α_i / [K⁻¹]_{ii}
            σ²_{-i} = 1 / [K⁻¹]_{ii}
        """
        gp.fit(X, y)
        n = X.shape[0]
        y_f = y.astype(float)
        preds: List[GPPrediction] = []

        for i in range(n):
            k_inv_ii = gp.K_inv[i, i]
            if abs(k_inv_ii) < 1e-15:
                preds.append(GPPrediction(mean=y_f[i], variance=1.0, std=1.0))
                continue
            mu_loo = y_f[i] - gp.alpha[i] / k_inv_ii
            var_loo = max(1.0 / k_inv_ii, 1e-15)
            preds.append(GPPrediction(mean=float(mu_loo),
                                      variance=float(var_loo),
                                      std=float(np.sqrt(var_loo))))
        return preds, y_f

    def _compute_ece(
        self, predictions: List[GPPrediction], y_true: np.ndarray
    ) -> float:
        """Expected Calibration Error via binning of absolute z-scores.

        For each of *n_bins* equally-spaced confidence levels in (0, 1) we
        compare the nominal coverage with the observed coverage.  ECE is the
        weighted average absolute difference.

        Args:
            predictions: Posterior predictive distributions.
            y_true: Ground-truth targets.

        Returns:
            ECE in [0, 1]; 0 means perfectly calibrated.
        """
        n = len(y_true)
        if n == 0:
            return 1.0

        residuals = np.array([y_true[i] - predictions[i].mean for i in range(n)])
        stds = np.array([max(predictions[i].std, 1e-10) for i in range(n)])
        z_scores = np.abs(residuals / stds)

        levels = np.linspace(1.0 / self.n_bins, 1.0 - 1.0 / self.n_bins, self.n_bins)
        ece = 0.0
        for conf in levels:
            z_crit = norm.ppf((1.0 + conf) / 2.0)
            observed = float(np.mean(z_scores <= z_crit))
            ece += abs(observed - conf) / self.n_bins
        return float(ece)

    def _compute_nlpd(
        self, predictions: List[GPPrediction], y_true: np.ndarray
    ) -> float:
        """Negative log predictive density.

        NLPD = −(1/n) Σ log 𝒩(y_i | μ_i, σ²_i).

        Lower values indicate better probabilistic predictions.

        Args:
            predictions: Posterior predictive distributions.
            y_true: Ground-truth targets.

        Returns:
            Mean negative log predictive density.
        """
        n = len(y_true)
        if n == 0:
            return float("inf")

        total = 0.0
        for i in range(n):
            var = max(predictions[i].variance, 1e-15)
            residual = y_true[i] - predictions[i].mean
            total += 0.5 * np.log(2.0 * np.pi * var) + 0.5 * residual ** 2 / var
        return float(total / n)

    def _compute_coverage(
        self, predictions: List[GPPrediction], y_true: np.ndarray, level: float
    ) -> float:
        """Empirical coverage at a given confidence level.

        Args:
            predictions: Posterior predictive distributions.
            y_true: Ground-truth targets.
            level: Nominal coverage probability (e.g. 0.90 or 0.95).

        Returns:
            Fraction of test points that fall within the credible interval.
        """
        n = len(y_true)
        if n == 0:
            return 0.0

        z_crit = norm.ppf((1.0 + level) / 2.0)
        count = 0
        for i in range(n):
            std = max(predictions[i].std, 1e-10)
            if abs(y_true[i] - predictions[i].mean) <= z_crit * std:
                count += 1
        return float(count / n)

    def _compute_calibration_curve(
        self, predictions: List[GPPrediction], y_true: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Calibration curve as (expected, observed) pairs.

        Evaluates ten equally-spaced nominal confidence levels from 0.1 to
        0.95 and records the observed empirical coverage at each.

        Args:
            predictions: Posterior predictive distributions.
            y_true: Ground-truth targets.

        Returns:
            List of ``(expected_coverage, observed_coverage)`` tuples.
        """
        levels = np.linspace(0.1, 0.95, self.n_bins)
        curve: List[Tuple[float, float]] = []
        for conf in levels:
            obs = self._compute_coverage(predictions, y_true, float(conf))
            curve.append((round(float(conf), 4), round(obs, 4)))
        return curve


# -------------------------------------------------------------------
# GPModelSelector
# -------------------------------------------------------------------


class GPModelSelector:
    """Compare GP kernel configurations and select the best one.

    Candidates are specified as dicts with keys understood by
    :class:`GPSurrogate`'s constructor (``length_scale``, ``use_ard``,
    ``signal_var``, ``noise_var``).  Selection is based on marginal
    log-likelihood plus a LOO-MSE tie-breaker.

    Args:
        candidates: Optional list of configuration dicts.  If *None* a
            sensible default set (isotropic at three scales + ARD) is used.
    """

    def __init__(
        self, candidates: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        self.candidates = candidates

    def select(
        self, X: np.ndarray, y: np.ndarray
    ) -> ModelSelectionResult:
        """Evaluate every candidate and return the best configuration.

        Each candidate GP is fitted, its marginal log-likelihood is
        computed, and LOO-MSE is recorded.  The candidate with the
        highest marginal log-likelihood is chosen; ties are broken by
        lower LOO-MSE.

        Args:
            X: Training inputs, shape ``(n, d)``.
            y: Training targets, shape ``(n,)``.

        Returns:
            A :class:`ModelSelectionResult` with the winning config,
            per-candidate metrics, and the selection criterion name.
        """
        candidates = self.candidates
        if candidates is None:
            candidates = self._default_candidates(X)

        y_f = y.astype(float)
        n = X.shape[0]
        all_results: List[Dict[str, Any]] = []

        for cfg in candidates:
            gp = GPSurrogate(
                length_scale=cfg.get("length_scale", 1.0),
                signal_var=cfg.get("signal_var", 1.0),
                noise_var=cfg.get("noise_var", 1e-6),
                use_ard=cfg.get("use_ard", False),
            )

            # Optimise length-scale(s) for this config.
            if cfg.get("use_ard", False):
                gp.optimize_ard_length_scales(X, y_f)
            else:
                gp.optimize_length_scale(X, y_f)

            gp.fit(X, y_f)

            mll = self._marginal_log_likelihood(gp, X, y_f)
            loo = gp.loo_cross_validation(X, y_f)

            result_entry: Dict[str, Any] = {
                "config": cfg,
                "marginal_log_likelihood": mll,
                "loo_mse": loo,
                "length_scale": gp.length_scale,
                "use_ard": gp.use_ard,
                "ard_length_scales": (
                    gp.length_scales.tolist()
                    if gp.length_scales is not None
                    else None
                ),
                "signal_var": gp.signal_var,
                "noise_var": gp.noise_var,
            }
            all_results.append(result_entry)

        # Rank: highest MLL first, then lowest LOO-MSE as tie-breaker.
        all_results.sort(
            key=lambda r: (-r["marginal_log_likelihood"], r["loo_mse"])
        )

        return ModelSelectionResult(
            best_config=all_results[0],
            all_results=all_results,
            selection_criterion="marginal_log_likelihood + loo_mse",
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_candidates(X: np.ndarray) -> List[Dict[str, Any]]:
        """Build a default set of candidate configurations.

        Includes isotropic kernels at three length-scale initialisations
        plus an ARD variant.
        """
        dists: List[float] = []
        n = min(X.shape[0], 100)
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(float(np.linalg.norm(X[i] - X[j])))
        median_dist = float(np.median(dists)) if dists else 1.0

        return [
            {"length_scale": median_dist * 0.3, "use_ard": False},
            {"length_scale": median_dist, "use_ard": False},
            {"length_scale": median_dist * 3.0, "use_ard": False},
            {"length_scale": median_dist, "use_ard": True},
        ]

    @staticmethod
    def _marginal_log_likelihood(
        gp: GPSurrogate, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute log marginal likelihood for an already-fitted GP.

        log p(y|X,θ) = −½ yᵀ K⁻¹ y − ½ log|K| − (n/2) log 2π

        The first term is available as ``yᵀ α`` and the log-determinant
        is recovered from the Cholesky factor.
        """
        n = X.shape[0]
        y_f = y.astype(float)

        if gp.use_ard and gp.length_scales is not None:
            from .surrogate import matern52_kernel_matrix_ard
            K = matern52_kernel_matrix_ard(X, gp.length_scales, gp.signal_var)
        else:
            from .surrogate import matern52_kernel_matrix
            K = matern52_kernel_matrix(X, gp.length_scale, gp.signal_var)
        K += gp.noise_var * np.eye(n)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_f))
        data_fit = -0.5 * float(y_f @ alpha)
        complexity = -np.sum(np.log(np.diag(L)))
        const = -0.5 * n * np.log(2.0 * np.pi)
        return float(data_fit + complexity + const)


# -------------------------------------------------------------------
# Diagnostic table formatter
# -------------------------------------------------------------------


def calibration_diagnostic_table(
    results: List[CalibrationResult],
) -> Dict[str, Any]:
    """Format a list of :class:`CalibrationResult` objects into a table dict.

    The returned dictionary is structured for easy conversion to a LaTeX
    table (e.g. via pandas or manual formatting):

    .. code-block:: python

        {
            "columns": ["ECE", "Sharpness", ...],
            "rows": [
                {"ECE": 0.03, "Sharpness": 0.12, ...},
                ...
            ],
        }

    Args:
        results: One or more :class:`CalibrationResult` instances,
            typically from different GP configurations or folds.

    Returns:
        Dictionary with ``"columns"`` and ``"rows"`` keys.
    """
    columns = [
        "ECE",
        "Sharpness",
        "Coverage_90",
        "Coverage_95",
        "NLPD",
        "LOO_MSE",
        "Signal_Var",
        "Noise_Var",
        "N_Train",
        "N_Test",
    ]

    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "ECE": round(r.ece, 4),
                "Sharpness": round(r.sharpness, 6),
                "Coverage_90": round(r.coverage_90, 4),
                "Coverage_95": round(r.coverage_95, 4),
                "NLPD": round(r.nlpd, 4),
                "LOO_MSE": round(r.loo_mse, 6),
                "Signal_Var": round(r.signal_variance, 6),
                "Noise_Var": r.noise_variance,
                "N_Train": r.n_train,
                "N_Test": r.n_test,
            }
        )

    return {"columns": columns, "rows": rows}
