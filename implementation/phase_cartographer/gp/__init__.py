"""GP-based surrogate and acquisition functions for adaptive exploration."""

from .surrogate import GPSurrogate, GPPrediction, matern52_kernel
from .acquisition import (
    expected_improvement, upper_confidence_bound,
    boundary_uncertainty, phase_boundary_score,
    AcquisitionOptimizer,
)
from .calibration import (
    CalibrationResult, GPCalibrator,
    ModelSelectionResult, GPModelSelector,
    calibration_diagnostic_table,
)

__all__ = [
    'GPSurrogate', 'GPPrediction', 'matern52_kernel',
    'expected_improvement', 'upper_confidence_bound',
    'boundary_uncertainty', 'phase_boundary_score',
    'AcquisitionOptimizer',
    'CalibrationResult', 'GPCalibrator',
    'ModelSelectionResult', 'GPModelSelector',
    'calibration_diagnostic_table',
]
