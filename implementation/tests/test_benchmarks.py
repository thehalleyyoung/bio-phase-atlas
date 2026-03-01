"""
Smoke tests for the benchmark runner.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.benchmarks.runner import (
    run_benchmark, run_ablation, run_scalability_study,
    run_depth_scaling, run_delta_validation,
)
from phase_cartographer.models.benchmark_models import list_benchmarks, get_benchmark


class TestBenchmarks:
    def test_list_benchmarks(self):
        bms = list_benchmarks()
        assert "toggle_switch" in bms
        assert "brusselator" in bms
        assert "selkov" in bms
        assert "repressilator" in bms
        assert "goodwin" in bms

    def test_toggle_smoke(self):
        """Quick smoke test on toggle switch (small grid)."""
        result = run_benchmark("toggle_switch", max_depth=2, max_cells=10)
        assert result.model_name == "toggle_switch"
        assert result.certified_cells >= 1
        assert result.tier1_pass_rate >= 0.0
        assert result.total_time_s > 0

    def test_selkov_smoke(self):
        result = run_benchmark("selkov", max_depth=2, max_cells=10)
        assert result.model_name == "selkov"
        assert result.certified_cells >= 1

    def test_repressilator_model_loads(self):
        bm = get_benchmark("repressilator")
        assert bm.n_states == 3
        assert bm.n_params == 4
        # Widened alpha range
        assert bm.parameter_domain[0] == (3.5, 5.5)

    def test_goodwin_model_loads(self):
        bm = get_benchmark("goodwin")
        assert bm.n_states == 3
        assert bm.n_params == 2
        assert bm.parameter_domain == [(3.0, 8.0), (0.8, 1.2)]

    def test_run_benchmark_e2e_selkov(self):
        """End-to-end: run_benchmark returns valid BenchmarkResult with all fields."""
        result = run_benchmark("selkov", max_depth=2, max_cells=10)
        assert result.model_name == "selkov"
        assert result.n_states == 2
        assert result.n_params >= 1
        assert result.certified_cells >= 1
        assert 0.0 <= result.coverage_fraction <= 1.0
        assert isinstance(result.regime_counts, dict)
        assert result.total_time_s > 0
        assert result.cells_per_second > 0
        assert result.tier1_pass_rate >= 0.0

    def test_run_benchmark_e2e_van_der_pol(self):
        """End-to-end: run_benchmark on van_der_pol."""
        result = run_benchmark("van_der_pol", max_depth=2, max_cells=10)
        assert result.model_name == "van_der_pol"
        assert result.certified_cells >= 1
        assert result.total_time_s > 0

    def test_depth_scaling_selkov(self):
        """Depth scaling produces monotonically non-decreasing coverage."""
        points = run_depth_scaling("selkov", depths=[2, 3], max_cells=20)
        assert len(points) == 2
        assert points[0].max_depth == 2
        assert points[1].max_depth == 3
        assert points[1].coverage_fraction >= points[0].coverage_fraction - 1e-9
        for p in points:
            assert p.model_name == "selkov"
            assert p.certified_cells >= 1
            assert p.total_time_s > 0

    def test_delta_validation_selkov(self):
        """Delta validation returns sensible structure."""
        result = run_delta_validation("selkov", max_depth=2, max_cells=10)
        assert result.model_name == "selkov"
        assert result.n_cells >= 1
        assert result.n_equilibria >= 1
        assert isinstance(result.all_sound, bool)
        assert result.min_eigenvalue_gap >= 0.0
        assert result.max_delta_required >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
