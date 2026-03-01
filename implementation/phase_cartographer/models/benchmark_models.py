"""
Benchmark biological model definitions.

Each model provides:
  - Symbolic RHS via SymPy
  - Parameter domain for phase-atlas construction
  - State domain for equilibrium search
  - Expected regimes (for validation, not certification)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sympy as sp

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..ode.rhs import SymbolicRHS


@dataclass
class BenchmarkModel:
    """Specification of a benchmark biological model."""
    name: str
    n_states: int
    n_params: int
    rhs: SymbolicRHS
    parameter_domain: List[Tuple[float, float]]
    state_domain: List[Tuple[float, float]]
    rhs_type: str  # "hill", "polynomial", "rational"
    expected_regimes: List[str]
    description: str = ""


def make_toggle_switch() -> BenchmarkModel:
    """Gardner toggle switch (2 states, 4 params)."""
    x1, x2 = sp.symbols('x0 x1')
    a1, a2, n1, n2 = sp.symbols('mu0 mu1 mu2 mu3')

    rhs = SymbolicRHS(
        n_states=2, n_params=4,
        state_symbols=[x1, x2],
        param_symbols=[a1, a2, n1, n2],
        name="toggle_switch",
    )
    rhs.set_expressions([
        a1 / (1 + x2**2) - x1,
        a2 / (1 + x1**2) - x2,
    ])

    return BenchmarkModel(
        name="toggle_switch",
        n_states=2, n_params=4,
        rhs=rhs,
        parameter_domain=[(2.0, 4.0), (2.0, 4.0), (2.0, 2.0), (2.0, 2.0)],
        state_domain=[(0.1, 4.5), (0.1, 4.5)],
        rhs_type="hill",
        expected_regimes=["bistable", "monostable"],
        description="Gardner toggle switch: mutual Hill repression (n1=n2=2 fixed)",
    )


def make_brusselator() -> BenchmarkModel:
    """Brusselator (2 states, 2 params)."""
    x1, x2 = sp.symbols('x0 x1')
    A, B = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x1, x2],
        param_symbols=[A, B],
        name="brusselator",
    )
    rhs.set_expressions([
        A - (B + 1) * x1 + x1**2 * x2,
        B * x1 - x1**2 * x2,
    ])

    return BenchmarkModel(
        name="brusselator",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.8, 2.0), (1.5, 3.5)],
        state_domain=[(0.3, 3.0), (0.5, 5.0)],
        rhs_type="polynomial",
        expected_regimes=["oscillatory", "monostable"],
        description="Brusselator: polynomial autocatalytic model",
    )


def make_selkov() -> BenchmarkModel:
    """Sel'kov glycolysis model (2 states, 2 params)."""
    x1, x2 = sp.symbols('x0 x1')
    a, b = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x1, x2],
        param_symbols=[a, b],
        name="selkov",
    )
    rhs.set_expressions([
        -x1 + a * x2 + x1**2 * x2,
        b - a * x2 - x1**2 * x2,
    ])

    return BenchmarkModel(
        name="selkov",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.05, 0.15), (0.4, 0.8)],
        state_domain=[(0.01, 2.5), (0.01, 3.5)],
        rhs_type="polynomial",
        expected_regimes=["oscillatory", "monostable"],
        description="Sel'kov glycolysis oscillator",
    )


def make_repressilator() -> BenchmarkModel:
    """Repressilator (3 states, 4 params): cyclic repression circuit."""
    x1, x2, x3 = sp.symbols('x0 x1 x2')
    alpha, K, n_h, gamma = sp.symbols('mu0 mu1 mu2 mu3')

    rhs = SymbolicRHS(
        n_states=3, n_params=4,
        state_symbols=[x1, x2, x3],
        param_symbols=[alpha, K, n_h, gamma],
        name="repressilator",
    )
    # Simplified repressilator with Hill coefficient fixed at 2:
    # dx1/dt = alpha/(1 + x3^2) - gamma*x1
    # dx2/dt = alpha/(1 + x1^2) - gamma*x2
    # dx3/dt = alpha/(1 + x2^2) - gamma*x3
    rhs.set_expressions([
        alpha / (1 + x3**2) - gamma * x1,
        alpha / (1 + x1**2) - gamma * x2,
        alpha / (1 + x2**2) - gamma * x3,
    ])

    return BenchmarkModel(
        name="repressilator",
        n_states=3, n_params=4,
        rhs=rhs,
        parameter_domain=[(3.5, 5.5), (1.0, 1.0), (2.0, 2.0), (1.0, 1.0)],
        state_domain=[(0.5, 5.0), (0.5, 5.0), (0.5, 5.0)],
        rhs_type="hill",
        expected_regimes=["oscillatory", "monostable"],
        description="Repressilator: 3-gene cyclic repression oscillator (Hill n=2, gamma=1 fixed)",
    )


def make_goodwin() -> BenchmarkModel:
    """Goodwin oscillator (3 states, 2 varying params).

    dx1/dt = a / (1 + x3^n) - b * x1
    dx2/dt = x1 - b * x2
    dx3/dt = x2 - b * x3

    K = 1 (fixed), n = 4 (fixed Hill coefficient).
    Varying parameters: a ∈ [3, 8], b ∈ [0.8, 1.2].
    """
    x1, x2, x3 = sp.symbols('x0 x1 x2')
    a, b = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=3, n_params=2,
        state_symbols=[x1, x2, x3],
        param_symbols=[a, b],
        name="goodwin",
    )
    # K=1 fixed, n=4 fixed (keeps bifurcation but tractable interval widths)
    rhs.set_expressions([
        a / (1 + x3**4) - b * x1,
        x1 - b * x2,
        x2 - b * x3,
    ])

    return BenchmarkModel(
        name="goodwin",
        n_states=3, n_params=2,
        rhs=rhs,
        parameter_domain=[(3.0, 8.0), (0.8, 1.2)],
        state_domain=[(0.5, 4.0), (0.5, 4.0), (0.5, 4.0)],
        rhs_type="hill",
        expected_regimes=["oscillatory", "monostable"],
        description="Goodwin oscillator: negative-feedback loop with Hill n=4",
    )


def make_lotka_volterra() -> BenchmarkModel:
    """Lotka-Volterra predator-prey (2 states, 3 params).

    dx/dt = alpha*x - beta*x*y
    dy/dt = delta*x*y - gamma*y

    Varying: alpha ∈ [0.5, 1.5], beta ∈ [0.8, 1.2], gamma = 1.0 (fixed).
    delta = 1.0 (fixed).
    Equilibria: (0, 0) unstable saddle and (gamma/delta, alpha/beta) center.
    """
    x, y = sp.symbols('x0 x1')
    alpha, beta, gamma = sp.symbols('mu0 mu1 mu2')

    rhs = SymbolicRHS(
        n_states=2, n_params=3,
        state_symbols=[x, y],
        param_symbols=[alpha, beta, gamma],
        name="lotka_volterra",
    )
    # delta = 1.0 fixed
    rhs.set_expressions([
        alpha * x - beta * x * y,
        x * y - gamma * y,
    ])

    return BenchmarkModel(
        name="lotka_volterra",
        n_states=2, n_params=3,
        rhs=rhs,
        parameter_domain=[(0.5, 1.5), (0.8, 1.2), (1.0, 1.0)],
        state_domain=[(0.1, 3.0), (0.1, 3.0)],
        rhs_type="polynomial",
        expected_regimes=["center", "monostable"],
        description="Lotka-Volterra predator-prey with delta=1 fixed",
    )


def make_fitzhugh_nagumo() -> BenchmarkModel:
    """FitzHugh-Nagumo neuron model (2 states, 2 params).

    dv/dt = v - v^3/3 - w + I
    dw/dt = epsilon * (v + a - b*w)

    Varying: I ∈ [0, 1] (external current), epsilon ∈ [0.05, 0.2].
    a = 0.7, b = 0.8 fixed.
    Rich bifurcation structure: excitable, oscillatory, monostable.
    """
    v, w = sp.symbols('x0 x1')
    I_ext, eps = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[v, w],
        param_symbols=[I_ext, eps],
        name="fitzhugh_nagumo",
    )
    # a = 0.7, b = 0.8 fixed
    a_val = sp.Rational(7, 10)
    b_val = sp.Rational(4, 5)
    rhs.set_expressions([
        v - v**3 / 3 - w + I_ext,
        eps * (v + a_val - b_val * w),
    ])

    return BenchmarkModel(
        name="fitzhugh_nagumo",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.0, 1.0), (0.05, 0.2)],
        state_domain=[(-2.5, 2.5), (-1.5, 2.0)],
        rhs_type="polynomial",
        expected_regimes=["excitable", "oscillatory", "monostable"],
        description="FitzHugh-Nagumo neuron: polynomial excitable/oscillatory model",
    )


def make_tyson_cell_cycle() -> BenchmarkModel:
    """Tyson cell cycle: simplified polynomial CDK model.

    Polynomial approximation of CDK oscillations:
    dx0/dt = mu0 - mu1*x0 - x0^2*x1  (synthesis - degradation - inhibition)
    dx1/dt = x0^2 - mu1*x1            (activation - degradation)
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="tyson_cell_cycle",
    )
    rhs.set_expressions([
        mu0 - mu1 * x0 - x0**2 * x1,
        x0**2 - mu1 * x1,
    ])

    return BenchmarkModel(
        name="tyson_cell_cycle",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.5, 2.0), (0.3, 1.0)],
        state_domain=[(0.1, 4.0), (0.1, 4.0)],
        rhs_type="polynomial",
        expected_regimes=["oscillatory", "monostable"],
        description="Tyson cell cycle: polynomial CDK/APC feedback model",
    )


def make_mapk_cascade() -> BenchmarkModel:
    """MAPK cascade: polynomial bistable switch model.

    Simplified model of MAPK signaling with cooperative activation
    producing ultrasensitive (bistable) response.
    dx0/dt = mu0*(1-x0)*x0 - mu1*x0     (cooperative activation - deactivation)
    dx1/dt = x0*(1-x1)*x1 - mu1*x1      (downstream cooperative activation)
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="mapk_cascade",
    )
    rhs.set_expressions([
        mu0 * (1 - x0) - mu1 * x0,
        x0 * (1 - x1) - mu1 * x1,
    ])

    return BenchmarkModel(
        name="mapk_cascade",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.2, 2.0), (0.2, 2.0)],
        state_domain=[(0.01, 1.5), (0.01, 1.5)],
        rhs_type="polynomial",
        expected_regimes=["monostable"],
        description="MAPK cascade: polynomial cooperative activation model",
    )


def make_p53_mdm2() -> BenchmarkModel:
    """p53-Mdm2 oscillator: core p53 negative feedback loop.

    Models the tumor-suppressor p53 and its negative regulator Mdm2,
    producing oscillations upon DNA damage signaling.
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="p53_mdm2",
    )
    rhs.set_expressions([
        mu0 - mu1 * x0 * x1 / (1 + x0),
        x0 - x1,
    ])

    return BenchmarkModel(
        name="p53_mdm2",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.5, 2.0), (0.5, 2.0)],
        state_domain=[(0.1, 5.0), (0.1, 5.0)],
        rhs_type="rational",
        expected_regimes=["oscillatory", "monostable"],
        description="p53-Mdm2 oscillator: tumor suppressor negative feedback",
    )


def make_calcium_oscillator() -> BenchmarkModel:
    """Calcium oscillator: polynomial calcium dynamics model.

    Simplified model of intracellular calcium oscillations with
    autocatalytic calcium release and buffering.
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="calcium_oscillator",
    )
    rhs.set_expressions([
        mu0 - x0 + x0**2 * x1,
        mu1 - x0**2 * x1,
    ])

    return BenchmarkModel(
        name="calcium_oscillator",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.5, 2.0), (1.0, 4.0)],
        state_domain=[(0.1, 5.0), (0.1, 5.0)],
        rhs_type="polynomial",
        expected_regimes=["oscillatory", "monostable"],
        description="Calcium oscillator: autocatalytic release with buffering",
    )


def make_circadian_clock() -> BenchmarkModel:
    """Circadian clock: simplified Goldbeter 2-variable model.

    Models the core circadian rhythm generator with transcriptional
    repression via a Hill function of the protein product.
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="circadian_clock",
    )
    rhs.set_expressions([
        mu0 / (1 + x1**4) - mu1 * x0,
        x0 - mu1 * x1,
    ])

    return BenchmarkModel(
        name="circadian_clock",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(1.0, 5.0), (0.5, 2.0)],
        state_domain=[(0.1, 5.0), (0.1, 5.0)],
        rhs_type="hill",
        expected_regimes=["oscillatory", "monostable"],
        description="Circadian clock: Goldbeter transcriptional repression oscillator",
    )


def make_genetic_oscillator() -> BenchmarkModel:
    """Genetic oscillator: 3-gene negative autoregulation with delay-like intermediate.

    A minimal genetic oscillator with a repressor feeding back through
    two intermediate species, creating effective delay for oscillations.
    """
    x0, x1, x2 = sp.symbols('x0 x1 x2')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=3, n_params=2,
        state_symbols=[x0, x1, x2],
        param_symbols=[mu0, mu1],
        name="genetic_oscillator",
    )
    rhs.set_expressions([
        mu0 / (1 + x2**2) - x0,
        x0 - mu1 * x1,
        x1 - mu1 * x2,
    ])

    return BenchmarkModel(
        name="genetic_oscillator",
        n_states=3, n_params=2,
        rhs=rhs,
        parameter_domain=[(2.0, 8.0), (0.5, 1.5)],
        state_domain=[(0.1, 5.0), (0.1, 5.0), (0.1, 5.0)],
        rhs_type="hill",
        expected_regimes=["oscillatory", "monostable"],
        description="Genetic oscillator: 3-gene negative autoregulation with delay",
    )


def make_mutual_activation_switch() -> BenchmarkModel:
    """Mutual activation switch: positive feedback bistable switch.

    Two species mutually activate each other via Hill functions,
    creating a bistable switch with hysteresis.
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="mutual_activation_switch",
    )
    rhs.set_expressions([
        mu0 * x0**2 / (1 + x0**2) + mu1 * x1**2 / (1 + x1**2) - x0,
        mu0 * x1**2 / (1 + x1**2) + mu1 * x0**2 / (1 + x0**2) - x1,
    ])

    return BenchmarkModel(
        name="mutual_activation_switch",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(1.0, 2.5), (1.0, 2.5)],
        state_domain=[(0.01, 4.0), (0.01, 4.0)],
        rhs_type="hill",
        expected_regimes=["bistable", "monostable"],
        description="Mutual activation switch: positive feedback bistable circuit",
    )


def make_van_der_pol() -> BenchmarkModel:
    """van der Pol oscillator: classic relaxation oscillator.

    A nonlinear oscillator with state-dependent damping, widely used
    as a model for biological rhythms and excitable systems.
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="van_der_pol",
    )
    rhs.set_expressions([
        x1,
        mu0 * (1 - x0**2) * x1 - x0 + mu1,
    ])

    return BenchmarkModel(
        name="van_der_pol",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.5, 3.0), (0.0, 1.0)],
        state_domain=[(-3.0, 3.0), (-5.0, 5.0)],
        rhs_type="polynomial",
        expected_regimes=["oscillatory", "monostable"],
        description="van der Pol oscillator: relaxation oscillator with nonlinear damping",
    )


ALL_BENCHMARKS = {
    "toggle_switch": make_toggle_switch,
    "brusselator": make_brusselator,
    "selkov": make_selkov,
    "repressilator": make_repressilator,
    "goodwin": make_goodwin,
    "lotka_volterra": make_lotka_volterra,
    "fitzhugh_nagumo": make_fitzhugh_nagumo,
    "tyson_cell_cycle": make_tyson_cell_cycle,
    "mapk_cascade": make_mapk_cascade,
    "p53_mdm2": make_p53_mdm2,
    "calcium_oscillator": make_calcium_oscillator,
    "circadian_clock": make_circadian_clock,
    "genetic_oscillator": make_genetic_oscillator,
    "mutual_activation_switch": make_mutual_activation_switch,
    "van_der_pol": make_van_der_pol,
}


def get_benchmark(name: str) -> BenchmarkModel:
    if name not in ALL_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(ALL_BENCHMARKS.keys())}")
    return ALL_BENCHMARKS[name]()


def list_benchmarks() -> List[str]:
    return list(ALL_BENCHMARKS.keys())
