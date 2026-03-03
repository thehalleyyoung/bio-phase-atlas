"""
Benchmark biological model definitions.

Each model provides:
  - Symbolic RHS via SymPy
  - Parameter domain for phase-atlas construction
  - State domain for equilibrium search
  - Expected regimes (for validation, not certification)
"""

import os
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


def make_michaelis_menten_switch() -> BenchmarkModel:
    """Michaelis-Menten enzyme switch: cooperative enzyme kinetics.

    A bistable switch arising from cooperative substrate inhibition,
    modeled via rational (Michaelis-Menten) kinetics.
    dx0/dt = mu0 * x0 / (1 + x0 + x0^2) - mu1 * x0
    dx1/dt = x0 - mu1 * x1
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="michaelis_menten_switch",
    )
    rhs.set_expressions([
        mu0 * x0 / (1 + x0 + x0**2) - mu1 * x0,
        x0 - mu1 * x1,
    ])

    return BenchmarkModel(
        name="michaelis_menten_switch",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(1.0, 4.0), (0.3, 1.5)],
        state_domain=[(0.01, 5.0), (0.01, 5.0)],
        rhs_type="rational",
        expected_regimes=["monostable", "bistable"],
        description="Michaelis-Menten enzyme switch: cooperative substrate inhibition",
    )


def make_erk_bistable() -> BenchmarkModel:
    """ERK bistable switch: simplified ERK signaling with positive feedback.

    Models ultrasensitive ERK activation with cooperative feedback.
    dx0/dt = mu0 * (1-x0) / (1 + (1-x0)) - x0 / (1 + x0) + mu1
    dx1/dt = x0 - x1
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="erk_bistable",
    )
    rhs.set_expressions([
        mu0 * (1 - x0) / (1 + (1 - x0)) - x0 / (1 + x0) + mu1,
        x0 - x1,
    ])

    return BenchmarkModel(
        name="erk_bistable",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.5, 3.0), (0.0, 0.5)],
        state_domain=[(0.01, 2.0), (0.01, 2.0)],
        rhs_type="rational",
        expected_regimes=["monostable", "bistable"],
        description="ERK bistable switch: ultrasensitive activation with feedback",
    )


def make_apoptosis_switch() -> BenchmarkModel:
    """Apoptosis toggle: simplified caspase activation switch.

    Models the caspase-3/IAP positive feedback loop creating a
    bistable life/death decision in apoptosis.
    dx0/dt = mu0 * x0^2 / (1 + x0^2) - mu1 * x0 * x1
    dx1/dt = mu0 / (1 + x0^2) - mu1 * x1
    """
    x0, x1 = sp.symbols('x0 x1')
    mu0, mu1 = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x0, x1],
        param_symbols=[mu0, mu1],
        name="apoptosis_switch",
    )
    rhs.set_expressions([
        mu0 * x0**2 / (1 + x0**2) - mu1 * x0 * x1,
        mu0 / (1 + x0**2) - mu1 * x1,
    ])

    return BenchmarkModel(
        name="apoptosis_switch",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(1.5, 3.5), (0.5, 1.5)],
        state_domain=[(0.1, 4.0), (0.1, 4.0)],
        rhs_type="hill",
        expected_regimes=["monostable", "bistable"],
        description="Apoptosis toggle: caspase-3/IAP bistable life/death switch",
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
    "michaelis_menten_switch": make_michaelis_menten_switch,
    "erk_bistable": make_erk_bistable,
    "apoptosis_switch": make_apoptosis_switch,
}


def get_benchmark(name: str) -> BenchmarkModel:
    if name not in ALL_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(ALL_BENCHMARKS.keys())}")
    return ALL_BENCHMARKS[name]()


def list_benchmarks() -> List[str]:
    return list(ALL_BENCHMARKS.keys())


def _mathml_to_sympy(element, nsmap, local_dict):
    """Convert a MathML element to a SymPy expression.

    Supports the subset: apply, ci, cn, plus, minus, times, divide,
    power, csymbol, and common unary functions (exp, ln, log, abs,
    sin, cos, sqrt).  This covers ~90% of BioModels rate laws.
    """
    tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

    if tag == 'cn':
        text = (element.text or '').strip()
        if element.get('type') == 'integer':
            return sp.Integer(int(text))
        if element.get('type') == 'e-notation':
            # <cn type="e-notation">1.5<sep/>3</cn>  →  1.5e3
            sep = element.find(f'{nsmap}sep') if nsmap else element.find('sep')
            if sep is not None and sep.tail:
                return sp.Float(float(text + 'e' + sep.tail.strip()))
            return sp.Float(float(text))
        return sp.Float(float(text)) if '.' in text else sp.Integer(int(text))

    if tag == 'ci':
        name = (element.text or '').strip()
        if name in local_dict:
            return local_dict[name]
        sym = sp.Symbol(name)
        local_dict[name] = sym
        return sym

    if tag == 'csymbol':
        name = (element.text or '').strip()
        if name in local_dict:
            return local_dict[name]
        sym = sp.Symbol(name)
        local_dict[name] = sym
        return sym

    if tag == 'apply':
        children = list(element)
        op_tag = children[0].tag.split('}')[-1] if '}' in children[0].tag else children[0].tag
        args = [_mathml_to_sympy(c, nsmap, local_dict) for c in children[1:]]

        ops = {
            'plus': lambda a: a[0] if len(a) == 1 else sum(a[1:], a[0]),
            'minus': lambda a: -a[0] if len(a) == 1 else a[0] - a[1],
            'times': lambda a: a[0] if len(a) == 1 else sp.Mul(*a),
            'divide': lambda a: a[0] / a[1],
            'power': lambda a: a[0] ** a[1],
            'exp': lambda a: sp.exp(a[0]),
            'ln': lambda a: sp.log(a[0]),
            'log': lambda a: sp.log(a[0]),
            'abs': lambda a: sp.Abs(a[0]),
            'sin': lambda a: sp.sin(a[0]),
            'cos': lambda a: sp.cos(a[0]),
            'sqrt': lambda a: sp.sqrt(a[0]),
            'root': lambda a: sp.sqrt(a[0]),
        }
        if op_tag in ops:
            return ops[op_tag](args)
        raise ValueError(f"Unsupported MathML operator: {op_tag}")

    if tag == 'piecewise':
        # Take the first <piece> branch as a simplification
        piece = element.find(f'{nsmap}piece') if nsmap else element.find('piece')
        if piece is not None:
            return _mathml_to_sympy(list(piece)[0], nsmap, local_dict)

    if tag in ('math', 'lambda', 'semantics', 'annotation', 'annotation-xml'):
        for child in element:
            return _mathml_to_sympy(child, nsmap, local_dict)

    raise ValueError(f"Unsupported MathML element: {tag}")


def load_sbml_file(path: str) -> BenchmarkModel:
    """Load an ODE model from an SBML file (Level 2 or Level 3).

    Uses only ``xml.etree.ElementTree`` (stdlib).  Extracts species,
    parameters, and kinetic rate laws.  MathML rate laws are converted
    to SymPy expressions via a simplified parser covering the common
    subset (plus, minus, times, divide, power, cn, ci, exp, ln).

    The returned ``BenchmarkModel`` has *all* SBML-level parameters
    treated as fixed constants baked into the rate expressions.  To
    sweep a parameter, list it in the optional ``<annotation>``
    ``parameter_bounds`` block (see ``examples/lotka_volterra.sbml``).

    Falls back to ``state_domain`` of ``[0, 10]`` per species and
    ``rhs_type = "general"`` unless overridden in the SBML annotation.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()

    # Detect SBML namespace
    ns = ''
    root_tag = root.tag
    if '}' in root_tag:
        ns = root_tag.split('}')[0] + '}'

    mathml_ns = '{http://www.w3.org/1998/Math/MathML}'

    # ---- Extract model element -------------------------------------------
    model_el = root.find(f'{ns}model')
    if model_el is None:
        raise ValueError("SBML file has no <model> element")
    model_name = model_el.get('id') or model_el.get('name') or 'sbml_model'

    # ---- Extract compartments (for volume scaling) -----------------------
    compartments: Dict[str, float] = {}
    list_of_comp = model_el.find(f'{ns}listOfCompartments')
    if list_of_comp is not None:
        for comp in list_of_comp.findall(f'{ns}compartment'):
            cid = comp.get('id')
            size = float(comp.get('size', '1'))
            compartments[cid] = size

    # ---- Extract species -------------------------------------------------
    species_list = []
    species_initial: Dict[str, float] = {}
    species_compartment: Dict[str, str] = {}
    list_of_species = model_el.find(f'{ns}listOfSpecies')
    if list_of_species is None:
        raise ValueError("SBML file has no <listOfSpecies>")
    for sp_el in list_of_species.findall(f'{ns}species'):
        sid = sp_el.get('id')
        species_list.append(sid)
        ic = sp_el.get('initialConcentration') or sp_el.get('initialAmount') or '0'
        species_initial[sid] = float(ic)
        species_compartment[sid] = sp_el.get('compartment', '')

    # ---- Extract global parameters ---------------------------------------
    global_params: Dict[str, float] = {}
    list_of_params = model_el.find(f'{ns}listOfParameters')
    if list_of_params is not None:
        for p in list_of_params.findall(f'{ns}parameter'):
            pid = p.get('id')
            val = p.get('value')
            if val is not None:
                global_params[pid] = float(val)

    # ---- Build symbol dicts ----------------------------------------------
    all_symbols: Dict[str, sp.Symbol] = {}
    for s in species_list:
        all_symbols[s] = sp.Symbol(s)
    for p in global_params:
        all_symbols[p] = sp.Symbol(p)
    for c in compartments:
        all_symbols[c] = sp.Symbol(c)

    # ---- Extract reactions & build ODE RHS per species -------------------
    ode_rhs: Dict[str, sp.Expr] = {s: sp.Integer(0) for s in species_list}
    list_of_reactions = model_el.find(f'{ns}listOfReactions')
    if list_of_reactions is None:
        raise ValueError("SBML file has no <listOfReactions>")

    for rxn in list_of_reactions.findall(f'{ns}reaction'):
        # Parse local parameters into symbol dict (override globals in scope)
        rxn_symbols = dict(all_symbols)
        kinetic_law = rxn.find(f'{ns}kineticLaw')
        if kinetic_law is None:
            continue
        local_params = kinetic_law.find(f'{ns}listOfParameters') or \
                       kinetic_law.find(f'{ns}listOfLocalParameters')
        if local_params is not None:
            for lp in local_params:
                lp_tag = lp.tag.split('}')[-1] if '}' in lp.tag else lp.tag
                if lp_tag in ('parameter', 'localParameter'):
                    lpid = lp.get('id')
                    lpval = lp.get('value')
                    if lpval is not None:
                        rxn_symbols[lpid] = sp.Float(float(lpval))
                    else:
                        rxn_symbols[lpid] = sp.Symbol(lpid)

        # Parse MathML rate expression
        math_el = kinetic_law.find(f'{mathml_ns}math')
        if math_el is None:
            continue
        rate_expr = _mathml_to_sympy(math_el, mathml_ns, rxn_symbols)

        # Substitute fixed local-parameter *values*
        for lsym_name, lsym in rxn_symbols.items():
            if isinstance(lsym, sp.Float):
                rate_expr = rate_expr.subs(sp.Symbol(lsym_name), lsym)

        # Distribute rate to reactants / products via stoichiometry
        reactants_el = rxn.find(f'{ns}listOfReactants')
        if reactants_el is not None:
            for sr in reactants_el.findall(f'{ns}speciesReference'):
                sid = sr.get('species')
                stoich = float(sr.get('stoichiometry', '1'))
                if sid in ode_rhs:
                    ode_rhs[sid] -= stoich * rate_expr

        products_el = rxn.find(f'{ns}listOfProducts')
        if products_el is not None:
            for sr in products_el.findall(f'{ns}speciesReference'):
                sid = sr.get('species')
                stoich = float(sr.get('stoichiometry', '1'))
                if sid in ode_rhs:
                    ode_rhs[sid] += stoich * rate_expr

    # ---- Substitute global parameter values into rate expressions --------
    subs_map = {sp.Symbol(p): sp.Float(v) for p, v in global_params.items()}
    for c, v in compartments.items():
        subs_map[sp.Symbol(c)] = sp.Float(v)

    # Determine which parameters are *varying* (have bounds annotation)
    # Default: bake everything in; annotation can override.
    varying_params: Dict[str, Tuple[float, float]] = {}

    # Check for a custom annotation block (optional)
    annot = model_el.find(f'{ns}annotation')
    if annot is not None:
        for child in annot:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child_tag == 'phaseCartographer':
                for pc_child in child:
                    pc_tag = pc_child.tag.split('}')[-1] if '}' in pc_child.tag else pc_child.tag
                    if pc_tag == 'parameterBounds':
                        pid = pc_child.get('id')
                        lo = float(pc_child.get('lower', '0'))
                        hi = float(pc_child.get('upper', '1'))
                        varying_params[pid] = (lo, hi)
                    elif pc_tag == 'stateDomain':
                        pass  # handled below
                    elif pc_tag == 'rhsType':
                        pass

    # Remove varying params from substitution map
    for p in varying_params:
        subs_map.pop(sp.Symbol(p), None)

    # Apply substitutions
    for s in species_list:
        ode_rhs[s] = sp.nsimplify(ode_rhs[s].subs(subs_map), rational=False)

    # If no varying params specified, promote all globals as fixed (n_params=0
    # doesn't make sense), so pick the first global param as varying [0.5x, 2x].
    if not varying_params and global_params:
        first = next(iter(global_params))
        val = global_params[first]
        lo = val * 0.5 if val > 0 else val - 1.0
        hi = val * 2.0 if val > 0 else val + 1.0
        varying_params[first] = (lo, hi)
        # un-substitute it
        for s in species_list:
            ode_rhs[s] = ode_rhs[s].subs(sp.Float(val), sp.Symbol(first))

    n_states = len(species_list)
    n_params = len(varying_params)
    param_names = list(varying_params.keys())

    state_symbols = [sp.Symbol(s) for s in species_list]
    param_symbols = [sp.Symbol(p) for p in param_names]

    rhs = SymbolicRHS(
        n_states=n_states, n_params=n_params,
        state_symbols=state_symbols,
        param_symbols=param_symbols,
        name=model_name,
    )
    expressions = [ode_rhs[s] for s in species_list]
    rhs.set_expressions(expressions)

    param_domain = [varying_params[p] for p in param_names]

    # State domain: from annotation or default
    state_domain_list: List[Tuple[float, float]] = []
    if annot is not None:
        for child in annot:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child_tag == 'phaseCartographer':
                for pc_child in child:
                    pc_tag = pc_child.tag.split('}')[-1] if '}' in pc_child.tag else pc_child.tag
                    if pc_tag == 'stateDomain':
                        for sd_child in pc_child:
                            sd_tag = sd_child.tag.split('}')[-1] if '}' in sd_child.tag else sd_child.tag
                            if sd_tag == 'bound':
                                lo = float(sd_child.get('lower', '0'))
                                hi = float(sd_child.get('upper', '10'))
                                state_domain_list.append((lo, hi))
    if not state_domain_list:
        state_domain_list = [(0.0, 10.0)] * n_states

    # rhs_type from annotation
    rhs_type = 'general'
    if annot is not None:
        for child in annot:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child_tag == 'phaseCartographer':
                for pc_child in child:
                    pc_tag = pc_child.tag.split('}')[-1] if '}' in pc_child.tag else pc_child.tag
                    if pc_tag == 'rhsType':
                        rhs_type = (pc_child.text or 'general').strip()

    return BenchmarkModel(
        name=model_name,
        n_states=n_states,
        n_params=n_params,
        rhs=rhs,
        parameter_domain=param_domain,
        state_domain=state_domain_list,
        rhs_type=rhs_type,
        expected_regimes=[],
        description=f"SBML model loaded from {os.path.basename(path)}",
    )


def load_model_file(path: str) -> BenchmarkModel:
    """Load a user-defined ODE model from a YAML, JSON, or SBML file.

    SBML files (``.sbml`` or ``.xml`` with an ``<sbml>`` root) are
    handled by :func:`load_sbml_file`.  Everything else is treated as
    YAML/JSON.

    The file must define:
      name: str
      states: list of variable name strings
      parameters: list of {name, bounds: [lo, hi]}
      equations: list of SymPy-parseable RHS strings (one per state)
      state_domain: list of [lo, hi] bounds for each state
      rhs_type: str (optional, default "general")
      expected_regimes: list of str (optional)
      description: str (optional)
    """
    if path.endswith('.sbml'):
        return load_sbml_file(path)
    if path.endswith('.xml'):
        # Peek at the root tag to auto-detect SBML
        import xml.etree.ElementTree as ET
        try:
            root_tag = ET.parse(path).getroot().tag
            if 'sbml' in root_tag.lower():
                return load_sbml_file(path)
        except ET.ParseError:
            pass  # Not XML; fall through to YAML/JSON

    import yaml

    with open(path) as f:
        if path.endswith('.json'):
            import json
            spec = json.load(f)
        else:
            spec = yaml.safe_load(f)

    # Validate required fields
    for field in ('name', 'states', 'parameters', 'equations', 'state_domain'):
        if field not in spec:
            raise ValueError(f"Model file missing required field: '{field}'")

    state_names = spec['states']
    n_states = len(state_names)

    param_entries = spec['parameters']
    n_params = len(param_entries)
    param_names = [p['name'] if isinstance(p, dict) else p for p in param_entries]
    param_domain = [tuple(p['bounds']) if isinstance(p, dict) else tuple(p)
                    for p in param_entries]

    state_symbols = [sp.Symbol(s) for s in state_names]
    param_symbols = [sp.Symbol(p) for p in param_names]

    rhs = SymbolicRHS(
        n_states=n_states, n_params=n_params,
        state_symbols=state_symbols,
        param_symbols=param_symbols,
        name=spec['name'],
    )

    # Parse equation strings into SymPy expressions
    local_dict = {s.name: s for s in state_symbols + param_symbols}
    expressions = [sp.sympify(eq, locals=local_dict) for eq in spec['equations']]
    rhs.set_expressions(expressions)

    state_domain = [tuple(b) for b in spec['state_domain']]

    return BenchmarkModel(
        name=spec['name'],
        n_states=n_states,
        n_params=n_params,
        rhs=rhs,
        parameter_domain=param_domain,
        state_domain=state_domain,
        rhs_type=spec.get('rhs_type', 'general'),
        expected_regimes=spec.get('expected_regimes', []),
        description=spec.get('description', ''),
    )


def validate_model_file(path: str) -> List[str]:
    """Parse and validate a model file, returning a list of issues (empty = valid)."""
    issues = []
    try:
        bm = load_model_file(path)
    except Exception as e:
        return [f"Parse error: {e}"]

    if bm.n_states < 1:
        issues.append("Model must have at least 1 state variable")
    if bm.n_params < 1:
        issues.append("Model must have at least 1 parameter")
    if len(bm.rhs.expressions) != bm.n_states:
        issues.append(f"Expected {bm.n_states} equations, got {len(bm.rhs.expressions)}")
    if len(bm.state_domain) != bm.n_states:
        issues.append(f"state_domain has {len(bm.state_domain)} entries, expected {bm.n_states}")
    if len(bm.parameter_domain) != bm.n_params:
        issues.append(f"parameter_domain has {len(bm.parameter_domain)} entries, expected {bm.n_params}")

    # Check that expressions only use declared symbols
    declared = set(s.name for s in bm.rhs.state_symbols + bm.rhs.param_symbols)
    for i, expr in enumerate(bm.rhs.expressions):
        free = {str(s) for s in expr.free_symbols}
        unknown = free - declared
        if unknown:
            issues.append(f"Equation {i} uses undeclared symbols: {unknown}")

    return issues
