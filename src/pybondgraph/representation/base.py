"""

Base classes and tools to represent dynamical systems

"""

### Local
from pybondgraph.utils import *

###

import sympy

###

sympy.init_printing()  # pretty print sympy expressions


"""

Toolbox

"""


def generate_invertible_function(funcname: str) -> tuple[sympy.Function]:
    """Generate an invertible sympy function and its inverse.

    Parameters
    ----------
    funcname : str
        Function name.

    Returns
    -------
    tuple[sympy.Function]
        Invertible function and its inverse.

    """

    def _eval_simplify(self, *args, **kwargs):
        """Workaround to recover: x = f(f_inv(x)) = f_inv(f(x)) FIXME won't work with e.g., f(f_inv(3*x))"""
        if isinstance(self.args[0], self.inverse()):
            return self.args[0].args[0]
        return self

    f = type(funcname, (sympy.Function,), {})
    f_inv = type(f"{funcname}_inv", (sympy.Function,), {})
    f._eval_simplify = _eval_simplify
    f_inv._eval_simplify = _eval_simplify
    f.inverse = lambda self, argindex=1: f_inv
    f_inv.inverse = lambda self, argindex=1: f
    return f, f_inv


"""Dummify an equation.
    `equations` : equations to dummify
    `vars`: variables to dummify (e.g., sympy.sympify("f(t)") ) 
    Returns `dummified_eqs`, `atom2dummy`, `dummy2atom`
"""


def dummify(
    equations: Sequence[sympy.Expr], all_vars: Sequence[sympy.Symbol | sympy.Function]
) -> Sequence[sympy.Expr]:
    """Dummify equations,
    i.e., replace variables (possibly functions, e.g., `f(t)`), derivatives and integrals by dummy symbols.
    Returns `dummified_eqs`, `atom2dummy`, `dummy2atom`

    Parameters
    ----------
    equations : Sequence[sympy.Expr]
        Sequence of equations (as sympy expressions) to be dummified.
    all_vars : Sequence[sympy.Symbol | sympy.Function]
        Variables to dummify. Necessary to keep undefined functions while dummifying variables of interest.
        (e.g., sympy.sympify("f(t)"))

    Returns
    -------
    Sequence[sympy.Expr]
        Dummified equations.
    dict[str, sympy.Symbol]
        atom2dummy
    dict[str, sympy.Symbol]
        dummy2atom
    """

    diff2dummy = {
        sym: sympy.Dummy()
        for eq in equations
        for sym in eq.atoms(sympy.Derivative, sympy.Integral)
    }
    sym2dummy = {sym: sympy.Dummy() for sym in all_vars}
    dummified_eqs = [eq.subs(diff2dummy).subs(sym2dummy) for eq in equations]
    dummy2atom = {v: k for k, v in (sym2dummy | diff2dummy).items()}
    return dummified_eqs, sym2dummy | diff2dummy, dummy2atom


def ss_is_admissible(
    transition_model, observation_model, states_and_inputs, verbose=False
):
    ss_variables = [str(v) for v in (transition_model | observation_model).keys()]
    states = [str(si) for si in states_and_inputs]
    for ss_var, expr in (transition_model | observation_model).items():
        for var_in_expr in expr.atoms(sympy.Function):
            if str(var_in_expr) in ss_variables and str(var_in_expr) not in states:
                if verbose:
                    log.warning(
                        f"State-Space not admissible (expression does not depend solely on states and inputs).\n{ss_var} = {expr}"
                    )
                return False
    return True


"""Symbolically convert equations into nonlinear state space form.
    `states` : states of the system
    `inputs` : inputs of the system (or outputs as exogeneous inputs)
    NOTE "f_", "e_", "p_", "q_" relate to variable prefixes (bond graph state & costate notations)
    Returns `transition_model` and `observation_model`
    NOTE can also return `dy_model` dY/dt = G(X, U, t) -> computed when the derived equations are provided
    // and `du_model` dU/dt = D(X, U, t) (--> useless)
    """


def eq2ss(
    equations: Sequence[sympy.Expr],
    states: set[sympy.Symbol | sympy.Function],
    inputs: set[sympy.Symbol | sympy.Function],
    all_models: bool = False,
):
    """Compute the state-space model of a system of equations.

    Parameters
    ----------
    equations : Sequence[sympy.Expr]
        Sequence of equations (as sympy expressions).
    states : set[sympy.Symbol | sympy.Function]
        System states.
    inputs : set[sympy.Symbol | sympy.Function]
        System inputs.
    all_models : bool, optional
        If set to True, extra observations are returned (dy(t)/dt),
        by default False

    Returns
    -------
    tuple[OrderedDict[str, sympy.Expr]]
        transition model `dX/dt = F(X, U, t)` ; observation model `Y = H(X, U, t)` ;
        derived observation model `dY/dt = G(X, U, t)`)
    """

    t = sympy.Symbol("t")

    # List all variables (sympy.Function) from the equations, and their first & second derivatives
    all_vars = {
        var
        for eq in equations
        for var in eq.atoms(sympy.Function)
        if str(var).startswith(("f_", "e_", "p_", "q_"))
    }
    all_vars_derivatives = {var.diff(t) for var in all_vars} | {
        var.diff(t, 2) for var in all_vars
    }

    dummified_eqs, atom2dummy, dummy2atom = dummify(equations, all_vars)
    # `solve_for`: all variables except states & inputs ; `pd_undefined`: partial derivatives of undefined functions
    solve_for, pd_undefined = [], []
    for eq in dummified_eqs:
        for var in eq.free_symbols:
            if var in dummy2atom.keys() and var not in set(
                [atom2dummy[v] for v in states | inputs]
            ):
                if dummy2atom[var] in all_vars | all_vars_derivatives:
                    solve_for.append(var)
                else:
                    pd_undefined.append(var)
    solve_for, pd_undefined = list(set(solve_for)), list(set(pd_undefined))
    sol = sympy.nonlinsolve(dummified_eqs, solve_for)

    transition_model, observation_model, dy_model = new(OrderedDict, 3)
    for var, expr in zip([var.subs(dummy2atom) for var in solve_for], list(sol)[0]):
        if var in [s.diff(t) for s in states]:
            transition_model[atom2name(var)] = sympy.simplify(expr.subs(dummy2atom))
        elif var in all_vars:
            observation_model[atom2name(var)] = sympy.simplify(expr.subs(dummy2atom))
        elif var in [s.diff(t) for s in inputs]:
            # du_model[atom2name(var)] = sympy.simplify(expr.subs(dummy2atom))
            ...
        else:
            dy_model[atom2name(var)] = sympy.simplify(expr.subs(dummy2atom))
    transition_model = OrderedDict(sorted(transition_model.items()))
    observation_model = OrderedDict(sorted(observation_model.items()))
    dy_model = OrderedDict(sorted(dy_model.items()))
    if all_models:
        return transition_model, observation_model, dy_model
    return transition_model, observation_model


def atom2name(atom: sympy.Expr | str) -> str:
    """Compute a sympy variable's name."""
    atom_ = atom if isinstance(atom, sympy.Expr) else sympy.sympify(atom)
    if isinstance(atom_, sympy.Symbol):
        return str(atom_)
    elif isinstance(atom_, sympy.Function):
        return str(atom_)[: str(atom_).index("(")]
    elif isinstance(atom_, sympy.Derivative):
        if atom_.args[1][1] > 1:
            return f"d{str(atom_.args[1][1])}{str(atom_.args[0])}/d{str(atom_.args[1][0])}{str(atom_.args[1][1])}"
        return f"d{str(atom2name(atom_.args[0]))}/d{str(atom_.args[1][0])}"
    else:
        return atom


def name2atom(name: str) -> sympy.Expr:
    """Create variable from name."""
    atom = sympy.sympify(name)
    if name.endswith("(t)") or isinstance(atom, (sympy.Derivative, sympy.Integral)):
        return atom
    else:
        return sympy.sympify(f"{name}(t)")


"""

Base classes

"""


class DynamicalSystem:
    """Base class for dynamical systems"""


###
###     Bond Graph components
###


class BGNode:
    """Bond Graph nodes act as aggregators.
    A physical behaviour `eq` is attached to each node (Sequence of equations as `str` or sympy expression `sympy.Expr`)
    """

    def __init__(self, eq: Optional[Sequence[str | sympy.Expr]] = None) -> None:
        self._eq = None
        self.eq = eq
        self.eq_global = None
        self.deq_global = None

    @property
    def eq(self):
        return self._eq

    @eq.setter
    def eq(self, expr: Sequence[str | sympy.Expr]):
        """Sympify node equations."""
        if expr is not None:
            self._eq_str = [e for e in expr if isinstance(e, str)]
            self._eq = [
                np.diff(sympy.sympify(eq.split("=")))[0]
                if isinstance(eq, str) and "=" in eq
                else sympy.sympify(eq)
                if isinstance(eq, str)
                else eq
                for eq in expr
            ]

    @property
    def is_linear(self):
        if self.eq is None:
            return True
        try:
            for eq in self.eq:
                all_vars = {
                    var
                    for var in eq.atoms(sympy.Function)
                    if str(var).startswith(
                        (
                            "p(t)",
                            "q(t)",
                            "e(t)",
                            "f(t)",
                            "e_in(t)",
                            "f_in(t)",
                            "e_out(t)",
                            "f_out(t)",
                        )
                    )
                }
                all_vars_derivatives = {var.diff(sympy.Symbol("t")) for var in all_vars}
                for var in [
                    "p(t)",
                    "q(t)",
                    "e(t)",
                    "f(t)",
                    "e_in(t)",
                    "f_in(t)",
                    "e_out(t)",
                    "f_out(t)",
                ]:
                    sympy_var = sympy.sympify(var)
                    dummy_eq, atom2dummy, _ = dummify(
                        [eq], all_vars | all_vars_derivatives
                    )
                    if sympy_var in atom2dummy.keys():
                        sympy.linsolve(dummy_eq, [atom2dummy[sympy_var]])
            return True
        except:
            return False

    @abstractproperty
    def short_name(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__


class BGJunction(BGNode):
    """Junction node: One, Zero, GY, TF"""


class BGIdealEnergyConverter(BGJunction):
    """Ideal Energy Converters (junction node): gyrators GY and transformers TF.
    (GY and TF can exhibit nonlinear behaviours)."""

    @BGJunction.eq.setter
    def eq(self, expr: Sequence[str | sympy.Expr]):
        assert not (expr is None or len(expr) != 2), "Two equations required for `eq`"
        super(BGIdealEnergyConverter, type(self)).eq.fset(self, expr)


class BGIdealMultiPortJunction(BGJunction):
    """Ideal Multi-Port junction node: 1- and 0- junctions."""


class BGComponent(BGNode):
    """Component node: R, I, C, SourceEffort, SourceFlow"""


class BGSourceComponent(BGComponent):
    """Source Component node: Se, Sf"""

    @BGComponent.eq.setter
    def eq(self, expr):
        assert expr is None, "Sources admit no behavioural equation"
        super(BGSourceComponent, type(self)).eq.fset(self, expr)


class BGEnergyComponent(BGComponent):
    """Energy Component node: R, I, C
    if no `eq` is provided, defaults to a linear component"""

    @BGComponent.eq.setter
    def eq(self, expr: Sequence[str | sympy.Expr]):
        if isinstance(expr, str | sympy.Expr):
            expr = [expr]
        assert not (expr is None or len(expr) != 1), "Single equation required for `eq`"
        if isinstance(self, Inductance):
            expr += [sympy.sympify(f"Derivative(p(t), t) - e(t)")]
        elif isinstance(self, Capacitance):
            expr += [sympy.sympify(f"Derivative(q(t), t) - f(t)")]
        super(BGEnergyComponent, type(self)).eq.fset(self, expr)


class Resistance(BGEnergyComponent):
    """`eq`: `f(t) = phi(e(t))`"""

    @property
    def short_name(self) -> str:
        return "R"

    @BGEnergyComponent.eq.setter
    def eq(self, expr: Sequence[str | sympy.Expr]):
        if expr is None:
            expr = f"f(t) = e(t) / R_{tmp_.id()}"
        super(Resistance, type(self)).eq.fset(self, expr)


class Capacitance(BGEnergyComponent):
    """`eq`: `e(t) = phi(q(t))`"""

    short_name = "C"

    @BGEnergyComponent.eq.setter
    def eq(self, expr: Sequence[str | sympy.Expr]):
        if expr is None:
            expr = f"e(t) = q(t) / C_{tmp_.id()}"
        super(Capacitance, type(self)).eq.fset(self, expr)

    # @property
    # def is_linear(self):
    #     return self._is_linear_wrt(["e(t)", "q(t)"])


class Inductance(BGEnergyComponent):
    """`eq`: `f(t) = phi(p(t))`"""

    short_name = "I"

    @BGEnergyComponent.eq.setter
    def eq(self, expr: Sequence[str | sympy.Expr]):
        if expr is None:
            expr = f"f(t) = p(t) / I_{tmp_.id()}"
        super(Inductance, type(self)).eq.fset(self, expr)


class SourceEffort(BGSourceComponent):
    """Source of effort."""

    short_name = "Se"


class SourceFlow(BGSourceComponent):
    """Source of flow."""

    short_name = "Sf"


class One(BGIdealMultiPortJunction):
    """One-junction: connected flows are equal, efforts algebraically sum to zero."""

    short_name = "One"


class Zero(BGIdealMultiPortJunction):
    """One-junction: connected efforts are equal, flows algebraically sum to zero."""

    short_name = "Zero"


class GY(BGIdealEnergyConverter):
    """Gyrator: relationship between flow in - effort out, and effort in - flow out.
    `eq`: [`e_out = phi(f_in)`, `e_in = theta(f_out)`] such that f_in * e_in = f_out * e_out.
    """

    short_name = "GY"

    @BGIdealEnergyConverter.eq.setter
    def eq(self, expr: Sequence[str | sympy.Expr]):
        if expr is None:
            expr = [
                f"e_out(t) = g_{tmp_.id()} * f_in(t)",
                f"e_in(t) = g_{tmp_.id()} * f_out(t)",
            ]
        super(GY, type(self)).eq.fset(self, expr)


class TF(BGIdealEnergyConverter):
    """Transformer: relationship between flow in - flow out, and effort in - effort out.
    `eq`: [`f_out(t) = phi(f_in(t))`, `e_in(t) = theta(e_out(t))`] such that f_in(t) * e_in(t) = f_out(t) * e_out(t).
    """

    short_name = "TF"

    @BGIdealEnergyConverter.eq.setter
    def eq(self, expr: Sequence[str | sympy.Expr]):
        if expr is None:
            expr = [
                f"f_out(t) = r_{tmp_.id()} * f_in(t)",
                f"e_in(t) = r_{tmp_.id()} * e_out(t)",
            ]
        super(TF, type(self)).eq.fset(self, expr)
