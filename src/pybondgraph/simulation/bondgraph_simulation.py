"""

Simulate a bond graph's behaviour

"""

### Local
from pybondgraph.utils import *
from pybondgraph.representation.base import *
from pybondgraph.representation.bondgraph import BondGraph


def lambdify_model(
    model: dict[str, sympy.Expr],
    args: Sequence[sympy.Symbol],
    params: dict = {},
) -> OrderedDict[sympy.Symbol, Callable]:
    """Lambdify a model (dict of sympy expressions).

    Parameters
    ----------
    model : dict[str, sympy.Expr]
        Dict (as `dict` or `OrderedDict`) of sympy expressions (free of derivatives/integrals).
    args : Sequence[sympy.Symbol]
        Arguments of the lambda functions to be returned.
    params : dict, optional
        Substitute fixed parameters by their value in the model's expressions,
        by default {}.

    Returns
    -------
    dict[sympy.Symbol, Callable]
        Dict (as `dict` or `OrderedDict`) of lambda functions taking `args` as arguments.
        NOTE if the model is provided as an OrderedDict, an order-preserving OrderedDict is returned.
    """

    return type(model)(
        [
            (atom2name(var), sympy.lambdify(args, expr.subs(params)))
            for var, expr in model.items()
            if not expr.atoms(sympy.Derivative, sympy.Integral)
        ]
    )


def simulate(
    bondgraph: BondGraph,
    x0: dict[str, Numeric],
    timespan: np.ndarray,
    control: dict[str, Callable] = {},
    params: dict = {},
    noise: dict = {},
    extended_ss=False,
):
    """Simulate a bond graph model.

    Parameters
    ----------
    bondgraph : BondGraph
        Bond graph model.
    x0 : dict[str, Numeric]
        Initial states.
    timespan : np.ndarray
        Sequence of time steps for which to solve for each state/observable of the bond graph.
    control : dict[str, Callable], optional
        Control/input functions (take state `x` and time step `t`),
        by default {}.
    params : dict, optional
        Values for all fixed parameters within the model,
        by default {}.
    noise : dict, optional
        Standard deviations of zero-mean Gaussian noises to be added to variables of interest (e.g. {"p_2_4": 1.0}),
        by default {}.
    extended_ss : bool, optional
        Whether to compute extended observations (flow/effort derivatives),
        by default False.

    Returns
    -------
    tuple[dict[str, np.ndarray]]
        Sequence of dict gathering time series (model simulated for each time step),
        x_t: states
        y_t: observations
        u_t: inputs
        dy_t, optional: additional observations (flow/effort derivatives), if `extended_ss` is set to `True`
    """

    # prepare: sort, ...
    state_names = list(OrderedDict(sorted(x0.items())).keys())
    input_names = list(OrderedDict(sorted(control.items())).keys())
    x0_sorted = list(OrderedDict(sorted(x0.items())).values())  # sort x0
    control_sorted = list(OrderedDict(sorted(control.items())).values())  # sort control

    # Compute the transition & observation models (symbolic equations)
    if extended_ss:
        trans_model, obs_model, dy_model = bondgraph.extended_state_space()
    else:
        trans_model, obs_model = bondgraph.state_space()

    # Lambdify models
    ctrl = lambda x, t: [c(x, t) for c in control_sorted]
    states_and_inputs = bondgraph.state_vector + bondgraph.input_vector
    lambda_trans = lambdify_model(trans_model, args=states_and_inputs, params=params)
    lambda_obs = lambdify_model(obs_model, args=states_and_inputs, params=params)
    if extended_ss:
        lambda_dy = lambdify_model(dy_model, args=states_and_inputs, params=params)

    # Integrate states, compute inputs
    def func(x, t):
        if control is None:
            return [dxdt(*x) for _, dxdt in lambda_trans.items()]
        return [dxdt(*x, *ctrl(x, t)) for _, dxdt in lambda_trans.items()]

    x_t_arr = scipy.integrate.odeint(func, x0_sorted, timespan)
    u_t_arr = np.array([ctrl(x, t) for x, t in zip(x_t_arr, timespan)])

    # Convert: ndarray to dict
    x_t = OrderedDict([(atom2name(var), x) for var, x in zip(state_names, x_t_arr.T)])
    u_t = OrderedDict([(atom2name(var), u) for var, u in zip(input_names, u_t_arr.T)])

    # Compute observations / extended observations
    y_t = OrderedDict(
        [
            (atom2name(var), np.array([h(*x, *u) for x, u in zip(x_t_arr, u_t_arr)]))
            for var, h in lambda_obs.items()
        ]
    )
    if extended_ss:
        dy_t = OrderedDict(
            [
                (
                    atom2name(var),
                    np.array([g(*x, *u) for x, u in zip(x_t_arr, u_t_arr)]),
                )
                for var, g in lambda_dy.items()
            ]
        )
        sim = x_t, y_t, u_t, dy_t
    else:
        sim = x_t, y_t, u_t

    # Add noise
    for var, std in noise.items():
        for ts in sim:
            if var in ts.keys():
                ts[var] += np.random.normal(0.0, std, ts[var].size)
                break
    return sim
