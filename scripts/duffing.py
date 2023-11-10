"""

Duffing Oscillator
==================
This example shows how to use the bond graph representation to model a Duffing oscillator.
Duffing equation: https://en.wikipedia.org/wiki/Duffing_equation

One-node model with:
- e_C(t) = alpha x(t) + beta x^3(t)
- e_I(t) = x_ddot(t)
- e_R(t) = delta x_dot(t)

Takeaways:
    - Declare a bond graph model in this library with custom relationships between variables;
    - Compute a state space model (symbolic equations);
    - Simulate the model;
    - Plot the results.




"""

from pybondgraph.representation.base import *
from pybondgraph.representation.bondgraph import BondGraph
from pybondgraph.simulation.bondgraph_simulation import simulate

import sympy.abc

### Declare the bond graph
bg = BondGraph()

bg.add_node(1, element=SourceEffort())
bg.add_node(2, element=One())
bg.add_node(3, element=Resistance(eq="e(t) = delta * f(t)"))
bg.add_node(4, element=Inductance(eq="f(t) = p(t)"))
bg.add_node(
    5,
    element=Capacitance(
        eq=sympy.sympify(
            "e(t) - alpha * q(t) - beta * q(t)**3", locals=sympy.abc._clash
        )
    ),
)

bg.add_edges_from(
    [
        (1, 2),
        (2, 3),
        (2, 4),
        (2, 5),
    ]
)

### Plot graph
from pybondgraph.visualisation import draw
draw(bg)

### Access diverse variables of the bond graph
print("\n------------------\n")
print("bg.equations=\n", *bg.equations, "\n", sep="\n")
print(f"\n{bg.vars=}")
print(f"\n{bg.state_vars=}")
print(f"\n{bg.costate_vars=}")
print(f"\n{bg.inputs=}")
print(f"\n{bg.parameters=}")
print(f"\n{bg.functions=}")
lin = {True: "linear", False: "nonlinear"}
print(
    *[
        f"\n{str(elt)} {n} is {lin[elt.is_linear]}"
        for n, elt in bg.nodes(data="element")
    ]
)
print("\n------------------\n")

### Compute the state space model (symbolic equations)
transition_model, observation_model, dy_model = bg.extended_state_space()

model_names = ("Transition model", "Observation model", "dy(t)/dt model")
for mn, model in zip(model_names, [transition_model, observation_model, dy_model]):
    print(f"\n---------  {mn}  ---------\n")
    for var, expr in model.items():
        sympy.pprint(sympy.Eq(sympy.sympify(var), expr))
    print("\n---------------------------------------\n")
print("\n\n")

### Run an experiment
timespan = np.linspace(0, 100, 1000)
# parameters, https://en.wikipedia.org/wiki/Duffing_equation
params = {"delta": 0.3, "alpha": -1.0, "beta": 1.0, "gamma": 0.37, "omega": 1.2}
# initial states
x0 = {"p_2_4(t)": 0.0, "q_2_5(t)": 1.0}
# noise
noise_std = 2e-2
y_noise = {
    "f_2_4": noise_std,
    "e_2_5": noise_std,
    "e_2_3": noise_std,
    "e_2_4": noise_std,
    "df_2_4/dt": noise_std,
}  # observation noise
dx_noise = {"p_2_4": noise_std, "q_2_5": noise_std}  # process noise
# inputs
input = {"e_1_2(t)": lambda x, t: params["gamma"] * np.cos(params["omega"] * t)}

x_t, y_t, u_t, dy_t = simulate(
    bg, x0, timespan, input, params, noise=y_noise | dx_noise, extended_ss=True
)

# list available data (collection of univariate time series)
ts_names = ("States x(t)", "Observations y(t)", "Inputs u(t)", "dy(t)/dt")
for tsn, ts in zip(ts_names, (x_t, y_t, u_t, dy_t)):
    print(f"\n---------  {tsn}  ---------\n")
    print(*((k, v.shape) for k, v in ts.items()), sep="\n")
    print("\n---------------------------------------\n")
print("\n\n")

### Plot data
# plot input force F(t)
plt.plot(
    timespan,
    u_t["e_1_2"],
    c="red",
    label=r"e_1_2 : $\Sigma e_{in} - \Sigma e_{out} = F(t) = \delta \cos(\omega t)$",
)
# plot momentum (mass=1) x_dot(t), acceleration x_ddot(t)
plt.plot(
    timespan,
    x_t["p_2_4"],
    c="green",
    alpha=0.6,
    label=r"p_2_4 = f_2_4 : $\dot{x}(t)$ (speed)",
)
plt.plot(
    timespan,
    # y_t["f_2_4"],
    dy_t["df_2_4/dt"],
    c="green",
    label=r"df_2_4/dt : $e_I(t) = \ddot{x}(t)$ (accel)",
)
# plot displacement x(t), speed x_dot(t)
plt.plot(timespan, x_t["q_2_5"], c="blue", alpha=0.6, label=r"q_2_5 : $x(t)$")
plt.plot(
    timespan, y_t["e_2_5"], c="blue", label=r"e_2_5 : $e_C(t) = \alpha x + \beta x^3$"
)
#
plt.xlabel("time (s)")
plt.legend()
plt.show()
