"""

DC Motor
========
This example shows how to use the bond graph representation to model a DC motor.

Takeaways:
    - Declare a bond graph model in this library;
    - Compute a state space model (symbolic equations);
    - Simulate the model;
    - Plot the results.

"""

from pybondgraph.representation.base import *
from pybondgraph.representation.bondgraph import BondGraph
from pybondgraph.simulation.bondgraph_simulation import simulate

sympy.init_printing()  # pretty print sympy expressions

### Declare the bond graph
bg = BondGraph()

bg.add_node(1, element=SourceEffort())
bg.add_node(2, element=One())
bg.add_node(3, element=Resistance())
bg.add_node(4, element=Inductance())
bg.add_node(5, element=GY())
bg.add_node(6, element=One())
bg.add_node(7, element=Resistance())
bg.add_node(8, element=Inductance())

bg.add_edges_from(
    [
        (1, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (5, 6),
        (6, 7),
        (6, 8),
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
print("\n------------------\n\n\n")

### Compute the state space model (symbolic equations)
# transition_model, observation_model = bg.state_space()
transition_model, observation_model, dy_model = bg.extended_state_space()

model_names = ("Transition model", "Observation model", "dy(t)/dt model")
for mn, model in zip(model_names, [transition_model, observation_model, dy_model]):
    print(f"\n---------  {mn}  ---------\n")
    for var, expr in model.items():
        sympy.pprint(sympy.Eq(sympy.sympify(var), expr))
    print("\n---------------------------------------\n")
print("\n\n")

### Run an experiment
timespan = np.linspace(0, 2, 1000)
input = {
    "e_1_2(t)": lambda x, t: 0.0 if t < 1 else 12.0
}  # (step input starting at t=1)
# initial states
x0 = {"p_2_4(t)": 0.0, "p_6_8(t)": 0.0}
# parameters
params = {"I_4": 1e-1, "I_8": 1e-5, "R_3": 1.0, "R_7": 1e-2, "g_5": 1e-1}
# noise
y_noise = {"f_2_4": 0.1, "f_6_8": 0.1}  # observation noise
dx_noise = {"p_2_4": 1e-2, "p_6_8": 1e-5}  # process noise
# sim
x_t, y_t, u_t, dy_t = simulate(
    bg, x0, timespan, input, params, extended_ss=True, noise=dx_noise | y_noise
)

# list available time series
ts_names = ("States x(t)", "Observations y(t)", "Inputs u(t)", "dy(t)/dt")
for tsn, ts in zip(ts_names, (x_t, y_t, u_t, dy_t)):
    print(f"\n---------  {tsn}  ---------\n")
    print(*((k, v.shape) for k, v in ts.items()), sep="\n")
    print("\n---------------------------------------\n")
print("\n\n")

### Plot data
# plot input voltage u(t)
plt.plot(timespan, u_t["e_1_2"], c="red", label=r"e_1_2 : $u(t)$")
# plot state p_2_4(t), current i(t)
plt.plot(timespan, x_t["p_2_4"], c="green", alpha=0.6, label=r"p_2_4")
plt.plot(timespan, y_t["f_2_4"], c="green", label=r"f_2_4 : $i(t)$")
# plot state p_6_8(t), speed omega(t)
plt.plot(timespan, x_t["p_6_8"], c="blue", alpha=0.6, label=r"p_6_8")
plt.plot(timespan, y_t["f_6_8"], c="blue", label=r"f_6_8 : $\omega(t)$")
#
plt.xlabel("time (s)")
plt.legend()
plt.show()
