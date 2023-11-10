"""

Generate dataset from Duffing equation

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


### Simulation parameters
timespan = np.linspace(0, 100, 1000)
# parameters, https://en.wikipedia.org/wiki/Duffing_equation
params = {"delta": 0.3, "alpha": -1.0, "beta": 1.0, "gamma": 0.37, "omega": 1.2}
# noise
noise_std = 3e-2
y_noise = {
    "e_1_2": noise_std,
    "f_2_4": noise_std,
    "e_2_5": noise_std,
    "e_2_3": noise_std,
    "e_2_4": noise_std,
    "df_2_4/dt": noise_std,
}  # observation noise
dx_noise = {"p_2_4": noise_std, "q_2_5": noise_std}  # process noise
init_state_range = {"p_2_4(t)": [-5, 5], "q_2_5(t)": [-np.pi, np.pi]}
# inputs (source)
source = {"e_1_2(t)": lambda x, t: params["gamma"] * np.cos(params["omega"] * t)}

### Generate data
dataset = {}

time_steps = timespan.size
num_samples = 2000

nn_inputs = np.zeros((num_samples, time_steps, 5))
groundtruth = np.zeros((num_samples, time_steps, 3))

for i in tqdm(range(num_samples)):
    x0 = {
        "p_2_4(t)": np.random.uniform(*init_state_range["p_2_4(t)"]),
        "q_2_5(t)": np.random.uniform(*init_state_range["q_2_5(t)"]),
    }

    x_t, y_t, u_t, dy_t = simulate(
        bondgraph=bg,
        x0=x0,
        timespan=timespan,
        control=source,
        params=params,
        noise=y_noise | dx_noise,
        extended_ss=True,
    )

    nn_inputs[i, :, 0] = x_t["q_2_5"]  # Integral(f(t), t)
    nn_inputs[i, :, 1] = y_t["f_2_4"]  # f(t)
    nn_inputs[i, :, 2] = dy_t["df_2_4/dt"]  # Derivative(f(t), t)
    nn_inputs[i, :, 3] = u_t["e_1_2"]  # e_in(t)
    # nn_inputs[i, :, 4] = ...            # e_out(t)     # NOTE no effort out
    groundtruth[i, :, 0] = y_t["e_2_5"]  # e_C(t)
    groundtruth[i, :, 1] = y_t["e_2_3"]  # e_R(t)
    groundtruth[i, :, 2] = y_t["e_2_4"]  # e_I(t)

dataset["data"] = nn_inputs
dataset["groundtruth"] = groundtruth


### Add metadata
dataset |= {
    "time": timespan,
    "params": params
    | {
        "noise": y_noise | dx_noise,
        "init_state_range": init_state_range,
    },
}

### Save dataset
pickle.dump(dataset, open(Path(__file__).parent / "data/duffing037.pickle", "wb"))
