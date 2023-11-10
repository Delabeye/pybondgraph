import pickle
from matplotlib.pyplot import *

### Local
from pybondgraph.utils import *


dataset = pickle.load(open(Path(__file__).parent / "data/duffing037.pickle", "rb"))

sample_index = 915
# Plot inputs x(t), x_dot(t), x_ddot(t), F(t)
plt.figure()
plt.plot(
    dataset["time"],
    dataset["data"][sample_index, :, 0],
    label=r"$x(t)$",
    lw=1,
    c="b",
    alpha=0.6,
)
plt.plot(
    dataset["time"],
    dataset["data"][sample_index, :, 1],
    label=r"$\dot{x}(t)$",
    lw=2,
    c="r",
)
plt.plot(
    dataset["time"],
    dataset["data"][sample_index, :, 2],
    label=r"$\ddot{x}(t)$",
    lw=1,
    c="k",
    alpha=0.6,
)
plt.ylabel("position, speed, acceleration", fontsize=18)
plt.xlabel("time (s)", fontsize=18)
plt.legend()

# Plot phase portrait
plt.figure()
plt.plot(dataset["data"][sample_index, :, 0], dataset["data"][sample_index, :, 1])
plt.ylabel(r"$\dot{x}$")
plt.xlabel(r"$x$")
plt.legend()

# Plot powers P_C(t), P_R(t), P_I(t), P_in(t)
plt.figure()
e_loss = dataset["groundtruth"][sample_index, :, :]  # e_C, e_R, e_I
e_loss_sum = np.sum(e_loss, axis=1)[:, None]
e_in = dataset["data"][sample_index, :, 3][:, None]
flow = dataset["data"][sample_index, :, 1][:, None]
plt.plot(
    dataset["time"],
    e_loss[:, 0][:, None] * flow,
    alpha=0.5,
    label=r"$f \times e_C = \dot{x} (\alpha x + \beta x^3)$",
    c="b",
)
plt.plot(
    dataset["time"],
    e_loss[:, 1][:, None] * flow,
    lw=2,
    alpha=1.0,
    label=r"$f \times e_R = \dot{x} (\delta \dot{x})$",
    c="r",
)
plt.plot(
    dataset["time"],
    e_loss[:, 2][:, None] * flow,
    alpha=0.5,
    label=r"$f \times e_I = \dot{x} (\ddot{x})$",
    c="k",
)

plt.ylabel("power (W)", fontsize=18)
plt.xlabel("time (s)", fontsize=18)
plt.legend()

plt.show()
