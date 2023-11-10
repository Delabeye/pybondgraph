"""

Library of nonlinear bond graph models

"""

from pybondgraph.representation.base import *
from pybondgraph.representation.bondgraph import BondGraph


###
###     Duffing
###

DUFFING = BondGraph()

DUFFING.add_node(1, element=SourceEffort())
DUFFING.add_node(2, element=One())
DUFFING.add_node(3, element=Resistance(eq="e(t) = delta * f(t)"))
DUFFING.add_node(4, element=Inductance(eq="f(t) = p(t)"))
DUFFING.add_node(
    5,
    element=Capacitance(
        eq=sympy.sympify(
            "e(t) - alpha * q(t) - beta * q(t)**3", locals=sympy.abc._clash
        )
    ),
)

DUFFING.add_edges_from(
    [
        (1, 2),
        (2, 3),
        (2, 4),
        (2, 5),
    ]
)

###
###     Simple Centrifugal Pump
###

PUMP = BondGraph()

PUMP.add_node(1, element=SourceEffort())
PUMP.add_node(2, element=One())
PUMP.add_node(3, element=Resistance())
PUMP.add_node(4, element=Inductance())
PUMP.add_node(5, element=GY())
PUMP.add_node(6, element=One())
PUMP.add_node(7, element=Resistance())
PUMP.add_node(8, element=Inductance())
PUMP.add_node(
    9,
    element=GY(
        eq=[
            "e_out(t) = (K1 * f_in(t) - K2 * f_out(t)) * f_in(t)",
            "e_in(t) = (K1 * f_in(t) - K2 * f_out(t)) * f_out(t)",
        ]
    ),
)
PUMP.add_node(10, element=One())
PUMP.add_node(11, element=Resistance(eq="xi * f(t)**2 = e(t)"))
PUMP.add_node(12, element=Capacitance())
PUMP.add_node(13, element=SourceEffort())


PUMP.add_edges_from(
    [
        (1, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (5, 6),
        (6, 7),
        (6, 8),
        (6, 9),
        (9, 10),
        (10, 11),
        (10, 12),
        (13, 10),
    ]
)
