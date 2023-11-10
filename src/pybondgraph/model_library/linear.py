"""

Library of linear bond graph models

"""

from pybondgraph.representation.base import *
from pybondgraph.representation.bondgraph import BondGraph


###
###     DC Motor
###

DCMOTOR = BondGraph()

DCMOTOR.add_node(1, element=SourceEffort())
DCMOTOR.add_node(2, element=One())
DCMOTOR.add_node(3, element=Resistance())
DCMOTOR.add_node(4, element=Inductance())
DCMOTOR.add_node(5, element=GY())
DCMOTOR.add_node(6, element=One())
DCMOTOR.add_node(7, element=Resistance())
DCMOTOR.add_node(8, element=Inductance())

DCMOTOR.add_edges_from(
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
