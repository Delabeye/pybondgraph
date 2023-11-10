### Local
from pybondgraph.utils import *
from pybondgraph.representation.base import *
from pybondgraph.visualisation.utils import *
from pybondgraph.representation.bondgraph import BondGraph, CausalityGraph


def draw(bondgraph: BondGraph | CausalityGraph):
    graph = nx.DiGraph()
    ### Format a static representation of the bond graph
    elt_panel = {
        "R": "red",
        "C": "green",
        "I": "green",
        "Se": "blue",
        "Sf": "blue",
    }
    get_elt = lambda n: bondgraph.nodes[n]["element"]
    for n, data in bondgraph.nodes(data=True):
        elt = data["element"].short_name
        node_name = f"{elt}_{n}"
        node_color = elt_panel.get(elt, "white")
        font_color = "white" if node_color != "white" else "black"
        options = {
            "label": node_name,
            "shape": "box",
            "color": node_color,
            "font": {"color": font_color},
        }
        graph.add_node(n, **options)
    for u, v, data in bondgraph.edges(data=True):
        edge_color = "black"
        if isinstance(elt_u := get_elt(u), BGComponent):
            edge_color = elt_panel.get(elt_u.short_name, "black")
        elif isinstance(elt_v := get_elt(v), BGComponent):
            edge_color = elt_panel.get(elt_v.short_name, "black")
        options = {"color": edge_color, "arrows": {"to": {"enabled": True}}}
        graph.add_edge(u, v, **options)

    ### Format a static causality representation of the bond graph
    if isinstance(bondgraph, CausalityGraph):
        ...

    viz_nx_graph(graph, deepcopy=False)
