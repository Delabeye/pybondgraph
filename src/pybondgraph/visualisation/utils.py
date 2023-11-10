from pybondgraph.utils import *

from pyvis.network import Network
import networkx as nx


def viz_nx_graph(
    graph: nx.Graph,
    attr_as_label: str | tuple[str, Callable] = None,
    attr_as_weight: str | tuple[str, Callable] = None,
    toggle_physics: bool = True,
    buttons: bool = True,
    height="900px",
    width="1800px",
    deepcopy=True,
) -> nx.Graph:
    """_summary_

    Parameters
    ----------
    graph : nx.Graph
        input NetworkX graph
    attr_as_label : str | tuple[str, Callable], optional
        attribute to use as node label (a formatting function can also be passed along). If None, will fetch the `label` attribute, by default None
    attr_as_weight : str | tuple[str, Callable], optional
        attribute to use as edge weight (a formatting function can also be passed along). If None, will fetch the `weight` attribute, by default None
    toggle_physics : bool, optional
        pyvis.Network option, by default True
    buttons : bool, optional
        pyvis.Network option, by default True
    height : str, optional
        display frame height, by default "900px"
    width : str, optional
        display frame width, by default "1800px"

    Returns
    -------
    nx.Graph
        input graph updated with `label` and `weight` data.
    """
    nx_graph = copy.deepcopy(graph) if deepcopy else graph
    lw = {"label": (attr_as_label, "nodes"), "weight": (attr_as_weight, "edges")}
    for k, (attr, node_or_edge) in lw.items():
        for *_, data in getattr(nx_graph, node_or_edge)(data=True):
            if attr is None:
                pass
            elif isinstance(attr, Sequence):
                data[k] = attr[1](data[attr[0]])
            else:
                data[k] = str(attr)

    nt = Network(height, width, directed=True)
    nt.toggle_physics(toggle_physics)
    nt.from_nx(nx_graph)
    nt.set_edge_smooth("dynamic")
    if buttons:
        nt.show_buttons()
    nt.show("nx_tmp.html")
    return nx_graph
