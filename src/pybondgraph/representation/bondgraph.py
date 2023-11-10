"""

Bond graph representation of a dynamical system

"""

### Local
from pybondgraph.utils import *
from pybondgraph.representation.base import *

###
import networkx as nx

###

###
###        Bond Graph
###


class BondGraph(nx.DiGraph, DynamicalSystem):
    """Bond graph representation of a dynamical system

    Parameters
    ----------
    NOTE *args, **kwargs are passed on to the `nx.DiGraph` parent class.

    Attributes
    ----------
    equations : Sequence
        Sequence of equations as sympy expressions (implicitely equal to zero).
    derived_equations : Sequence
        Derivatives of `equations`.
    causal_graph : CausalityGraph
        Causal graph used to derive state space equations. Picked among `all_causal_graphs`.
    all_causal_graphs : Sequence[CausalityGraph]
        All causal graphs (for the system's rank).

    var : MultiKeyDict
        Access all variables of the model (by their name or alias, e.g. "p_2_4" and "p_2_4(t)" will point to the same symbol).
    parameters : set
        All fixed parameters of the model.
    functions : set
        All undefined functions of the model.
    derivatives : set
        All derivatives appearing in the model.
    vars : set
        All variables of the model (state and costate variables).
    state_vars : set
        All state variables of the model (generalised displacements and momenta).
    costate_vars : set
        All co-state variables of the model (generalised flows and efforts).
    states : set
        All states of the model (differs from `state_vars` when causal conflicts occur).
    costates : set
        All co-states of the model (differs from `costate_vars` when causal conflicts occur).
    inputs : set
        All inputs of the model (variables associated to flow/effort sources)
    outputs : set
        All outputs of the model (variables associated to flow/effort sinks)

    state_vector : Sequence[sympy.Function]
        Ordered sequence of state symbols.
    input_vector : Sequence[sympy.Function]
        Ordered sequence of input symbols.

    Methods
    -------
    add_node(node: Any, *args, **kwargs)
        Add a node with id `node` and node type `element` (bond graph element).
    add_nodes_from(*args, **kwargs)
        Add multiple nodes at once.
    add_edge(node: Any, *args, **kwargs)
        Connect a pair of nodes (directed).
    add_edges_from(*args, **kwargs)
        Connect multiple pairs of nodes at once (directed).
    state_space()
        Compute state space equations
        (transition model `dX/dt = F(X, U, t)` and observation model `Y = H(X, U, t)`).
    extended_state_space()
        Compute state space equations with extra observations
        (transition model `dX/dt = F(X, U, t)` and observation model `Y = H(X, U, t)`
        and derived observation model `dY/dt = G(X, U, t)`)

    """

    def __init__(self, *args, **kwargs) -> None:
        self.var = MultiKeyDict()
        self._all_atoms = set()  # all atoms (parameters, states, inputs...)
        (
            self.parameters,
            self.functions,
            self.derivatives,
            self.vars,
            self.state_vars,
            self.costate_vars,
            self.states,
            self.costates,
            self.inputs,
            self.outputs,
        ) = new(set, 10)
        self.equations, self.dummy_equations = new(list, 2)
        self.all_causal_graphs = None
        self.causal_graph = None
        super().__init__(*args, **kwargs)

    def add_node(self, node: Any, element: BGNode, *args, **kwargs):
        assert isinstance(node, int), f"Node id must be an integer, not `{node}`."
        # assert "element" in kwargs.keys(), f"Which element is node {node}?"
        kwargs["element"] = element
        parent_output = super().add_node(node, *args, **kwargs)
        self._wrap()
        return parent_output

    def add_nodes_from(self, *args, **kwargs):
        parent_output = super().add_nodes_from(*args, **kwargs)
        self._wrap()
        return parent_output

    def add_edge(self, *args, **kwargs):
        parent_output = super().add_edge(*args, **kwargs)
        self._wrap()
        return parent_output

    def add_edges_from(self, *args, **kwargs):
        parent_output = super().add_edges_from(*args, **kwargs)
        self._wrap()
        return parent_output

    @property
    def state_vector(self) -> Sequence[sympy.Function]:
        return sorted(self.states, key=lambda v: str(v))

    @property
    def input_vector(self) -> Sequence[sympy.Function]:
        return sorted(self.inputs, key=lambda v: str(v))

    def state_space(self):
        """Bond graph to state-space.
        Returns `transition_model` [dX/dt = F(X, U, t)], `observation_model` [Y = H(X, U, t)]
        """
        self._wrap(warn=True)
        # Try to solve [bond graph without causal conflict]
        trans_model, obs_model = eq2ss(
            equations=self.equations,
            states=self.states,
            inputs=self.inputs,
        )
        # Try to solve [bond graph with causal conflicts] (twice as many equations; slower)
        if not ss_is_admissible(trans_model, obs_model, self.states | self.inputs):
            trans_model, obs_model, *_ = self.extended_state_space(wrap=False)
        return trans_model, obs_model

    def extended_state_space(self, wrap=True):
        """Bond graph to 'extended state-space'
        Returns `transition_model` [dX/dt = F(X, U, t)], `observation_model` [Y = H(X, U, t)] and `dy_model` [dY/dt = G(X, U, t)]
        """
        if wrap:
            self._wrap(warn=True)
        trans_model, obs_model, dy_model = eq2ss(
            equations=self.equations + self.derived_equations,
            states=self.states,
            inputs=self.inputs,
            all_models=True,
        )
        return trans_model, obs_model, dy_model

    def _wrap(self, warn=False):
        if self._is_consistent(warn=warn):
            self._generate_equations()
            self._classify_atoms()
            self._make_functions_invertible()

    def _classify_atoms(self):
        """Fetch & distribute atoms (variables, parameters, etc.)
        from the system of equations to the relevant sets."""
        # Reset & fetch variables (and state & co-state varables), parameters, functions FROM equations
        (
            self.parameters,
            self.functions,
            self.derivatives,
            self.vars,
            self.state_vars,
            self.costate_vars,
            self.states,
            self.costates,
            self.inputs,
            self.outputs,
        ) = new(set, 10)
        for eq in self.equations + self.derived_equations:
            self.parameters |= eq.free_symbols - {sympy.Symbol("t")}
            self.derivatives |= eq.atoms(sympy.Derivative)
            for func in eq.atoms(sympy.Function):
                if str(func).startswith(("f_", "e_", "p_", "q_")):
                    self.vars |= {func}
                    if str(func).startswith(("p_", "q_")):
                        self.state_vars |= {func}
                    else:
                        self.costate_vars |= {func}
                else:
                    self.functions |= {
                        f
                        for f in func.atoms(sympy.Function)
                        if not str(f).startswith(("f_", "e_", "p_", "q_"))
                    }

        self.inputs |= self._find_inputs()
        self.outputs |= self._find_outputs()
        self.states |= self._find_states()
        self.costates |= self._find_costates()

        self.vars |= self.inputs | self.outputs

        self._all_atoms = (
            self.parameters
            | self.functions
            | self.derivatives
            | self.vars
            | self.state_vars
            | self.costate_vars
            | self.states
            | self.costates
            | self.inputs
            | self.outputs
        )
        # Warn about duplicates (e.g., sympy function stored as different types)
        for s in [self.parameters, self.vars, self.functions]:
            if len(list(map(str, s))) != len(s):
                log.warning(f"Duplicates in\n{s}")
        # create multi-key dict for each variable
        self.var = MultiKeyDict()
        for atom in self._all_atoms:
            self.var[atom2name(atom), str(atom)] = atom

    def _fetch_var(self, varname: str):
        if varname in list(map(str, self.vars)):
            return [v for v in self.vars if str(v) == varname].pop()

    def _find_inputs(self):
        short2var = {"Se": "e", "Sf": "f"}
        inputs = set()
        for u, v in self.edges():
            elt_in, elt_out = self.nodes[u]["element"], self.nodes[v]["element"]
            if isinstance(elt_in, (SourceFlow, SourceEffort)):
                varname = f"{short2var[elt_in.short_name]}_{u}_{v}(t)"
                inputs |= {self._fetch_var(varname)}
        return inputs

    def _find_outputs(self):
        short2var = {"Se": "e", "Sf": "f"}
        outputs = set()
        for u, v in self.edges():
            elt_in, elt_out = self.nodes[u]["element"], self.nodes[v]["element"]
            if isinstance(elt_out, (SourceFlow, SourceEffort)):
                varname = f"{short2var[elt_out.short_name]}_{u}_{v}(t)"
                outputs |= {self._fetch_var(varname)}
        return outputs

    def _find_states(self):
        shortname2state = {"I": "p", "C": "q"}
        states = set()
        for u, v in self.edges():
            elt_in, elt_out = self.nodes[u]["element"], self.nodes[v]["element"]
            if (
                isinstance(elt_out, (Inductance, Capacitance))
                and v in self.causal_graph.state_nodes
            ):
                varname = f"{shortname2state[elt_out.short_name]}_{u}_{v}(t)"
                states |= {self._fetch_var(varname)}
        return states

    def _find_costates(self):
        shortname2costate = {"I": "f", "C": "e"}
        states = set()
        for u, v in self.edges():
            elt_in, elt_out = self.nodes[u]["element"], self.nodes[v]["element"]
            if isinstance(elt_out, (Inductance, Capacitance)):
                varname = f"{shortname2costate[elt_out.short_name]}_{u}_{v}(t)"
                states |= {self._fetch_var(varname)}
        return states

    def _compute_causality_graphs(self):
        self.all_causal_graphs = CausalityGraph(self).compute()
        self.causal_graph = self.all_causal_graphs[0]

    def _make_functions_invertible(self):
        # Define inverse functions (assuming they exist...) -> ease sympy solve
        f2invf = dict()
        for func_tmp in self.functions:
            stem_funcname = str(func_tmp)[: str(func_tmp).index("(")]
            func, _ = generate_invertible_function(stem_funcname)
            f2invf |= {sympy.Function(stem_funcname): func}
        for ieq, eq in enumerate(self.equations):
            for f, invf in f2invf.items():
                self.equations[ieq] = eq.replace(f, invf)

    def _generate_equations(self):
        """Compute the system of equations from the bond graph representation.
        - All bond graph variables are defined as sympified functions of time.
        - Naming convention for global bond graph variables:
            > [vartype]_[node_in]_[node_out](t)
            > e.g., `e_2_3(t)`
        """
        t = sympy.Symbol("t")
        self.equations = []
        for node_id, element in self.nodes(data="element"):
            eqs = []
            # deqs = []
            in_edges = list(self.in_edges(node_id))
            out_edges = list(self.out_edges(node_id))
            ### Replace local symbols with global names (graph-wise)
            if (
                isinstance(element, BGIdealMultiPortJunction)
                and len(in_edges) > 0
                and len(out_edges) > 0
            ):
                ef, fe = ("e", "f") if isinstance(element, One) else ("f", "e")
                # efforts (flows respectively) sum to zero
                algeb_sum = sympy.sympify(
                    " + ".join(["", *[f"{ef}_{u}_{v}(t)" for u, v in in_edges]])
                    + " - ".join(["", *[f"{ef}_{u}_{v}(t)" for u, v in out_edges]])
                )
                eqs.append(algeb_sum)
                # flows (efforts respectively) are equal (generate minimum number of equations)
                all_edges = in_edges + out_edges
                for (u1, v1), (u2, v2) in zip(all_edges[:-1], all_edges[1:]):
                    eq_local = sympy.sympify(f"{fe}_{u1}_{v1}(t) - {fe}_{u2}_{v2}(t) ")
                    eqs.append(eq_local)
            else:
                for eq_local in element.eq or []:
                    eq_global = copy.deepcopy(eq_local)
                    # replace (local) temporary ids in variable names
                    for symb in eq_global.free_symbols:
                        if tmp_.contains_tmp_id(symb):
                            eq_global = eq_global.replace(
                                symb, sympy.Symbol(f"{tmp_.stem(symb)}{node_id}")
                            )
                    # replace local names by global ones using in/out nodes
                    if (
                        isinstance(element, BGEnergyComponent)
                        and len(in_edges) == 1
                        and len(out_edges) == 0
                    ):
                        in_node_id = in_edges[0][0]
                        for bgvar in ["f", "e", "p", "q"]:
                            eq_global = eq_global.replace(
                                sympy.sympify(f"{bgvar}(t)"),
                                sympy.sympify(f"{bgvar}_{in_node_id}_{node_id}(t)"),
                            )
                    elif (
                        isinstance(element, BGIdealEnergyConverter)
                        and len(in_edges) == 1
                        and len(out_edges) == 1
                    ):
                        in_node_id, out_node_id = in_edges[0][0], out_edges[0][1]
                        for bgvar in ["f_in", "e_in", "p_in", "q_in"]:
                            eq_global = eq_global.replace(
                                sympy.sympify(f"{bgvar}(t)"),
                                sympy.sympify(f"{bgvar[0]}_{in_node_id}_{node_id}(t)"),
                            )
                        for bgvar in ["f_out", "e_out", "p_out", "q_out"]:
                            eq_global = eq_global.replace(
                                sympy.sympify(f"{bgvar}(t)"),
                                sympy.sympify(f"{bgvar[0]}_{node_id}_{out_node_id}(t)"),
                            )

                    eqs.append(eq_global)
            element.eq_global = eqs

        ### Apply causality
        self._compute_causality_graphs()

        ### List all equations, as well as derived equations
        eqs = []
        self.equations = [
            eq for _, elt in self.nodes(data="element") for eq in elt.eq_global
        ]
        self.derived_equations = [eq.diff(t) for eq in self.equations]
        return self.equations

    def _wellconnected(self, node_id, element):
        """Check whether a node is well connected"""
        in_edges = list(self.in_edges(node_id))
        out_edges = list(self.out_edges(node_id))
        if (
            (
                isinstance(element, BGIdealMultiPortJunction)
                and len(in_edges) > 0
                and len(out_edges) > 0
            )
            or (
                isinstance(element, BGEnergyComponent)
                and len(in_edges) == 1
                and len(out_edges) == 0
            )
            or (
                isinstance(element, BGSourceComponent)
                and (
                    (len(in_edges) == 1 and len(out_edges) == 0)
                    or (len(in_edges) == 0 and len(out_edges) == 1)
                )
            )
            or (
                isinstance(element, BGIdealEnergyConverter)
                and len(in_edges) == 1
                and len(out_edges) == 1
            )
        ):
            return True
        return False

    def _is_consistent(self, warn=False):
        """Check that all nodes are properly connected"""
        flag = False
        for node_id, element in self.nodes(data="element"):
            if not self._wellconnected(node_id, element):
                if warn:
                    log.warning(f"Node {node_id} not properly connected.")
                flag = True
        if flag:
            return False
        else:
            return True


###
###        Causality Graph
###


class CausalityGraph(nx.DiGraph):
    """_summary_


    Parameters
    ----------
    bondgraph : BondGraph
        Bond graph onto which causality should be applied.
    NOTE *args, **kwargs are passed on to the `nx.DiGraph` parent class.

    Attributes
    ----------
    rank : int
        Rank of the model.
    state_nodes : Sequence[int]
        Node ids associated to the model's states (i.e., Inductance and Capacitance nodes in integral causality).

    Methods
    -------
    compute()
        Compute causality, i.e., infer all causal graphs and select one.
        Causality assignment steps:
            1. Assign mandatory causality (flow/effort sources)
            For rank in number_of_storage_elements ... 1:
                2. Compute all permutations of storage elements to be assigned
                   with integral/derived causality depending on the rank of the system;
                   Assign causality on Inductance/Capacitance elements accordingly
                   and propagate causality.
                3. Compute all permutations of resistive elements to be assigned with a causality or the other;
                   Assign and propagate causality accordingly.
                4. List all valid causality graphs. If none, decrement the rank.

        NOTE
        - causality assignment is not sensitive to causal conflicts;
        - nonlinear/non-revertible resistances are treated as linear ones (arbitrary causality);
        - all possible causal graphs are computed together with the (highest possible) rank.

        TODO include non-revertible resistances to mandatory assignment
        TODO references / methods [SCAP, MCAP, RCAP...] / remarks

    References
    ----------


    """

    def __init__(self, bondgraph: BondGraph, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Make a snapshot of the bondgraph's elements
        self.add_nodes_from((n, copy.deepcopy(d)) for n, d in bondgraph._node.items())
        self.add_edges_from(
            (u, v, copy.deepcopy(datadict))
            for u, nbrs in bondgraph._adj.items()
            for v, datadict in nbrs.items()
        )
        self._apply_mandatory_causal_strokes()

    @property
    def state_nodes(self) -> Sequence[int]:
        return [n for n in self.nodes() if self._is_in_integral_causality(n)]

    def _is_in_integral_causality(self, node):
        if not isinstance(self.nodes[node]["element"], (Inductance, Capacitance)):
            return None
        _, _, stroke = list(self.in_edges(node, data="stroke"))[0]
        if (
            isinstance(self.nodes[node]["element"], Inductance) and stroke == "out"
        ) or (isinstance(self.nodes[node]["element"], Capacitance) and stroke == "in"):
            return True
        elif (
            isinstance(self.nodes[node]["element"], Inductance) and stroke == "in"
        ) or (isinstance(self.nodes[node]["element"], Capacitance) and stroke == "out"):
            return False

    def _apply_mandatory_causal_strokes(self):
        # TODO include sinks
        for u, v, data in self.edges(data=True):
            if isinstance(self.nodes[u]["element"], SourceEffort):
                data["stroke"] = "out"
            elif isinstance(self.nodes[u]["element"], SourceFlow):
                data["stroke"] = "in"
            else:
                data["stroke"] = None

    @staticmethod
    def _update_graph(graph: nx.DiGraph) -> nx.DiGraph:
        """Propagate causality / update causal strokes with constraints
        Returns the updated graph OR None if a constraint turns out not to be fulfilled
        """
        if graph is None:
            return None
        initial_strokes = [s for _, _, s in graph.edges(data="stroke")]
        ggraph = copy.deepcopy(graph)
        for n, elt in ggraph.nodes(data="element"):
            if isinstance(elt, One):
                bonds = [
                    s
                    for _, _, s in list(ggraph.in_edges(n, data="stroke"))
                    + list(ggraph.out_edges(n, data="stroke"))
                ]
                strong_bond = [
                    s for _, _, s in ggraph.in_edges(n, data="stroke") if s == "in"
                ] + [s for _, _, s in ggraph.out_edges(n, data="stroke") if s == "out"]
                if len(strong_bond) > 1 or (
                    bonds.count(None) == 0 and len(strong_bond) == 0
                ):
                    # More than one strong bond, or zero strong bond despite all nodes being assigned
                    return None
                elif len(strong_bond) == 1:
                    # One strong bond already assigned
                    for _, _, data in ggraph.in_edges(n, data=True):
                        if data["stroke"] is None:
                            data["stroke"] = "out"
                    for _, _, data in ggraph.out_edges(n, data=True):
                        if data["stroke"] is None:
                            data["stroke"] = "in"
                elif bonds.count(None) == 1:
                    # Last unassigned bond is the strong bond
                    for _, _, data in ggraph.in_edges(n, data=True):
                        if data["stroke"] is None:
                            data["stroke"] = "in"
                    for _, _, data in ggraph.out_edges(n, data=True):
                        if data["stroke"] is None:
                            data["stroke"] = "out"

            elif isinstance(elt, Zero):
                bonds = [
                    s
                    for _, _, s in list(ggraph.in_edges(n, data="stroke"))
                    + list(ggraph.out_edges(n, data="stroke"))
                ]
                strong_bond = [
                    s for _, _, s in ggraph.in_edges(n, data="stroke") if s == "out"
                ] + [s for _, _, s in ggraph.out_edges(n, data="stroke") if s == "in"]
                if len(strong_bond) > 1 or (
                    bonds.count(None) == 0 and len(strong_bond) == 0
                ):
                    # More than one strong bond, or zero strong bond despite all nodes being assigned
                    return None
                elif len(strong_bond) == 1:
                    # One strong bond already assigned
                    for _, _, data in ggraph.in_edges(n, data=True):
                        if data["stroke"] is None:
                            data["stroke"] = "in"
                    for _, _, data in ggraph.out_edges(n, data=True):
                        if data["stroke"] is None:
                            data["stroke"] = "out"
                elif bonds.count(None) == 1:
                    # Last unassigned bond is the strong bond
                    for _, _, data in ggraph.in_edges(n, data=True):
                        if data["stroke"] is None:
                            data["stroke"] = "out"
                    for _, _, data in ggraph.out_edges(n, data=True):
                        if data["stroke"] is None:
                            data["stroke"] = "in"

            elif isinstance(elt, GY):
                _, _, in_data = list(ggraph.in_edges(n, data=True))[0]
                _, _, out_data = list(ggraph.out_edges(n, data=True))[0]
                if (in_data["stroke"] == "in" and out_data["stroke"] == "in") or (
                    in_data["stroke"] == "out" and out_data["stroke"] == "out"
                ):
                    return None
                elif (in_data["stroke"] == "in" and out_data["stroke"] == None) or (
                    in_data["stroke"] == None and out_data["stroke"] == "out"
                ):
                    in_data["stroke"] = "in"
                    out_data["stroke"] = "out"
                elif (in_data["stroke"] == "out" and out_data["stroke"] == None) or (
                    in_data["stroke"] == None and out_data["stroke"] == "in"
                ):
                    in_data["stroke"] = "out"
                    out_data["stroke"] = "in"

            elif isinstance(elt, TF):
                _, _, in_data = list(ggraph.in_edges(n, data=True))[0]
                _, _, out_data = list(ggraph.out_edges(n, data=True))[0]
                if (in_data["stroke"] == "in" and out_data["stroke"] == "out") or (
                    in_data["stroke"] == "out" and out_data["stroke"] == "in"
                ):
                    return None
                elif (in_data["stroke"] == "in" and out_data["stroke"] == None) or (
                    in_data["stroke"] == None and out_data["stroke"] == "in"
                ):
                    in_data["stroke"] = "in"
                    out_data["stroke"] = "in"
                elif (in_data["stroke"] == "out" and out_data["stroke"] == None) or (
                    in_data["stroke"] == None and out_data["stroke"] == "out"
                ):
                    in_data["stroke"] = "out"
                    out_data["stroke"] = "out"

        # perform several iterations until the graph stabilises
        updated_strokes = [s for _, _, s in ggraph.edges(data="stroke")]
        if updated_strokes == initial_strokes:
            return ggraph
        return CausalityGraph._update_graph(ggraph)

    @staticmethod
    def _assign_causality_IC(candidates, rank):
        """Assign integral or derivative causality to I/C elements (candidates).
        Provide candidates as [(node, element), ...]."""

        def causality(element, integral_causality):
            if (isinstance(element, Inductance) and integral_causality) or (
                isinstance(element, Capacitance) and not integral_causality
            ):
                return "out"
            return "in"

        one_solution = [True] * rank + [False] * (len(candidates) - rank)
        perms = list(set(itertools.permutations(one_solution, len(candidates))))
        strokes = [
            {c[0]: causality(c[1], p) for c, p in zip(candidates, perm)}
            for perm in perms
        ]
        return strokes

    @staticmethod
    def _assign_causality_R(graph):
        nodes_R_undefined = [
            v
            for u, v, strk in graph.in_edges(data="stroke")
            if strk is None and isinstance(graph.nodes[v]["element"], Resistance)
        ]
        perms = list(itertools.product(["in", "out"], repeat=len(nodes_R_undefined)))
        strokes = [{c: p for c, p in zip(nodes_R_undefined, perm)} for perm in perms]
        return strokes

    @staticmethod
    def _causality_completed(graph):
        return (
            True
            if graph is None
            else not (None in [stroke for _, _, stroke in graph.edges(data="stroke")])
        )

    def _self_copy_causality(self, graph):
        for u, v, data in self.edges(data=True):
            data["stroke"] = graph.edges[u, v]["stroke"]

    def compute(self):
        """Compute the rank and all possible causality graphs"""
        candidates = [
            (n, elt)
            for n, elt in self.nodes(data="element")
            if isinstance(elt, (Inductance, Capacitance))
        ]
        causality_graphs = []
        rank = len(candidates)
        while rank >= 0:
            # Try all possible I/C assignements according to the current `rank`
            for causality_setup in CausalityGraph._assign_causality_IC(
                candidates, rank
            ):
                graph = copy.deepcopy(self)
                for n, strk in causality_setup.items():
                    _, _, data = list(graph.in_edges(n, data=True))[0]
                    data["stroke"] = strk
                graph = CausalityGraph._update_graph(graph)
                if not CausalityGraph._causality_completed(graph):
                    # Try all possible (unassigned) R assignments
                    for causality_R_setup in CausalityGraph._assign_causality_R(graph):
                        ggraph = copy.deepcopy(graph)
                        for n, strk in causality_R_setup.items():
                            _, _, data = list(ggraph.in_edges(n, data=True))[0]
                            data["stroke"] = strk
                        ggraph = CausalityGraph._update_graph(ggraph)
                        if ggraph is not None:
                            causality_graphs.append(ggraph)
                else:
                    if graph is not None:
                        causality_graphs.append(graph)
            if causality_graphs:
                break
            rank -= 1
        self.rank = rank
        self._self_copy_causality(causality_graphs[0])
        return causality_graphs
