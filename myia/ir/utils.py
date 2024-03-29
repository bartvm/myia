"""Utilities for manipulating and inspecting the IR."""

from typing import Any, Iterable, Set

from ..graph_utils import EXCLUDE, FOLLOW, NOFOLLOW, dfs as _dfs, \
    toposort as _toposort
from .anf import ANFNode, Apply, Constant, Graph, Parameter, Special

#######################
# Successor functions #
#######################


def succ_deep(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming and graph references.

    A node's successors are its `incoming` set, or the return node of a graph
    when a graph Constant is encountered.
    """
    if is_constant_graph(node):
        return [node.value.return_] if node.value.return_ else []
    else:
        return node.incoming


def succ_deeper(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming and graph references.

    Unlike `succ_deep` this visits all encountered graphs thoroughly, including
    those found through free variables.
    """
    if is_constant_graph(node):
        return [node.value.return_] if node.value.return_ else []
    elif node.graph:
        return list(node.incoming) + [node.graph.return_]
    else:
        return node.incoming


def succ_incoming(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming."""
    return node.incoming


#################################
# Inclusion/exclusion functions #
#################################


def exclude_from_set(stops):
    """Avoid visiting nodes in the stops set."""
    if not isinstance(stops, (set, frozenset, dict)):
        stops = frozenset(stops)

    def include(node):
        return EXCLUDE if node in stops else FOLLOW

    return include


def freevars_boundary(graph, include_boundary=True):
    """Stop visiting when encountering free variables.

    Arguments:
        graph: The main graph from which we want to include nodes.
        include_boundary: Whether to yield the free variables or not.
    """
    def include(node):
        g = node.graph
        if g is None or g is graph:
            return FOLLOW
        elif include_boundary:
            return NOFOLLOW
        else:
            return EXCLUDE

    return include


#####################
# Search algorithms #
#####################


def dfs(root: ANFNode, follow_graph: bool = False) -> Iterable[ANFNode]:
    """Perform a depth-first search."""
    return _dfs(root, succ_deep if follow_graph else succ_incoming)


def toposort(root: ANFNode) -> Iterable[ANFNode]:
    """Order the nodes topologically."""
    return _toposort(root, succ_incoming)


###############
# Isomorphism #
###############


def _same_node_shallow(n1, n2, equiv):
    # Works for Constant, Parameter and nodes previously seen
    if n1 in equiv and equiv[n1] is n2:
        return True
    elif is_constant_graph(n1) and is_constant_graph(n2):
        # Note: we provide current equiv so that nested graphs can properly
        # match their free variables, using the equiv of their parent graph.
        return isomorphic(n1.value, n2.value, equiv)
    elif is_constant(n1):
        return n1.value == n2.value
    elif is_parameter(n1):
        # Parameters are matched together in equiv when we ask whether two
        # graphs are isomorphic. Therefore, we only end up here when trying to
        # match free variables.
        return False
    else:
        raise TypeError(n1)  # pragma: no cover


def _same_node(n1, n2, equiv):
    # Works for Apply (when not seen previously) or other nodes
    if is_apply(n1):
        return all(_same_node_shallow(i1, i2, equiv)
                   for i1, i2 in zip(n1.inputs, n2.inputs))
    else:
        return _same_node_shallow(n1, n2, equiv)


def _same_subgraph(root1, root2, equiv):
    # Check equivalence between two subgraphs, starting from root1 and root2,
    # using the given equivalence dictionary. This is a modified version of
    # toposort that walks the two graphs in lockstep.

    done: Set = set()
    todo = [(root1, root2)]

    while todo:
        n1, n2 = todo[-1]
        if n1 in done:
            todo.pop()
            continue
        cont = False

        s1 = list(succ_incoming(n1))
        s2 = list(succ_incoming(n2))
        if len(s1) != len(s2):
            return False
        for i, j in zip(s1, s2):
            if i not in done:
                todo.append((i, j))
                cont = True

        if cont:
            continue
        done.add(n1)

        res = _same_node(n1, n2, equiv)
        if res:
            equiv[n1] = n2
        else:
            return False

        todo.pop()

    return True


def isomorphic(g1, g2, equiv=None):
    """Return whether g1 and g2 are structurally equivalent.

    Constants are isomorphic iff they contain the same value or are isomorphic
    graphs.

    g1.return_ and g2.return_ must represent the same node under the
    isomorphism. Parameters must match in the same order.
    """
    if equiv and (g1, g2) in equiv:
        return equiv[(g1, g2)] is not False

    if len(g1.parameters) != len(g2.parameters):
        return False

    prev_equiv = equiv
    equiv = dict(zip(g1.parameters, g2.parameters))
    if prev_equiv:
        equiv.update(prev_equiv)

    equiv[(g1, g2)] = 'PENDING'
    rval = _same_subgraph(g1.return_, g2.return_, equiv)
    equiv[(g1, g2)] = rval

    return rval


##################
# Misc utilities #
##################


def is_apply(x: ANFNode) -> bool:
    """Return whether x is an Apply."""
    return isinstance(x, Apply)


def is_parameter(x: ANFNode) -> bool:
    """Return whether x is a Parameter."""
    return isinstance(x, Parameter)


def is_constant(x: ANFNode, cls: Any = object) -> bool:
    """Return whether x is a Constant, with value of given cls."""
    return isinstance(x, Constant) and isinstance(x.value, cls)


def is_constant_graph(x: ANFNode) -> bool:
    """Return whether x is a Constant with a Graph value."""
    return is_constant(x, Graph)


def is_special(x: ANFNode, cls: Any = object) -> bool:
    """Return whether x is a Special, with value of given cls."""
    return isinstance(x, Special) and isinstance(x.special, cls)
