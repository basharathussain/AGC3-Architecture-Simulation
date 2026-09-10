"""Graph transformation operations — the executable form of G_{t+1} ≠ G_t.

Raw_planning §12 Guideline 3: adaptation must be *real* structural change, not
"choose another agent". Every function here returns a new graph; none mutates.
Only the microkernel calls these, and only on a decision that has already passed
the governance filter.
"""

from __future__ import annotations

from .model import CoordinationGraph


def add_node(g: CoordinationGraph, node: str) -> CoordinationGraph:
    return CoordinationGraph(g.nodes | {node}, g.edges, g.entry)


def remove_node(g: CoordinationGraph, node: str) -> CoordinationGraph:
    if node == g.entry:
        raise ValueError("cannot remove the entry node")
    return CoordinationGraph(
        g.nodes - {node},
        frozenset(e for e in g.edges if node not in e),
        g.entry,
    )


def add_edge(g: CoordinationGraph, src: str, dst: str) -> CoordinationGraph:
    return CoordinationGraph(g.nodes | {src, dst}, g.edges | {(src, dst)}, g.entry)


def remove_edge(g: CoordinationGraph, src: str, dst: str) -> CoordinationGraph:
    return CoordinationGraph(g.nodes, g.edges - {(src, dst)}, g.entry)


def insert_after(g: CoordinationGraph, anchor: str, node: str) -> CoordinationGraph:
    """Splice `node` between `anchor` and every current successor of `anchor`.

    This is the move that activates the Security agent in the paper's worked
    example: Tester -> Reviewer becomes Tester -> Security -> Reviewer. It is
    the operation a supervisor topology cannot express, because a star has no
    notion of "between".
    """
    if anchor not in g.nodes:
        raise ValueError(f"anchor {anchor!r} is not in the graph")
    successors = g.successors(anchor)
    edges = set(g.edges)
    for s in successors:
        edges.discard((anchor, s))
        edges.add((node, s))
    edges.add((anchor, node))
    return CoordinationGraph(g.nodes | {node}, frozenset(edges), g.entry)


def insert_before(g: CoordinationGraph, anchor: str, node: str) -> CoordinationGraph:
    if anchor not in g.nodes:
        raise ValueError(f"anchor {anchor!r} is not in the graph")
    predecessors = g.predecessors(anchor)
    edges = set(g.edges)
    for p in predecessors:
        edges.discard((p, anchor))
        edges.add((p, node))
    edges.add((node, anchor))
    entry = node if anchor == g.entry else g.entry
    return CoordinationGraph(g.nodes | {node}, frozenset(edges), entry)


def reenter(g: CoordinationGraph, source: str, target: str) -> CoordinationGraph:
    """Add a back-edge so execution can return to an earlier stage.

    Re-entry is what turns a failure into a controlled restructuring rather than
    a restart: the shared artefact survives, so the work already done is kept.
    """
    if source not in g.nodes or target not in g.nodes:
        raise ValueError("re-entry endpoints must already be nodes")
    return CoordinationGraph(g.nodes, g.edges | {(source, target)}, g.entry)


def replace_subgraph(
    g: CoordinationGraph,
    removed: set[str],
    added_nodes: set[str],
    added_edges: set[tuple[str, str]],
) -> CoordinationGraph:
    nodes = (g.nodes - removed) | added_nodes
    edges = frozenset(e for e in g.edges if not (set(e) & removed)) | frozenset(added_edges)
    if g.entry in removed:
        raise ValueError("cannot replace the entry node")
    return CoordinationGraph(nodes, edges, g.entry)
