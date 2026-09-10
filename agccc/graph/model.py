"""The coordination graph as a first-class runtime object.

Raw_planning §12 Guideline 2 is binding here: orchestration is never expressed
as Python call order. The active topology is this object, the executor reads it
to decide who runs next, and the only way to change who runs next is to change
this object through `graph.ops`.

    G_t = (V_t, E_t),  V_t ⊆ A,  E_t ⊆ V_t × V_t        (paper Eq. 1)

Frozen, so that G_t and G_{t+1} genuinely coexist and can be diffed. A mutable
graph would make `graph_before` and `graph_after` in the audit record the same
object, and the auditability metric would be measuring nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinationGraph:
    nodes: frozenset[str]
    edges: frozenset[tuple[str, str]]
    entry: str

    def __post_init__(self) -> None:
        if self.entry not in self.nodes:
            raise ValueError(f"entry {self.entry!r} is not a node")
        for src, dst in self.edges:
            if src not in self.nodes or dst not in self.nodes:
                raise ValueError(f"edge ({src}, {dst}) references an unknown node")

    # --- queries --------------------------------------------------------

    def successors(self, node: str) -> tuple[str, ...]:
        return tuple(sorted(d for s, d in self.edges if s == node))

    def predecessors(self, node: str) -> tuple[str, ...]:
        return tuple(sorted(s for s, d in self.edges if d == node))

    def is_terminal(self, node: str) -> bool:
        return not self.successors(node)

    @property
    def order(self) -> tuple[str, ...]:
        """Linear execution order, when the graph is a simple chain.

        Returns the chain from `entry` while each node has exactly one
        successor. Used for presentation and for the sequential strategies;
        the adaptive strategy does not rely on it, because its graphs branch
        and re-enter.
        """
        seen: list[str] = [self.entry]
        node = self.entry
        while True:
            succ = self.successors(node)
            if len(succ) != 1 or succ[0] in seen:
                return tuple(seen)
            node = succ[0]
            seen.append(node)

    # --- identity -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": sorted(self.nodes),
            "edges": sorted([list(e) for e in self.edges]),
            "entry": self.entry,
        }

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def __str__(self) -> str:
        chain = self.order
        if len(chain) == len(self.nodes) and len(self.edges) == len(self.nodes) - 1:
            return " -> ".join(chain)
        return f"V={sorted(self.nodes)} E={sorted(tuple(e) for e in self.edges)}"


def chain(*nodes: str) -> CoordinationGraph:
    """Build a linear pipeline: a -> b -> c."""
    if not nodes:
        raise ValueError("a chain needs at least one node")
    return CoordinationGraph(
        nodes=frozenset(nodes),
        edges=frozenset(zip(nodes, nodes[1:])),
        entry=nodes[0],
    )


def star(hub: str, *spokes: str) -> CoordinationGraph:
    """Build a supervisor topology: hub <-> each spoke.

    This is B2's shape. The hub can route to any spoke and each spoke returns
    to the hub — that is agent *selection*, and it is the most a supervisor can
    express. It cannot place one spoke after another, which is the structural
    move B2 is missing and AGCCC has.
    """
    edges = {(hub, s) for s in spokes} | {(s, hub) for s in spokes}
    return CoordinationGraph(
        nodes=frozenset((hub,) + spokes),
        edges=frozenset(edges),
        entry=hub,
    )
