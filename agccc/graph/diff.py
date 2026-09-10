"""Graph difference — the "graph difference" the paper requires every topology
change to log alongside its triggering evidence and policy decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import CoordinationGraph


@dataclass(frozen=True)
class GraphDiff:
    nodes_added: tuple[str, ...]
    nodes_removed: tuple[str, ...]
    edges_added: tuple[tuple[str, str], ...]
    edges_removed: tuple[tuple[str, str], ...]

    @property
    def is_empty(self) -> bool:
        return not (
            self.nodes_added or self.nodes_removed or self.edges_added or self.edges_removed
        )

    @property
    def magnitude(self) -> int:
        """Number of elementary changes — feeds the transition cost κ."""
        return (
            len(self.nodes_added)
            + len(self.nodes_removed)
            + len(self.edges_added)
            + len(self.edges_removed)
        )

    def to_dict(self) -> dict:
        return {
            "nodes_added": list(self.nodes_added),
            "nodes_removed": list(self.nodes_removed),
            "edges_added": [list(e) for e in self.edges_added],
            "edges_removed": [list(e) for e in self.edges_removed],
        }

    def __str__(self) -> str:
        if self.is_empty:
            return "(no change)"
        parts = (
            [f"+{n}" for n in self.nodes_added]
            + [f"-{n}" for n in self.nodes_removed]
            + [f"+{s}->{d}" for s, d in self.edges_added]
            + [f"-{s}->{d}" for s, d in self.edges_removed]
        )
        return ", ".join(parts)


def diff(before: CoordinationGraph, after: CoordinationGraph) -> GraphDiff:
    return GraphDiff(
        nodes_added=tuple(sorted(after.nodes - before.nodes)),
        nodes_removed=tuple(sorted(before.nodes - after.nodes)),
        edges_added=tuple(sorted(after.edges - before.edges)),
        edges_removed=tuple(sorted(before.edges - after.edges)),
    )
