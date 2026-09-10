"""Audit logger.

The paper requires every topology change to record its triggering evidence, the
policy decision, the graph difference, and the resulting execution state. That
record *is* the auditability metric, so it is produced by the kernel at the
moment of mutation rather than reconstructed afterwards — a reconstruction could
be complete by construction and would measure nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdaptationRecord:
    step: int
    trigger: str
    confidence: float
    candidates: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)   # candidate -> policy id
    scores: dict[str, float] = field(default_factory=dict)
    selected: str | None = None
    graph_before: str = ""
    graph_after: str = ""
    graph_diff: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Every field the paper names must be present and non-trivial."""
        return bool(
            self.trigger
            and self.candidates
            and self.selected
            and self.graph_before
            and self.graph_after
            and self.state_after
        )

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "trigger": self.trigger,
            "confidence": round(self.confidence, 6),
            "candidates": list(self.candidates),
            "rejected": dict(self.rejected),
            "scores": {k: round(v, 6) for k, v in self.scores.items()},
            "selected": self.selected,
            "graph_before": self.graph_before,
            "graph_after": self.graph_after,
            "graph_diff": self.graph_diff,
            "state_after": self.state_after,
        }


class AuditLogger:
    def __init__(self) -> None:
        self._records: list[AdaptationRecord] = []

    def record(self, r: AdaptationRecord) -> None:
        self._records.append(r)

    @property
    def records(self) -> tuple[AdaptationRecord, ...]:
        return tuple(self._records)

    @property
    def completeness(self) -> float:
        """Fraction of adaptation records carrying every required field.

        Metric 12 in V3 §21. Vacuously 1.0 when no adaptation occurred, which is
        correct: a non-adaptive arm has nothing to fail to record.
        """
        if not self._records:
            return 1.0
        return sum(r.is_complete for r in self._records) / len(self._records)

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self._records]
