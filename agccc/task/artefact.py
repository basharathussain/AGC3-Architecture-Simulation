"""The artefact under construction.

This is the object the agents build and the suites measure. It is deliberately a
plain data structure with no behaviour: agents mutate it, suites read it, and
nothing else in the system may do either.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace



@dataclass(frozen=True)
class Artefact:
    """An immutable snapshot of the generated API.

    Frozen so that a run's trace holds genuinely distinct states rather than
    repeated references to one mutated object — the reproducibility record
    (V3 §19.1) stores per-round artefacts and they must not alias.
    """

    endpoints: frozenset[str] = frozenset()
    defects: frozenset[str] = frozenset()

    @property
    def units(self) -> frozenset[str]:
        """Task-neutral name for `endpoints`.

        The field keeps its original name so that stored run records and the
        reproducibility manifest stay readable against the results already
        published; new code reads `units`, which is what it always meant.
        """
        return self.endpoints

    def with_endpoints(self, added) -> "Artefact":
        return replace(self, endpoints=self.endpoints | frozenset(added))

    def without_defects(self, removed) -> "Artefact":
        return replace(self, defects=self.defects - frozenset(removed))

    def with_defects(self, added) -> "Artefact":
        return replace(self, defects=self.defects | frozenset(added))

    def is_complete_for(self, spec) -> bool:
        return self.endpoints >= spec.all_units

    @property
    def is_complete(self) -> bool:
        """Completeness against the default task. New code uses `is_complete_for`."""
        from .catalogue import DEFAULT_TASK
        return self.is_complete_for(DEFAULT_TASK)

    def to_dict(self) -> dict:
        return {
            "endpoints": sorted(str(e) for e in self.endpoints),
            "defects": sorted(str(d) for d in self.defects),
        }

    def digest(self) -> str:
        """Stable hash, so an artefact can be identified in the audit trail."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


EMPTY = Artefact()
