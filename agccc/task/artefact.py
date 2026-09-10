"""The artefact under construction.

This is the object the agents build and the suites measure. It is deliberately a
plain data structure with no behaviour: agents mutate it, suites read it, and
nothing else in the system may do either.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from .defects import ALL_ENDPOINTS, Defect, Endpoint


@dataclass(frozen=True)
class Artefact:
    """An immutable snapshot of the generated API.

    Frozen so that a run's trace holds genuinely distinct states rather than
    repeated references to one mutated object — the reproducibility record
    (V3 §19.1) stores per-round artefacts and they must not alias.
    """

    endpoints: frozenset[Endpoint] = frozenset()
    defects: frozenset[Defect] = frozenset()

    def with_endpoints(self, added: frozenset[Endpoint] | set[Endpoint]) -> "Artefact":
        return replace(self, endpoints=self.endpoints | frozenset(added))

    def without_defects(self, removed: frozenset[Defect] | set[Defect]) -> "Artefact":
        return replace(self, defects=self.defects - frozenset(removed))

    def with_defects(self, added: frozenset[Defect] | set[Defect]) -> "Artefact":
        return replace(self, defects=self.defects | frozenset(added))

    @property
    def is_complete(self) -> bool:
        return self.endpoints == ALL_ENDPOINTS

    def to_dict(self) -> dict:
        return {
            "endpoints": sorted(e.value for e in self.endpoints),
            "defects": sorted(d.value for d in self.defects),
        }

    def digest(self) -> str:
        """Stable hash, so an artefact can be identified in the audit trail."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


EMPTY = Artefact()
