"""Task specifications — what the agents build and the suites measure.

The architecture never knew anything about REST authentication; it only ever
needed *work units*, two *disjoint* defect families, and a 1:1 mapping from each
defect to one acceptance assertion. This module makes that explicit so the same
runtime can be exercised on more than one authored task.

**Units and defects are plain strings, deliberately.** The original task defined
them as `str`-subclassed enums, whose members compare and hash exactly as their
string values do. Keeping strings therefore leaves the seeded draws in
`agents/sim/world.py` byte-identical for the original task: `_unit(seed,
"inject", "register_broken")` is the same call it always was. That is what lets
the pre-existing results be reproduced exactly after this refactor, which is the
only honest way to add tasks to a study that has already reported numbers.

**Disjointness is enforced, not assumed.** A defect that sits in both suites
would make the derived-confidence weights meaningless, because one cause would
move both terms. `__post_init__` refuses such a spec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    """One acceptance assertion: passes iff its unit exists and its defect is absent."""

    id: str
    description: str
    unit: str
    defect: str


@dataclass(frozen=True)
class TaskSpec:
    key: str
    name: str
    units: tuple[str, ...]
    functional: tuple[Check, ...]
    security: tuple[Check, ...]
    suite_version: str = "1.0.0"
    # Competence profile for this task. Held here rather than in WorldParams'
    # defaults because "how hard is the security surface" is a property of the
    # task, not of the simulator.
    world_overrides: dict[str, float] | None = None

    def __post_init__(self) -> None:
        f, s = self.functional_defects, self.security_defects
        assert not set(f) & set(s), f"{self.key}: suites must not share a cause"
        assert len(set(f)) == len(f) and len(set(s)) == len(s), f"{self.key}: duplicate defect"
        ids = [c.id for c in (*self.functional, *self.security)]
        assert len(set(ids)) == len(ids), f"{self.key}: duplicate check id"
        unknown = {c.unit for c in (*self.functional, *self.security)} - set(self.units)
        assert not unknown, f"{self.key}: checks reference unknown units {sorted(unknown)}"

    @property
    def functional_defects(self) -> tuple[str, ...]:
        return tuple(c.defect for c in self.functional)

    @property
    def security_defects(self) -> tuple[str, ...]:
        return tuple(c.defect for c in self.security)

    @property
    def all_defects(self) -> tuple[str, ...]:
        return (*self.functional_defects, *self.security_defects)

    @property
    def all_units(self) -> frozenset[str]:
        return frozenset(self.units)

    def is_security(self, defect: str) -> bool:
        return defect in set(self.security_defects)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "units": list(self.units),
            "functional_checks": len(self.functional),
            "security_checks": len(self.security),
            "suite_version": self.suite_version,
            "world_overrides": dict(self.world_overrides or {}),
        }
